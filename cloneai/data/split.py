"""
- Convert all audio to the same format and sampling rate (might need to batch use raw.dat or re-sample)
- Break up audio into short sentence phrases
    - cut audio whenever there is a gap of more than couple of seconds
- For consistency into whisper and training make sure audio files:
    - Are in WAV format (not necessary, can be flac as well)
    - Have a sample rate of 16kHz (this should be done)
    - Are mono-channel (this should also be done)
"""

import re
import subprocess
import shutil
import os 


def get_duration(infile):
    """Command to extract time duration of ffmpeg input file
    
    Interestingly this seemed to give a different value than the end_time 
    obtained in the get_silences() function

    This is because this just uses the metadata, but not the actual 
    data which is processed when we calculate the silences
    """
    cmd = ["ffprobe",
        "-i", infile,
        "-show_entries", "format=duration",
        "-v", "quiet", "-of", "csv=p=0",
    ]
    result = subprocess.run(cmd, capture_output=True)
    duration = result.stdout.decode("utf-8").strip()
    return float(duration)

def get_sample_rate(filename):
    cmd = ["ffprobe",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        "-i",
        "pipe:"]
    result = subprocess.run(cmd, capture_output=True, text=True, stdin=filename)
    sr = result.stdout.strip()
    return sr


def get_silences(infile, silence_db=-60, min_silence_s=2.0, verbose=False):
    """Use ffmpeg to detect silence regions of audio
    
    This is based on an example using ffmpeg-python located here:
    https://github.com/kkroening/ffmpeg-python/blob/master/examples/split_silence.py
    
    Some differences include avoiding ffmpeg-python library (raw subprocess calls)
    and also just collecting the raw silence start/end timestamps instead of converting that 
    to chunk_start and chunk_end times (this is done later). It's less efficient because
    we have to reiterate through the array but for development purposes it's easier
    """

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        infile,
        "-filter_complex",
        f"silencedetect=d={min_silence_s}:n={silence_db}dB",
        "-f",
        "null",
        "-"
    ]

    if verbose:
        print(" ".join(cmd))

    silence_start_re = re.compile(r' silence_start: (?P<start>[0-9]+(\.?[0-9]*))$')
    silence_end_re = re.compile(r' silence_end: (?P<end>[0-9]+(\.?[0-9]*)) ')
    total_duration_re = re.compile(r'size=[^ ]+ time=(?P<hours>[0-9]{2}):(?P<minutes>[0-9]{2}):(?P<seconds>[0-9\.]{5}) bitrate=')

    result = subprocess.run(cmd, capture_output=True)
    lines = result.stderr.decode("utf-8").splitlines()

    starts = []
    ends = []
    end_time = float("inf")

    for line in lines:
        if verbose:
            print(line)

        silence_start_match = silence_start_re.search(line)
        silence_end_match = silence_end_re.search(line)
        total_duration_match = total_duration_re.search(line)

        if silence_start_match:
            starts.append(float(silence_start_match.group('start')))
        elif silence_end_match:
            ends.append(float(silence_end_match.group('end')))
        elif total_duration_match:
            hours = int(total_duration_match.group('hours'))
            minutes = int(total_duration_match.group('minutes'))
            seconds = float(total_duration_match.group('seconds'))
            end_time = hours * 3600 + minutes * 60 + seconds

    assert len(starts) >= len(ends), "Less number of starts than ends should never occur"

    # to assure that len(starts) == len(ends) for iteration later
    # this handles ending with silence
    if len(starts) > len(ends):
        ends.append(end_time) 

    return starts, ends, end_time



def convert_silences_to_nonsilences(starts, ends, end_time, min_nonsilence_s=2):
    """Convert all audio files to specific format and split for ML training"""

    len_starts = len(starts)
    len_ends = len(ends)

    splits = []

    assert len_starts == len_ends, "Silence detection did not work correctly because uneven lengths"

    if len_starts == 0: # implicitly == len_ends
        return splits
    for i in range(len_starts):
        if i == 0 and starts[0] != 0:
            # handles non-silence at beginning of track
            ss = 0.0
            to = starts[0]
        elif i+1 == len_starts:
            # handles end of track (avoids out-of-range indexing)
            if ends[i] > end_time:
                # because end_time is slightly different than the parsed time
                continue
            ss = ends[i]
            to = end_time
        else:
            ss = ends[i]
            to = starts[i + 1]

        if (to - ss) >= min_nonsilence_s:
            # print(i, "|", starts[i], ends[i], "|", ss, to)
            splits.append([ss, to])
            
    return splits


def write(infile, outfile, ss=None, to=None, verbose=False, sample_rate_hz=44100):
    cmd = ["ffmpeg",
        "-v",
        "verbose",
        "-y", # overwrite files without asking
        "-i",
        infile]
    
    if ss is not None and to is not None:
        cmd.extend(["-ss", str(ss), "-to", str(to)])

    if sample_rate_hz is not None:
        cmd.extend(["-ar", str(sample_rate_hz)])
        
    cmd.append(outfile)

    result = subprocess.run(cmd, capture_output=True)

    if verbose:
        print(result.stderr.decode("utf-8"))


def run(speaker_dir, processed_dir, out_format, sample_rate_hz,
        silence_db, min_silence_s, min_nonsilence_s):
    
    print("Starting audio splitting into smaller segments")

    shutil.rmtree(processed_dir, ignore_errors=True)
    os.makedirs(processed_dir)

    for root, file_dir, files in os.walk(speaker_dir):
        if len(file_dir) != 0:
            # because we are at the directory not the sub-directory level
            continue
        
        sentence_dir = os.path.join(processed_dir, os.path.basename(root))
        os.makedirs(sentence_dir, exist_ok=True)

        for file in files:
            print("Processing", file)
            name = re.match(r"(\w+)\.", file).group(1)
            ext = re.search(r"\w+$", file).group(0)
            infile = os.path.join(root, file)

            starts, ends, end_time = get_silences(infile, silence_db, min_silence_s)
            
            if starts == 0: # implicitly == len_ends
                print("No silence detected - copying as is")
                shutil.copy2(infile, os.path.join(sentence_dir, file))
                continue

            nonsilences = convert_silences_to_nonsilences(starts, ends, end_time, min_nonsilence_s)

            for i, (start, end) in enumerate(nonsilences):
                outfile = os.path.join(sentence_dir, f"{name}_{i}.{out_format}")
                write(infile, outfile, ss=start, to=end, verbose=False, sample_rate_hz=sample_rate_hz)
 








    

    



