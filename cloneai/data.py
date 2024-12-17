"""
Unzip and prepare raw data that is collected from Craig recordings

These folders were auto uploaded to my Google Drive account and now I will
proceed to unzip them and create data that is ready to be processed
by the ML models

- Unzip everything from raw data location
- Create destination folders for speaker1, speaker2, speaker3 in data/processed folder
- Determine speaker unique IDs by iterating through every folder and hashing them as names
- Separate tracks into the respective folders (map speaker names to processed folder) -> only process individual tracks
- Delete recordings under a certain size/size
- Delete recordings with a single speaker
- Convert all audio to the same format and sampling rate (might need to batch use raw.dat or re-sample)
- Break up audio into short sentence phrases
    - cut audio whenever there is a gap of more than couple of seconds? pause in dialog? look at algorithms for this
    - pydub is apparently one library to do this. Let's see if there is any sort of low-level usage with ffmpeg? silence.split_on_silence()
    - https://stackoverflow.com/questions/45526996/split-audio-files-using-silence-detection
-For consistency into whisper and training make sure audio files:
    - Are in WAV format (not necessary, can be flac as well)
    - Have a sample rate of 16kHz (this should be done)
    - Are mono-channel (this should also be done)
"""

import zipfile
import yaml
import os
import glob
import re
import shutil
import librosa
import subprocess
import ffmpeg
import matplotlib.pyplot as plt
import soundfile as sf

def extract_zips(in_dir, out_dir, nested=False, clean=False):
    """Extract zip files from in_dir to out_dir
    
    Also creates a log file in the out_dir to monitor extraction

    Params:
        in_dir  : folder with zip files
        out_dir : output directory
        nested  : extract a nested zip file 
        clean   : delete zip archive we just extracted

    Returns:
        number of extractions
    """
    os.makedirs(out_dir, exist_ok=True)
    extraction_num = 0 
    zip_files = glob.glob(f"{in_dir}/*.zip")

    # zip_files = zip_files[2:4] # for debugging

    with open(os.path.join(out_dir, f"extract_{nested}_{clean}.log"), "w") as log:
        for file in zip_files:
            with zipfile.ZipFile(file, "r") as zip_ref:
                if nested is False:
                    extraction_num += 1
                    wrap_dir = os.path.join(out_dir, str(extraction_num))
                    os.makedirs(wrap_dir, exist_ok=True)
                    print(f"Extract #{extraction_num}: {zip_ref.filename}", file=log, flush=True)
                    zip_ref.extractall(wrap_dir)
                else: 
                    for member in zip_ref.infolist():
                        extraction_num += 1
                        data = zip_ref.read(member)
                        out_file = os.path.join(out_dir, os.path.basename(member.filename))
                        with open(out_file, 'wb') as target:
                            print(f"Extract #{extraction_num}: {member.filename} ---> {out_file}", file=log, flush=True)
                            target.write(data)
            if clean is True:
                os.remove(file)

        print("\nNumber of extractions:", extraction_num, file=log, flush=True)
    return extraction_num

def group_tracks(in_dir, out_dir, merge, ignore, clean=False):
    """Group all individual tracks into a central location

    Return dictionary for unique track speakers based on track name

    Each key will be the unique name of the speaker, and the value will
    be another dictionary that has a key for the directory location for all
    tracks belonging to that speaker, and the number of tracks for that speaker

    output files are copied to speaker specific directory and are of the format:
    "REC_TRACK.EXT" where:
        REC    : recording number
        TRACK  : track number for this speaker
        EXT    : {"acc", "flac"}
    
    with REC and TRACK we can determine information on the distribution of speakers
    for each recording later

    Params:
        in_dir  : root folder with extracted audio files
        out_dir : output directory
        merge   : list of speaker names to merge into one
        ignore  : list of speakers to ignore
        clean   : delete extracted audio file folders as we group

    Returns:
        speaker dictionary
    """
    speaker = dict()

    for root, file_dir, files in os.walk(os.path.join(in_dir)):
        if len(file_dir) != 0:
            # because we are at the directory not the sub-directory level
            continue
        
        for file in files:
            # because we want to identify just the string part of the name
            name = re.match(r"\d+-(\D+)_?\d+", file)
            if name is None:
                # raw.dat or info.txt
                continue
            
            # first capturing group above just gets the discord name
            # because some usernames have _ we just get the text name
            name = name.group(1)
            name = re.sub(r"_", r"", name) 

            if name in ignore:
                continue
            if name in merge:
                name = merge[0]

            ext = re.search(r"\w+$", file).group(0)

            if name not in speaker:
                speaker_dir = os.path.join(out_dir, f"{str(len(speaker))}-{name}")
                os.makedirs(speaker_dir, exist_ok=True)
                speaker[name] = dict(dir=speaker_dir, num=0)
            
            src = os.path.join(root, file)
            dest = os.path.join(speaker[name]["dir"], f"{os.path.basename(root)}_{str(speaker[name]['num'])}.{ext}")
            shutil.copy2(src, dest)
            speaker[name]["num"] += 1

        if clean is True:
            shutil.rmtree(root)
        
    return speaker

def get_silences(infile):
    """Use ffmpeg to detect silence regions of audio
    
    This is based on an example using ffmpeg-python located here:
    https://github.com/kkroening/ffmpeg-python/blob/master/examples/split_silence.py
    
    Some differences include avoiding ffmpeg-python library (raw subprocess calls)
    and also just collecting the raw silence start/end timestamps instead of converting that 
    to chunk_start and chunk_end times (this is done later)
    """

    cmd = ["ffmpeg", "-hide_banner",
        "-i", infile,
        "-filter_complex",
        "silencedetect=d=2.0:n=-60dB",
        "-f", "null", "-"
    ]
    # print(" ".join(cmd))

    silence_start_re = re.compile(r' silence_start: (?P<start>[0-9]+(\.?[0-9]*))$')
    silence_end_re = re.compile(r' silence_end: (?P<end>[0-9]+(\.?[0-9]*)) ')
    total_duration_re = re.compile(r'size=[^ ]+ time=(?P<hours>[0-9]{2}):(?P<minutes>[0-9]{2}):(?P<seconds>[0-9\.]{5}) bitrate=')

    result = subprocess.run(cmd, capture_output=True)
    lines = result.stderr.decode("utf-8").splitlines()

    starts = []
    ends = []

    for line in lines:

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

def process_audio(in_dir, out_dir, config):
    """Convert all audio files to specific format and split for ML training
    
    Uses librosa and ffmpeg utility to load in audio files
    and split on silence to detect sentence boundaries

    ffmpeg usage is required because the raw audio files are in .aac format
    and soundread (library that librosa uses for loading) does not support
    MP3 and MP4 formats

    Params:
        config   : dict for controlling audio preprocessing
            format       : convert to this audio format
            sampling     : convert to this sampling rate
            top_db       : threshold in decibels below reference to consider as silence
            frame_length : the number of samples per analysis frame
            hop_length   : the number of samples between anlysis frames

    Also we could avoid using librosa (kind of unnecessary)
    and just use ffmpeg and use the existing filter:
    https://github.com/kkroening/ffmpeg-python/blob/master/examples/split_silence.py
    https://ffmpeg.org/ffmpeg-filters.html
    silencedetect 

    that actually splits silence based on dB and a silence duration
    then once we have that information we can output the right timestamps

    there is also "silenceremove" filter but this doesn't split into
    separate files which is what we want... we need the timestamps 
    """
    for root, file_dir, files in os.walk(os.path.join(in_dir)):
        if len(file_dir) != 0:
            # because we are at the directory not the sub-directory level
            continue

        sentence_dir = os.path.join(out_dir, os.path.basename(root))
        os.makedirs(sentence_dir, exist_ok=True)

        for file in files:
            print("Processing", file)
            name = re.match(r"(\w+)\.", file).group(1)
            ext = re.search(r"\w+$", file).group(0)
            infile = os.path.join(root, file)

            starts, ends, end_time = get_silences(infile)

            len_starts = len(starts)
            len_ends = len(ends)

            assert len_starts == len_ends, "Silence detection did not work correctly because uneven lengths"

            if len_starts == 0: # implicitly == len_ends
                print("No silence detected - copying as is")
                shutil.copy2(infile, os.path.join(sentence_dir, file))
                continue
            for i in range(len_starts):

                outfile = os.path.join(sentence_dir, f"{name}_{i}.{ext}")
        
                if i == 0 and starts[0] != 0:
                    # handles non-silence at beginning of track
                    ss = 0.0
                    to = starts[0]
                elif i+1 == len_starts:
                    # handles end of track (avoids out-of-range indexing)
                    ss = ends[i]
                    to = end_time
                else:
                    ss = ends[i]
                    to = starts[i + 1]
                    
                print(i, "|", starts[i], ends[i], "|", ss, to, outfile)

                cmd = ["ffmpeg", "-hide_banner", "-i", infile, "-ss", str(ss), "-to", str(to), outfile]
                result = subprocess.run(cmd, capture_output=False)     


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


if __name__ == "__main__":
    ROOT = os.getcwd()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    raw_dir = os.path.join(ROOT, config["dir"]["data"]["raw"])
    processed_dir = os.path.join(ROOT, config["dir"]["data"]["processed"])
    out_dir = os.path.join(raw_dir, "extracted")
    
    print("Extracting audio data archives... this might take a while")

    # we will call this function twice as we have a nested 
    # zip archive. the first time will be False and the second time True
    # extract_zips(raw_dir, out_dir, nested=True, clean=False)
    # extract_zips(out_dir, out_dir, nested=False, clean=True)

    speaker_dir = os.path.join(raw_dir, "speaker")
    # group_tracks(out_dir, speaker_dir, config["data"]["merge"], config["data"]["ignore"], clean=True)
    print("Extraction completed")

    print("Starting audio preprocessing")
    # process_audio(os.path.join(speaker_dir, "2-scott"), processed_dir, config["data"]["audio"])

    shutil.rmtree(os.path.join(processed_dir, "processed"), ignore_errors=True)
    process_audio(processed_dir, processed_dir, config["data"]["audio"])



