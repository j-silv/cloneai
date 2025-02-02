import io
import re
import librosa
import numpy as np 
import subprocess
import matplotlib.pyplot as plt

class FFmpeg:


    def get_silences(infile, dB=-60, dur=2.0):
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
            f"silencedetect=d={dur}:n={dB}dB",
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
    



    def get_nonsilences(infile, dB=-60, min_silence=2.0, min_nonsilence=2):
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

        starts, ends, end_time = FFmpeg.get_silences(infile, dB, min_silence)

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

            if (to - ss) >= min_nonsilence:
                # print(i, "|", starts[i], ends[i], "|", ss, to)
                splits.append([ss, to])
                
        return splits

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
    

    def write(infile, outfile, ss=None, to=None, verbose=False):
        cmd = ["ffmpeg",
            "-v",
            "verbose",
            "-y", # overwrite files without asking
            "-i",
            infile]
        
        if ss is not None and to is not None:
            cmd.extend(["-ss", str(ss), "-to", str(to)])

        cmd.append(outfile)

        result = subprocess.run(cmd, capture_output=True)

        if verbose:
            print(result.stderr.decode("utf-8"))



def matplotlib_plot(arr):
    plt.plot(arr)
    plt.show()

def matplotlib_plot_with_intervals(arr, intervals):
    plt.plot(arr)
    # Plot vertical bars for each interval
    for start, end in intervals:
        plt.axvspan(start, end, color='blue', alpha=0.3, label='Interval')  # Shaded bar
        plt.axvline(x=start, color='red', linestyle='--', label='Start Line')
        plt.axvline(x=end, color='green', linestyle='--', label='End Line')
    plt.show()

if __name__ == "__main__":
    silences = (FFmpeg.get_silences("/home/justin/files/test_ffmpeg/in.wav", dB=-30, dur=1.5))
    nonsilences = (FFmpeg.get_nonsilences("/home/justin/files/test_ffmpeg/in.wav", dB=-40, min_silence=1.0, min_nonsilence=1))

    print(nonsilences)
    for i, (start, end) in enumerate(nonsilences):
        print(i, start, end)
        outfile = f"/home/justin/files/test_ffmpeg/in_split_{i}.wav"
        FFmpeg.write("/home/justin/files/test_ffmpeg/in.wav", outfile, ss=start, to=end, verbose=False)








    

    



