import io
import re
import librosa
import numpy as np 
import subprocess
import matplotlib.pyplot as plt

class FFmpeg:
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

    def read_chunk(filename, chunk_size_s=5, hop_length_s=1.25, sr=16000, audio="s16le", verbose=False):
        ss = 0

        while(1):
            cmd = ["ffmpeg",
                "-v",
                "verbose",
                "-accurate_seek",
                "-i",
                filename,
                "-f",
                audio, # PCM signed 16-bit little-endian samples
                "-ss",
                str(ss),
                "-t",
                str(chunk_size_s),
                "-ac",   # number of channels
                "1",
                "-ar",   # sampling rate (Hz)
                str(sr), 
                "pipe:"]
            
            result = subprocess.run(cmd, capture_output=True)
            stdout = result.stderr.decode("utf-8")

            if verbose:
                print(stdout)

            if re.search("Output file is empty", stdout):
                break

            yield result.stdout

            ss += hop_length_s
    
    def write(filename, out_buffer, sr=16000, audio="s16le", ss=None, to=None, verbose=False):
        # TODO -> might need the -c copy option if out_format is same as in_format?
        #         I'm not sure if ffmpeg is smart enough to not reencode 
        cmd = ["ffmpeg",
            "-v",
            "verbose",
            "-y", # overwrite files without asking
            "-f",
            audio,
            "-ar",
            str(sr),
            "-i",
            "pipe:"]
        if ss is not None and to is not None:
            cmd.extend(["-ss", str(ss), "-to", str(to)])
        cmd.append(filename)

        result = subprocess.run(cmd, capture_output=True, input=out_buffer)

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
    chunk_size_s = 5
    hop_length_s = 1.25
    gen_chunks = FFmpeg.read_chunk("/home/justin/files/test_ffmpeg/in.aac",
                                   chunk_size_s=chunk_size_s, hop_length_s=hop_length_s, verbose=False)


    concat_times = [] 
    pos = 0
    for i, chunk in enumerate(gen_chunks):
        print(f"Processing chunk {i}")
        arr = np.frombuffer(chunk, dtype="<i2")
        splits_samples = librosa.effects.split(arr, top_db=60, frame_length=2048, hop_length=512)
        splits_time = librosa.samples_to_time(splits_samples, sr=16000)

        for (start, end) in splits_time:
            print(start + pos, end + pos)
        # print(splits_time)
        
        pos += hop_length_s

        # min_length = 2
        # mask = (splits_time[:,1]-splits_time[:,0]) > min_length
        # splits_time_filtered = splits_time[mask]
        # for i, (split_start, split_end) in enumerate(splits_time_filtered):
        #     outfile = f"/home/justin/files/test_ffmpeg/in_split_{i}.aac"
        #     FFmpeg.write(outfile, arr.tobytes(), ss=split_start, to=split_end)






    

    



