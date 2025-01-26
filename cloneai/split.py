import io
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

    def read_chunk(filename, chunk_size=1024*1024, sr=16000, audio="s16le", verbose=False):
        with open(filename, "rb") as f:
            in_buffer = io.BytesIO(f.read(chunk_size))
        
        cmd = ["ffmpeg",
            "-v",
            "verbose",
            "-i",
            "pipe:",
            "-f",
            audio, # PCM signed 16-bit little-endian samples
            "-ac",   # number of channels
            "1",
            "-ar",   # sampling rate (Hz)
            str(sr), 
            "pipe:"]
        
        result = subprocess.run(cmd, capture_output=True, input=in_buffer.getbuffer())

        if verbose:
            print(result.stderr.decode("utf-8"))

        return result.stdout
    
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
    

def write_chunk(filename, in_buffer):
    with open(filename, "wb") as f:
        f.write(in_buffer.read())

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
    in_buffer = FFmpeg.read_chunk("/home/justin/files/test_ffmpeg/in.aac")

    
    arr = np.frombuffer(in_buffer, dtype="<i2")

    # matplotlib_plot(arr)

    splits_samples = librosa.effects.split(arr, top_db=60, frame_length=2048, hop_length=512)
    splits_time = librosa.samples_to_time(splits_samples, sr=16000)

    

    min_length = 2
    mask = (splits_time[:,1]-splits_time[:,0]) > min_length
    

    splits_time_filtered = splits_time[mask]

    matplotlib_plot_with_intervals(arr, librosa.time_to_samples(splits_time_filtered, sr=16000))


    for i, (split_start, split_end) in enumerate(splits_time_filtered):
        outfile = f"/home/justin/files/test_ffmpeg/in_split_{i}.aac"
        FFmpeg.write(outfile, arr.tobytes(), ss=split_start, to=split_end)






    

    



