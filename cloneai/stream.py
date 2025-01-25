import io
import librosa
import numpy as np 
import subprocess
import matplotlib.pyplot as plt
import sys

def ffmpeg_get_sample_rate(filename):
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

def ffmpeg_read_chunk(filename, chunk_size=1024*1024):
    with open(filename, "rb") as f:
        in_buffer = io.BytesIO(f.read(chunk_size))
        
    cmd = ["ffmpeg",
           "-v",
           "verbose",
           "-i",
           "pipe:",
           "-f",
           "s16le", # PCM signed 16-bit little-endian samples
           "-ac",   # number of channels
           "1",
           "-ar",   # sampling rate (Hz)
           "16000", 
           "pipe:"]
    result = subprocess.run(cmd, capture_output=True, input=in_buffer.getbuffer())
    print(result.stderr.decode("utf-8"))

    return result.stdout

def ffmpeg_write(filename, out_buffer):
    cmd = ["ffmpeg",
           "-v",
           "verbose",
           "-y", # overwrite files without asking
           "-f",
           "s16le",
           "-ar",
           "16000",
           "-i",
           "pipe:",
           filename]
    result = subprocess.run(cmd, stderr=subprocess.STDOUT, input=out_buffer)
    return result



def numpy_load(bytes):
    # here is where i do the silence splitting
    # i guess i'll gather from here the timestamps where it is zero and then
    # create it i guess... yeah 
    arr = np.frombuffer(bytes, dtype="<i2")
    print(arr)
    return arr










def write_chunk(filename, in_buffer):
    with open(filename, "wb") as f:
        f.write(in_buffer.read())


def librosa_load(filename):
    audio, sampr = librosa.load(filename, sr=48000)
    print(audio, sampr) 
    return audio, sampr


def matplotlib_plot(arr):
    plt.plot(arr)
    plt.show()

def matplotlib_plot_with_intervals(arr, intervals):
    plt.plot(arr)

    # Plot vertical bars for each interval
    for start, end in intervals:
        # if start < arr[0]:
        #     continue
        # if start > arr[-1]:
        #     break
        # if end > arr[-1]:
        #     end = arr[-1]

        plt.axvspan(start, end, color='blue', alpha=0.3, label='Interval')  # Shaded bar
        plt.axvline(x=start, color='red', linestyle='--', label='Start Line')
        plt.axvline(x=end, color='green', linestyle='--', label='End Line')

    plt.show()

def numpy_find_silence(arr):
    """Find silence in array that are below certain threshold
    
    I guess we can find decibels so anything below threshold cut-off
    """
    #librosa.effects.split(y, top_db=60, ref, frame_length=2048, hop_length=512)
    pass 



if __name__ == "__main__":
    # np.set_printoptions(threshold=5000)

    in_buffer = ffmpeg_read_chunk("/home/justin/files/test_ffmpeg/in.aac")
    # ffmpeg_write("/home/justin/files/test_ffmpeg/out.aac", in_buffer)

    arr = numpy_load(in_buffer)

    # print(arr.dtype)
    # assert arr.dtype == np.int16, "Input data must be 16-bit signed PCM"

    # arr = arr.astype(np.float32)

    # matplotlib_plot(arr)

    
    arr = arr // 4

    # arr = arr.astype(np.int16)
    
    # matplotlib_plot(arr)

    ffmpeg_write("/home/justin/files/test_ffmpeg/out.aac", arr.tobytes())
    



    # audio, sampr = librosa_load("data/processed/clip.aac")
    # frame_length = librosa.time_to_samples(1.0, sr=sampr)
    # hop_length = frame_length//4
    # print(frame_length, hop_length)
    
    # splits = librosa.effects.split(audio,
    #                                top_db=60,
    #                                frame_length=frame_length,
    #                                hop_length=hop_length)

    # plt.figure()
    # librosa.display.waveshow(audio, sr=sampr)

    # for start, end in librosa.samples_to_time(splits, sr=sampr):
    #     plt.axvspan(start, end, color='red', alpha=0.3, label='Interval')  # Shaded bar
    #     # plt.axvline(x=start, color='red', linestyle='--', label='Start Line')
    #     # plt.axvline(x=end, color='green', linestyle='--', label='End Line')
    # print(librosa.samples_to_time(splits, sr=sampr))
    # plt.xlim(70, 90) 
    # plt.show()



    # print(splits)
    # matplotlib_plot_with_intervals(audio, splits)
    # matplotlib_plot(audio)
    # fig, ax = plt.plot()
    # plt.figure()
    
    # matplotlib_plot(arr)



    # in_buffer = np.zeros((10000,), dtype=">i2")
    # nonsilent = in_buffer.copy()
    # nonsilent[0:5000] = 30000
    # ffmpeg_write("data/processed/clip_nonsilence.aac", nonsilent.tobytes())

    # silence = in_buffer.copy()
    # silence[5000:] = 30000
    # ffmpeg_write("data/processed/clip_silence.aac", silence.tobytes())

    # out_buffer = arr.tobytes()
    # ffmpeg_write("data/processed/clip2.aac", out_buffer)

    # plt.show()
    # plt.plot()





    

    



