# Other thoughts

Maybe I should go ahead and fine-tune whisper as well? This would improve transcription
and I already have all the data

So basically I could fine tune whisper, llama, tacotron2, and wavenet to all be specific to 
the speaker.

But hmmm then would I have 4 separate models? Hmm idk. Well I would need that for tacotron2 and wavenet.
I think I read something about multispeaker in the textbook and we could switch between which speaker
we want -> oh wait:

multi-speaker TTS

on page 356 of textbook

Hmmm well that would be kind of nice but i think we should have a separate one for now...
yeah that would be a nice to have but let's first get a single speaker and then maybe we can fine tune it later
also once I fine tune one model i think the rest will be good. Let's first just try to fine tune to one 

eventually it will be all fine tuning though when you think about it because then it is supposed to be a clone 


```



    inter_file = os.path.join(sentence_dir, f"{name}.{config['inter_format']}")
    
    y, sr = librosa.load(inter_file, sr=config["sr"], mono=True)

    frame_length = librosa.time_to_samples(config["frame_length_s"], sr=sr)
    hop_length = librosa.time_to_samples(config["hop_length_s"], sr=sr)

    print(frame_length, hop_length)

    sentences = librosa.effects.split(y, top_db=config["top_db"],
                                        frame_length=frame_length,
                                        hop_length=hop_length)

    print(librosa.samples_to_time(sentences, sr=sr))

    # librosa.display.waveshow(y, sr=sr)

    # for start, end in librosa.samples_to_time(sentences, sr=sr):
    #     plt.axvspan(start, end, color='red', alpha=0.3, label='Interval')  # Shaded bar

    # plt.xlim(70, 90)  
    # plt.show()


    mask = (sentences[:,1]-sentences[:,0]) > librosa.time_to_samples(config["min_length_s"], sr=sr)
    filtered_sentences = sentences[mask]
    
    for i, sentence in enumerate(filtered_sentences):
        outfile = os.path.join(sentence_dir, f"{i}.{config['out_format']}")

        start_time_s = librosa.samples_to_time(sentence[0], sr=sr)
        end_time_s = librosa.samples_to_time(sentence[1], sr=sr)

        
        # TODO -> might need the -c copy option if out_format is same as in_format?
        #         I'm not sure if ffmpeg is smart enough to not reencode 


Processing clip_nonsilence_start.aac
[silencedetect @ 0x57686e40f480] silence_start: 0.140204
[silencedetect @ 0x57686e40f480] silence_end: 0.25542 | silence_duration: 0.115215

Processing clip_silence_start.aac
[silencedetect @ 0x6380bfb4a700] silence_start: 0
[silencedetect @ 0x6380bfb4a700] silence_end: 0.135556 | silence_duration: 0.135556

```


```
(.venv) justin@ThinkPad-W530:~/files/projects/cloneai$ ffmpeg -hide_banner -i /home/justin/files/projects/cloneai/data/processed/clip.aac -filter_complex silencedetect=d=0.02:n=-60dB -f null -
[aac @ 0x6516ca9e4780] Estimating duration from bitrate, this may be inaccurate
Input #0, aac, from '/home/justin/files/projects/cloneai/data/processed/clip.aac':
  Duration: 00:31:50.85, bitrate: 3 kb/s
  Stream #0:0: Audio: aac (LC), 44100 Hz, mono, fltp, 3 kb/s
Stream mapping:
  Stream #0:0 (aac) -> silencedetect
  silencedetect -> Stream #0:0 (pcm_s16le)
Press [q] to stop, [?] for help
[silencedetect @ 0x6516ca9f5e80] silence_start: 0
Output #0, null, to 'pipe:':
  Metadata:
    encoder         : Lavf58.76.100
  Stream #0:0: Audio: pcm_s16le, 44100 Hz, mono, s16, 705 kb/s
    Metadata:
      encoder         : Lavc58.134.100 pcm_s16le
[silencedetect @ 0x6516ca9f5e80] silence_end: 5.59501 | silence_duration: 5.59501
[silencedetect @ 0x6516ca9f5e80] silence_start: 5.74844
[silencedetect @ 0x6516ca9f5e80] silence_end: 7.00211 | silence_duration: 1.25367
[silencedetect @ 0x6516ca9f5e80] silence_start: 7.22084
[silencedetect @ 0x6516ca9f5e80] silence_end: 7.2581 | silence_duration: 0.0372562
[silencedetect @ 0x6516ca9f5e80] silence_start: 7.31041
[silencedetect @ 0x6516ca9f5e80] silence_end: 12.2005 | silence_duration: 4.89011
[silencedetect @ 0x6516ca9f5e80] silence_start: 12.2102
[silencedetect @ 0x6516ca9f5e80] silence_end: 12.243 | silence_duration: 0.0327664
[silencedetect @ 0x6516ca9f5e80] silence_start: 12.3378
[silencedetect @ 0x6516ca9f5e80] silence_end: 12.382 | silence_duration: 0.0441723
[silencedetect @ 0x6516ca9f5e80] silence_start: 12.4094
[silencedetect @ 0x6516ca9f5e80] silence_end: 12.4439 | silence_duration: 0.0344671
[silencedetect @ 0x6516ca9f5e80] silence_start: 12.4787
[silencedetect @ 0x6516ca9f5e80] silence_end: 12.5032 | silence_duration: 0.0244898
[silencedetect @ 0x6516ca9f5e80] silence_start: 12.5167
[silencedetect @ 0x6516ca9f5e80] silence_end: 15.2704 | silence_duration: 2.75365
[silencedetect @ 0x6516ca9f5e80] silence_start: 15.4375
[silencedetect @ 0x6516ca9f5e80] silence_end: 74.4563 | silence_duration: 59.0189
[silencedetect @ 0x6516ca9f5e80] silence_start: 74.9806
[silencedetect @ 0x6516ca9f5e80] silence_end: 75.3279 | silence_duration: 0.347324
[silencedetect @ 0x6516ca9f5e80] silence_start: 76.6634
[silencedetect @ 0x6516ca9f5e80] silence_end: 76.8514 | silence_duration: 0.188005
[silencedetect @ 0x6516ca9f5e80] silence_start: 77.7884
[silencedetect @ 0x6516ca9f5e80] silence_end: 78.3991 | silence_duration: 0.610726
[silencedetect @ 0x6516ca9f5e80] silence_start: 78.9398
[silencedetect @ 0x6516ca9f5e80] silence_end: 80.0507 | silence_duration: 1.11088
[silencedetect @ 0x6516ca9f5e80] silence_start: 80.189
[silencedetect @ 0x6516ca9f5e80] silence_end: 80.5065 | silence_duration: 0.317551
[silencedetect @ 0x6516ca9f5e80] silence_start: 81.4591
[silencedetect @ 0x6516ca9f5e80] silence_end: 81.636 | silence_duration: 0.176848
[silencedetect @ 0x6516ca9f5e80] silence_start: 83.2296
[silencedetect @ 0x6516ca9f5e80] silence_end: 83.9701 | silence_duration: 0.740499
[silencedetect @ 0x6516ca9f5e80] silence_start: 84.3964
[silencedetect @ 0x6516ca9f5e80] silence_end: 84.8745 | silence_duration: 0.478095
[silencedetect @ 0x6516ca9f5e80] silence_start: 84.9526
[silencedetect @ 0x6516ca9f5e80] silence_end: 84.9738 | silence_duration: 0.0212018
[silencedetect @ 0x6516ca9f5e80] silence_start: 85.9062
[silencedetect @ 0x6516ca9f5e80] silence_end: 85.9502 | silence_duration: 0.044059
[silencedetect @ 0x6516ca9f5e80] silence_start: 85.9985
[silencedetect @ 0x6516ca9f5e80] silence_end: 90.3667 | silence_duration: 4.36821
[silencedetect @ 0x6516ca9f5e80] silence_start: 92.0419
[silencedetect @ 0x6516ca9f5e80] silence_end: 92.0767 | silence_duration: 0.0348753
[silencedetect @ 0x6516ca9f5e80] silence_start: 92.0774
[silencedetect @ 0x6516ca9f5e80] silence_end: 92.1039 | silence_duration: 0.0264853
[silencedetect @ 0x6516ca9f5e80] silence_start: 93.9359
[silencedetect @ 0x6516ca9f5e80] silence_end: 99.2185 | silence_duration: 5.28252
[silencedetect @ 0x6516ca9f5e80] silence_start: 100.639
[silencedetect @ 0x6516ca9f5e80] silence_end: 100.743 | silence_duration: 0.103537
[silencedetect @ 0x6516ca9f5e80] silence_start: 101.298
[silencedetect @ 0x6516ca9f5e80] silence_end: 103.619 | silence_duration: 2.32011
[silencedetect @ 0x6516ca9f5e80] silence_start: 104.336
[silencedetect @ 0x6516ca9f5e80] silence_end: 105.739 | silence_duration: 1.40313
[silencedetect @ 0x6516ca9f5e80] silence_start: 106.923
[silencedetect @ 0x6516ca9f5e80] silence_end: 106.967 | silence_duration: 0.0447392
[silencedetect @ 0x6516ca9f5e80] silence_start: 106.97
[silencedetect @ 0x6516ca9f5e80] silence_end: 107.083 | silence_duration: 0.113832
[silencedetect @ 0x6516ca9f5e80] silence_start: 107.298
[silencedetect @ 0x6516ca9f5e80] silence_end: 107.785 | silence_duration: 0.486667
[silencedetect @ 0x6516ca9f5e80] silence_start: 107.99
[silencedetect @ 0x6516ca9f5e80] silence_end: 108.329 | silence_duration: 0.338866
[silencedetect @ 0x6516ca9f5e80] silence_start: 109.886
[silencedetect @ 0x6516ca9f5e80] silence_end: 114.381 | silence_duration: 4.49497
[silencedetect @ 0x6516ca9f5e80] silence_start: 115.875
[silencedetect @ 0x6516ca9f5e80] silence_end: 115.976 | silence_duration: 0.100068
[silencedetect @ 0x6516ca9f5e80] silence_start: 116.128
[silencedetect @ 0x6516ca9f5e80] silence_end: 117.075 | silence_duration: 0.947551
[silencedetect @ 0x6516ca9f5e80] silence_start: 117.561
[silencedetect @ 0x6516ca9f5e80] silence_end: 119.301 | silence_duration: 1.74041
[silencedetect @ 0x6516ca9f5e80] silence_start: 120.1
[silencedetect @ 0x6516ca9f5e80] silence_end: 122.25 | silence_duration: 2.1493
[silencedetect @ 0x6516ca9f5e80] silence_start: 123.333
[silencedetect @ 0x6516ca9f5e80] silence_end: 123.367 | silence_duration: 0.0340816
[silencedetect @ 0x6516ca9f5e80] silence_start: 123.388
[silencedetect @ 0x6516ca9f5e80] silence_end: 123.416 | silence_duration: 0.0284127
[silencedetect @ 0x6516ca9f5e80] silence_start: 123.418
[silencedetect @ 0x6516ca9f5e80] silence_end: 123.608 | silence_duration: 0.190249
[silencedetect @ 0x6516ca9f5e80] silence_start: 124.409
[silencedetect @ 0x6516ca9f5e80] silence_end: 129.294 | silence_duration: 4.88488
[silencedetect @ 0x6516ca9f5e80] silence_start: 131.484
[silencedetect @ 0x6516ca9f5e80] silence_end: 134.413 | silence_duration: 2.9288
[silencedetect @ 0x6516ca9f5e80] silence_start: 134.99
[silencedetect @ 0x6516ca9f5e80] silence_end: 135.075 | silence_duration: 0.0848299
[silencedetect @ 0x6516ca9f5e80] silence_start: 135.083
[silencedetect @ 0x6516ca9f5e80] silence_end: 135.105 | silence_duration: 0.0214739
[silencedetect @ 0x6516ca9f5e80] silence_start: 135.578
[silencedetect @ 0x6516ca9f5e80] silence_end: 138.504 | silence_duration: 2.92599
[silencedetect @ 0x6516ca9f5e80] silence_start: 140.521
[silencedetect @ 0x6516ca9f5e80] silence_end: 140.796 | silence_duration: 0.275125
[silencedetect @ 0x6516ca9f5e80] silence_start: 142.981
[silencedetect @ 0x6516ca9f5e80] silence_end: 143.094 | silence_duration: 0.113696
[silencedetect @ 0x6516ca9f5e80] silence_start: 143.11
[silencedetect @ 0x6516ca9f5e80] silence_end: 143.137 | silence_duration: 0.0269615
[silencedetect @ 0x6516ca9f5e80] silence_start: 143.199
[silencedetect @ 0x6516ca9f5e80] silence_end: 144.111 | silence_duration: 0.911293
[silencedetect @ 0x6516ca9f5e80] silence_start: 145.27
[silencedetect @ 0x6516ca9f5e80] silence_end: 147.798 | silence_duration: 2.52703
[silencedetect @ 0x6516ca9f5e80] silence_start: 149.32
[silencedetect @ 0x6516ca9f5e80] silence_end: 178.903 | silence_duration: 29.5833
[silencedetect @ 0x6516ca9f5e80] silence_start: 179.244
[silencedetect @ 0x6516ca9f5e80] silence_end: 179.265 | silence_duration: 0.0212018
[silencedetect @ 0x6516ca9f5e80] silence_start: 181.011
[silencedetect @ 0x6516ca9f5e80] silence_end: 200.74 | silence_duration: 19.7286
[silencedetect @ 0x6516ca9f5e80] silence_start: 200.746
[silencedetect @ 0x6516ca9f5e80] silence_end: 200.808 | silence_duration: 0.0617687
[silencedetect @ 0x6516ca9f5e80] silence_start: 200.867
[silencedetect @ 0x6516ca9f5e80] silence_end: 200.975 | silence_duration: 0.1078
[silencedetect @ 0x6516ca9f5e80] silence_start: 201.065
[silencedetect @ 0x6516ca9f5e80] silence_end: 201.151 | silence_duration: 0.0858277
[silencedetect @ 0x6516ca9f5e80] silence_start: 201.904
[silencedetect @ 0x6516ca9f5e80] silence_end: 203.199 | silence_duration: 1.29517
[silencedetect @ 0x6516ca9f5e80] silence_start: 205.396
[silencedetect @ 0x6516ca9f5e80] silence_end: 205.633 | silence_duration: 0.237732
[silencedetect @ 0x6516ca9f5e80] silence_start: 207.058
[silencedetect @ 0x6516ca9f5e80] silence_end: 225.068 | silence_duration: 18.0107
[silencedetect @ 0x6516ca9f5e80] silence_start: 225.518
[silencedetect @ 0x6516ca9f5e80] silence_end: 225.576 | silence_duration: 0.0580952
[silencedetect @ 0x6516ca9f5e80] silence_start: 225.606
[silencedetect @ 0x6516ca9f5e80] silence_end: 231.984 | silence_duration: 6.37841
[silencedetect @ 0x6516ca9f5e80] silence_start: 232.522
[silencedetect @ 0x6516ca9f5e80] silence_end: 234.217 | silence_duration: 1.69447
[silencedetect @ 0x6516ca9f5e80] silence_start: 235.038
[silencedetect @ 0x6516ca9f5e80] silence_end: 244.536 | silence_duration: 9.498
[silencedetect @ 0x6516ca9f5e80] silence_start: 244.8
[silencedetect @ 0x6516ca9f5e80] silence_end: 244.907 | silence_duration: 0.107029
[silencedetect @ 0x6516ca9f5e80] silence_start: 245.269
[silencedetect @ 0x6516ca9f5e80] silence_end: 249.609 | silence_duration: 4.33955
[silencedetect @ 0x6516ca9f5e80] silence_start: 249.879
[silencedetect @ 0x6516ca9f5e80] silence_end: 249.936 | silence_duration: 0.0568481
[silencedetect @ 0x6516ca9f5e80] silence_start: 249.952
[silencedetect @ 0x6516ca9f5e80] silence_end: 249.975 | silence_duration: 0.0230612
[silencedetect @ 0x6516ca9f5e80] silence_start: 250.395
[silencedetect @ 0x6516ca9f5e80] silence_end: 250.443 | silence_duration: 0.048254
[silencedetect @ 0x6516ca9f5e80] silence_start: 250.553
[silencedetect @ 0x6516ca9f5e80] silence_end: 254.141 | silence_duration: 3.58807
[silencedetect @ 0x6516ca9f5e80] silence_start: 254.447
[silencedetect @ 0x6516ca9f5e80] silence_end: 254.556 | silence_duration: 0.108844
[silencedetect @ 0x6516ca9f5e80] silence_start: 257.271
[silencedetect @ 0x6516ca9f5e80] silence_end: 257.868 | silence_duration: 0.596531
[silencedetect @ 0x6516ca9f5e80] silence_start: 258.292
[silencedetect @ 0x6516ca9f5e80] silence_end: 258.41 | silence_duration: 0.118481
[silencedetect @ 0x6516ca9f5e80] silence_start: 260.927
[silencedetect @ 0x6516ca9f5e80] silence_end: 265.89 | silence_duration: 4.96306
[silencedetect @ 0x6516ca9f5e80] silence_start: 267.889
[silencedetect @ 0x6516ca9f5e80] silence_end: 267.93 | silence_duration: 0.0415193
[silencedetect @ 0x6516ca9f5e80] silence_start: 267.93
[silencedetect @ 0x6516ca9f5e80] silence_end: 267.963 | silence_duration: 0.0322449
[silencedetect @ 0x6516ca9f5e80] silence_start: 267.963
[silencedetect @ 0x6516ca9f5e80] silence_end: 272.487 | silence_duration: 4.52483
[silencedetect @ 0x6516ca9f5e80] silence_start: 272.713
[silencedetect @ 0x6516ca9f5e80] silence_end: 273.128 | silence_duration: 0.415125
[silencedetect @ 0x6516ca9f5e80] silence_start: 274.556
[silencedetect @ 0x6516ca9f5e80] silence_end: 274.682 | silence_duration: 0.126667
[silencedetect @ 0x6516ca9f5e80] silence_start: 275.35
[silencedetect @ 0x6516ca9f5e80] silence_end: 276.406 | silence_duration: 1.05567
[silencedetect @ 0x6516ca9f5e80] silence_start: 276.701
[silencedetect @ 0x6516ca9f5e80] silence_end: 277.081 | silence_duration: 0.379819
[silencedetect @ 0x6516ca9f5e80] silence_start: 277.484
[silencedetect @ 0x6516ca9f5e80] silence_end: 285.394 | silence_duration: 7.91088
[silencedetect @ 0x6516ca9f5e80] silence_start: 289.042
[silencedetect @ 0x6516ca9f5e80] silence_end: 289.07 | silence_duration: 0.0282766
[silencedetect @ 0x6516ca9f5e80] silence_start: 289.08
[silencedetect @ 0x6516ca9f5e80] silence_end: 296.962 | silence_duration: 7.8817
[silencedetect @ 0x6516ca9f5e80] silence_start: 300.256
[silencedetect @ 0x6516ca9f5e80] silence_end: 302.411 | silence_duration: 2.15544
[silencedetect @ 0x6516ca9f5e80] silence_start: 303.516
[silencedetect @ 0x6516ca9f5e80] silence_end: 313.913 | silence_duration: 10.3973
[silencedetect @ 0x6516ca9f5e80] silence_start: 314.349
[silencedetect @ 0x6516ca9f5e80] silence_end: 314.412 | silence_duration: 0.0626304
[silencedetect @ 0x6516ca9f5e80] silence_start: 315.68
[silencedetect @ 0x6516ca9f5e80] silence_end: 315.711 | silence_duration: 0.0307256
[silencedetect @ 0x6516ca9f5e80] silence_start: 315.873
[silencedetect @ 0x6516ca9f5e80] silence_end: 316.24 | silence_duration: 0.36737
[silencedetect @ 0x6516ca9f5e80] silence_start: 318.09
[silencedetect @ 0x6516ca9f5e80] silence_end: 318.342 | silence_duration: 0.251587
[silencedetect @ 0x6516ca9f5e80] silence_start: 319.699
[silencedetect @ 0x6516ca9f5e80] silence_end: 328.65 | silence_duration: 8.95077
[silencedetect @ 0x6516ca9f5e80] silence_start: 330.958
[silencedetect @ 0x6516ca9f5e80] silence_end: 331.358 | silence_duration: 0.400726
size=N/A time=00:05:33.43 bitrate=N/A speed= 698x    
video:0kB audio:28722kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown

```


I guess I could output the timestamps of the original file 
and then the output file and basically find the location when it changes?
then I will basically have the output there

Basically whenever the output skips 

1       1
2       2
3       3
8       4
        5
        6
        7
        8
9
10
20

Maybe this information is just present in the file itself? Let's see

ffmpeg -hide_banner -i /home/justin/files/projects/cloneai/data/processed/clip.aac -filter_complex ashowinfo -f null - 2> /home/justin/files/projects/cloneai/data/processed/processed/clip_ashowinfo.txt


ffmpeg -hide_banner -i /home/justin/files/projects/cloneai/data/processed/clip.aac -filter_complex silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-60dB
 /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed.aac


ffmpeg -hide_banner -i /home/justin/files/projects/cloneai/data/processed/clip.aac -filter_complex silenceremove=stop_periods=-1:timestamp=copy:stop_duration=1:stop_threshold=-60dB
 /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed.aac


ffmpeg -hide_banner -i /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed.aac -filter_complex ashowinfo -f null - 2> /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed_ashowinfo.txt


so it looks like i need a newer version of ffmpeg. That's okay, we should just do that
the version i am using is like 2021

how can i install it? lets see 

here we see that the timestamp option was added

https://github.com/FFmpeg/FFmpeg/commits/16d4945e9aea8dab89ec7f83011379ae08712580/libavfilter/af_silenceremove.c

I want to try it and see if it would work. I would still need to post process it
https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

but it would be pretty nice and I don't have to worry too much about silence splitting
basically I'd look at when the timestamps don't line up 

i think i have to compile by source though 


# 01/24/2025

I haven't worked on this a little bit, I need a little bit of a reminder of what we are doing.
I think I left off with using a new version of ffmpeg so I have that filter that does the work I need
with a lot of different options. However the issue is that Ubuntu 22.04 doesn't have this, so I think I should
upgrade my Ubuntu actually to get the new version.

So I currently have
ffmpeg version 4.4.2-0ubuntu0.22.04.1+esm6 Copyright (c) 2000-2021 the FFmpeg developers

I need to have at least a version after 2023 because the commit was 
richardpl committed on May 28, 2023 

I know that I have Jammy Jellyfish which is Ubuntu 22.04.

On the package I see that I have: 7:4.4.2-0ubuntu0.22.04.1
https://launchpad.net/ubuntu/+source/ffmpeg

This follows what I expect. So I need to see which version I would need to get that feature. It's version 4.4.2.

I need at least version FFmpeg 6.1.2 "Heaviside" since it was cut from master branch on 2023-10-29
https://git.ffmpeg.org/gitweb/ffmpeg.git/shortlog/n6.1.2?pg=21
So let's compile it from source:
https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

This was the installation configure:

```
cd ~/ffmpeg_sources && \
wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
tar xjvf ffmpeg-snapshot.tar.bz2 && \
cd ffmpeg && \
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-gnutls \
  --enable-libfdk-aac \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-nonfree && \
PATH="$HOME/bin:$PATH" make && \
make install && \
hash -r
```

I also had to run this command:

```
sudo apt-get install libunistring-dev
```
https://askubuntu.com/questions/1252997/unable-to-compile-ffmpeg-on-ubuntu-20-04

Okay I compiled latest version. Now let's try our silenceremove filter thing:

```
ffmpeg -hide_banner -i /home/justin/files/projects/cloneai/data/processed/clip.aac -filter_complex silenceremove=stop_periods=-1:timestamp=copy:stop_duration=1:stop_threshold=-60dB /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed.aac
```

Now we need to see the timestamps to see if that worked:


```
ffmpeg -hide_banner -i /home/justin/files/projects/cloneai/data/processed/clip.aac -filter_complex ashowinfo -f null - 2> /home/justin/files/projects/cloneai/data/processed/processed/clip_ashowinfo.txt

ffmpeg -hide_banner -i /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed.aac -filter_complex ashowinfo -f null - 2> /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed_ashowinfo.txt
```

Hmm this doesn't seem to have worked necessarily.

Let's try with write

```
ffmpeg -hide_banner -i /home/justin/files/projects/cloneai/data/processed/clip.aac -filter_complex silenceremove=stop_periods=-1:timestamp=write:stop_duration=1:stop_threshold=-60dB /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed.aac

ffmpeg -hide_banner -i /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed.aac -filter_complex ashowinfo -f null - 2> /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed_ashowinfo.txt
```

No change... I think I'm not looking at the timestamps correctly maybe?

Hmm didn't seem to have worked... let's see. Maybe I am just not looking at the option correctly?

Okay I am trying with -loglevel debug:

```
ffmpeg -loglevel debug -hide_banner -i /home/justin/files/projects/cloneai/data/processed/clip.aac -filter_complex silenceremove=stop_periods=-1:timestamp=copy:stop_duration=1:stop_threshold=-60dB /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed_copy.aac &> /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed_copy.log

then press ENTER
```

OH actually it does look like it worked in the log. Am I sure I am outputting the correct afshowinfo? 

So is the ashowinfo working or actually is it a real bug where we are keeping it but it is getting overwritten? 

I'm not really sure. I'm not too confident that we are getting the actual PTS though... i want to see how to output the real metadata not just looking at chatgpt

there is this:

```
ffprobe -hide_banner -report -i "http://distribution.bbb3d.renderfarming.net/video/mp4/bbb_sunflower_1080p_30fps_normal.mp4" -show_entries "packet=pts,pts_time,dts,dts_time,pos" -read_intervals 00:00:00.000%+#2


ffprobe -hide_banner -report -i "/home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed_copy.aac" -show_entries "packet=pts,pts_time,dts,dts_time,pos" -read_intervals 00:00:00.000%+#2
```

I used this:

```
justin@ThinkPad-W530:~/ffmpeg_sources/ffmpeg$ ffprobe -hide_banner -report -i "/home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed_copy.aac" -show_entries "packet=pts,pts_time,dts,dts_time,pos" -read_intervals 00:00:00.000% > /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed_copy.report
```

Wait I think that worked! 

```
justin@ThinkPad-W530:~/ffmpeg_sources/ffmpeg$ ffprobe -hide_banner -report -i "/home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed_write.aac" -show_entries "packet=pts,pts_time,dts,dts_time,pos" -read_intervals 00:00:00.000% > /home/justin/files/projects/cloneai/data/processed/processed/clip_silence_removed_write.report
```

Nope nevermind didn't seem to work....

Okay here is what I'm going to send:

```
ffmpeg -loglevel debug -hide_banner -i "~/files/test_ffmpeg/in.wav" -filter_complex silenceremove=stop_periods=-1:timestamp=copy:stop_duration=1:stop_threshold=-60dB ~/files/test_ffmpeg/out.mp4 &> ~/files/test_ffmpeg/out.log


justin@ThinkPad-W530:~/ffmpeg_sources/ffmpeg$ ffprobe -hide_banner -report -i "http://distribution.bbb3d.renderfarming.net/video/mp4/bbb_sunflower_1080p_30fps_normal.mp4" -show_entries "packet=pts,pts_time,dts,dts_time,pos" -read_intervals 00:00:00.000%10
```

https://freesound.org/people/frosthardr/sounds/253067/



```
https://freesound.org/people/frosthardr/sounds/253067/

ffmpeg -y -i ~/files/test_ffmpeg/in.wav -filter_complex silenceremove=stop_periods=-1:timestamp=copy:stop_threshold=-30dB ~/files/test_ffmpeg/out_copy.wav

ffmpeg -y -i ~/files/test_ffmpeg/in.wav -filter_complex silenceremove=stop_periods=-1:timestamp=write:stop_threshold=-30dB ~/files/test_ffmpeg/out_write.wav

ffmpeg -i ~/files/test_ffmpeg/in.wav -filter_complex ashowinfo -f null - 2> ~/files/test_ffmpeg/in_ashowinfo.txt
ffmpeg -i ~/files/test_ffmpeg/out_copy.wav -filter_complex ashowinfo -f null - 2> ~/files/test_ffmpeg/out_copy_ashowinfo.txt
ffmpeg -i ~/files/test_ffmpeg/out_write.wav -filter_complex ashowinfo -f null - 2> ~/files/test_ffmpeg/out_write_ashowinfo.txt
```

I think the easiest thing to do now is just to do it myself. with my own function to calculate the numpy values. It is a bit too much of a headache with ffmpeg because i would still need to parse and implement that.

If I just go block by block with my own numpy operation, I can do things a lot more simply. I can reject samples that are too short in length.

OKAY ! yes it is working. I have it basically dividing down the numpy array and converting it back. So I think we could do this.

High level how am I going to do this...

1. go through and generate a window and see what the energy in that window is. Okay I'm kind of copying what is going on 

numpy.lib.stride_tricks.sliding_window_view

So we can easily get this and then with that view we just calculate the RMS. Well it's strided basically 

Okay I can do this... create a sliding window... lets see

Ahh okay let's do this:

https://stackoverflow.com/a/8260297

we convolve to get the RMS of each section and then the only thing is is that those sections are separated
but that's okay.

I think the issue is that i'm not sure how the overlap or frame thing works. Does it really matter? I'm not too sure. Can't I just do it one by one ? 

So basically i have the RMS of segments... hmmm doesn't it not matter? then I'll go through and see how long the seconds are. If the seconds are less than the time, I will basically merge them. 

Oh no it is an issue. Because if i happen to clip then I'll miss out important information basically.  But not if I make the window small enough, then it'll be fine I think. And I'll add padding on the left and right of the window basically time wise? 

https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html

https://numpy.org/doc/stable/reference/arrays.ndarray.html#arrays-ndarray

Okay let's just think about this. Do I need to do it? Yeah I guess so? 


I'm a little unsure about how the strides are necessary. I mean I have one reason ->

if i have a large frame length, like 1 second, then, if the speech is very short I will absolutely miss this.

But if I have RMS from multiple then it sort of averages and makes the RMS values much smoother. the RMS transitions are much smoother. that's for sure. Ok it sort of makes sense. I'll miss sharp transients if I don't have a moving window thing. I'll miss the transients basically because of speech.

But maybe I don't even need the short transients when I think about it though. Because... I am already removing those short transients anyway. If I'm in the middle of a phrase, and I have silence protection.

I don't think I need this. I think I should just try the RMS thing and filter base on that and then on the seconds. I am kind of adding it back in anyway.

Okay I'll just do the convolving thing and then I will get a bunch of RMS windows. and I'll convert to db, and then I'll go through each window size and I'll see how long in seconds it is. If it's under the threshold for longer than 1 second of silence I'll cut it out, but leave a little room on both ends for the end of a sentence. 

Okay okay okay I think I see it. I imagine like a step function of audio. and lets say i define my hoplength to be no overlap. Then basically what happens is that lets say the window just barely clips the step funciton. the energy will be too low so we will cut all of that silence out. Then the next portion will be max energy right? so then we will basically have missed the beginning of that sentence. If it's a sharp transient. 

Now compare this to if we slowly move a big window around. Then basically we will never "miss" any audio because we aren't missing that big step funciton. we'll always have some wiggle room. It makes a little more sense. It will basically neatly overlap with everything. and I won't miss a sharp step transient from sound. we can try like 50% or something. I don't need to analyze more than 1 second actually. I don't need very fine windows. I need like 1 second windows basically, calculate the RMS, and then the hop_length can be 50%. So within 0.5 seconds.

If I didn't then I would miss transients which is fine, but i would miss sharp transients in the next frame. imagine "My name ISSSSSSSSSSSSSSSSSSSSSS" I would miss the "my" without overlap. or maybe even the My name but I would get the ISSSSSSSSSSSSSSSSSSSSSSSSSSSSS. Now whereas if I did it slowly then some of the ISSSSSSSS is affecting the My name and I will be sure to get the My name. Basically I'm making sure I don't miss the influence of some edges on the rest 

~~~~VVVVVVV~~~~~~-----~~~~VVVVV

so in the above waveform we see that unless my window is perfect we will cut out some of the ~~~~ but we want the ~~~~ and to do that we can
still use a big window but just slide over it to align everything properly. Otherwise we'd need a really small window to do so. So basically it is an improvement to use a big window liek 2 seconds but then overlap to make sure we get all the different areas.

Okay no actually this is a perfect example

----~~~~VVVV

So let's say the window length is 2 characters

-- -- ~~ ~~ VV VV
|| || || || || ||

Here we miss the ~~ because we have no overlap

compare to

----~~~~VVVV
||||||||||||

which is 50% overlap (moving every 1 character)

now we analyze ~V and we see that actually this is higher than we expect
so we can now include the ~V part right before.

And if we make the window larger, then perhaps

~~~~V would be detected (5 characters). But it's unlikely that we would lign up perfectly to

-~~~~ is totally possible or 
--~~~ 

or 
----~

then we would have ~~~~V and sure we might get that but not the first ~ because we have no windowing in effect
but if we windowed 50% we would have

----~ then --~~~ then ~~~~~ then ~~~VV and maybe the second to last one is enough to trip. but for sure not without windowing! 

Okay I need to learn how to create strides, which is actually pretty simple with numpy. So I should do that then calculate the RMS using that window technique, and then I should go through, maybe overlap 


# 01/25/2025

Okay so Paul says the filter works fine. Ok... go ahead and show this with MP4.

Ehhh okay I don't want to do this. Meh I think I'm going to do it myself. Cleaner that way... and i learn a lot more

So let's first understand strides -> 

https://stackoverflow.com/questions/53097952/how-to-understand-numpy-strides-for-layman

OK first I'm just going to implement the db filtering thing and see how that goes

maybe I use librosa and just adapt some of the functions for my own use.

Okay I like just using librosa. I have the outputs all in numpy arrays so I don't need to do much and I can just add my silence thing on top of it. I just think that i'll be spending so much time trying to make my own function which basically does the same thing. And I think this is super efficient. I still need to stack things in a buffer way and know how to do that, but at least this is nice 

# 01/26/2025

Okay today I am going to go ahead and implement this using librosa. I'll get the silence bits removed. it works fine 

i first need to remind myself how to sample using librosa. is it a function?

I think I'll go ahead and just use ss and to to seek audio using the ffmpeg -ss and -to.
The advantage here is that I'll handle data in like 30 second chunks. The exact data will be flucuating but
it should be fine enough. Otherwise I'm not sure if I read the aac file with raw data I will end up
at the right location (what if I'm in the middle of a frame, ya know?). At least with this
I know I have 30 seconds of audio, or whatever. And I'll overlap and maybe do accurate seek to align things correctly
(-accurate_seek). 

okay where i left off was I was just verifying that we had the correct position seeking.
If I understand this correctly looks like we are slightly off the position seeking actually... i might be misunderstanding
though. I think the issue is that i am printing numpy arrays with not all digits


# 02/01/2025

So we should go through and at the start we start at the first position and go until the end for sure. Then if on the next one, we have something started at 0 then we should include it basically. Then same for the next one, so we'll always be comparing the next? although I think it gets a little complciated because what about the next next one.

wait i forgot I am trying to recreate the one without a chunk.

let's see what that looks like:

# MINE:

Processing chunk 0
[[0.224 2.912]
 [4.032 5.   ]]
Processing chunk 1
[[0.    1.664]
 [2.784 5.   ]]
Processing chunk 2
[[0.    0.416]
 [1.536 4.256]]
Processing chunk 3
[[0.288 2.976]
 [4.096 5.   ]]
Processing chunk 4
[[0.    1.728]
 [2.848 5.   ]]
Processing chunk 5
[[0.    0.48 ]
 [1.568 4.   ]]
Processing chunk 6
[[0.32  2.752]
 [4.128 5.   ]]
Processing chunk 7
[[0.    1.504]
 [2.88  5.   ]]
Processing chunk 8
[[0.    0.32 ]
 [1.632 4.256]]
Processing chunk 9
[[0.384 3.008]]
Processing chunk 10
[[0.   1.76]]
Processing chunk 11
[[0.      1.52875]]
Processing chunk 12
[[0.      0.27875]]


# GOLDEN

Processing chunk 0
[[ 0.224  2.912]
 [ 4.032  6.72 ]
 [ 7.84  10.24 ]
 [11.648 14.208]]

 I need to convert the previous to the final and then 
 cut based on length

 Okay no difference on where I put the -input or the -output


 .venv/bin/python -m cloneai.split
Processing chunk 0
[[0.224 2.912]
 [4.032 5.   ]]
Processing chunk 1
[[0.    1.92 ]
 [3.008 5.   ]]
Processing chunk 2
[[0.    0.928]
 [2.016 4.768]]
Processing chunk 3
[[1.024 3.776]
 [4.832 5.   ]]
Processing chunk 4
[[0.032 2.72 ]
 [3.84  5.   ]]
Processing chunk 5
[[0.    1.728]
 [2.848 5.   ]]
Processing chunk 6
[[0.    0.704]
 [1.824 4.256]]
Processing chunk 7
[[0.832 3.232]
 [4.64  5.   ]]
Processing chunk 8
[[0.    2.24 ]
 [3.648 5.   ]]
Processing chunk 9
[[0.    1.312]
 [2.656 5.   ]]
Processing chunk 10
[[0.    0.32 ]
 [1.632 4.256]]
Processing chunk 11
[[0.64  3.264]]
Processing chunk 12
[[0.   2.24]]
Processing chunk 13
[[0.    1.408]]
Processing chunk 14
[[0.      1.27875]]
Processing chunk 15
[[0.      0.27875]]
(.venv) justin@ThinkPad-W530:~/files/projects/cloneai$ make split
.venv/bin/python -m cloneai.split
Processing chunk 0
[[0.224 2.912]
 [4.032 5.   ]]
Processing chunk 1
[[0.    1.92 ]
 [3.008 5.   ]]
Processing chunk 2
[[0.    0.928]
 [2.016 4.768]]
Processing chunk 3
[[1.024 3.776]
 [4.832 5.   ]]
Processing chunk 4
[[0.032 2.72 ]
 [3.84  5.   ]]
Processing chunk 5
[[0.    1.728]
 [2.848 5.   ]]
Processing chunk 6
[[0.    0.704]
 [1.824 4.256]]
Processing chunk 7
[[0.832 3.232]
 [4.64  5.   ]]
Processing chunk 8
[[0.    2.24 ]
 [3.648 5.   ]]
Processing chunk 9
[[0.    1.312]
 [2.656 5.   ]]
Processing chunk 10
[[0.    0.32 ]
 [1.632 4.256]]
Processing chunk 11
[[0.64  3.264]]
Processing chunk 12
[[0.   2.24]]
Processing chunk 13
[[0.    1.408]]
Processing chunk 14
[[0.      1.27875]]
Processing chunk 15
[[0.      0.27875]]

I don't think I am accurately seeking here


Processing chunk 0
0.224 3.456
4.032 5.0
Processing chunk 1
1.25 3.458
4.034 6.25
Processing chunk 2
2.5 3.46
4.036 7.3
Processing chunk 3
4.038 7.302
7.846 8.75
Processing chunk 4
5.0 7.304
7.848 10.0
Processing chunk 5
6.25 7.306
7.818 10.826
Processing chunk 6
7.82 10.828
11.628 12.5
Processing chunk 7
8.75 10.83
11.629999999999999 13.75
Processing chunk 8
10.0 10.832
11.632 14.832
Processing chunk 9
11.634 14.834
Processing chunk 10
12.5 14.836
Processing chunk 11
13.75 14.838000000000001
Processing chunk 12


0.224 3.456
4.032 7.296
7.84 10.816
11.648 14.848


# okay one way is to just have 2 pointers
# we'll go through and check the end and the start.
# if the end is greater than the start and its greater than 5
seconds

so we are at 5 seconds, we shouldn't look at anything before 5 seconds
3.75 and it goes until 6 so we should include that 
