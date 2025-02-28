"""
Prepare dataset for Tacotron2 training

Consists of converting audio files to mel spectograms

We will be converting all the audio files to spectograms before training
and saving them to a data directory (pickled?)
"""

import librosa
import re
import os 
import torch
import whisper


def log_mel_spectogram(audio, n_fft, hop_length, win_length, sampling_rate_hz, n_mels):
    window = torch.hann_window(win_length)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_filters = librosa.filters.mel(sr=sampling_rate_hz, n_fft=n_fft, n_mels=n_mels),
    mel_spec = mel_filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec
    
    

def run(indir, sample_rate_hz, pad_sample_length, n_fft, hop_length, win_length, n_mels):
    
    re_transcription = re.compile(r"(.*)\|(.*)")
    
    with open(os.path.join(indir, f"transcriptions.txt"), "r") as f:
        lines = f.readlines()
        
    num_audio_files = len(lines)
    
    # we will have each audio sample with length: pad_sample_length
    # we will have a window length of win_length
    # so it's roughly pad_sample_length / win_length
    n_frames = pad_sample_length/hop_length
    
    mels = torch.tensor(num_audio_files, n_mels, n_frames)
    
    for line in lines:
        re_result = re_transcription.match(line)
        filename = os.path.join(indir, re_result.group(1))
        text = re_result.group(2)
        
        audio = whisper.load_audio(filename, sample_rate_hz)
        audio = whisper.pad_or_trim(audio, length=pad_sample_length)
        
        
        # mel = whisper.log_mel_spectrogram(audio, n_mels=80)
        mel = log_mel_spectogram()
        
        # calculate what mel should be so we can specify the n_frames
        
        
        print(mel.shape)
        

                
                
                
                
                
# import torch
# import torchaudio


# torch.random.manual_seed(0)
# device = "cuda" if torch.cuda.is_available() else "cpu"

# bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
# processor = bundle.get_text_processor()
# tacotron2 = bundle.get_tacotron2().to(device)


# from cloneai.tts.tacotron2.hparams import create_hparams
# from cloneai.tts.tacotron2.train import train

# create training and validation set
# transcription = os.path.join(params['dir']['processed'], 'transcriptions.txt')
# training = os.path.join(params['dir']['processed'], "training.txt")
# validation = os.path.join(params['dir']['processed'], "validation.txt")

# with open(transcription, "r") as f:
#     transcription_lines = f.readlines()
    
# training_size = int(len(transcription_lines)*params["test_split"])

# with open(training, "w") as f:
#     for line in transcription_lines[:training_size]:
#         f.write(f"{params['dir']['processed']}/")
#         f.write(line)

# with open(validation, "w") as f:
#     for line in transcription_lines[training_size:]:
#         f.write(f"{params['dir']['processed']}/")
#         f.write(line)            

# # training_path = os.path.join("..", "..", training)
# # validation_path = os.path.join("..", "..", validation)
# training_path = training
# validation_path = validation

# override_hparams = f"training_files={training_path},validation_files={validation_path},batch_size={params['batch_size']}"
# hparams = create_hparams(override_hparams)

# torch.backends.cudnn.enabled = hparams.cudnn_enabled
# torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

# args = params["args"]
# # train(args["output_dir"], args["log_dir"], None,
# #   False, args["n_gpus"], args["rank"], args["group_name"], hparams)
# train(args["output_dir"], args["log_dir"], args["checkpoint_path"],
#     args["warm_start"], args["n_gpus"], args["rank"], args["group_name"], hparams)
