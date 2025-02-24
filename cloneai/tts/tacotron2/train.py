"""
Prepare dataset for Tacotron2 training

Consists of converting audio files to mel spectograms

We will be converting all the audio files to spectograms before training
and saving them to a data directory (pickled?)
"""

import librosa
import re
import os 

def run(indir):
    
    for root, file_dir, files in os.walk(indir):
        if len(files) == 0:
            # because we are at the directory not the sub-directory level
            continue

        name = re.match(r"\d+-(\D+)_?(\d+)?", os.path.basename(root))
        name = name.group(1)

        with open(os.path.join(root, f"transcriptions.txt"), "r") as f:
            for line in f:
                print(line)
                
                
                
                
                
                
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
