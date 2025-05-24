"""Create dataset.json for dataset which is expected by WaveRNN submodule

The WaveRNN submodule expects a dataset.json which splits training, validation, and testing samples.

Here we follow that convention and also use this for tacotron2 training.
"""

import json
import os
import random

def run(in_dir, out_dir, train_split, val_split, seed):

    # Generate dataset.json with train / validation / test split.
    
    # use transcriptions.txt to split up the datasets randomly
    in_file = os.path.join(in_dir, "transcriptions.txt")
    
    audio_files = []

    with open(in_file, "r") as handle:
        for line in handle:
            audio_file = line.split('|', 1)[0]
            audio_files.append(audio_file)

    num_audio_files = len(audio_files)
    
    random.seed(seed)
    random.shuffle(audio_files)
    
    train_start = 0
    train_end = int(train_split*num_audio_files)
    
    val_start = train_end
    val_end = val_start + int(val_split*num_audio_files)
    
    test_start = val_end
    test_end = num_audio_files
    
    train = audio_files[train_start:train_end]
    val = audio_files[val_start:val_end]
    test = audio_files[test_start:test_end]
    

    out_file = os.path.join(out_dir, "dataset.json")
    with open(out_file, "w") as handle:
        json.dump(
            {
                "train": train,
                "valid": val,
                "test": test,
            },
            handle,
            indent=2,
        )
