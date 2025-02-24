import sys
sys.path.append("./cloneai/tts/tacotron2")
import torch
import os
import yaml
import cloneai.data.extract as extract
import cloneai.data.split as split
import cloneai.data.transcribe as transcribe
from cloneai.tts.tacotron2.hparams import create_hparams
from cloneai.tts.tacotron2.train import train


def load_config(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)



if __name__ == "__main__":
    config = load_config("config.yaml")
    ROOT = os.getcwd()

    # -------------------------------------------------------------------

    params = config["data"]["extract"]
    raw_dir = os.path.join(ROOT, params["dir"]["raw"])
    extracted_dir = os.path.join(ROOT, params["dir"]["extracted"])
    zip_relpath = params["dir"]["zip_relpath"]
    if params["enable"]:
        extract.run(raw_dir, extracted_dir,
                    params["merge"], params["ignore"], params["clean"], zip_relpath)

    # -------------------------------------------------------------------

    params = config["data"]["split"]
    processed_dir = os.path.join(ROOT, params["dir"]["processed"])
    speaker_dir = os.path.join(ROOT, params["dir"]["speaker"])
    if params["enable"]:
        split.run(speaker_dir, processed_dir,
                  params["out_format"],  params["sample_rate_hz"],
                  params["silence_db"], params["min_silence_s"],
                  params["min_nonsilence_s"], clean=params["clean"], progress=params["progress"], num_channels=params["num_channels"],
                  verbose=params["verbose"], accurate=params["accurate"], max_splits=params["max_splits"], ignore=params["ignore"])


    # -------------------------------------------------------------------

    params = config["data"]["transcribe"]
    if params["enable"]:
        transcribe.run(params["dir"]["processed"], params["model"], params["verbose"],
                       params["ignore"], params["min_confidence"], params["del_wav_if_no_transcribe"])

    # -------------------------------------------------------------------
    
    
    params = config["model"]["tacotron2"]
    if params["enable"]:
        
        # create training and validation set
        transcription = os.path.join(params['dir']['processed'], 'transcriptions.txt')
        training = os.path.join(params['dir']['processed'], "training.txt")
        validation = os.path.join(params['dir']['processed'], "validation.txt")
        
        with open(transcription, "r") as f:
            transcription_lines = f.readlines()
            
        training_size = int(len(transcription_lines)*params["test_split"])
        
        with open(training, "w") as f:
            for line in transcription_lines[:training_size]:
                f.write(f"{params['dir']['processed']}/")
                f.write(line)
 
        with open(validation, "w") as f:
            for line in transcription_lines[training_size:]:
                f.write(f"{params['dir']['processed']}/")
                f.write(line)            
        
        # training_path = os.path.join("..", "..", training)
        # validation_path = os.path.join("..", "..", validation)
        training_path = training
        validation_path = validation
        
        override_hparams = f"training_files={training_path},validation_files={validation_path},batch_size={params['batch_size']}"
        hparams = create_hparams(override_hparams)
        
        torch.backends.cudnn.enabled = hparams.cudnn_enabled
        torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
        
        args = params["args"]
        # train(args["output_dir"], args["log_dir"], None,
        #   False, args["n_gpus"], args["rank"], args["group_name"], hparams)
        train(args["output_dir"], args["log_dir"], args["checkpoint_path"],
          args["warm_start"], args["n_gpus"], args["rank"], args["group_name"], hparams)

    # -------------------------------------------------------------------