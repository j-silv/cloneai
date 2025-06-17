import yaml
import cloneai.extract as extract
import cloneai.split as split
import cloneai.transcribe as transcribe
import cloneai.load as load
import cloneai.tacotron2 as tacotron2
import cloneai.wavernn as wavernn


def load_config(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)



if __name__ == "__main__":
    config = load_config("config.yaml")

    # -------------------------------------------------------------------

    params = config["data"]["extract"]
    if params["enable"]:
        speakers = extract.run(params["input_zip"], params["dir"]["in"], params["dir"]["out"],
                               params["merge"], params["ignore"], params["clean"])

    # -------------------------------------------------------------------

    params = config["data"]["split"]
    if params["enable"]:
        split.run(params["dir"]["in"], params["dir"]["out"],
                  params["out_format"],  params["sample_rate_hz"],
                  params["silence_db"], params["min_silence_s"],
                  params["min_nonsilence_s"], clean=params["clean"],
                  progress=params["progress"], num_channels=params["num_channels"],
                  verbose=params["verbose"], accurate=params["accurate"],
                  max_splits=params["max_splits"], ignore=params["ignore"])


    # -------------------------------------------------------------------

    params = config["data"]["transcribe"]
    if params["enable"]:
        transcribe.run(params["dir"]["in"], params["model"], params["verbose"],
                       params["ignore"], params["min_confidence"],
                       params["del_wav_if_no_transcribe"])

    # -------------------------------------------------------------------
    
    params = config["load"]
    if params["enable"]:
        data = load.run(params["dir"]["in"], params["dir"]["out"],
                           params["resample"], params["audio"], params["dummy_data"])

    # -------------------------------------------------------------------  
    
    params = config["tacotron2"]
    if params["enable"]:
        tacotron2.run(data, params["dir"]["out"],
                      params["seed"],
                      params["load_checkpoint"],
                      params["save_checkpoint"],
                      params["hyperparams"],
                      params["tokenizer"])

    # -------------------------------------------------------------------  
    
    params = config["wavernn"]
    if params["enable"]:
        wavernn.run(data, params["dir"]["out"],
                    params["seed"],
                    params["load_checkpoint"],
                    params["save_checkpoint"],
                    params["hyperparams"],)

    # -------------------------------------------------------------------  