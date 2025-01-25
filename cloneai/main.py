import os
import yaml
import cloneai.data.extract as extract
import cloneai.data.split as split
import cloneai.data.transcribe as transcribe

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
    if params["enable"]:
        extract.run(raw_dir, extracted_dir,
                    params["merge"], params["ignore"], params["clean"])

    # -------------------------------------------------------------------

    params = config["data"]["split"]
    processed_dir = os.path.join(ROOT, params["dir"]["processed"])
    speaker_dir = os.path.join(ROOT, params["dir"]["speaker"])
    if params["enable"]:
        split.run(speaker_dir, processed_dir,
                  params["out_format"],  params["sample_rate_hz"],
                  params["silence_db"], params["min_silence_s"],
                  params["min_nonsilence_s"], clean=params["clean"], progress=params["progress"],
                  accurate=params["accurate"], ignore=params["ignore"])


    # -------------------------------------------------------------------

    params = config["data"]["transcribe"]
    if params["enable"]:
        transcribe.run()

    # -------------------------------------------------------------------