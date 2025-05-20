import yaml
import cloneai.data.extract as extract
import cloneai.data.split as split
import cloneai.data.transcribe as transcribe


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