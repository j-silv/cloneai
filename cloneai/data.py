"""
Unzip and prepare raw data that is collected from Craig recordings

These folders were auto uploaded to my Google Drive account and now I will
proceed to unzip them and create data that is ready to be processed
by the ML models

- Unzip everything from raw data location
- Create destination folders for speaker1, speaker2, speaker3 in data/processed folder
- Determine speaker unique IDs by iterating through every folder and hashing them as names
- Separate tracks into the respective folders (map speaker names to processed folder) -> only process individual tracks
- Delete recordings under a certain size/size
- Delete recordings with a single speaker
- Convert all audio to the same format and sampling rate (might need to batch use raw.dat or re-sample)
- Break up audio into short sentence phrases
    - cut audio whenever there is a gap of more than couple of seconds? pause in dialog? look at algorithms for this
    - pydub is apparently one library to do this. Let's see if there is any sort of low-level usage with ffmpeg? silence.split_on_silence()
    - https://stackoverflow.com/questions/45526996/split-audio-files-using-silence-detection
-For consistency into whisper and training make sure audio files:
    - Are in WAV format (not necessary, can be flac as well)
    - Have a sample rate of 16kHz (this should be done)
    - Are mono-channel (this should also be done)

    
TODO: when iterating through the intermediate data folders, we should just use the os.walk() method
      instead of relying on the number of zips that are determined from each folder.
      I guess it is pretty equivalent but meh. I guess it doesn't really matter both are correct

TODO: save disk space. Right now I am just making sure all the steps work
      but for final script I want to delete zip files as we go and move files as we go
      otherwise I will need a really big drive to do this. Right now I see that the data folder is taking up
      150 GBs!!! And we started with 20 GBs. So delete zip files as we go once I know everything works 
"""

import zipfile
import yaml
import os
import glob
import re
import shutil

def extract_drive_zips(raw_dir):
    """Google drive downloads big folders in multiple zips
    
    At the start we have Craig-####-001.zip, Craig-####-002.zip, etc.
    Afterwards, we have 0/Craig, 1/Craig, etc. and within each "Craig"
    folder we have the zip files for each recording

    Returns num_zips because later we iterate through all of the drive zips
    """
    zip_files = glob.glob(f"{raw_dir}/*.zip")
    num_zips = len(zip_files)

    for i, file in enumerate(zip_files):
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(os.path.join(raw_dir, str(i)))
    
    return num_zips



def extract_craig_zips(raw_dir, num_zips):
    """Craig packages all recordings in a zip file
    
    After here we have all/0, all/1, etc. and within each # folder,
    we have the individual tracks for each speaker

    Returns num_recordings because later we iterate through all craig zips 
    """

    all_dir = os.path.join(raw_dir, "all")
    os.makedirs(all_dir, exist_ok=True)

    offset = 0
    bad_files = []

    with open(os.path.join(raw_dir, "extract.log"), "w") as log:
              
        for i in range(num_zips):
            zip_files = glob.glob(f"{raw_dir}/{i}/Craig/*.zip")

            for j, file in enumerate(zip_files):
                with zipfile.ZipFile(file, "r") as zip_ref:

                    print(f"Extract #{j+offset}:", zip_ref.filename, file=log, flush=True)
                    ok_zip = zip_ref.testzip()

                    if ok_zip is not None:
                        bad_files.append(ok_zip)
                        print(f"ERROR: could not extract {zip_ref.filename}", file=log, flush=True)
                        offset = offset-1 # skip the one that was invalid
                    else:
                        zip_ref.extractall(os.path.join(all_dir, str(j + offset)))
                    
            # so that we extract to the next number in order
            offset = offset + j + 1


        num_recordings = offset
        print("\nNumber of recordings:", num_recordings, file=log, flush=True)
        print("\nSummary of bad zip files:", bad_files, file=log, flush=True)

    return num_recordings


def group_tracks(raw_dir, num_recordings, merge, ignore):
    """Group all individual tracks into a central location

    Return dictionary for unique track speakers based on track name

    Each key will be the unique name of the speaker, and the value will
    be another dictionary that has a key for the directory location for all
    tracks belonging to that speaker, and the number of tracks for that speaker
    """
    speaker = dict()

    for i in range(num_recordings):
        for file_dir, _, files in os.walk(os.path.join(raw_dir, "all", str(i))):
            for file in files:
                # because we want to identify just the string part of the name
                name = re.match(r"\d+-(\D+)_?\d+", file)
                if name is None:
                    # raw.dat or info.txt
                    continue
                
                name = name.group(1)


                # because some usernames have _ 
                name = re.sub(r"_", r"", name) 

                if name in ignore:
                    continue

                if name in merge:
                    name = merge[0]

                ext = re.search(r"\w+$", file).group(0)

                if name not in speaker:
                    speaker_dir = os.path.join(raw_dir, "speaker", f"{str(len(speaker))}-{name}")
                    os.makedirs(speaker_dir, exist_ok=True)
                    speaker[name] = dict(dir=speaker_dir, num=0)
                
                # output file is of format REC#_NUM#.ext where
                # ext == {"acc", "flac"}
                # REC# == recording number
                # NUM# == track number for this speaker
                # with these 2 pieces of information we can determine which how many recordings each person is in

                # os.rename(os.path.join(file_dir, file), os.path.join(speaker[name]["dir"], f"{str(i)}_{str(speaker[name]['num'])}.{ext}"))
                
                # we might want to replace this with the one above because then we have extra space 
                # but we can just delete this stuff later i guess 
                shutil.copy2(os.path.join(file_dir, file), os.path.join(speaker[name]["dir"], f"{str(i)}_{str(speaker[name]['num'])}.{ext}"))
                speaker[name]["num"] += 1

    return speaker

def combine_speakears():
    """Merge speaker tracks together if same person
    
    
    This is an optional step because some usernames had changed perhaps
    This will merge together tracks from speakers. We probably should just do this 
    in the previous function though because that is when we are creating it
    We will also have an ignore speaker name (like Maki, Alistair, Craig, etc.)
    So ignore speaker and also merge speakers 
    """
    pass


if __name__ == "__main__":
    ROOT = os.getcwd()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    raw_dir = os.path.join(ROOT, config["dir"]["data"]["raw"])
    
    print("Extracting audio data archives... this might take a while")

    # num_zips = extract_drive_zips(raw_dir)

    # num_zips = 7
    # num_recordings = extract_craig_zips(raw_dir, num_zips)

    num_recordings = 877 # hard-coded to test individual functions


    speaker = group_tracks(raw_dir, num_recordings, config["data"]["merge"], config["data"]["ignore"])


