"""
Unzip and prepare raw data that is collected from Craig recordings

These folders were auto uploaded to my Google Drive account, and now
this script proceeds to unzip them and create data that is ready to be processed
by the ML models

- Unzip everything from raw data location
- Create destination folders for speaker1, speaker2, speaker3 in data/processed folder
- Determine speaker unique IDs by iterating through every folder and hashing them as names
- Separate tracks into the respective folders (map speaker names to processed folder) -> only process individual tracks
- Delete recordings under a certain file size
- Delete recordings with a single speaker
"""

import zipfile
import os
import glob
import re
import shutil

def extract_zips(in_dir, out_dir, nested=False, clean=False):
    """Extract zip files from in_dir to out_dir
    
    Also creates a log file in the out_dir to monitor extraction

    Params:
        in_dir  : folder with zip files
        out_dir : output directory
        nested  : extract a nested zip file 
        clean   : delete zip archive we just extracted

    Returns:
        number of extractions
    """
    os.makedirs(out_dir, exist_ok=True)
    extraction_num = 0 
    zip_files = glob.glob(f"{in_dir}/*.zip")

    # for debugging, just use a few zip files instead of the whole directory 
    # zip_files = zip_files[2:4] 

    for file in zip_files:
        with open(os.path.join(out_dir, f"extract_{nested}_{clean}_{os.path.basename(file)}.log"), "w") as log:
            with zipfile.ZipFile(file, "r") as zip_ref:
                if nested is False:
                    extraction_num += 1
                    wrap_dir = os.path.join(out_dir, str(extraction_num))
                    os.makedirs(wrap_dir, exist_ok=True)
                    print(f"Extract #{extraction_num}: {zip_ref.filename}", file=log, flush=True)
                    zip_ref.extractall(wrap_dir)
                else:
                    for member in zip_ref.infolist():
                        extraction_num += 1
                        out_file = os.path.join(out_dir, member.filename)
                        print(f"Extract #{extraction_num}: {member.filename} ---> {out_file}", file=log, flush=True)
                    zip_ref.extractall(out_dir)
            if clean is True:
                os.remove(file)

            print("\nNumber of extractions:", extraction_num, file=log, flush=True)
    return extraction_num

def group_tracks(in_dir, out_dir, merge, ignore, clean=False):
    """Group all individual tracks into a central location

    Return dictionary for unique track speakers based on track name

    Each key will be the unique name of the speaker, and the value will
    be another dictionary that has a key for the directory location for all
    tracks belonging to that speaker, and the number of tracks for that speaker

    output files are copied to speaker specific directory and are of the format:
    "REC_TRACK.EXT" where:
        REC    : recording number
        TRACK  : track number for this speaker
        EXT    : {"acc", "flac"}
    
    with REC and TRACK we can determine information on the distribution of speakers
    for each recording later

    Params:
        in_dir  : root folder with extracted audio files
        out_dir : output directory
        merge   : list of speaker names to merge into one
        ignore  : list of speakers to ignore
        clean   : delete extracted audio file folders as we group

    Returns:
        speaker dictionary
    """
    speaker = dict()

    for root, file_dir, files in os.walk(os.path.join(in_dir)):
        if len(file_dir) != 0:
            # because we are at the directory not the sub-directory level
            continue
        
        for file in files:
            # because we want to identify just the string part of the name
            name = re.match(r"\d+-(\D+)_?\d+", file)
            if name is None:
                # raw.dat or info.txt
                continue
            
            # first capturing group above just gets the discord name
            # because some usernames have _ we just get the text name
            name = name.group(1)
            name = re.sub(r"_", r"", name) 

            if name in ignore:
                continue
            if name in merge:
                name = merge[0]

            ext = re.search(r"\w+$", file).group(0)

            if name not in speaker:
                speaker_dir = os.path.join(out_dir, f"{str(len(speaker))}-{name}")
                os.makedirs(speaker_dir, exist_ok=True)
                speaker[name] = dict(dir=speaker_dir, num=0)
            
            src = os.path.join(root, file)
            dest = os.path.join(speaker[name]["dir"], f"{os.path.basename(root)}_{str(speaker[name]['num'])}.{ext}")
            shutil.copy2(src, dest)
            speaker[name]["num"] += 1

        if clean is True:
            shutil.rmtree(root)
        
    return speaker



def run(raw_dir, out_dir, merge, ignore, clean, zip_relpath):
    """
        Params:
            in_dir  : root folder with extracted audio files
            out_dir : output directory
            merge   : list of speaker names to merge into one
            ignore  : list of speakers to ignore
            clean   : delete extracted audio file folders as we group
    """

    print("Extracting audio data archives... this might take a while")

    # we will call this function thrice as we have a nested zip archive
    # put clean to False the first time because this is the raw audio data
    extract_zips(raw_dir, out_dir, nested=True, clean=False)
    extract_zips(out_dir, out_dir, nested=True, clean=clean)

    # because we might have a wrapper around each zip file
    out_dir = os.path.join(out_dir, zip_relpath)
    extract_zips(out_dir, out_dir, nested=False, clean=clean)

    speaker_dir = os.path.join(raw_dir, "speaker")
    group_tracks(out_dir, speaker_dir, merge, ignore, clean)
    print("Extraction completed")