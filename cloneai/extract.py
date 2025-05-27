"""
Unzip and prepare raw data that is collected from Craig recordings

These folders were auto uploaded to my Google Drive account, and now
this script proceeds to unzip them and create data that is ready to be processed
by the ML models

- Unzip everything from raw data location
- Create destination folders for speaker1, speaker2, speaker3 in output folder
- Determine speaker unique IDs by iterating through every folder and hashing them as names
- Separate tracks into the respective folders (map speaker names to processed folder) -> only process individual tracks
- TODO: Delete recordings under a certain file size ?
- TODO: Delete recordings with a single speaker ? 
"""

import zipfile
import os
import re
import shutil

def extract_zips(input_zip, in_dir, out_dir, log, recurse=False, clean=False):
    """Extract zip files from in_dir to out_dir
    
    Also creates a log file in the out_dir to monitor extraction

    Params:
        input_zip  : file to unzip (relative to 'in_dir')
        in_dir     : input directory
        out_dir    : output directory
        recurse    : extract a nested zip file and continue until there is no more zipfile
        clean      : delete zip archive we just extracted

    Returns:
        full path of extracted file as elements in a list
    """
    result = []
    
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(os.path.join(in_dir, input_zip), "r") as zip_ref:
        print(f"Extracting: {zip_ref.filename}\n", file=log, end="")
        members = zip_ref.namelist()
        for member in members:
            print(f"\t{os.path.join(out_dir, member)}", file=log)
        print(file=log, flush=True)
        zip_ref.extractall(out_dir)

        if clean is True:
            os.remove(zip_ref.filename)

        
        for member in members:
            member_name, member_ext = os.path.splitext(member)
            if member_ext == ".zip" and recurse is True:
                member_outdir = os.path.join(out_dir, member_name)
                result.extend(extract_zips(member, out_dir, member_outdir, log, recurse, clean))
            elif member_ext.strip() != "":
                member_outpath = os.path.join(out_dir, member)
                result.append(member_outpath)
                
        return result
                

def group_tracks(audio_files, out_dir, merge, ignore, clean=False):
    """Group all individual tracks into a central location

    Return dictionary for unique track speakers based on track name

    Each key will be the unique name of the speaker, and the value will
    be another dictionary that has a key for the directory location for all
    tracks belonging to that speaker, and the number of tracks for that speaker

    output files are copied to speaker specific directory and are of the format:
    "REC_TRACK.EXT" where:
        REC    : recording name and number
        TRACK  : track number for this speaker
        EXT    : {"acc", "flac"}
    
    with REC and TRACK we can determine information on the distribution of speakers
    for each recording later

    Params:
        audio_files  : list of full path extracted audio files
        out_dir      : output directory
        merge        : list of speaker names to merge into one
        ignore       : list of speakers to ignore
        clean        : delete extracted audio file folders as we group

    Returns:
        speaker dictionary
    """
    speaker = dict()

    
    for file in audio_files:
        
        # because we want to identify just the string part of the name
        basename = os.path.basename(file)
        name = re.search(r"\d+-(\D+)_?\d+", basename)
        
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
        
        output_filename = os.path.basename(os.path.dirname(file)) 
        output_filename = f"{output_filename}_{str(speaker[name]['num'])}.{ext}"
        output_filename = os.path.join(speaker[name]["dir"], output_filename)
    
        shutil.copy2(file, output_filename)
        speaker[name]["num"] += 1

        if clean is True:
            shutil.rmtree(file)
    
    dirs = []
    for dir in speaker.values():
        dirs.append(dir['dir'])
    return dirs



def run(input_zip, in_dir, out_dir, merge, ignore, clean):
    """
        Params:
            input_zip  : file containing audio to unzip
            in_dir     : input directory
            out_dir    : output directory
            merge      : list of speaker names to merge into one
            ignore     : list of speakers to ignore
            clean      : delete extracted audio file folders as we group
    """

    print("Extracting audio data archives... this might take a while")
    
    input_name, _ = os.path.splitext(input_zip)
    out_dir = os.path.join(out_dir, input_name)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, f"extract.log"), "w") as log:

        audio_files = extract_zips(input_zip, in_dir, out_dir, log, recurse=True, clean=clean)
        speakers = group_tracks(audio_files, os.path.join(out_dir, "speaker"), merge, ignore, clean)
        return speakers

    print("Extraction completed")