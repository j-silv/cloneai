"""
Use Whisper to transcribe short sentence audio clips 

- check confidence level (i.e. probability) for transcription and discard if 
  below a certain threshold

Runs on Google Collab sheet. Make sure that the following commands are ran:

!pip install git+https://github.com/openai/whisper.git
!sudo apt update && sudo apt install ffmpeg

"""

import os
import re
import textwrap
import torch
import whisper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run(speaker_dir, model, verbose=False, ignore=[], min_confidence=-0.5, del_wav_if_no_transcribe=False):

    print("Starting transcription")
    model = whisper.load_model(model)

    for root, file_dir, files in os.walk(speaker_dir):
        if len(files) == 0:
            # because we are at the directory not the sub-directory level
            continue

        name = re.match(r"\d+-(\D+)_?(\d+)?", os.path.basename(root))
        name = name.group(1)
        name = re.sub(r"_", r"", name)

        if name in ignore:
            continue

        with open(os.path.join(root, f"transcriptions.txt"), "w") as f:

            for idx, filename in enumerate(files):
                if filename == "transcriptions.txt":
                    continue
                print("Processing", f"{filename:<15}", end="")
                name = re.match(r"(\w+)\.", filename).group(1)
                ext = re.search(r"\w+$", filename).group(0)
                infile = os.path.join(root, filename)

                result = run_whisper(model, infile)
                avg_logprob = result.avg_logprob
                text = result.text

                if verbose:
                    shortened = textwrap.shorten(text, width=50)
                    print(f" {avg_logprob=:.5f} | text={shortened}")

                if avg_logprob > min_confidence:
                    print(f"{filename}|{text}", file=f)
                elif del_wav_if_no_transcribe:
                    print(f"File removed due to low confidence")
                    os.remove(infile)


def run_whisper(model, filename):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    return result














