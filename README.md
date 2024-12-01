# CloneAI

This is a program which can digitally "clone" you or your friends voice and personality. The inspiration for this project came after my two best friends and I decided to record all of our voice conversations through Discord over the span of a year. Having GBs of audio data, I thought of possible applications I could target and one of them was the creation of voice clones of ourselves. Some further enhancements were to add real time (online) processing so that we could interact with these voice clones via Discord, Large Language Model (LLM) support so we could ask questions to these clones, and fine-tuning personality of the LLM so that it responds with our personality

# Pipeline

1. Speech to text 
2. Text to response
3. Response tweaked with personality
4. Text to speech

# Other possibles names

- CloneAI
- CloneBot
- CloneBotAI
- AIme
- AIandMe


# How to run

The `data.py` script processes Craig audio data (multi-track audio recorded from a Discord bot) and gathers the audio files into separate folders within the `data/raw` directory. You can modify the `config.yaml` to change the directory naming convention of the raw data directories.

## 0. Set up environment

I am using `venv` and installing the project specific dependencies into that.

```
python3 -m venv .venv
source .venv/bin/activate
```
This is optional however if you want to install the packages system-wide through pip. The Makefile will check whether or not a `.venv` exists in the root directory, so you don't need to manually activate everytime.

## 1. Download audio data

To do this, I went to Google Drive and downloaded the Craig folder into multiple zip files to the data/raw directory.

I then used a shell script to unzip everything into a single folder.

# References

- [Stanford Speech Language Processing textbook](https://web.stanford.edu/~jurafsky/slp3/ed3bookaug20_2024.pdf)