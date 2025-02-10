# CloneAI

This is a program which can digitally "clone" you or your friends voice and personality. The inspiration for this project came after my two best friends and I decided to record all of our voice conversations through Discord over the span of a year. Having GBs of audio data, I thought of possible applications I could target and one of them was the creation of voice clones of ourselves. Some further enhancements were to add real time (online) processing so that we could interact with these voice clones via Discord, Large Language Model (LLM) support so we could ask questions to these clones, and fine-tuning personality of the LLM so that it responds with our personality

# Pipeline

1. Speech to text
2. Text to response
3. Response tweaked with personality
4. Text to speech

# Text to speech (TTS)

I decided to start backwards from the pipeline and have a TTS engine which is fine-tuned on pre-recorded audio from a single speaker. This would be a fun little program which runs as a Discord bot and lets you and your friends generate speech from text by sending slash commands.

## Pre-requisites

To handle fine-tuning the TTS models and for quickly processing each audio file, I created a Google Compute VM instance with an added T4 GPU. This is not strictly necessary, but the audio processing and the model training will be significantly slower if you do it on a CPU-only machine.

### Create a Google Compute VM

Going to the Google Cloud console, we can select a standard VM with GPU support. Here are the specs that I chose:

- n1-standard-4 (4 vCPU, 2 core, 15 GB memory)
- 1 NVIDIA T4 GPU
- 200 GBs storage (for holding all audio data, libraries, and model checkpoints)
- spot provision (way cheaper, and the VM doesn't need to run continuously)
- Debian 11 Deep Learning VM disk image with Python 3.10 and Cuda 11.8 preinstalled

### Accessing VM

I found the easiest way to connect is to first run the `gcloud ssh` command which you can find by going to your VM instances in the Google Compute Engine console and clicking on SSH gcloud command line:

```
gcloud compute ssh --zone "YOUR_ZONE" "VM_INSTANCE_NAME" --project "YOUR_PROJECT_NAME"
```

On subsequent connections, you can then use basic SSH to connect to the machine's external IP.

```
ssh VM_IP
```

You can also install the Remote SSH connection for VScode to acces the VM through the external IP. If you do use this flow, I went ahead and installed the VS code Python extension directly on the remote machine.


### Setting up SSH keys for GitHub

To access my GitHub repo, I set up SSH keys on the VM by generating an SSH keypair and adding it to my GitHub profile.

```
ssh-keygen -t ed25519 -C "YOUR_EMAIL"
```

### Setting up VM environment

First step is to clone this GitHub repository. In my home directory, I ran the following command. Note that the submodules flag is required since we are using NVidia's tacotron2 and waveglow implementations and fine-tuning them.

A note on the sub-module used here:

To support TensorFlow 2.0, I have used a small fork from user abalanonline instead of the main NVidia tacotron2 which removes some depreciated code.

```
git submodule add git@github.com:abalanonline/tacotron2.git cloneai/tts/tacotron2
git submodule update --init --recursive
```

```
git clone --recurse-submodules git@github.com:j-silv/cloneai.git
```

Next we need to install the required Python modules and libraries. There are slightly different requirements depending on if you are running on a CPU-only machine or a GPU machine. The following instructions are for GPU machines.

I created everything in a separate `venv` within this repository's root folder.

```
cd cloneai
python -m venv venv
source ./venv/bin/activate
```

#### GPU requirements

- PyTorch (Cuda 11.8) (`torchaudio` needed so we can download weights for pretrained model)
    - `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118`
- cloneai pip requirements
    - Discord library, YAML parser, Pandas, and Whisper for transcription
    - `pip install "py-cord[voice]" PyYAML openai-whisper pandas`
- Install requirements for Nvidia's tacotron2 implementation
    - Note that we can't directly install the `requirements.txt` due to an incompatibility with the latest `tensorflow` package. So just manually install them. To support the Cuda 11.8 drivers, we are installing tf==2.14:
    - `pip install "matplotlib" "tensorflow==2.14" "numpy<2" librosa scipy pillow Unidecode inflect`
    - The requirement that the `apex` library be installed is only if you want to run with `fp16` precision. In my case, I ignored this installation.
- Install requirements for waveglow
    - Although there also is a `requirements.txt`, I think the previous installations cover this, so you can ignore this installation
- ffmpeg to process audio files
    - `sudo apt update && sudo apt install ffmpeg`

#### Download checkpoints for fine-tuning

To have a starting point for adjusting the tacotron2 and waveglow model, I instantiated the models from PyTorch's documentation and saved it to a `state_dict` within the projects main directory.

You can run the following command to do this:

```
make get_weights
```

#### Download raw audio data

These are nested zipped directories directly from Google Drive and generated by a Discord bot called Craig. This is a manual step to download them to the VM. I used `scp` to copy from my computer. While on your remote machine:

```
mkdir -p data/raw
```

And while on your local machine:

```
cd PATH_TO_ZIPPED_FILES
zip ./audio_data.zip ./Craig-*.zip
scp ./audio_data.zip YOURNAME@VM_IP:~/cloneai/data/raw
```

You can clean up the individual zip files after you copy it over:

```
rm -f ./Craig-*.zip
```



## How to run - text to speech

Starting from scratch, you are going to need lots of single track audio for the particular speaker you want to clone. In my case, this was hours of audio recorded through Discord with the Craig bot which were saved to my Google Drive. My raw data was thus multiple zip files from Google Drive which contained nested zip files each with a recording that has multiple tracks.

So the first step is this to unzip all of these directories and create a single directory for each speaker which holds all the raw audio. We also need to convert these audio files into a standardized format for subsequent model training.


### Prepare data

In the `config.yaml` file, you can enable the following steps:

- extract
- split
- transcribe

Then you can run:

```
make main
```
and the previous steps will be ran.

### Converting to mono:

```
for i in *.wav; do ffmpeg -y -i "$i" -ac 1 "mono_${i}"; done
for i in mono_*; do mv "$i" "${i/#mono_/}"; done
```

### Tensorboard

```
https://www.montefischer.com/2020/02/20/tensorboard-with-gcp.html
gcloud compute ssh [INSTANCE_NAME] -- -NfL 6006:localhost:6006
tensorboard --logdir=./results/results
```

### Fine-tuning tacotron2

As mentionned before, the Nvidia implementation of tacotron2 is used. 

