# CloneAI

This is a program which can digitally "clone" voices. The inspiration for this project came after my two best friends and I decided to record all of our voice conversations through Discord over the span of 2 years. Having GBs and hours of single-track audio data, voice cloning seemed like the perfect application.

To summarize the project: a TTS engine which is trained (or fine-tuned) on pre-recorded audio from a single speaker. An inference program then runs as a Discord bot and lets you and your friends generate speech from text by sending slash commands.

# Environment setup

To handle the compute requirements, I created a Google Compute VM instance with a single T4 GPU. Using a GPU is not strictly necessary, but the audio processing and the model training will be significantly slower if you do it on a CPU-only machine.

## Google Compute VM setup

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

You can also install the Remote SSH connection for VScode to acces the VM through the external IP. If you do use this flow, I recommend installing the VS code Python extension directly on the remote machine. Also note that if you stop your machine via the Google Cloud console and start it back up, sometimes the host key will change and subsequent SSH connections via VS code will fail. You either have to delete the `~/.ssh/known_hosts` entry from the local machine and/or rerun the `gcloud` command shown above.


### Setting up SSH keys for GitHub

To access my GitHub repo, I set up SSH keys on the VM by generating an SSH keypair and adding it to my GitHub profile.

```
ssh-keygen -t ed25519 -C "YOUR_EMAIL"
```

## Pre-requisites

First step is to clone this GitHub repository.

```
git clone https://github.com/j-silv/cloneai.git
# git clone git@github.com:j-silv/cloneai.git # for dev
```

Next, we need to install the required Python modules and libraries.

I installed everything in a separate `venv` within this repository's root folder.

```
cd cloneai
python -m venv venv
source ./venv/bin/activate
```

### Dependencies

- PyTorch (Cuda 11.8) (`torchaudio` needed so we can download weights for pretrained model)
- Discord library, YAML parser, Matplotlib, and Whisper for transcription
- ffmpeg to process audio files

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 # GPU
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu # local computer

pip install "py-cord[voice]" PyYAML openai-whisper matplotlib
```

# How to run

## Download checkpoints for fine-tuning

To have a starting point for adjusting the tacotron2 and waveglow model, I instantiated the models from PyTorch's documentation and saved it to a `state_dict` within the projects main directory.

You can run the following command to do this:

```
make get_weights
```

## Download raw audio data

Starting from scratch, you are going to need lots of single-track audio for the particular speaker you want to clone. In my case, this was hours of audio recorded through Discord with the Craig bot which were saved to my Google Drive. My raw data was thus multiple zip files from Google Drive which contained nested zip files each with a recording that has multiple tracks.

So the first step is this to unzip all of these directories and create a single directory for each speaker which holds all the raw audio. We also need to convert these audio files into a standardized format for subsequent model training.

You first need to manually download the audio files from Craig to the machine you are using. Since I am using a Google VM, I used `scp` to copy the zipped folders from my local computer:

```
# mkdir data/raw # on remote machine so that scp doesn't fail
cd PATH_TO_ZIPPED_FILES
# zip ./audio_data.zip ./Craig-*.zip # if Google Drive gives you separated .zips 
scp ./audio_data.zip YOURNAME@VM_IP:~/cloneai/data/raw
```

Note that you don't *have* to use Craig as the source of audio. However, the data extraction, splitting, and transcription steps of the TTS pipeline expect the format that Craig uses to package the data. Namely, individual folders for each separate recording with separate speaker tracks:

```
├── craig_RECID_DATE_TIME
    ├── 1-SPEAKER_USERNAME.aac
    ├── 2-SPEAKER_USERNAME.aac
    └── ...etc.
```

## Prepare data

In the `config.yaml` file, you can enable the following steps:

- extract:
    - extract data from the compressed zip files
- split:
    - group each audio file per speaker in their own directory
    - process audio with FFmpeg and split audio files into short chunks based on silences
- transcribe:
    - run OpenAI Whisper transcription on the resulting chunks

Then you can run:

```
make main
```
and the enabled steps will run.

## Tacotron2 (text-to-spectrogram)

After getting the audio files and their associated transcriptions, we can now train the tacotron2 model as the first step of the TTS pipeline.

We are downloading a pre-trained model from PyTorch audio. This is because the API for obtaining the model is straightforward. We can simply wrap the model around a fine-tuning optimization and the goal is that this speeds up training time required.

We might have to scratch this though and train a tacotron2 model from the ground-up, if we find that using a pre-trained model is degrading the performance.

The code in the `tacotron2.py` module might benefit from comparision with [PyTorch's example pipeline](https://github.com/pytorch/audio/tree/release/0.12/examples/pipeline_tacotron2). I think the general flow is similar, but I haven't compared in depth between the two.

## WaveRNN (spectrogram-to-speech / vocoder)

This part of the project has been arduous. In a similar vein as above, I wanted to create my own (simple) vocoder training loop so that I could learn more than just integrating someone elses.

I got very close but then got stuck because of some odd GPU memory utilization which I can't reconcile. I included a very bare-bones example which shows how PyTorch cannot allocate enough GPU memory (apparently?) to support a reasonable amount of time samples. Nobody has provided any input on the PyTorch forums, and this was inhibiting a lot of my progress.

I briefly explored the WaveRNN repository created by Andrew Gibiansky, but this had too many bells and whistles for my simple application. The repository was also somewhat out of date. Eventually we can explore some of the optimizations, but initially I just wanted to get a working pipeline.

Thus I decided that despite the still lingering GPU usage issue, I will mitigate it by simply splitting up the sequence lengths of the melspectrograms small enough. This is what Andrew's code did and the example pipeline from PyTorch.

## Note about smaller dataset

To evaluate the training loops without getting bogged down by the high number of training samples, I am using a small dataset with just my friend Scott's voice. Once the module works for this small dataset and we can confirm the model's outputs, then we can transition to the entire Discord dataset.


## Converting to mono:

As of 05/27/2025, there is a small bug which doesn't standardize the tracks to mono-channel. It's on my todo list!

In the meantime, the files can be converted to mono with the following terminal commands:

```
for i in *.wav; do ffmpeg -y -i "$i" -ac 1 "mono_${i}"; done
for i in mono_*; do mv "$i" "${i/#mono_/}"; done
```
