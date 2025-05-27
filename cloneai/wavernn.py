# Python packages

import re
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchaudio
from torchaudio.models.wavernn import WaveRNN
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from google.colab import drive

drive.mount('/content/drive')
device = "cuda" if torch.cuda.is_available() else "cpu"





# get dataset, print some info, and plot first waveform

in_dir = "/content/drive/MyDrive/cloneai/processed/2-scott-small"
dataset = AudioDataset(in_dir, "transcriptions.txt", resample=False)

print(f"{dataset.num_audio_files=}",
      f"{dataset.audio_data.shape=}",
      f"{dataset.audio_lengths.shape=}",
      f"{dataset.audio_data[0].shape=}",
      f"{dataset.audio_lengths[0]=}",
      f"{dataset.specgram_lengths[0]=}",
      f"{dataset.sr=}",
      f"{dataset.audio_length_samples=}",
      f"{dataset.frame_hop_samples=}",
      f"{dataset.frame_size_samples=}",
      f"{dataset.mels.shape=}",
      f"{dataset.log_mels.shape=}",
      f"{dataset.log_mels.max()=}",
      f"{dataset.log_mels.min()=}\n",

      sep="\n")

audio = pad_or_trim(dataset.audio_data[0], 200000)
plot_waveform(audio,xlim=(0,25), title="Trimmed waveform")
plot_waveform(dataset.audio_data[0],xlim=(0,25),  title="Padded waveform")
plot_spectrogram(dataset.mels[0],  title="Melspectrogram")





###########################################################################


# try training

bits = 8
epochs = 10
criterion = nn.CrossEntropyLoss()
lr = 0.001
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 1e-06



model = WaveRNN(
    upsample_scales=[5, 5, 8], # gives hop_length samples of 200
    n_classes=2**bits, # 8 bits
    hop_length=dataset.frame_hop_samples, # by default hop_length == 200
    n_freq=dataset.n_mels,
    kernel_size=dataset.kernel_size
)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas,
                             eps=eps, weight_decay=weight_decay)


model = model.to(device)
model.train()


# Training loop

for epoch in range(epochs):
  print(f"Epoch {epoch+1}\n-------------------------------")

  # note that the torchaudio has a bg_iterator which I can't find any docs on
  # but there is this discussion thread:
  # https://discuss.pytorch.org/t/audio-dataset-load-large-file-into-memory-in-background/85943/2
  for waveform, specgram in dataset:



    waveform = waveform[:,0:479400] # hacky way to make sure alignment

    target = normalized_waveform_to_bits(waveform, bits)

    waveform = torch.unsqueeze(waveform, 0).to(device)
    specgram = specgram.unsqueeze(0).to(device)
    target = target.to(device)


    # print(waveform.device, target.device, specgram.device)
    # print(waveform.shape, specgram.shape, target.shape)
    output = model(waveform, specgram)


    output, target = output.squeeze(1), target.squeeze(1)

    loss = criterion(output, target)


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # if epoch % 10 == 0:
    #     loss = loss.item()
    #     print(f"loss: {loss:>7f}")

    break
  break

print("Done!")
