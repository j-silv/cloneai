# Utility functions
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchaudio.functional as F
from torchaudio.prototype.functional import oscillator_bank
import torchaudio.transforms as T
import os

def pad_tensor(array, length):
    """Pad audio tensor to length"""
    if array.shape[-1] < length:
        pad_width = (0, length - array.shape[-1])
        array = nn.functional.pad(array, pad_width)
    return array

def trim_tensor(array, length):
    """Trim audio tensor to length"""
    if array.shape[-1] > length:
        array = array[..., :length]
    return array

def save_waveform(out_dir, filename, plot_title, waveform, **kwargs):
  """Wrapper around plot_waveform with additional stdout"""
  
  outputImg = os.path.join(out_dir, filename)
  print(f"{plot_title}:", outputImg)
  plot_waveform(waveform, outputImg=outputImg, title=plot_title, **kwargs)

def save_spectrogram(out_dir, filename, plot_title, specgram, **kwargs):
  """Wrapper around plot_spectrogram with additional stdout"""
  
  outputImg = os.path.join(out_dir, filename)
  print(f"{plot_title}:", outputImg)
  plot_spectrogram(specgram, outputImg=outputImg, title=plot_title, **kwargs)


def plot_waveform(waveform, sample_rate=22050, title="Waveform", xlim=None, ylim=None, outputImg=""):
  """Taken from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html"""
  waveform = waveform.cpu().detach().numpy()

  if len(waveform.shape) == 2:
    num_channels, num_frames = waveform.shape
    assert num_channels == 1, "Plot waveform only supports mono-chanel"
    waveform = waveform.squeeze(0)

  num_frames = waveform.shape[0]

  time_axis = torch.arange(0, num_frames) / sample_rate
  fig, ax = plt.subplots() # only plot one channel

  ax.plot(time_axis, waveform, linewidth=1)
  ax.grid(True)

  if xlim is None:
    xlim = (0, time_axis[-1])
  if ylim is None:
    ylim = (-1, 1)

  ax.set_xlim(xlim)
  ax.set_ylim(ylim)

  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Amplitude")

  ax.set_title(title)

  if outputImg != "":
    fig.savefig(outputImg)

  return fig, ax

def plot_spectrogram(spec, title="Specgram", logCompressed=False,
                     xmax=None, outputImg="", ):
  spec = spec.cpu().detach()
  fig, ax = plt.subplots()
  ax.set_title(title or 'Spectrogram (dB)')
  ax.set_ylabel("Freq bin")
  ax.set_xlabel('Frame')

  if len(spec.shape) == 3:
    num_channels, num_mels, num_frames = spec.shape
    assert num_channels == 1, "Plot spectogram only supports mono-chanel"
    spec = spec.squeeze(0)

  num_mels, num_frames = spec.shape

  if logCompressed is False:
    label = "dB (0 == spec.max())"
    amin = 0.0 # minimum power level
    multiplier = 10.0 # input is power
    db_multiplier = torch.log10(spec.max()) # spec.max() is the ref value and dB values are scaled relative to spec.max()
    data = F.amplitude_to_DB(spec, multiplier, amin, db_multiplier) # supports channel dimension
  else:
    # we already clipped and applied log compression as per tacotron2 paper
    label = "log"
    data = spec

  time_axis = torch.arange(0, num_frames)

  im = ax.imshow(data, origin='lower', aspect="auto",
                  extent=[0, time_axis.max(), 0, num_mels-1])
  if xmax:
    ax.set_xlim((0, xmax))
  fig.colorbar(im, ax=ax, label=label)

  if outputImg != "":
    fig.savefig(outputImg)
    plt.close() # to allow for overwriting

  return fig, ax

def normalized_waveform_to_bits(waveform: torch.Tensor, bits: int) -> torch.Tensor:
    """Transform waveform [-1, 1] to label [0, 2 ** bits - 1]

    we have to convert the target waveform to the class because
    we are doing cross entropy loss
    """

    assert abs(waveform).max() <= 1.0
    waveform = (waveform + 1.0) * (2**bits - 1) / 2
    return torch.clamp(waveform, 0, 2**bits - 1).int()

def bits_to_normalized_waveform(label: torch.Tensor, bits: int) -> torch.Tensor:
    r"""Transform label [0, 2 ** bits - 1] to waveform [-1, 1]"""

    return 2 * label / (2**bits - 1.0) - 1.0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dummy_data(mel_config,
                      random_seed = 1337, num_data_samples=10,
                      min_audio_samples=50000, max_audio_samples=480000,
                      mod_type="staircase", # or triangle
                      f0=2000, fmod=2000, num_steps=10,
                      token_vocab_size=26, # number of symbols we could have
                      min_token_size=10, max_token_size=100): 
  """Generate dummy (but easily visual) data to train on
  
  The waveform generated steps between multiple frequencies and thus the resulting
  spectrogram gives a triangular (ramp) or a more abrupt staircase plot
  Adapted from: https://docs.pytorch.org/audio/stable/tutorials/oscillator_tutorial.html

  The text that is generated is random integers from [0, token_vocab_size)
  """
  random.seed(random_seed)
  
  transcriptions = []
  waveforms = []
  mels = []

  for i in range(num_data_samples):
    num_samples = random.randint(min_audio_samples, max_audio_samples)

    if mod_type == "staircase":
      freq = torch.linspace(f0-fmod, f0+fmod, num_steps)
      freq = freq.repeat_interleave(num_samples//num_steps)
      freq = freq[freq > 0.0].unsqueeze(-1) # if we can't fit interleave then we will have 0s
      num_samples = freq.shape[0]
    elif mod_type == "triangle":
      freq = torch.linspace(f0-fmod, f0+fmod, num_samples).unsqueeze(-1)
    else:
      raise ValueError("Not a valid mod_type:", mod_type)

    amp = torch.ones((num_samples, 1))
    waveform = oscillator_bank(freq, amp, sample_rate=mel_config.sr)

    transform = T.MelSpectrogram(
      sample_rate=mel_config.sr,
      n_fft=mel_config.frame_size_samples,
      hop_length=mel_config.frame_hop_samples,
      n_mels=mel_config.n_mels, 
      window_fn=mel_config.window
    )
    mel = transform(waveform)
    
    text_size = torch.randint(min_token_size, max_token_size+1, (1,)).item()
    text = torch.randint(0, token_vocab_size, (text_size,))
    
    transcriptions.append(text)
    waveforms.append(waveform)
    mels.append(mel)
    
    
  return transcriptions, waveforms, mels
        
  