
# Utility functions
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torchaudio.functional as F


def pad_or_trim(array, length):
    """Pad or trim the audio tensor to length"""
    if array.shape[-1] > length:
        array = array[..., :length]

    elif array.shape[-1] < length:
        pad_width = (0, length - array.shape[-1])
        array = nn.functional.pad(array, pad_width)

    return array

def plot_waveform(waveform, sample_rate=22050, title="Waveform", xlim=None, ylim=None, outputImg=""):
  """Taken from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html"""
  waveform = waveform.cpu().detach().numpy()

  num_channels, num_frames = waveform.shape
  
  time_axis = torch.arange(0, num_frames) / sample_rate
  fig, ax = plt.subplots() # only plot one channel

  ax.plot(time_axis, waveform[0], linewidth=1)
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

  num_channels, num_mels, num_frames = spec.shape
  assert num_channels == 1, "Plot spectogram only supports mono-chanel"

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
  
  im = ax.imshow(data[0], origin='lower', aspect="auto",
                  extent=[0, time_axis.max(), 0, num_mels-1])
  if xmax:
    ax.set_xlim((0, xmax))
  fig.colorbar(im, ax=ax, label=label)

  if outputImg != "":
    fig.savefig(outputImg)

  return fig, ax

def normalized_waveform_to_bits(waveform: torch.Tensor, bits: int) -> torch.Tensor:
    """Transform waveform [-1, 1] to label [0, 2 ** bits - 1]

    we have to convert the target waveform to the class because
    we are doing cross entropy loss
    """

    assert abs(waveform).max() <= 1.0
    waveform = (waveform + 1.0) * (2**bits - 1) / 2
    return torch.clamp(waveform, 0, 2**bits - 1).int()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)