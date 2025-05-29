
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

def plot_waveform(waveform, sample_rate=22050, title="Waveform", xlim=None, ylim=None):
  """Taken from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html"""
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)
  
  return figure

def plot_spectrogram(spec, sample_rate=22050, hop_length=200, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('time')

  num_channels, num_mels, num_frames = spec.shape

  assert num_channels == 1, "Plot spectogram only supports mono-chanel"

  db_multiplier = torch.log10(torch.clamp(spec.max(), min=1e-10))
  db = F.amplitude_to_DB(spec.squeeze(0), 10.0, 1e-10, db_multiplier)

  time_axis = torch.arange(0, num_frames)*(hop_length / sample_rate)

  im = axs.imshow(db, origin='lower', aspect=aspect,
                  extent=[0, time_axis.max(), 0, num_mels-1])
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

  return fig

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