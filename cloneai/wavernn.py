import random
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from cloneai.utils import plot_spectrogram, plot_waveform, normalized_waveform_to_bits, bits_to_normalized_waveform
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.prototype.functional import oscillator_bank
import os
from torchaudio.models.wavernn import WaveRNN

device = "cuda" if torch.cuda.is_available() else "cpu"


class WaveRNNDataset(Dataset):
    """PyTorch wrapper for WaveRNN data so we can use DataLoader"""

    def __init__(self, data, dummy_data=False):
        if dummy_data:
          self.data = []

          config = dict(
              sample_rate = 22050,
              n_fft = int(0.05*22050),
              hop_length = int(0.0125*22050),
              n_mels = 80,
              window_fn = torch.hann_window
          )
          print(config)

          transform = T.MelSpectrogram(**config)

          for i in range(10):
            # adopted from https://docs.pytorch.org/audio/stable/tutorials/oscillator_tutorial.html?highlight=sine+wave
            # generates a staircase frequency plot
            num_samples = random.randint(50000, 480000)
            num_steps = 10
            f0 = 2000
            fmod = 2000
            mod_type = "staircase" # or triangle

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
            waveform = oscillator_bank(freq, amp, sample_rate=config["sample_rate"])

            mel = transform(waveform)
            print(waveform.shape, mel.shape)
            self.data.append((torch.randint(0, 20, (1, 20)), waveform, mel))
        else:
          self.data = data


        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        _, waveform, mel = self.data[idx]
        return (waveform, mel)


def collate_fn_wrapper(frames_per_forward, hop_length, kernel_size, min_mag):
  """Wrapper around collate because we need additional arguments"""
  def collate_fn(batch):
    """Randomly chunk max frames from batch for input to WaveRNN

    Taken from PyTorch's example pipeline_wavernn
    """

    waveforms = []
    mels = []

    #######################################
    # unpack batch into waveform and mels
    #######################################

    for idx, (waveform, mel) in enumerate(batch):
        assert mel.shape[-1] >= frames_per_forward, "Not enough frames per forward"

        # as per tacotron2 paper, clip to minimum value and compress with log
        mel = torch.clamp(mel, min=min_mag).log10()

        mels.append(mel)
        waveforms.append(waveform)

    #######################################
    # prepare slicing
    #######################################

    pad = (kernel_size - 1) // 2

    # input waveform length
    wave_length = hop_length * frames_per_forward
    # input spectrogram length
    spec_length = frames_per_forward + pad * 2

    # max start postion in spectrogram
    max_offsets = [mel.shape[-1] - (spec_length + pad * 2) for mel in mels]

    # random start postion in spectrogram
    spec_offsets = [random.randint(0, offset) for offset in max_offsets]
    # random start postion in waveform
    wave_offsets = [(offset + pad) * hop_length for offset in spec_offsets]

    waveform_combine = [waveform[offset : offset + wave_length + 1] for waveform, offset in zip(waveforms, wave_offsets)]
    specgram = [mel[:, offset : offset + spec_length] for mel, offset in zip(mels, spec_offsets)]

    specgram = torch.stack(specgram)
    waveform_combine = torch.stack(waveform_combine)

    waveform = waveform_combine[:, :wave_length]
    target = waveform_combine[:, 1:]

    target = normalized_waveform_to_bits(target, 8)

    # print(f"{max_offsets=}")
    # print(f"{spec_offsets=}")
    # print(f"{wave_offsets=}")

    #######################################
    # sanity check
    #######################################

    # plot_waveform(waveform[0], xlim=(0, 0.01))
    # plot_waveform(target[0], xlim=(1.01, 1.02), ylim=(0, 255))
    # plot_spectrogram(mels[0], logCompressed=True)[0]
    # plot_spectrogram(specgram[0], logCompressed=True)[0]

    return waveform.unsqueeze(1), specgram.unsqueeze(1), target.unsqueeze(1)

  return collate_fn




#######################################################################

def run(data):



  batch_size = 4
  frames_per_forward = 100 # number of frames to feed into WaveRNN
  sr = 22050
  min_mag = 0.01
  hop_length = int(0.0125*sr)
  kernel_size = 5 # first conv1d kernel size reduces spectrogram by kernel_size-1 samples
  dataset = WaveRNNDataset(None, True)

  train = DataLoader(dataset,
                    batch_size=batch_size,
                    collate_fn=collate_fn_wrapper(frames_per_forward, hop_length, kernel_size, min_mag))

  elem = next(iter(train))

  ##############################################################################

  bits = 8
  epochs = 10
  criterion = nn.CrossEntropyLoss()
  lr = 0.001
  betas = (0.9, 0.999)
  eps = 1e-08
  weight_decay = 1e-06


  model = WaveRNN(
      upsample_scales=[5, 5, 11], # gives hop_length samples of 275
      n_classes=2**bits, # 8 bits
      hop_length=275, # by default hop_length == 200
      n_freq=80,
      kernel_size=5
  )

  optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas,
                              eps=eps, weight_decay=weight_decay)
  model = model.to(device)
  model.train()

  print("")

  ########################################

  # Training loop

  for epoch in range(epochs):
    for waveform, specgram, target in train:

      waveform = waveform.to(device)
      specgram = specgram.to(device)
      target = target.to(device)

      output = model(waveform, specgram)

      # output.shape == (N, time, 256) -> need to transpose for CrossEntropyLoss
      # target.shape == (N, time)
      output, target = output.squeeze(1), target.squeeze(1)
      output = output.transpose(1, 2)
      target = target.long() # has to be long for CrossEntropyLoss

      loss = criterion(output, target)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if epoch % 1 == 0:
          loss = loss.item()
          print(f"Epoch {epoch+1} | loss: {loss:>7f}")


  print("Done!")



  ####################################

  probs = torch.softmax(output.transpose(1, 2), dim=-1)
  probs = probs.view(-1, 256)
  indices = torch.multinomial(probs, num_samples=1)
  indices = indices.view(target.shape[0], -1)

  # predicted = bits_to_normalized_waveform(indices, 8)

  plot_waveform(target[0], xlim=(0, 0.001), ylim=(0, 255))
  plot_waveform(indices[0], xlim=(0, 0.001), ylim=(0, 255))


  plot_waveform(target[0], xlim=(1.0, 1.001), ylim=(0, 255))
  plot_waveform(indices[0], xlim=(1.0, 1.001), ylim=(0, 255))
