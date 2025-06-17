import random
import torch
from torch import nn
from cloneai.utils import save_spectrogram, save_waveform, normalized_waveform_to_bits, bits_to_normalized_waveform
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.models.wavernn import WaveRNN

device = "cuda" if torch.cuda.is_available() else "cpu"


class WaveRNNDataset(Dataset):
    """PyTorch wrapper for WaveRNN data so we can use DataLoader"""

    def __init__(self, data):
        self.data = data
        self.num_samples = len(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # texts are unnecessary for wavernn training
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

    return waveform.unsqueeze(1), specgram.unsqueeze(1), target.unsqueeze(1)

  return collate_fn




#######################################################################

def run(data, out_dir, seed, load_checkpoint_path, save_checkpoint_path, hyperparams):
  #######################################
  # load data
  #######################################

  torch.manual_seed(seed)  
  
  dataset = WaveRNNDataset(data)
  
  print(f"{hyperparams['split']=}")
  print(f"{hyperparams['batch_size']=}")

  train_split, val_split, test_split = random_split(dataset, hyperparams["split"]) 

  train = DataLoader(train_split,
                     batch_size=hyperparams["batch_size"],
                     collate_fn=collate_fn_wrapper(hyperparams["frames_per_forward"],
                                                  dataset.data.audio_config.frame_hop_samples,
                                                  hyperparams["kernel_size"],
                                                  dataset.data.audio_config.min_mag))

  #######################################
  # sanity check
  #######################################

  elem = next(iter(train))

  save_waveform(out_dir, "wavernn_waveform.png", "Chunked waveform", elem[0][0], xlim=(0, 0.01))
  save_spectrogram(out_dir, "wavernn_specgram.png", "Chunked wavernn specgram", elem[1][0], logCompressed=True)
  save_waveform(out_dir, "wavernn_target.png", "Target waveform shifted", elem[2][0], xlim=(0, 0.01), ylim=(0, 255))

  #######################################
  # Set up model
  #######################################

  criterion = nn.CrossEntropyLoss()

  model = WaveRNN(
      upsample_scales=hyperparams["upsample_scales"], 
      n_classes=2**hyperparams["bits"],
      hop_length=dataset.data.audio_config.frame_hop_samples,
      n_freq=dataset.data.audio_config.n_mels,
      kernel_size=hyperparams["kernel_size"]
  )

  optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"],
                              betas=hyperparams["betas"],
                              eps=float(hyperparams["eps"]),
                              weight_decay=float(hyperparams["weight_decay"]))
  
  model = model.to(device)


  #######################################
  # Training loop
  #######################################  

  for epoch in range(hyperparams["epochs"]):
    model.train()
    loss = 0.0
    data_samples_seen = 0
    
    for batch, (waveform, specgram, target) in enumerate(train):
      data_samples_seen += waveform.shape[0]
      
      
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
          print(f"Training | Epoch {epoch+1} | batch {batch} | samples {data_samples_seen}/{len(train_split)} | loss: {loss:>7f}")


  print("Done!")



  ####################################

  probs = torch.softmax(output.transpose(1, 2), dim=-1)
  probs = probs.view(-1, 256)
  indices = torch.multinomial(probs, num_samples=1)
  indices = indices.view(target.shape[0], -1)

  predicted = bits_to_normalized_waveform(indices, 8)

  save_waveform(out_dir, "wavernn_target_waveform.png", "Target", target[0], xlim=(0, 0.001), ylim=(0, 255))
  save_waveform(out_dir, "wavernn_predicted_waveform.png", "Predicted", indices[0], xlim=(0, 0.001), ylim=(0, 255))

  # plot_waveform(target[0], xlim=(1.0, 1.001), ylim=(0, 255))
  # plot_waveform(indices[0], xlim=(1.0, 1.001), ylim=(0, 255))
