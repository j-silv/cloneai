import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import re
import os
from dataclasses import dataclass
from cloneai.utils import pad_or_trim
import os
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class AudioConfig:
  sr: int = 16000
  frame_size_s: float = 0.05
  frame_hop_s: float = 0.0125
  audio_length_s: float = 30.0
  n_mels: int = 80
  window: str = "hann"
  kernel_size: int = 5
  min_mag: float = 0.01

  def __post_init__(self):
    """Add new fields after creating AudioConfig instance"""
    self.audio_length_samples = int(self.audio_length_s*self.sr)
    # TODO: set n_fft to a power of 2 for FFT speedup
    self.frame_size_samples = int(self.frame_size_s*self.sr)
    self.frame_hop_samples = int(self.frame_hop_s*self.sr)
    
    if self.window == "hann":
      self.window = torch.hann_window
    else:
      raise ValueError("Invalid window function for AudioConfig")
    


class AudioDataset(Dataset):
  """PyTorch dataset for training tacotron2
  
  We define pytorch dataset for easy to use dataloader and for
  batch processing later. This is where we load in the dataset
  from the transcription file and convert it to mel spectograms
  and symbolized text.
  
  The audio files are packaged with a PyTorch dataset wrapper: AudioDataset.
  The waveform path locations are specified in a transcription file with the following format:

    8_56_368.wav|We start with, we rethink everything.
    8_56_243.wav|Yes, I understand that. I'm going to use the Tinder pouch...
    8_56_331.wav|Sorry, he as in you. He as in you as in you as in Justin...
    8_56_368.wav|We start with, we rethink everything.

  Note that these audio files were transcribed with Whisper.
  Part of that flow is to pad audio samples to at least 30 seconds.
  The audio files themselves are shorter or longer, but the transcriptions correspond to padded audio.

  To leverage batch spectogram processing, we still pad/trim (using torchaudio utilities)
  and just keep track of the valid lengths of the waveforms and spectograms
  (one infers the other, but for simplicitly we will track both).
  This will result in a single spectogram call, and when we call the model we will splice with the valid indexes.  
  
  To create the spectrograms, we leverage torchaudio's melspectogram API.
  Originally librosa was used, but to avoid extra packages I decided
  to just stick with torchaudio.
  
  Note about lengths of audio and spectrograms:
  we need to adjust the lengths of the audio and specgrams.
  let's say we have N_valid, which is number of audio samples which are valid
  (e.g. 282020), and N which are the total number of padded audio samples
  (e.g. 480000). We have a fixed hop length of 200, and a kernel size of 5.
  so we have an audio sample size of 282020.
  And a hop length of 200, so how many of those frames can we fit? 282020 / 200 + 1 == 1411.1 == 1411
  so how do we deal with frames that don't fit? we integer it.
  so we need to actually determine the appropiate waveform shape which is
  (n_time - kernel_size + 1)*hop_length to be appropiate
  
  """  
  def __init__(self,
               in_dir,                   # root directory for audio file paths and transcription file
               transcript,               # map file containing audio file paths and transcriptions
               resample=False,           # if True, resample audio to specified sample rate
               processor=None,           # text processor (tokenizer). If None, use default
               **kwargs                  # keyword arguments that apply to AudioConfig dataclass
              ):

    self.audio_config = AudioConfig(**kwargs)

    if processor is None or processor == "WAVERNN_CHAR_LJSPEECH":
      processor = (
          torchaudio
          .pipelines
          .TACOTRON2_WAVERNN_CHAR_LJSPEECH
          .get_text_processor()
      )
    else:
      raise ValueError("Invalid processor value for AudioDataset")

    audio_files, transcriptions = self.load_text(in_dir, transcript)
    
    self.num_samples = len(audio_files)
    
    self.tokens, self.token_lens = self.text_to_tokens(transcriptions, processor)
    
    # TODO: we need to set this per batch -> otherwise one line in transcript
    #       which is long adds a lot of unnecessary memory for other batches
    self.max_token_len = self.token_lens[0] # equivalently, self.tokens.shape[-1]
    
    self.waveforms, self.waveform_lens, self.mel_lens =\
      self.load_audio(audio_files, resample, self.audio_config)    
    
    self.raw_mels, self.mels = self.create_mel(self.waveforms, self.audio_config)

    
    self.tokens = self.tokens.to(device)
    self.token_lens = self.token_lens.to(device)
    self.waveforms = self.waveforms.to(device)
    self.waveform_lens = self.waveform_lens.to(device)
    self.mels = self.mels.to(device)
    self.mel_lens = self.mel_lens.to(device)
    

  @staticmethod
  def load_text(in_dir, transcript):

    # e.g.: 8_56_368.wav|We start with, we rethink everything.
    re_transcription = re.compile(r"(.*)\|(.*)")

    audio_files    = []  # full path to audio files
    transcriptions = []  # raw transcription texts from Whisper
    
    with open(os.path.join(in_dir, transcript), "r") as f:
      for line in f:
        re_result = re_transcription.match(line)
        audio_files.append(os.path.join(in_dir, re_result.group(1)))
        transcriptions.append(re_result.group(2))
     
    return audio_files, transcriptions

  @staticmethod
  def text_to_tokens(transcriptions, processor):
    """Convert text to tokens
    
    tacotron2 forward pass expects the following shapes:
      tokens               - (n_batch, max of token_lengths)
      token_lengths        - (n_batch, )

    when a list is passed in, the text is automatically zero-padded
    to the max length of the list
    
    TODO: do this per batch when the model is called instead of the entire dataset
          at initialization
          
    """
    tokens, token_lengths = processor(transcriptions)  
    return tokens, token_lengths

  @staticmethod
  def load_audio(audio_files, resample, config):
    """load audio data into memory from audio_files path"""
    
    num_audio_files = len(audio_files)

    audio_data = torch.zeros((num_audio_files, 1, config.audio_length_samples))
    audio_lengths = torch.zeros((num_audio_files), dtype=torch.long)
    specgram_lengths = torch.zeros((num_audio_files), dtype=torch.long)

    for idx, audio_file in enumerate(audio_files):
        audio, native_sr = torchaudio.load(audio_file)
        assert audio.shape[0] == 1, "Only mono-channel audio files are supported"
        if resample is True:
          resampler = T.Resample(native_sr, config.sr, dtype=audio.dtype)
          audio = resampler(audio)
          config.sr = native_sr

        # use the full waveform shape to compute the expected number of specgrams
        # then update the audio_lengths to fit the pytorch API -> upsampling due to kernel
        specgram_lengths[idx] = (audio.shape[-1] // config.frame_hop_samples) + 1
        audio_lengths[idx] = (specgram_lengths[idx] - config.kernel_size + 1)*config.frame_hop_samples

        # padding is not optional because we are storing result in a tensor
        audio = pad_or_trim(audio, config.audio_length_samples)
        audio_data[idx,...] = audio
        
    return audio_data, audio_lengths, specgram_lengths


  @staticmethod
  def create_mel(waveforms, config):
      """generate the mel spectogram from the audio file

      waveRNN forward pass expects the following shapes:
        waveform         - (n_batch, 1, (n_time - kernel_size + 1)*hop_length
        specgram         - (n_batch, 1, n_freq, n_time)
      the specgram is upsampled in the last conv layer and this is all done
      automatically. we don't really need to take care of anything here
      """

      # +1 because that's how the STFT calculation works
      # tensor because this is actually an input into the model
      assert config.audio_length_samples % config.frame_hop_samples == 0
      num_frames = torch.tensor((config.audio_length_samples // config.frame_hop_samples) + 1)

      mel_transform = T.MelSpectrogram(
          sample_rate=config.sr,
          n_fft=config.frame_size_samples,
          hop_length=config.frame_hop_samples,
          n_mels=config.n_mels,
          window_fn=config.window
      )

      mels = mel_transform(waveforms)

      # as per tacotron2 paper, clip to minimum value and compress with log
      log_mels = torch.clamp(mels, min=config.min_mag).log10()
      
      return mels, log_mels

  def split_train_val_test(self, split, seed):
    """Generate train / validation / test split indices
    
    Unused since PyTorch already has a helper function
    """
    
    random.seed(seed)
    
    assert sum(split) == 1, "Train/val/test splits must add up to 1"
    
    train_split, val_split, test_split = split
    
    samples = range(self.num_samples)
    samples = random.sample(samples, self.num_samples)
    
    train_start = 0
    train_end = int(train_split*self.num_samples) + 1
    
    val_start = train_end
    val_end = val_start + int(val_split*self.num_samples) + 1
    
    test_start = val_end
    test_end = val_end + int(test_split*self.num_samples) + 1
    
    assert test_end >= self.num_samples, "We are missing some samples when generating test split"
    
    train = samples[train_start:train_end]
    val = samples[val_start:val_end]
    test = samples[test_start:test_end]
    
    return train,val,test
  

  # functions required for pytorch
  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    return (self.tokens[idx], self.token_lens[idx],
            self.waveforms[idx], self.waveform_lens[idx],
            self.mels[idx], self.mel_lens[idx])














