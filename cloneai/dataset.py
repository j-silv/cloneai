import torch
import torchaudio
import torchaudio.transforms as T
import re
import os
from dataclasses import dataclass
from cloneai.utils import plot_waveform, plot_spectrogram

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class AudioConfig:
  """Small dataclass which contains audio parameters for STFT"""
  sr: int = 16000               # sampling rate
  frame_size_s: float = 0.05    # how many seconds do we use to calculate STFT  
  frame_hop_s: float = 0.0125   # how many seconds do we hop to calculate next STFT frame
  audio_length_s: float = 30.0  # padded/max length of audio (for batch processing)
  n_mels: int = 80              # how many frequency bins for mel spectrogram
  window: str = "hann"          # which windowing function to use for calculating STFT
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
    

class AudioDataset():
  """Loads dataset for TTS training
  
  We define pytorch dataset for easy to use dataloader and for
  batch processing later. This is where we load in the dataset
  from the transcription file, load the tokenized text, and convert
  the waveforms to mel spectograms
  
  Note that the tacotron2 (text-to-spectrogram) and wavernn (vocoder) 
  models require different inputs and targets. There are separate subclassed
  pytorch datasets which will take an instance of this class as input.
  The reasoning behind this was to decouple the audio loading and spectrogram
  creation from the specific model requirements.
  
  The waveform path locations are specified in a transcription file with the following format:

    8_56_368.wav|We start with, we rethink everything.
    8_56_243.wav|Yes, I understand that. I'm going to use the Tinder pouch...
    ...

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
  """  
  def __init__(self,
               in_dir,                   # root directory for audio file paths and transcription file
               transcript,               # map file containing audio file paths and transcriptions
               audio_config,             # dict of options that applies to AudioConfig dataclass
               resample=False,           # if True, resample audio to specified sample rate
              
              ):

    self.audio_config = AudioConfig(**audio_config)
    self.audio_files, self.transcriptions = self.load_text(in_dir, transcript)
    self.num_samples = len(self.audio_files)
    self.waveforms = self.load_audio(self.audio_files, resample, self.audio_config)    
    self.mels = self.create_mel(self.waveforms, self.audio_config)

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
  def load_audio(audio_files, resample, config):
    """load audio data into memory from audio_files path"""

    audio_data = []

    for idx, audio_file in enumerate(audio_files):
      
        audio, native_sr = torchaudio.load(audio_file)
        
        if audio.shape[0] > 1: 
          print("Warning: did stereo to mono conversion because only single channel audio files are supported")
          audio = torch.mean(audio, dim=0, keepdim=True)
          
        if resample is True:
          resampler = T.Resample(native_sr, config.sr, dtype=audio.dtype)
          audio = resampler(audio)
          config.sr = native_sr

        audio_data.append(audio)    
        
    return audio_data


  @staticmethod
  def create_mel(waveforms, config):
      """generate the mel spectogram from the audio file
      
      note that for now, we are pre-computing the melspectrograms
      ahead of time, and storing them as a member variable of this class.
      alternatively, we could calculate the spectrograms on the fly
      while collating the data
      """
      mels = []
      
      for waveform in waveforms:
        mel_transform = T.MelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.frame_size_samples,
            hop_length=config.frame_hop_samples,
            n_mels=config.n_mels,
            window_fn=config.window
        )
        mels.append(mel_transform(waveform))
        
        print(waveform.shape, mels[-1].shape)
      return mels
  
  # not strictly necessary, but simplifies data accessing
  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    return (self.transcriptions[idx], self.waveforms[idx], self.mels[idx])


def run(in_dir, out_dir, resample, audio_config):
  """Load dataset, plot waveforms/specgrams, and prepare batch data"""
  
  dataset = AudioDataset(in_dir, "transcriptions.txt", audio_config, resample)
  
  print(  f"{len(dataset)=}",
          f"{dataset.waveforms[0].shape=}",
          f"{dataset.transcriptions[0]=}",
          f"{len(dataset.transcriptions[0])=}",
          f"{dataset.mels[0].shape=}",
          sep="\n")

  outputImg = os.path.join(out_dir, "waveform.png")
  print("Raw waveform:", outputImg)
  plot_waveform(dataset.waveforms[0], outputImg=outputImg, title="Raw waveform")
  
  outputImg = os.path.join(out_dir, "raw_specgram.png")
  print("Raw specgram:", outputImg)
  plot_spectrogram(dataset.mels[0], outputImg=outputImg, title="Raw spectrogram")    

  return dataset






