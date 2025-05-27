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
  
  def __init__(self, in_dir, sr=16000, frame_size_s=0.05, frame_hop_s=0.0125,
               audio_length_s=30, n_mels=80, window='hann', min_mag=0.01,
               transcript_file="transcriptions.txt", processor=None, max_files=10):

    self.in_dir = in_dir
    self.sr = sr
    self.n_mels = n_mels
    self.audio_length_s = audio_length_s
    self.frame_size_s = frame_size_s
    self.frame_hop_s = frame_hop_s
    self.window = window
    self.min_mag = min_mag
    self.transcript_file = transcript_file
    self.max_files = 10

    if processor is None:
      self.processor = (
          torchaudio
          .pipelines
          .TACOTRON2_WAVERNN_CHAR_LJSPEECH
          .get_text_processor()
      )

    self.load_text()
    self.text_to_tokens()
    self.create_mel()


  def load_text(self):

    # e.g.: 8_56_368.wav|We start with, we rethink everything.
    re_transcription = re.compile(r"(.*)\|(.*)")

    with open(os.path.join(self.in_dir, self.transcript_file), "r") as f:
        lines = f.readlines()

    self.num_samples = min(self.max_files, len(lines))
    self.audio_files    = []  # full path to audio files
    self.transcriptions = []  # raw transcription texts from Whisper

    count = 0
    for line in lines:
        re_result = re_transcription.match(line)
        self.audio_files.append(os.path.join(self.in_dir, re_result.group(1)))
        self.transcriptions.append(re_result.group(2))

        count += 1

        if count == self.max_files:
          break


  def text_to_tokens(self):
    # tacotron2 forward pass expects the following shapes:
    #   tokens               - (n_batch, max of token_lengths)
    #   token_lengths        - (n_batch, )


    # when a list is passed in, the text is automatically zero-padded
    # to the max length of the list
    # TODO: do this per batch when the model is called instead of the entire dataset
    #       at initialization
    self.tokens, self.token_lengths = self.processor(self.transcriptions)

    # the tacotron2 model expects input batches with decreasing size of token arrays
    # so we have to sort the array and keep track of indexes for aligning the mel spectograms
    # """ `lengths` array must be sorted in decreasing order when `enforce_sorted` is True."""
    self.token_lengths, self.idx_sorted_samples = torch.sort(self.token_lengths, descending=True)
    self.max_token_length = self.token_lengths[0]

    # simple PyTorch indexing lets us reindex based on the sorted samples above
    self.tokens = self.tokens[self.idx_sorted_samples]

    self.tokens = self.tokens.to(device)
    self.token_lengths = self.token_lengths.to(device)


  def create_mel(self):
    # tacotron2 forward pass expects the following shapes:
    #   mel_specgram         - (n_batch, n_mels, max of mel_specgram_lengths)
    #   mel_specgram_lengths - (n_batch, )

    # TODO: set n_fft to a power of 2 for FFT speedup
    frame_size_samples = librosa.time_to_samples(self.frame_size_s, sr=self.sr)
    frame_hop_samples = librosa.time_to_samples(self.frame_hop_s, sr=self.sr)
    audio_length_samples = librosa.time_to_samples(self.audio_length_s, sr=self.sr)


    # +1 because that's how the STFT calculation works
    # tensor because this is actually an input into the model
    assert audio_length_samples % frame_hop_samples == 0
    self.num_frames = torch.tensor((audio_length_samples // frame_hop_samples) + 1)


    # technically we don't need to keep track of mels
    # but this is just for debugging
    self.mels = torch.zeros((self.num_samples,
                                 self.n_mels,
                                 self.num_frames))

    self.log_mels = torch.zeros_like(self.mels)


    # TODO: might need to do this as a batch processing step,
    #       instead of loading every single spectogram into memory (costly)
    for idx, audio_file in enumerate(self.audio_files):
      print(f"Loading {os.path.basename(audio_file)}")

      # everything is resampled, also
      # all transcriptions were padded or trimmed to 30 seconds, so
      # we should match that for training as well. otherwise if we use
      # a different size, then the transcript might not align with the spectogram
      audio = whisper.load_audio(audio_file)
      audio = whisper.pad_or_trim(audio)

      mel_spec = librosa.feature.melspectrogram(
          y=audio,
          sr=self.sr,
          n_fft=frame_size_samples,
          hop_length=frame_hop_samples,
          window=self.window,
          n_mels=self.n_mels
      )

      # as per paper, compress with log and clip to minimum value of 0.01
      log_spec = torch.clamp(torch.tensor(mel_spec), min=self.min_mag).log10()

      # make sure to use the appropiate index into the sorted tensor
      self.mels[self.idx_sorted_samples[idx]] = torch.tensor(mel_spec)
      self.log_mels[self.idx_sorted_samples[idx]] = log_spec

    self.log_mels = self.log_mels.to(device)
    self.num_frames = self.num_frames.to(device)


  # functions required for pytorch
  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    return (self.tokens[idx], self.token_lengths[idx],
            self.log_mels[idx], self.num_frames)





class AudioDataset(Dataset):
  """PyTorch dataset wrapper for training WaveRNN"""

  def __init__(self,
               in_dir,                   # root directory for audio file paths and transcription file
               transcript,               # map file containing audio file paths and transcriptions
               resample=False,           # if True, resample audio to specified sample rate
               sr=16000,                 # sampling rate
               frame_size_s=0.05,        # melspectogram frame size in seconds
               frame_hop_s=0.0125,       # melspectogram hop size in seconds
               audio_length_s=30,        # padded audio length in seconds
               n_mels=80,                # melspectogram frequency bins
               window=torch.hann_window, # melspectogram window function
               kernel_size=5,            # kernel size for upsampling in WaveRNN
               min_mag=0.01              # melspectogram clipped magnitude
              ):

    self.sr = sr
    self.audio_length_samples = int(audio_length_s*sr)

    # TODO: set n_fft to a power of 2 for FFT speedup
    self.frame_size_samples = int(frame_size_s*sr)

    self.frame_hop_samples = int(frame_hop_s*sr)

    self.n_mels = n_mels
    self.window = window
    self.min_mag = min_mag

    self.kernel_size = kernel_size


    self.load_audio(in_dir, transcript, resample)


    self.create_mel()


  def load_audio(self, in_dir, transcript, resample):
    """load audio data from transcription mapping file"""

    regex = re.compile(r"(.*)\|(.*)")

    with open(os.path.join(in_dir, transcript), "r") as f:
        lines = f.readlines()

    self.num_audio_files = len(lines)

    self.audio_data = torch.zeros((self.num_audio_files, 1, self.audio_length_samples))
    self.audio_lengths = torch.zeros((self.num_audio_files))
    self.specgram_lengths = torch.zeros((self.num_audio_files))

    for idx, line in enumerate(lines):
        re_result = regex.match(line)

        audio, native_sr = torchaudio.load(os.path.join(in_dir, re_result.group(1)))
        if resample is True:
          resampler = T.Resample(native_sr, self.sr, dtype=audio.dtype)
          audio = resampler(audio)
          self.sr = native_sr

        # use the full waveform shape to compute the expected number of specgrams
        # then update the audio_lengths to fit the pytorch API -> upsampling due to kernel
        self.specgram_lengths[idx] = (audio.shape[-1] // self.frame_hop_samples) + 1
        self.audio_lengths[idx] = (self.specgram_lengths[idx] - self.kernel_size + 1)*self.frame_hop_samples

        # padding is not optional because we are storing result in a tensor
        audio = pad_or_trim(audio, self.audio_length_samples)
        self.audio_data[idx,...] = audio


  def create_mel(self):
      """generate the mel spectogram from the audio file

      waveRNN forward pass expects the following shapes:
        waveform         - (n_batch, 1, (n_time - kernel_size + 1)*hop_length
        specgram         - (n_batch, 1, n_freq, n_time)
      the specgram is upsampled in the last conv layer and this is all done
      automatically. we don't really need to take care of anything here
      """

      # +1 because that's how the STFT calculation works
      # tensor because this is actually an input into the model
      assert self.audio_length_samples % self.frame_hop_samples == 0
      self.num_frames = torch.tensor((self.audio_length_samples // self.frame_hop_samples) + 1)

      mel_transform = T.MelSpectrogram(
          sample_rate=self.sr,
          n_fft=self.frame_size_samples,
          hop_length=self.frame_hop_samples,
          n_mels=self.n_mels,
          window_fn=self.window
      )

      self.mels = mel_transform(self.audio_data)

      # as per tacotron2 paper, clip to minimum value and compress with log
      self.log_mels = torch.clamp(self.mels, min=self.min_mag).log10()


  # functions required for pytorch
  def __len__(self):
    return self.num_audio_files

  def __getitem__(self, idx):
    return (self.audio_data[idx], self.audio_lengths[idx],
            self.log_mels[idx], self.specgram_lengths[idx])









"""Create dataset.json for dataset which is expected by WaveRNN submodule

The WaveRNN submodule expects a dataset.json which splits training, validation, and testing samples.

Here we follow that convention and also use this for tacotron2 training.
"""

import json
import os
import random

def split_train_val_test(in_dir, out_dir, train_split, val_split, seed):

    # Generate dataset.json with train / validation / test split.
    
    # use transcriptions.txt to split up the datasets randomly
    in_file = os.path.join(in_dir, "transcriptions.txt")
    
    audio_files = []

    with open(in_file, "r") as handle:
        for line in handle:
            audio_file = line.split('|', 1)[0]
            audio_files.append(audio_file)

    num_audio_files = len(audio_files)
    
    random.seed(seed)
    random.shuffle(audio_files)
    
    train_start = 0
    train_end = int(train_split*num_audio_files)
    
    val_start = train_end
    val_end = val_start + int(val_split*num_audio_files)
    
    test_start = val_end
    test_end = num_audio_files
    
    train = audio_files[train_start:train_end]
    val = audio_files[val_start:val_end]
    test = audio_files[test_start:test_end]
    

    out_file = os.path.join(out_dir, "dataset.json")
    with open(out_file, "w") as handle:
        json.dump(
            {
                "train": train,
                "valid": val,
                "test": test,
            },
            handle,
            indent=2,
        )
