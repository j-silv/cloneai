import torch
from torch import nn
import torchaudio
from cloneai.utils import plot_spectrogram, pad_tensor, trim_tensor
from torch.utils.data import Dataset, DataLoader, random_split
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class Tacotron2Dataset(Dataset):
    """PyTorch wrapper for Tacotron2 data so we can use DataLoader"""
    
    def __init__(self, data):    
        self.data = data
        self.num_samples = len(data)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # waveforms are unnecessary for tacotron2 training
        transcription, _, mel = self.data[idx]
        return (transcription, mel)
    

def collate_fn_wrapper(tokenizer, min_mag):
    """Wrapper around collate because we need additional arguments
    
    Similar to https://github.com/pytorch/audio/blob/release/0.12/examples/pipeline_wavernn/datasets.py,
    we are wrapping a function and returning it since the collate_fn should only receive a single batch argument
    but we need additional arguments such as text processor, audio config, etc.
    
    """
    def collate_fn(batch):
        batch_size = len(batch)
        transcriptions = []
        mels = []
        mel_lens = torch.zeros(batch_size, dtype=torch.long)
        
        #######################################
        # unpack batch into text and mels
        #######################################
        
        for idx, (transcription, mel) in enumerate(batch):
            transcriptions.append(transcription)
            mels.append(mel.squeeze(0)) # we don't need the 'channel' dimension
            mel_lens[idx] = min(2401, mel.shape[-1])
        
        #######################################
        # zero-pad mel spectrograms to max length
        #######################################
        
        max_mel_len = torch.max(mel_lens).item()
        num_freq_bins = mels[0].shape[0]
        padded_mels = torch.zeros(batch_size ,num_freq_bins, max_mel_len)
        
        for idx, mel in enumerate(mels):
            # I don't think there is a better way to
            # do this since mels are packed in a list
            padded_mels[idx, :, :mel.shape[-1]] = mel[:, :mel.shape[-1]]
        
        # print(f"{padded_mels.shape=}, {max_mel_len=}, {num_freq_bins=}, {mel_lens.shape=}")
        
        #######################################
        # get tokens -> tokenizer already pads for us
        #######################################
        
        tokens, token_lens = tokenizer(transcriptions)
        
        # print(f"{tokens.shape=}, {token_lens.shape=}")
    
        # the tacotron2 model expects input batches with decreasing size of token arrays
        # so we have to sort the array and keep track of indexes for aligning the other data
        #`lengths` array must be sorted in decreasing order when `enforce_sorted` is True
        _, idx_sorted = torch.sort(token_lens, descending=True)
        
        # simple PyTorch indexing lets us reindex based on the sorted samples above
        # we could do this in the DataLoader directly for efficiency
        tokens = tokens[idx_sorted]
        token_lens = token_lens[idx_sorted]
        
        padded_mels = padded_mels[idx_sorted]
        mel_lens = mel_lens[idx_sorted]
        
        #######################################
        # log compression of specgram
        #######################################
        # as per tacotron2 paper, clip to minimum value and compress with log
        log_mels = torch.clamp(padded_mels, min=min_mag).log10()
        
        #######################################
        # set up stop prediction (gate)
        #######################################
        
        gates = torch.zeros((batch_size, max_mel_len), dtype=torch.float32)
        # this tells the tacotron2 to stop generating during inference whenever
        # we exceed the actual specgram lengths (before it is 0.0 so don't stop)
        
        # fun little hack to set stop token only after the valid mel lengths
        # -1 because we need to have at least 1 stop token when we have output all required mel frames
        mask = torch.arange(max_mel_len) >= (mel_lens.unsqueeze(1) - 1)
        gates[mask] = 1.0
        
        return tokens, token_lens, log_mels, mel_lens, gates
        
    return collate_fn
      

def run(data, out_dir, seed, load_checkpoint_path, save_checkpoint_path, hyperparams, tokenizer=None):
    #######################################
    # load data
    #######################################
    
    torch.manual_seed(seed)
    
    if tokenizer is None or tokenizer == "WAVERNN_CHAR_LJSPEECH":
        tokenizer = (
            torchaudio
            .pipelines
            .TACOTRON2_WAVERNN_CHAR_LJSPEECH
            .get_text_processor()
        )
    else:
        raise ValueError("Invalid tokenizer value for AudioDataset")
    
    dataset = Tacotron2Dataset(data)

    print(f"{hyperparams['split']=}")
    print(f"{hyperparams['batch_size']=}")

    train_split, val_split, test_split = random_split(dataset, hyperparams["split"]) 
    
    train = DataLoader(train_split,
                       batch_size=hyperparams["batch_size"],
                       collate_fn=collate_fn_wrapper(tokenizer, dataset.data.audio_config.min_mag))
    val = DataLoader(val_split,
                     batch_size=hyperparams["batch_size"],
                     collate_fn=collate_fn_wrapper(tokenizer, dataset.data.audio_config.min_mag))
    test = DataLoader(test_split,
                      batch_size=hyperparams["batch_size"],
                      collate_fn=collate_fn_wrapper(tokenizer, dataset.data.audio_config.min_mag))
    
    
    #######################################
    # Set up model
    #######################################
    
    # TODO: load Tacotron2 explicitly so that we can set
    #       some of the parameters like decoder_max_step
    model = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_tacotron2()
    model = model.to(device)
    
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    # TODO: learning rate decay
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"],
                                betas=hyperparams["betas"],
                                eps=float(hyperparams["eps"]),
                                weight_decay=float(hyperparams["weight_decay"]))
    
    #######################################
    # load checkpoint
    #######################################
    
    if load_checkpoint_path:
        checkpoint_path = os.path.join(out_dir, load_checkpoint_path)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epoch = checkpoint['epoch']
            # loss = checkpoint['loss']
        else:
            print(f"Specified checkpoint does not exist - {checkpoint_path}")
                
    #######################################
    # Training loop
    #######################################  
    for epoch in range(hyperparams["epochs"]):
        model.train()
        
        loss = 0.0
        data_samples_seen = 0
        for batch, (tokens, token_lens, mels, mel_lens, gates) in enumerate(train):
            data_samples_seen += tokens.shape[0]
            
            tokens = tokens.to(device)
            token_lens = token_lens.to(device)
            mels = mels.to(device)
            mel_lens = mel_lens.to(device)
            gates = gates.to(device)
            
            mel_prenet, mel_postnet, gate_out, _ = model(tokens, token_lens, mels, mel_lens)
            
            loss = mse_loss(mel_prenet, mels)
            loss += mse_loss(mel_postnet, mels)
            loss += bce_loss(gate_out, gates)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 1 == 0:
                loss = loss.item()
                print(f"Training | Epoch {epoch+1} | batch {batch} | samples {data_samples_seen}/{len(train_split)} | loss: {loss:>7f}")

        ###########################################################
        # validation loss
        ###########################################################
        model.eval()

        with torch.inference_mode():
            loss = 0.0
            data_samples_seen = 0
            
            # Validation with teacher forcing mode
            for batch, (tokens, token_lens, mels, mel_lens, gates) in enumerate(val):
                data_samples_seen += tokens.shape[0]
                
                tokens = tokens.to(device)
                token_lens = token_lens.to(device)
                mels = mels.to(device)
                mel_lens = mel_lens.to(device)
                gates = gates.to(device)
                
                mel_prenet, mel_postnet, gate_out, _ = model(tokens, token_lens, mels, mel_lens)
                
                loss = mse_loss(mel_prenet, mels)
                loss += mse_loss(mel_postnet, mels)
                loss += bce_loss(gate_out, gates)

                loss = loss.item()
                print(f"Validation (teacher-forcing) | Epoch {epoch+1} | batch {batch} | samples {data_samples_seen}/{len(val_split)} | loss: {loss:>7f}")

            # validation with inference
            loss = 0.0
            data_samples_seen = 0
            for batch, (tokens, token_lens, mels, mel_lens, gates) in enumerate(val):
                data_samples_seen += tokens.shape[0]
                
                tokens = tokens.to(device)
                token_lens = token_lens.to(device)
                mels = mels.to(device)
                mel_lens = mel_lens.to(device)
                gates = gates.to(device)
                
                predicted_mel, predicted_mel_lens, _ = model.infer(tokens, token_lens)
                
                # TODO: -> we are assuming that the predicted mel is padded with zeros
                # after the predicted_mel_lens. If so, then this is valid because 
                # the extra zeros don't add to the loss, but we still account for 
                # the uneven lengths. I'll have to sanity check this...
                if predicted_mel.shape[-1] < mels.shape[-1]:
                    predicted_mel = pad_tensor(predicted_mel, mels.shape[-1])
                elif predicted_mel.shape[-1] > mels.shape[-1]:
                    mels = pad_tensor(mels, predicted_mel.shape[-1])
                    
                loss = mse_loss(predicted_mel, mels)
                loss = loss.item()
                print(f"Validation (inference) | Epoch {epoch+1} | batch {batch} | samples {data_samples_seen}/{len(val_split)} | loss: {loss:>7f}")

            outputImg = os.path.join(out_dir, "golden_train_specgram.png")
            print("Golden train specgram:", outputImg)
            plot_spectrogram(mels[0], outputImg=outputImg, logCompressed=True, title="Golden train spectrogram")   
            
            outputImg = os.path.join(out_dir, "predicted_val_inference_specgram.png")
            print("Predicted val inference specgram:", outputImg)
            plot_spectrogram(predicted_mel[0], outputImg=outputImg, logCompressed=True, title="Predicted val inference specgram")    
        
    print("Done!")
    
    
    #######################################
    # save model
    #######################################
    
    if save_checkpoint_path:
        save_checkpoint_path = os.path.join(out_dir, save_checkpoint_path)
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        checkpoint = os.path.join(out_dir, "tacotron2_checkpoint.pth")
        torch.save(state, checkpoint)
      
    

    
    
    
    