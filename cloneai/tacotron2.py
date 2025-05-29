import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchaudio
from cloneai.dataset import AudioDataset
from cloneai.utils import plot_waveform, plot_spectrogram
import os

device = "cuda" if torch.cuda.is_available() else "cpu"



def run(in_dir, out_dir, seed, resample, processor, audio_config, hyperparams):
    torch.manual_seed(seed)
    
    dataset = AudioDataset(in_dir, "transcriptions.txt", resample, processor, **audio_config)
    
    print(  f"{dataset.waveforms.shape=}",
            f"{dataset.tokens.shape=}",
            f"{dataset.token_lens.shape=}",
            f"{dataset.mels.shape=}",
            f"{dataset[0][0].shape=}",
            f"{dataset[0][1]=}",
            f"{dataset[0][2].shape=}",
            f"{dataset[0][3]=}",
            sep="\n")

    outputImg = os.path.join(out_dir, "waveform.png")
    print("Raw waveform:", outputImg)
    plot_waveform(dataset.waveforms[0, :, :int(dataset.waveform_lens[0])], outputImg=outputImg, title="Raw waveform")
    
    outputImg = os.path.join(out_dir, "padded_waveform.png")
    print("Padded waveform:", outputImg)
    plot_waveform(dataset.waveforms[0], outputImg=outputImg, title="Padded waveform")    
    
    outputImg = os.path.join(out_dir, "raw_specgram.png")
    print("Raw specgram:", outputImg) # note extra dimension due to WaveRNN.forward requirements
    plot_spectrogram(dataset.raw_mels[0, :, :, :int(dataset.mel_lens[0])], outputImg=outputImg, title="Raw spectrogram")    

    outputImg = os.path.join(out_dir, "log_specgram.png")
    print("Log specgram:", outputImg) # note extra dimension due to WaveRNN.forward requirements
    plot_spectrogram(dataset.mels[0, :, :, :int(dataset.mel_lens[0])], outputImg=outputImg, logCompressed=True, title="Log spectrogram")  
    
    print("Hyperparams:")
    print(hyperparams)  
    
    # train,val,test = dataset.split_train_val_test(hyperparams["split"], seed)
    train, val, test = random_split(dataset, hyperparams["split"]) # pytorch implementation vs mine
    
    # TODO: could explore shuffling data after every epoch
    #       but then we have to be careful to avoid the enforce_sorted? not sure why this is True
    # TODO: we have to chunk up waveforms and specgrams when passing into WaveRNN due to GPU bug
    #       I'll have to think how I want to set this up -> I think it should be totally on the WaveRNN
    #       side because when we do inference I'll still have to do the same thing. 
    train = DataLoader(train, batch_size=hyperparams["batch_size"])
    val = DataLoader(val, batch_size=hyperparams["batch_size"])
    test = DataLoader(test, batch_size=hyperparams["batch_size"])
    
    print(f"{next(iter(train))[0].shape}")

    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH

    model = bundle.get_tacotron2()
    model.train()
    model = model.to(device)
    
    loss_fn = nn.MSELoss()
    # TODO: learning rate decay
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"],
                                betas=hyperparams["betas"],
                                eps=float(hyperparams["eps"]),
                                weight_decay=float(hyperparams["weight_decay"]))


    for epoch in range(hyperparams["epochs"]):
        for batch, (tokens, token_lens, _, _,  mels, mel_lens) in enumerate(train):

            # print(tokens.shape, token_lens.shape, mels.shape, mel_lens.shape, sep="\n")
            
            # the tacotron2 model expects input batches with decreasing size of token arrays
            # so we have to sort the array and keep track of indexes for aligning the other data
            #`lengths` array must be sorted in decreasing order when `enforce_sorted` is True
            _, idx_sorted_samples = torch.sort(token_lens, descending=True)
            
            # simple PyTorch indexing lets us reindex based on the sorted samples above
            # we could do this in the DataLoader directly for efficiency
            tokens = tokens[idx_sorted_samples]
            token_lens = token_lens[idx_sorted_samples]
            mels = mels[idx_sorted_samples]
            mel_lens = mel_lens[idx_sorted_samples]
            
            # NOTE: tacotron2 differs from what wavernn expects
            mels = mels.squeeze(1)
            
            # print(tokens.device, token_lens.device, mels.device, mel_lens.device)
            
            mel_prenet, mel_postnet, _, _ = model(tokens, token_lens, mels, mel_lens)

            loss = loss_fn(mel_prenet, mels)
            loss += loss_fn(mel_postnet, mels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 1 == 0:
                loss = loss.item()
                print(f"Epoch {epoch+1} | loss: {loss:>7f}")

    print("Done!")
