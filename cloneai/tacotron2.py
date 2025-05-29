import torch
from torch import nn
import torchaudio
from cloneai.utils import plot_spectrogram
import os

device = "cuda" if torch.cuda.is_available() else "cpu"



def run(data, out_dir, hyperparams):
    dataset, train, val, test = data
    
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
    processor = bundle.get_text_processor()

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
    
    
    # sanity check compare spectrogram output with golden
    # compare against a sample that the tacotron2 was trained on
    test_token, test_token_len = next(iter(train))[0][:1], next(iter(train))[1][:1]
    golden_spec, golden_spec_len = next(iter(train))[4][:1], next(iter(train))[5][:1]
    with torch.inference_mode():
        spec, spec_len, _ = model.infer(test_token, test_token_len)
    
    outputImg = os.path.join(out_dir, "golden_log_specgram.png")
    print("Golden log specgram:", outputImg)
    plot_spectrogram(golden_spec[0, :, :, :int(golden_spec_len[0])], outputImg=outputImg, logCompressed=True, title="Golden log spectrogram")  
    
    outputImg = os.path.join(out_dir, "predicted_log_specgram.png")
    print("Predicted log specgram:", outputImg)
    spec = spec.unsqueeze(1)
    plot_spectrogram(spec[0, :, :, :int(spec_len[0])], outputImg=outputImg, logCompressed=True, title="Predicted log spectrogram")  
    
    
    
    