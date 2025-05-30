import torch
from torch import nn
import torchaudio
from cloneai.utils import plot_spectrogram
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def run(data, out_dir, load_checkpoint_path, save_checkpoint_path, hyperparams):
    
    #######################################
    # Set up model
    #######################################
    
    dataset, train, val, test = data
    
    # TODO: load Tacotron2 explicitly so that we can set
    #       some of the parameters like decoder_max_step
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
    processor = bundle.get_text_processor()

    model = bundle.get_tacotron2()
    model.train()
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
    
    if load_checkpoint_path != "":
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
    
    loss = 0.0
    for epoch in range(hyperparams["epochs"]):
        for batch, (tokens, token_lens, _, _,  mels, mel_lens, gates) in enumerate(train):

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
            
            mel_prenet, mel_postnet, gate_out, _ = model(tokens, token_lens, mels, mel_lens)

            loss = mse_loss(mel_prenet, mels)
            loss += mse_loss(mel_postnet, mels)
            loss += bce_loss(gate_out, gates)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 1 == 0:
                loss = loss.item()
                print(f"Epoch {epoch+1} | loss: {loss:>7f}")

    print("Done!")
    
    
    #######################################
    # save model
    #######################################
    
    if save_checkpoint_path != "":
        save_checkpoint_path = os.path.join(out_dir, save_checkpoint_path)
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        checkpoint = os.path.join(out_dir, "tacotron2_checkpoint.pth")
        torch.save(state, checkpoint)
      
    
    #######################################
    # sanity check
    #######################################
    
    # compare spectrogram output with golden
    # compare against a sample that the tacotron2 was trained on
    
    model.eval()
    
    train_batch_sample = next(iter(train))
    
    test_token, test_token_len = train_batch_sample[0][:1], train_batch_sample[1][:1]
    golden_spec, golden_spec_len = train_batch_sample[4][:1], train_batch_sample[5][:1]
  
    print("Testing 1st batch sample for training data")
    print("transcription:")
    print(f"{''.join([processor.tokens[i] for i in test_token[0, : test_token_len[0]]])}")
 
    outputImg = os.path.join(out_dir, "golden_log_specgram.png")
    print("Golden log specgram:", outputImg)
    plot_spectrogram(golden_spec[0, :, :, :int(golden_spec_len[0])], outputImg=outputImg, logCompressed=True, title="Golden log spectrogram") 
    
    with torch.inference_mode():
        spec, spec_len, _ = model.infer(test_token, test_token_len) 
    
    print(f"{spec.shape=}, {spec_len.shape=}")
    outputImg = os.path.join(out_dir, "predicted_log_specgram.png")
    print("Predicted log specgram:", outputImg)
    spec = spec.unsqueeze(1)
    plot_spectrogram(spec[0, :, :, :int(spec_len[0])], outputImg=outputImg, logCompressed=True, title="Predicted log spectrogram")  
    
    
    
    