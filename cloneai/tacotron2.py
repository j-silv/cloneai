import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchaudio
from cloneai.dataset import AudioDataset
from cloneai.utils import plot_spectrogram
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def run(in_dir, out_dir, train_split, val_split, seed, resample, processor, audio_config):
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
    
    fig = plot_spectrogram(dataset.mels[0])
    fig.savefig(os.path.join(out_dir, "specgram.png"))
    # plot_spectrogram(dataset.mels[-1])


# def google_colab():       



#     ########################################################################################
#     ########################################################################################


#     # As per tacotron2 paper, we will set up optimizer, loss function, and split our data into training and test


#     # hyperparameters
#     seed = 0
#     batch_size = 64
#     split = [0.8, 0.2] # [train_fraction, test_fraction]
#     lr = 0.001
#     betas = (0.9, 0.999)
#     eps = 1e-08
#     weight_decay = 1e-06
#     epochs = 100


#     torch.random.manual_seed(seed)

#     # can't shuffle randomly because we need arrays to go in ascending order
#     # this is another reason to do the descending stuff not when we init
#     # the data but when we are about to perform a batch pass
#     # for now it's fine to test with it though
#     # TODO: shuffle up data beforehand, but not here
#     # train_dataset, test_dataset = random_split(dataset, split)
#     train_dataset = Subset(dataset, range(int(len(dataset)*split[0])))
#     test_dataset = Subset(dataset, range(int(len(dataset)*split[0]), len(dataset)))

#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH

#     model = bundle.get_tacotron2().to(device)
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas,
#                                 eps=eps, weight_decay=weight_decay)

#     # TODO: use learning rate decay... for now don't though because not sure
#     #       best way to implement exponential decay here. i don't know if
#     #       well this would work. By gamma should be 1/2.71, so we have 2.17^-x, not sure
#     #       1*1/2.71 = 1*e-1. not sure
#     # scheduler = torch.ExponentialLR(optimizer, gamma=0.9)


#     ########################################################################################
#     ########################################################################################


#     # Finally we will run a loop and see how our loss is from the initial pre-trained state
#     for epoch in range(epochs):
#         print(f"Epoch {epoch+1}\n-------------------------------")

#         # important for batch normalization and dropout layers
#         model.train()
#         for batch, (tokens, token_lengths, mel, mel_length) in enumerate(train_dataloader):

#             # print(tokens, token_lengths, mel, mel_length, sep="\n")
#             mel_prenet, mel_postnet, _, _ = model(tokens, token_lengths, mel, mel_length)

#             loss = loss_fn(mel_prenet, mel)
#             loss += loss_fn(mel_postnet, mel)

#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#             if epoch % 1 == 0:
#                 loss = loss.item()
#                 print(f"loss: {loss:>7f}")

#     print("Done!")


#     # def test_loop(dataloader, model, loss_fn):
#     #     # Set the model to evaluation mode - important for batch normalization and dropout layers
#     #     # Unnecessary in this situation but added for best practices
#     #     model.eval()
#     #     num_batches = len(dataloader)
#     #     test_loss = 0

#     #     # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
#     #     # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
#     #     with torch.no_grad():
#     #         for tokens, token_lengths, mel_specgram, mel_specgram_lengths in dataloader:
#     #             mel_spec_prenet, mel_spec_postnet, _, _ = model(tokens, token_lengths, mel_specgram, mel_specgram_lengths)
#     #             test_loss += loss_fn(mel_spec_postnet, mel_specgram).item()

#     #     test_loss /= num_batches
#     #     print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")


#     ########################################################################################
#     ########################################################################################

#     # To save additional information, such as the epoch number or loss, you can create a dictionary:
#     state = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss
#     }
#     torch.save(state, 'tacotron2_checkpoint_colab_05202025.pth')