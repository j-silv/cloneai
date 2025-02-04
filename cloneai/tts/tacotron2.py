# https://pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html

# Make sure to run the following commands on google collab:
#
# %%bash
# pip3 install deep_phonemizer
# pip3 install torch torchaudio

import torch
import torchaudio

def run(text, outfile):
    torch.random.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(torch.__version__)
    print(torchaudio.__version__)
    print(device)

    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2().to(device)
    vocoder = bundle.get_vocoder().to(device)

    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device)
        lengths = lengths.to(device)
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
        waveforms, lengths = vocoder(spec, spec_lengths)

    torchaudio.save(outfile, waveforms, vocoder.sample_rate)
