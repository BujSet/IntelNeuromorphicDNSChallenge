"""
Music Source Separation with Hybrid Demucs
==========================================

**Author**: `Sean Kim <https://github.com/skim0514>`__

This tutorial shows how to use the Hybrid Demucs model in order to
perform music separation

"""

######################################################################
# 1. Overview
# -----------
#
# Performing music separation is composed of the following steps
#
# 1. Build the Hybrid Demucs pipeline.
# 2. Format the waveform into chunks of expected sizes and loop through
#    chunks (with overlap) and feed into pipeline.
# 3. Collect output chunks and combine according to the way they have been
#    overlapped.
#
# The Hybrid Demucs [`Défossez, 2021 <https://arxiv.org/abs/2111.03600>`__]
# model is a developed version of the
# `Demucs <https://github.com/facebookresearch/demucs>`__ model, a
# waveform based model which separates music into its
# respective sources, such as vocals, bass, and drums.
# Hybrid Demucs effectively uses spectrogram to learn
# through the frequency domain and also moves to time convolutions.
#


######################################################################
# 2. Preparation
# --------------
#
# First, we install the necessary dependencies. The first requirement is
# ``torchaudio`` and ``torch``
#

import sys
import torch
import torchaudio

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

######################################################################
# In addition to ``torchaudio``, ``mir_eval`` is required to perform
# signal-to-distortion ratio (SDR) calculations. To install ``mir_eval``
# please use ``pip3 install mir_eval``.
#

# from IPython.display import Audio
from mir_eval import separation
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.utils import download_asset

######################################################################
# 3. Construct the pipeline
# -------------------------
#
# Pre-trained model weights and related pipeline components are bundled as
# :py:func:`torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS`. This is a
# :py:class:`torchaudio.models.HDemucs` model trained on
# `MUSDB18-HQ <https://zenodo.org/record/3338373>`__ and additional
# internal extra training data.
# This specific model is suited for higher sample rates, around 44.1 kHZ
# and has a nfft value of 4096 with a depth of 6 in the model implementation.

bundle = HDEMUCS_HIGH_MUSDB_PLUS

model = bundle.get_model()

print("CUDA is available? " + str(torch.cuda.is_available()))
deviceString = "cuda:0" if torch.cuda.is_available() else "cpu"
print(deviceString)
device = torch.device(deviceString)

model.to(device)

sample_rate = bundle.sample_rate


######################################################################
# 4. Configure the application function
# -------------------------------------
#
# Because ``HDemucs`` is a large and memory-consuming model it is
# very difficult to have sufficient memory to apply the model to
# an entire song at once. To work around this limitation,
# obtain the separated sources of a full song by
# chunking the song into smaller segments and run through the
# model piece by piece, and then rearrange back together.
#
# When doing this, it is important to ensure some
# overlap between each of the chunks, to accommodate for artifacts at the
# edges. Due to the nature of the model, sometimes the edges have
# inaccurate or undesired sounds included.
#
# We provide a sample implementation of chunking and arrangement below. This
# implementation takes an overlap of 1 second on each side, and then does
# a linear fade in and fade out on each side. Using the faded overlaps, I
# add these segments together, to ensure a constant volume throughout.
# This accommodates for the artifacts by using less of the edges of the
# model outputs.
#
# .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/HDemucs_Drawing.jpg

from torchaudio.transforms import Fade


def separate_sources(
    model,
    mix,
    segment=10.0,
    overlap=0.1,
    device=None,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        print("chunk dims: " + str(chunk.size()))
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def plot_spectrogram(stft, title="Spectrogram", savefile="spectogram"):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    fig = plt.figure(figsize=(8, 4))
    axis = fig.add_subplot(1, 1, 1)
    axis.imshow(spectrogram, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto")
    axis.set_title(title)
    plt.tight_layout()
    fig.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    


######################################################################
# 5. Run Model
# ------------
#
# Finally, we run the model and store the separate source files in a
# directory
#
# As a test song, we will be using A Classic Education by NightOwl from
# MedleyDB (Creative Commons BY-NC-SA 4.0). This is also located in
# `MUSDB18-HQ <https://zenodo.org/record/3338373>`__ dataset within
# the ``train`` sources.
#
# In order to test with a different song, the variable names and urls
# below can be changed alongside with the parameters to test the song
# separator in different ways.
#

data_loader = None
assert(len(sys.argv) == 2)
test_or_train = sys.argv[1]
if test_or_train == "train":
    musdb_hq_data = torchaudio.datasets.MUSDB_HQ('.',
            subset="train",  
            sources=['mixture', 'drums', 'bass', 'other', 'vocals'],
            #sources=["bass", "drums", "other", "mixture", "vocals"],
            download=False)

    data_loader = torch.utils.data.DataLoader(
        musdb_hq_data,
        batch_size=1,
        shuffle=False,
        num_workers=4)
else:
    musdb_hq_data = torchaudio.datasets.MUSDB_HQ('.',
            subset="test",  
            sources=['mixture', 'drums', 'bass', 'other', 'vocals'],
            #sources=["bass", "drums", "other", "mixture", "vocals"],
            download=False)

    data_loader = torch.utils.data.DataLoader(
        musdb_hq_data,
        batch_size=1,
        shuffle=False,
        num_workers=4)


# parameters
segment: int = 10
overlap = 0.1


sdr_file = open("hybrid_demucs_" + test_or_train + "_sdr_scores.csv", "w")
# write header
sdr_file.write("track ID, train/test set")

for i in range(len(model.sources)):
    sdr_file.write(", " + model.sources[i])


#### CNN Model
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
    #   self.conv1 = nn.Conv2d(1, 32, 3, 1)
    #   self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.conv1 = nn.Conv1d(2, 32, 2, 1)
      self.conv2 = nn.Conv1d(32, 64, 2, 1)
      self.dropout1 = nn.Dropout1d(0.25)
      self.dropout2 = nn.Dropout1d(0.5)
      self.fc1 = nn.Linear(9216, 128) # TODO: need constant width 
      self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through ``fc1``
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output
    
    

#instantiate model
net = torch.nn.DataParallel(Net().to(device),  # use data parallel for gpu
                    device_ids=[0])

optimizer = torch.optim.RAdam(net.parameters(),
                                  lr=0.001, # start with "high" learning rate 
                                  weight_decay=1e-5) # how to adjust weights to minimize error
 
# stochastic gradient descent
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
 
num_epochs = 5
for epoch in range(num_epochs):
    for i, sample in enumerate(data_loader):      
        assert(len(sample) == 4)
        waveform = sample[0].to(device) # (1,5,2,numFrames)
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()  # normalization
        mix = waveform[0,0,:,:].squeeze()
        
        print("mix has dims " + str(mix.size()))
        print("mix[None] has dims " + str(mix[None].size()))
        
        


        # pass to network 
        prediction = net(mix[None])
        print("prediction dims: " + prediction.size())
        # output is tensor with 4 tracks

        sources = separate_sources(
            model,
            mix[None],
            device=device,
            segment=segment,
            overlap=overlap,
        )[0]
        sources = sources * ref.std() + ref.mean()
        print("sources has dims " + str(sources.size())) # (4, 2, numFrames)
        if (i == 5):
            sys.exit()
        # # print(model.sources)
        # sdr_file.write("\n" + str(i)+ ", " + test_or_train)
        # for j in range(len(model.sources)):
        #     # print(model.sources[j])
        #     sdr_score = separation.bss_eval_sources(waveform[0,j+1,:,:].cpu().detach().numpy(), sources[j,:,:].cpu().detach().numpy())[0].mean()
        #     #write sdr score
        #     sdr_file.write(", "+str(sdr_score))

    sdr_file.close()
