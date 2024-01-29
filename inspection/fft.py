import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

import librosa
import matplotlib.pyplot as plt

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


NOISY_FILE = './training_set/noisy/SLR61_es_ar_female_arf_06592_00774535859_48kHz-l_ile_mysterieuse_2_06_f000145-book_00394_chp_0008_re_f872gDrfumM-fan_Freesound_validated_437341_3-door_Freesound_validated_349356_0_snr16_tl-23_fileid_1542.wav'
CLEAN_FILE = './training_set/clean/clean_fileid_1542.wav'
NOISE_FILE = './training_set/noise/noise_fileid_1542.wav'
METADATA = {'snr': 16, 'target_level': -23}


noisy, nysr = torchaudio.load(NOISY_FILE)
clean, clsr = torchaudio.load(CLEAN_FILE)
noise, nesr = torchaudio.load(NOISE_FILE)

# Define transform
spectrogram = T.Spectrogram(
        n_fft=512,
        power=2)

spectrogram_complex = T.Spectrogram(
        n_fft=512,
        power=None)
# Perform transform
spec_noisy = spectrogram(noisy)
spec_clean = spectrogram(clean)
spec_noise = spectrogram(noise)

fig, axs = plt.subplots(2, 1)
plot_waveform(noisy, nysr, title="Original waveform", ax=axs[0])
plot_spectrogram(spec_noisy[0], title="spectrogram", ax=axs[1])
fig.tight_layout()
plt.savefig("spec_noisy.png")
plt.close()

fig, axs = plt.subplots(2, 1)
plot_waveform(clean, clsr, title="Original waveform", ax=axs[0])
plot_spectrogram(spec_clean[0], title="spectrogram", ax=axs[1])
fig.tight_layout()
plt.savefig("spec_clean.png")
plt.close()

fig, axs = plt.subplots(2, 1)
plot_waveform(noise, nesr, title="Original waveform", ax=axs[0])
plot_spectrogram(spec_noise[0], title="spectrogram", ax=axs[1])
fig.tight_layout()
plt.savefig("spec_noise.png")
plt.close()
