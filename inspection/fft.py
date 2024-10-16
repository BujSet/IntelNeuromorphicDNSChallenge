import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import sys
import numpy as np

def plot_waveform(waveform, sr, title="Waveform", ax=None, ylabel=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    if (ylabel != None):
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (sec)")

def plot_pitch(waveform, pitch, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)

    axis2 = ax.twinx()
    pitch = pitch.numpy()
    num_vals, num_time_steps = pitch.shape
    time_axis = torch.arange(0, num_time_steps) / sr
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")
    axis2.legend(loc=0)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    ax.set_xlabel("Time (sec)")

def plot_spectrogram(specgram, title=None, ylabel="Frequency (kHz)", ax=None, yticks=[], yticklabels=[]):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (FFT Frame)")
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def plot_phase(specgram, title=None, ylabel="Frequency (kHz)", yticks=[], yticklabels=[], filename="dummy.png"):
    ampls = torch.sqrt(torch.square(specgram.real) + torch.square(specgram.imag))
    phase = torch.atan(torch.div(specgram.imag, specgram.real))
    print(specgram.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    for i in range(len(phase)):
        theta =  phase[i].numpy() * 180.0 / np.pi
        r = ampls[i].numpy()
        c = ax.scatter(theta, r, c='b', s=0.5)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


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
        power=None)

mel_spectrogram = T.MelSpectrogram(
    sample_rate=44100,
    n_fft=512,
    win_length=None,
    hop_length=256,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=64,
    mel_scale="htk",
)

# Perform transform
spec_noisy = spectrogram(noisy).squeeze()
spec_clean = spectrogram(clean)
spec_noise = spectrogram(noise)

mel_spec_noisy = mel_spectrogram(noisy).squeeze()
mel_spec_clean = mel_spectrogram(clean).squeeze()
mel_spec_noise = mel_spectrogram(noise).squeeze()

pitch_noisy = F.detect_pitch_frequency(noisy, 44100)
pitch_clean = F.detect_pitch_frequency(clean, 44100)
pitch_noise = F.detect_pitch_frequency(noise, 44100)

freq_map = librosa.fft_frequencies(sr=44100, n_fft=512)
yticks = [i for i in range(0, len(freq_map), int(len(freq_map)/5))]
yticklabels = [round(freq_map[i]/1000.0, 2) for i in yticks]

mel_map = librosa.mel_frequencies(n_mels=64, htk=True)
mel_yticks = [i for i in range(0, len(mel_map), int(len(mel_map)/5))]
mel_yticklabels = [round(mel_map[i]/1000.0, 2) for i in mel_yticks]

fig, axs = plt.subplots(2, 2, figsize=(30,30))
plot_waveform(noisy, nysr, title="Original waveform", ax=axs[0, 0])
plot_waveform(pitch_noisy, nysr, title="Pitch", ax=axs[1, 0], ylabel="Frequency (Hz)")
plot_spectrogram(spec_noisy, title="Power Spectrogram", ax=axs[0, 1], yticks=yticks, yticklabels=yticklabels)
plot_spectrogram(mel_spec_noisy, title="Power Mel Spectrogram", ax=axs[1, 1], yticks=mel_yticks, yticklabels=mel_yticklabels)
fig.tight_layout()
plt.savefig("spec_noisy.png")
plt.close()
#plot_phase(spec_noisy, title="Phase Spectrogram", yticks=yticks, yticklabels=yticklabels, filename="spec_noisy_phase.png")

fig, axs = plt.subplots(2, 2, figsize=(30,30))
plot_waveform(clean, clsr, title="Original waveform", ax=axs[0, 0])
plot_waveform(pitch_clean, clsr, title="Original waveform", ax=axs[1, 0], ylabel="Frequency (Hz)")
plot_spectrogram(spec_clean[0], title="spectrogram", ax=axs[0,1], yticks=yticks, yticklabels=yticklabels)
plot_spectrogram(mel_spec_clean, title="Power Mel Spectrogram", ax=axs[1, 1], yticks=mel_yticks, yticklabels=mel_yticklabels)
fig.tight_layout()
plt.savefig("spec_clean.png")
plt.close()
#plot_phase(spec_clean, title="Phase Spectrogram", yticks=yticks, yticklabels=yticklabels, filename="spec_clean_phase.png")

fig, axs = plt.subplots(2, 2, figsize=(30,30))
plot_waveform(noise, nesr, title="Original waveform", ax=axs[0, 0])
plot_waveform(pitch_noise, nesr, title="Original waveform", ax=axs[1, 0], ylabel="Frequency (Hz)")
plot_spectrogram(spec_noise[0], title="spectrogram", ax=axs[0,1], yticks=yticks, yticklabels=yticklabels)
plot_spectrogram(mel_spec_noise, title="Power Mel Spectrogram", ax=axs[1,1], yticks=mel_yticks, yticklabels=mel_yticklabels)
fig.tight_layout()
plt.savefig("spec_noise.png")
plt.close()
#plot_phase(spec_noise, title="Phase Spectrogram", yticks=yticks, yticklabels=yticklabels, filename="spec_noise_phase.png")
