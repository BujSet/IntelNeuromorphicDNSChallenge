import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
from hrtfs.cipic_db import CipicDatabase 
import soundfile as sf
import numpy as np

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

def plot_magnitude(mag, title="Magnitude FFT", ax=None):
    ax.set_title(title)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Time")
    ax.imshow(mag, cmap='viridis', interpolation='nearest', aspect='auto')

def plot_phase(phase, title="Phase FFT", ax=None):
    ax.set_title(title)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Time")
    ax.imshow(phase, cmap='viridis', interpolation='nearest', aspect='auto')

def plot_complex(complex_tensor, title="Complex FFT", ax=None):
    ax.set_title(title)
    ax.set_ylabel("Imag")
    ax.set_xlabel("Real")
    ax.scatter(complex_tensor.real.numpy(), complex_tensor.imag.numpy())

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

EPS = 2.220446049250313e-16

def mixer(clean, noise, snr,
        target_level, 
        target_level_lower,
        target_level_higher,
        clipping_threshold=0.99):
    ' ''Function to mix clean speech and noise at various segmental SNR levels'''
    clean_div = torch.max(torch.abs(clean)) + EPS
    noise_div = torch.max(torch.abs(noise)) + EPS
    ssl_clean = torch.div(clean, clean_div.item())
    ssl_noise = torch.div(noise, noise_div.item())
    # TODO should only calculate the RMS of the 'active' windows, but
    # for now we just use the whole audio sample
    clean_rms = torch.sqrt(torch.mean(torch.square(ssl_clean))).item()
    noise_rms = torch.sqrt(torch.mean(torch.square(ssl_noise))).item()
    clean_scalar = 10 ** (target_level / 20) / (clean_rms + EPS)
    noise_scalar = 10 ** (target_level / 20) / (noise_rms + EPS)
    ssl_clean = torch.mul(ssl_clean, clean_scalar)
    ssl_noise = torch.mul(ssl_noise, noise_scalar)
    # Adjust noise to SNR level
    noise_scalar = clean_rms / (10**(snr/20)) / (noise_rms+EPS)
    ssl_noise = torch.mul(ssl_noise, noise_scalar)
    ssl_noisy = torch.add(ssl_clean, ssl_noise)
    noisy_rms_level = torch.randint(
        target_level_lower,
        target_level_higher,
        (1,))
    noisy_rmsT = torch.sqrt(torch.mean(torch.square(ssl_noisy)))
    noisy_rms = torch.sqrt(torch.mean(torch.square(ssl_noisy))).item()
    noisy_scalar = 10 ** (noisy_rms_level / 20) / (noisy_rms + EPS)
    ssl_noisy = torch.mul(ssl_noisy, noisy_scalar.item())
    ssl_clean = torch.mul(ssl_clean, noisy_scalar.item())
    ssl_noise = torch.mul(ssl_noise, noisy_scalar.item())
    # check if any clipping happened
    needToClip = torch.gt(torch.abs(ssl_noisy), 0.99).any() # 0.99 is the clipping threshold 
    if (needToClip):
        noisyspeech_maxamplevel = torch.max(torch.abs(ssl_noisy)).item() / (0.99 - EPS)
        ssl_noisy = torch.div(ssl_noisy, noisyspeech_maxamplevel)
        ssl_noise = torch.div(ssl_noise, noisyspeech_maxamplevel)
        ssl_clean = torch.div(ssl_clean, noisyspeech_maxamplevel)
        noisy_rms_level = int(20 * np.log10(noisy_scalar/noisyspeech_maxamplevel * (noisy_rms + EPS)))
    return ssl_clean, ssl_noise, ssl_noisy, noisy_rms_level


device = torch.device('cuda:{}'.format(0))
conv_transform = torchaudio.transforms.Convolve("same").to(device)
CIPICSubject = CipicDatabase.subjects[12]
speechFilter  = torch.from_numpy(CIPICSubject.getHRIRFromIndex(624, 0)).float()
speechFilter  = speechFilter.unsqueeze(0).to(device)
noiseFilter   = torch.from_numpy(CIPICSubject.getHRIRFromIndex(600, 0)).float()
noiseFilter   = noiseFilter.unsqueeze(0).to(device) 
NOISY_FILE = './training_set/noisy/SLR61_es_ar_female_arf_06592_00774535859_48kHz-l_ile_mysterieuse_2_06_f000145-book_00394_chp_0008_re_f872gDrfumM-fan_Freesound_validated_437341_3-door_Freesound_validated_349356_0_snr16_tl-23_fileid_1542.wav'
CLEAN_FILE = './training_set/clean/clean_fileid_1542.wav'
NOISE_FILE = './training_set/noise/noise_fileid_1542.wav'
METADATA = {'snr': 16, 'target_level': -23}

# Define transform
spectrogram = T.Spectrogram(
        n_fft=512,
        power=2).to(device)

complex_spectrogram = T.Spectrogram(
        n_fft=512,
        power=None).to(device)

noisy, nysr = torchaudio.load(NOISY_FILE)
clean, clsr = torchaudio.load(CLEAN_FILE)
noise, nesr = torchaudio.load(NOISE_FILE)

noisy = noisy.to(device)
clean = clean.to(device)
noise = noise.to(device)

spec_noisy = spectrogram(noisy).cpu()
spec_clean = spectrogram(clean).cpu()
spec_noise = spectrogram(noise).cpu()

complex_noisy = complex_spectrogram(noisy).cpu()
complex_clean = complex_spectrogram(clean).cpu()
complex_noise = complex_spectrogram(noise).cpu()
mel_noisy_mag = librosa.feature.melspectrogram(
        S=complex_noisy.abs()[0].numpy(),
        sr=16000)
mel_noisy_phase = librosa.feature.melspectrogram(
        S=complex_noisy.angle()[0].numpy(),
        sr=16000)
mel_clean_mag = librosa.feature.melspectrogram(
        S=complex_clean.abs()[0].numpy(),
        sr=16000)
mel_clean_phase = librosa.feature.melspectrogram(
        S=complex_clean.angle()[0].numpy(),
        sr=16000)
mel_noise_mag = librosa.feature.melspectrogram(
        S=complex_noise.abs()[0].numpy(),
        sr=16000)
mel_noise_phase = librosa.feature.melspectrogram(
        S=complex_noise.angle()[0].numpy(),
        sr=16000)

ssl_noise = conv_transform(noise, noiseFilter)
ssl_clean = conv_transform(clean, speechFilter)

ssl_clean, ssl_noise, ssl_noisy, rms = mixer(
        ssl_clean, ssl_noise, METADATA['snr'],
        METADATA['target_level'],
        -35,
        -15)

# Perform transform
complex_ssl_noisy = complex_spectrogram(ssl_noisy).cpu()
complex_ssl_clean = complex_spectrogram(ssl_clean).cpu()
complex_ssl_noise = complex_spectrogram(ssl_noise).cpu()
mel_ssl_noisy_mag = librosa.feature.melspectrogram(
        S=complex_ssl_noisy.abs()[0].numpy(),
        sr=16000)
mel_ssl_noisy_phase = librosa.feature.melspectrogram(
        S=complex_ssl_noisy.angle()[0].numpy(),
        sr=16000)
mel_ssl_clean_mag = librosa.feature.melspectrogram(
        S=complex_ssl_clean.abs()[0].numpy(),
        sr=16000)
mel_ssl_clean_phase = librosa.feature.melspectrogram(
        S=complex_ssl_clean.angle()[0].numpy(),
        sr=16000)
mel_ssl_noise_mag = librosa.feature.melspectrogram(
        S=complex_ssl_noise.abs()[0].numpy(),
        sr=16000)
mel_ssl_noise_phase = librosa.feature.melspectrogram(
        S=complex_ssl_noise.angle()[0].numpy(),
        sr=16000)

fig  = plt.figure(figsize=(20,20))
gs = fig.add_gridspec(4,6)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[0, 4])
ax6 = fig.add_subplot(gs[0, 5])
ax7 = fig.add_subplot(gs[1, 0])
ax8 = fig.add_subplot(gs[1, 1])
ax9 = fig.add_subplot(gs[1, 2])
ax10 = fig.add_subplot(gs[1, 3])
ax11 = fig.add_subplot(gs[1, 4])
ax12 = fig.add_subplot(gs[1, 5])
ax13 = fig.add_subplot(gs[2, 0])
ax14 = fig.add_subplot(gs[2, 1])
ax15 = fig.add_subplot(gs[2, 2])
ax16 = fig.add_subplot(gs[2, 3])
ax17 = fig.add_subplot(gs[2, 4])
ax18 = fig.add_subplot(gs[2, 5])
ax19 = fig.add_subplot(gs[3, 0])
ax20 = fig.add_subplot(gs[3, 1])
ax21 = fig.add_subplot(gs[3, 2])
ax22 = fig.add_subplot(gs[3, 3])
ax23 = fig.add_subplot(gs[3, 4])
ax24 = fig.add_subplot(gs[3, 5])
#ax25 = fig.add_subplot(gs[4, 0])
#ax26 = fig.add_subplot(gs[4, 1])
#ax27 = fig.add_subplot(gs[4, 2])
#ax28 = fig.add_subplot(gs[4, 3])
#ax29 = fig.add_subplot(gs[4, 4])
#ax30 = fig.add_subplot(gs[4, 5])
plot_magnitude(complex_clean.abs()[0].numpy()  , title="Magnitude Clean", ax=ax1)
plot_phase(    complex_clean.angle()[0].numpy(), title="Phase Clean"    , ax=ax2)
plot_magnitude(complex_noise.abs()[0].numpy()  , title="Magnitude Noise", ax=ax3)
plot_phase(    complex_noise.angle()[0].numpy(), title="Phase Noise"    , ax=ax4)
plot_magnitude(complex_noisy.abs()[0].numpy()  , title="Magnitude Noisy", ax=ax5)
plot_phase(    complex_noisy.angle()[0].numpy(), title="Phase Noisy"    , ax=ax6)
plot_magnitude(complex_ssl_clean.abs()[0].numpy()  , title="Magnitude SS Clean", ax=ax7)
plot_phase(    complex_ssl_clean.angle()[0].numpy(), title="Phase SS Clean"    , ax=ax8)
plot_magnitude(complex_ssl_noise.abs()[0].numpy()  , title="Magnitude SS Noise", ax=ax9)
plot_phase(    complex_ssl_noise.angle()[0].numpy(), title="Phase SS Noise"    , ax=ax10)
plot_magnitude(complex_ssl_noisy.abs()[0].numpy()  , title="Magnitude SS Noisy", ax=ax11)
plot_phase(    complex_ssl_noisy.angle()[0].numpy(), title="Phase SS Noisy"    , ax=ax12)
plot_magnitude(mel_clean_mag, title="Magnitude Clean (Mel Scale)", ax=ax13)
plot_phase(    mel_clean_phase, title="Phase Clean (Mel Scale)"    , ax=ax14)
plot_magnitude(mel_noise_mag  , title="Magnitude Noise (Mel Scale)", ax=ax15)
plot_phase(    mel_noise_phase, title="Phase Noise (Mel Scale)"    , ax=ax16)
plot_magnitude(mel_noisy_mag, title="Magnitude Noisy (Mel Scale)", ax=ax17)
plot_phase(    mel_noisy_phase, title="Phase Noisy (Mel Scale)"    , ax=ax18)
plot_magnitude(mel_ssl_clean_mag  , title="Magnitude SS Clean (Mel Scale)", ax=ax19)
plot_phase(    mel_ssl_clean_phase, title="Phase SS Clean (Mel Scale)"    , ax=ax20)
plot_magnitude(mel_ssl_noise_mag  , title="Magnitude SS Noise (Mel Scale)", ax=ax21)
plot_phase(    mel_ssl_noise_phase, title="Phase SS Noise (Mel Scale)"    , ax=ax22)
plot_magnitude(mel_ssl_noisy_mag  , title="Magnitude SS Noisy (Mel Scale)", ax=ax23)
plot_phase(    mel_ssl_noisy_phase, title="Phase SS Noisy (Mel Scale)"    , ax=ax24)
#plot_magnitude(spec_clean.abs()[0].numpy()  , title="Magnitude Clean", ax=ax25)
#plot_phase(    spec_clean.angle()[0].numpy(), title="Phase Clean"    , ax=ax26)
#plot_magnitude(spec_noise.abs()[0].numpy()  , title="Magnitude Noise", ax=ax27)
#plot_phase(    spec_noise.angle()[0].numpy(), title="Phase Noise"    , ax=ax28)
#plot_magnitude(spec_noisy.abs()[0].numpy()  , title="Magnitude Noisy", ax=ax29)
#plot_phase(    spec_noisy.angle()[0].numpy(), title="Phase Noisy"    , ax=ax30)
fig.tight_layout()
plt.savefig("complex.png")
plt.close()
