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

complex_spec = T.Spectrogram(
        n_fft=512,
        power=1).to(device)

mel_spectrogram = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    win_length=None,
    hop_length=256,
    center=True,
    pad_mode="reflect",
    power=1.0,
    norm="slaney",
    n_mels=128,
    mel_scale="htk",
).to(device)

melscale = transforms.MelScale(sample_rate=16000, 
        n_stft=257).to(device)


noisy, nysr = torchaudio.load(NOISY_FILE)
clean, clsr = torchaudio.load(CLEAN_FILE)
noise, nesr = torchaudio.load(NOISE_FILE)

noisy = noisy.to(device)
clean = clean.to(device)
noise = noise.to(device)

mel_noisy = mel_spectrogram(noisy).cpu()
mel_clean = mel_spectrogram(clean).cpu()
mel_noise = mel_spectrogram(noise).cpu()

complex_noisy = mel_spectrogram(noisy).cpu()
complex_clean = mel_spectrogram(clean).cpu()
complex_noise = mel_spectrogram(noise).cpu()

ssl_noise = conv_transform(noise, noiseFilter)
ssl_clean = conv_transform(clean, speechFilter)

ssl_clean, ssl_noise, ssl_noisy, rms = mixer(
        ssl_clean, ssl_noise, METADATA['snr'],
        METADATA['target_level'],
        -35,
        -15)

# Perform transform
spec_noisy = spectrogram(ssl_noisy).cpu()
spec_clean = spectrogram(ssl_clean).cpu()
spec_noise = spectrogram(ssl_noise).cpu()

mel_spec_noisy = mel_spectrogram(ssl_noisy).cpu()
mel_spec_clean = mel_spectrogram(ssl_clean).cpu()
mel_spec_noise = mel_spectrogram(ssl_noise).cpu()
print(mel_spec_noisy.size())
print(mel_spec_noisy)

noisy = ssl_noisy.cpu()
clean = ssl_clean.cpu()
noise = ssl_noise.cpu()

fig  = plt.figure()
gs = fig.add_gridspec(5,2)
#fig, axs = plt.subplots(3, 1)
plot_waveform(noisy, nysr, title="Noisy", ax=axs[0])
plot_spectrogram(spec_noisy[0], title="Spectrogram", ax=axs[1])
plot_spectrogram(mel_spec_noisy[0], title="Mel Spectrogram", ax=axs[2], ylabel='mel freq')
fig.tight_layout()
plt.savefig("mel_spec_noisy_ssl.png")
plt.close()

fig, axs = plt.subplots(3, 1)
plot_waveform(clean, clsr, title="Clean", ax=axs[0])
plot_spectrogram(spec_clean[0], title="Spectrogram", ax=axs[1])
plot_spectrogram(mel_spec_clean[0], title="Mel Spectrogram", ax=axs[2], ylabel='mel freq')
fig.tight_layout()
plt.savefig("mel_spec_clean_ssl.png")
plt.close()

fig, axs = plt.subplots(3, 1)
plot_waveform(noise, nesr, title="Noise", ax=axs[0])
plot_spectrogram(spec_noise[0], title="Spectrogram", ax=axs[1])
plot_spectrogram(mel_spec_noise[0], title="Mel Spectrogram", ax=axs[2], ylabel='mel freq')
fig.tight_layout()
plt.savefig("mel_spec_noise_ssl.png")
plt.close()

#sf.write('ssl_noisy.wav', np.ravel(noisy.numpy()), nysr)
#sf.write('ssl_clean.wav', np.ravel(clean.numpy()), clsr)
#sf.write('ssl_noise.wav', np.ravel(noise.numpy()), nesr)

