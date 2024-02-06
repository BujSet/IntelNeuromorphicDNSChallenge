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

def plot_complex(complex_tensor, title="Complex FFT", ax=None):
    ax.set_title(title)
    ax.set_ylabel("Imag")
    ax.set_ylabel("Real")
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
#spectrogram = T.Spectrogram(
#        n_fft=512,
#        power=2).to(device)

complex_spectrogram = T.Spectrogram(
        n_fft=512,
        power=None).to(device)

#mel_spectrogram = T.MelSpectrogram(
#    sample_rate=16000,
#    n_fft=512,
#    win_length=None,
#    hop_length=256,
#    center=True,
#    pad_mode="reflect",
#    power=1.0,
#    norm="slaney",
#    n_mels=128,
#    mel_scale="htk",
#).to(device)

#melscale = transforms.MelScale(sample_rate=16000, 
#        n_stft=257).to(device)


noisy, nysr = torchaudio.load(NOISY_FILE)
clean, clsr = torchaudio.load(CLEAN_FILE)
noise, nesr = torchaudio.load(NOISE_FILE)

noisy = noisy.to(device)
clean = clean.to(device)
noise = noise.to(device)

complex_noisy = complex_spectrogram(noisy).cpu()
complex_clean = complex_spectrogram(clean).cpu()
complex_noise = complex_spectrogram(noise).cpu()

cplx_noisy_real = torch.zeros(complex_noisy.size(), dtype=torch.float64, layout=complex_noisy.layout, device=complex_noisy.device)
cplx_noisy_imag = torch.zeros(complex_noisy.size(), dtype=torch.float64, layout=complex_noisy.layout, device=complex_noisy.device)
cplx_noise_real = torch.zeros(complex_noise.size(), dtype=torch.float64, layout=complex_noise.layout, device=complex_noise.device)
cplx_noise_imag = torch.zeros(complex_noise.size(), dtype=torch.float64, layout=complex_noise.layout, device=complex_noise.device)
cplx_clean_real = torch.zeros(complex_clean.size(), dtype=torch.float64, layout=complex_clean.layout, device=complex_clean.device)
cplx_clean_imag = torch.zeros(complex_clean.size(), dtype=torch.float64, layout=complex_clean.layout, device=complex_clean.device)
for i in range(257):
    for j in range(1876):
        cplx_noisy_real[0, i, j] = complex_noisy[0, i, j].real
        cplx_noisy_imag[0, i, j] = complex_noisy[0, i, j].imag
        cplx_noise_real[0, i, j] = complex_noise[0, i, j].real
        cplx_noise_imag[0, i, j] = complex_noise[0, i, j].imag
        cplx_clean_real[0, i, j] = complex_clean[0, i, j].real
        cplx_clean_imag[0, i, j] = complex_clean[0, i, j].imag
cplx_noisy = torch.complex(cplx_noisy_real, cplx_noisy_imag)
cplx_noise = torch.complex(cplx_noise_real, cplx_noise_imag)
cplx_clean = torch.complex(cplx_clean_real, cplx_clean_imag)
        

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

print(complex_noisy.size())
print(complex_ssl_noisy.size())
print(complex_noisy)
print(complex_noisy.dtype)
print(type(complex_noisy))
print(torch.view_as_real(complex_noisy))

noisy = noisy.cpu()
clean = clean.cpu()
noise = noise.cpu()
#noisy = ssl_noisy.cpu()
#clean = ssl_clean.cpu()
#noise = ssl_noise.cpu()

fig  = plt.figure()
gs = fig.add_gridspec(1,3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
plot_complex(cplx_clean[0], title="Complex FFT Clean", ax=ax1)
plot_complex(cplx_noise[0], title="Complex FFT Noise", ax=ax2)
plot_complex(cplx_noisy[0], title="Complex FFT Noisy", ax=ax3)
fig.tight_layout()
plt.savefig("complex.png")
plt.close()
#fig, axs = plt.subplots(3, 1)
#plot_waveform(noisy, nysr, title="Noisy", ax=axs[0])
#plot_spectrogram(spec_noisy[0], title="Spectrogram", ax=axs[1])
#plot_spectrogram(mel_spec_noisy[0], title="Mel Spectrogram", ax=axs[2], ylabel='mel freq')
#fig.tight_layout()
#plt.savefig("mel_spec_noisy_ssl.png")
#plt.close()

#fig, axs = plt.subplots(3, 1)
#plot_waveform(clean, clsr, title="Clean", ax=axs[0])
#plot_spectrogram(spec_clean[0], title="Spectrogram", ax=axs[1])
#plot_spectrogram(mel_spec_clean[0], title="Mel Spectrogram", ax=axs[2], ylabel='mel freq')
#fig.tight_layout()
#plt.savefig("mel_spec_clean_ssl.png")
#plt.close()

#fig, axs = plt.subplots(3, 1)
#plot_waveform(noise, nesr, title="Noise", ax=axs[0])
#plot_spectrogram(spec_noise[0], title="Spectrogram", ax=axs[1])
#plot_spectrogram(mel_spec_noise[0], title="Mel Spectrogram", ax=axs[2], ylabel='mel freq')
#fig.tight_layout()
#plt.savefig("mel_spec_noise_ssl.png")
#plt.close()

#sf.write('ssl_noisy.wav', np.ravel(noisy.numpy()), nysr)
#sf.write('ssl_clean.wav', np.ravel(clean.numpy()), clsr)
#sf.write('ssl_noise.wav', np.ravel(noise.numpy()), nesr)

