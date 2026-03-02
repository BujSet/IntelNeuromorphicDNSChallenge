import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import sys
import numpy as np
import parselmouth

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

def plot_pitch(waveform, sr, title="Waveform", ax=None, ylabel=None, praat_pitch=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) * (30.0 / num_frames)

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1, color="blue", label="Torchaudio")
    ax.grid(True)
    ax.set_xlim([0, 30.0])
    ax.set_title(title)
    if (ylabel != None):
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (sec)")
    if (praat_pitch == None):
    	return
    ax.plot(time_axis, praat_pitch, color="red", linewidth=2, label="Praat")
    ax.legend()

def plot_spectrogram(specgram, title=None, ylabel="Frequency (kHz)", ax=None, yticks=[], yticklabels=[], praat_pitch=None, torchaudio_pitch=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (FFT Frame)")
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    if (praat_pitch == None and torchaudio_pitch == None):
    	return
    if (praat_pitch != None and torchaudio_pitch != None):
        ax2 = ax.twinx()
        ax2.set_ylim([0, 8.0])
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.plot(range(0, len(praat_pitch)), praat_pitch, linewidth=1, color="red", label="Praat")
        ax2.plot(range(0, len(torchaudio_pitch)), torchaudio_pitch, linewidth=1, color="magenta", label="TorchAudio")
        ax2.legend()
    if (praat_pitch != None and torchaudio_pitch == None):
        ax2 = ax.twinx()
        ax2.set_ylim([0, 8.0])
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.plot(range(0, len(praat_pitch)), praat_pitch, linewidth=1, color="red", label="Praat")
        ax2.legend()
    if (praat_pitch == None and torchaudio_pitch != None):
        ax2 = ax.twinx()
        ax2.set_ylim([0, 8.0])
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.plot(range(0, len(torchaudio_pitch)), torchaudio_pitch, linewidth=1, color="magenta", label="TorchAudio")
        ax2.legend()

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

def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

def draw_praat_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")


NOISY_FILE = './training_set/noisy/SLR61_es_ar_female_arf_06592_00774535859_48kHz-l_ile_mysterieuse_2_06_f000145-book_00394_chp_0008_re_f872gDrfumM-fan_Freesound_validated_437341_3-door_Freesound_validated_349356_0_snr16_tl-23_fileid_1542.wav'
CLEAN_FILE = './training_set/clean/clean_fileid_37978.wav' # 
CLEAN_FILE = './training_set/clean/clean_fileid_1542.wav'
NOISE_FILE = './training_set/noise/noise_fileid_1542.wav'
METADATA = {'snr': 16, 'target_level': -23}

noisy, nysr = torchaudio.load(NOISY_FILE)
clean, clsr = torchaudio.load(CLEAN_FILE)
print("Torchaudio clean speech size=" + str(clean.size()))
print("Torchaudio clean speech sampling rate=" + str(clsr))
noise, nesr = torchaudio.load(NOISE_FILE)

noisy_snd = parselmouth.Sound(NOISY_FILE)
clean_snd = parselmouth.Sound(CLEAN_FILE)
noise_snd = parselmouth.Sound(NOISE_FILE)

# If desired, pre-emphasize the sound fragment before calculating the spectrogram
pre_emphasized_snd = noisy_snd.copy()
pre_emphasized_snd.pre_emphasize()
noisy_spectrogram_praat = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=16000)

pre_emphasized_snd = clean_snd.copy()
pre_emphasized_snd.pre_emphasize()
clean_spectrogram_praat = pre_emphasized_snd.to_spectrogram(window_length=0.004, maximum_frequency=16000)
X, Y = clean_spectrogram_praat.x_grid(), clean_spectrogram_praat.y_grid()
print("Praat clean speech spectrogram X dim=" + str(len(X)))
print("Praat clean speech spectrogram Y dim=" + str(len(Y)))

pre_emphasized_snd = noise_snd.copy()
pre_emphasized_snd.pre_emphasize()
noise_spectrogram_praat = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=16000)


# Human fundamental pitch frequency is very limited
# 100 - 500 for males, 130-800 for females from https://pmc.ncbi.nlm.nih.gov/articles/PMC4911068/
#  80 - 200 for males, 150-350 for femalers from https://www.sciencedirect.com/topics/computer-science/fundamental-frequency
noisy_pitch_praat = noisy_snd.to_pitch()
clean_pitch_praat = clean_snd.to_pitch(time_step=0.004, pitch_floor=50.0, pitch_ceiling=1000.0)
noise_pitch_praat = noise_snd.to_pitch()

#print(clean_pitch_praat.selected_array['frequency'])
#print(clean_pitch_praat.xs())

print("Praat clean spech freq array len=" + str(len(clean_pitch_praat.selected_array['frequency'])))
print("Praat clean spech freq x array len=" + str(len(clean_pitch_praat.xs())))
clean_period = 1.0 / (1.0 * clsr)
clean_pitch_freq = []
# think this should be n_fft/2???  
for i in range(0, clean.size()[1], 256):
    t = i * clean_period
#    clean_pitch_freq.append(clean_pitch_praat.get_value_at_time(t)  )
    clean_pitch_freq.append(round(clean_pitch_praat.get_value_at_time(t)/1000.0,2)  )

clean_pitch_freq.append(np.nan)
print(len(clean_pitch_freq))

clean_pitch_time = []
for i in range(0, clean.size()[1], 64):
	t = i * clean_period
	clean_pitch_time.append(clean_pitch_praat.get_value_at_time(t))
print("Praat clean pitch time len =" + str(len(clean_pitch_time)))
clean_pitch_time = clean_pitch_time[0:len(clean_pitch_time)-8]
clean_pitch_time = clean_pitch_time[7:]
print("Praat clean pitch time len =" + str(len(clean_pitch_time)))

# Define transform
spectrogram = T.Spectrogram(
        n_fft=512,
        power=None)

mel_spectrogram = T.MelSpectrogram(
    sample_rate=16000,
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


plt.figure()
draw_praat_spectrogram(noisy_spectrogram_praat)
plt.twinx()
draw_pitch(noisy_pitch_praat)
plt.savefig("praat_noisy.png")
plt.close()

plt.figure()
draw_praat_spectrogram(clean_spectrogram_praat)
plt.twinx()
draw_pitch(clean_pitch_praat)
plt.savefig("praat_clean.png")
plt.close()

plt.figure()
draw_praat_spectrogram(noise_spectrogram_praat)
plt.twinx()
draw_pitch(noise_pitch_praat)
plt.savefig("praat_noise.png")
plt.close()

pitch_noisy = F.detect_pitch_frequency(noisy, 16000)
pitch_clean = F.detect_pitch_frequency(clean, 16000, frame_time=0.004, freq_low=50.0, freq_high=1000.0)
pitch_noise = F.detect_pitch_frequency(noise, 16000)

pitch_clean_freq = F.detect_pitch_frequency(clean, 16000, frame_time=0.016, win_length=30, freq_low=50.0, freq_high=1000.0)
pitch_clean_freq = pitch_clean_freq.squeeze().tolist()
print("TorchAudio clean pitch freq len =" + str(len(pitch_clean_freq)))
pitch_clean_freq = [0,0,0,0,0,0,0,0]  + pitch_clean_freq + [0,0,0,0,0,0,0,0]
print("TorchAudio clean pitch freq len =" + str(len(pitch_clean_freq)))
pitch_clean_freq = [i/1000.0 for i in pitch_clean_freq]

freq_map = librosa.fft_frequencies(sr=16000, n_fft=512)
yticks = [i for i in range(0, len(freq_map), int(len(freq_map)/60))]
yticklabels = [round(freq_map[i]/1000.0, 2) for i in yticks]

mel_map = librosa.mel_frequencies(n_mels=64, htk=True)
mel_yticks = [i for i in range(0, len(mel_map), int(len(mel_map)/5))]
mel_yticklabels = [round(mel_map[i]/1000.0, 2) for i in mel_yticks]

fig, axs = plt.subplots(2, 2, figsize=(60,30))
plot_waveform(noisy, nysr, title="Original waveform", ax=axs[0, 0])
plot_waveform(pitch_noisy, nysr, title="Pitch", ax=axs[1, 0], ylabel="Frequency (Hz)")
plot_spectrogram(spec_noisy, title="Power Spectrogram", ax=axs[0, 1], yticks=yticks, yticklabels=yticklabels)
plot_spectrogram(mel_spec_noisy, title="Power Mel Spectrogram", ax=axs[1, 1], yticks=mel_yticks, yticklabels=mel_yticklabels)
fig.tight_layout()
plt.savefig("spec_noisy.png")
plt.close()
#plot_phase(spec_noisy, title="Phase Spectrogram", yticks=yticks, yticklabels=yticklabels, filename="spec_noisy_phase.png")

fig, axs = plt.subplots(2, 2, figsize=(60,30))
plot_waveform(clean, clsr, title="Original waveform", ax=axs[0, 0])
plot_pitch(pitch_clean, clsr, title="Pitch", ax=axs[1, 0], ylabel="Frequency (Hz)", praat_pitch=clean_pitch_time)
plot_spectrogram(spec_clean[0], title="spectrogram", ax=axs[0,1], yticks=yticks, yticklabels=yticklabels)
plot_spectrogram(mel_spec_clean, title="Power Mel Spectrogram", ax=axs[1,1], yticks=mel_yticks, yticklabels=mel_yticklabels)
# plot_spectrogram(spec_clean[0], title="spectrogram", ax=axs[0,1], yticks=yticks, yticklabels=yticklabels, praat_pitch=clean_pitch_freq, torchaudio_pitch=None)
# plot_spectrogram(spec_clean[0], title="spectrogram", ax=axs[1,1], yticks=yticks, yticklabels=yticklabels, praat_pitch=None, torchaudio_pitch=pitch_clean_freq)
#plot_spectrogram(mel_spec_clean, title="Power Mel Spectrogram", ax=axs[1, 1], yticks=mel_yticks, yticklabels=mel_yticklabels)
fig.tight_layout()
plt.savefig("spec_clean.png")
plt.close()
#plot_phase(spec_clean, title="Phase Spectrogram", yticks=yticks, yticklabels=yticklabels, filename="spec_clean_phase.png")

fig, axs = plt.subplots(2, 2, figsize=(60,30))
plot_waveform(noise, nesr, title="Original waveform", ax=axs[0, 0])
plot_waveform(pitch_noise, nesr, title="Original waveform", ax=axs[1, 0], ylabel="Frequency (Hz)")
plot_spectrogram(spec_noise[0], title="spectrogram", ax=axs[0,1], yticks=yticks, yticklabels=yticklabels)
plot_spectrogram(mel_spec_noise, title="Power Mel Spectrogram", ax=axs[1,1], yticks=mel_yticks, yticklabels=mel_yticklabels)
fig.tight_layout()
plt.savefig("spec_noise.png")
plt.close()
#plot_phase(spec_noise, title="Phase Spectrogram", yticks=yticks, yticklabels=yticklabels, filename="spec_noise_phase.png")
