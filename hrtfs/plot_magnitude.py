from cipic_db import CipicDatabase
import matplotlib.pyplot as plt
import torch
import torchaudio
import numpy as np
from scipy.ndimage import gaussian_filter
import librosa

SUBJECT = 9  
SAMPLE_RATE = 44100.0
NFFT = 256

def format_azimuth_angle(azimuth):
    if azimuth > 180.0:
        azimuth -= 360.0
    if azimuth > 0.0:
        azimuth = "+" + str(azimuth) + "°"
    else:
        azimuth = str(azimuth) + "°"
    return azimuth

def get_db_response(filter_freq, input_freq):
    filter_mag = torch.sqrt(torch.square(filter_freq.real) + torch.square(filter_freq.imag))
    spectrogram = 20 * torch.log10(filter_mag).numpy()
    # only care about first frame
    spectrogram = np.squeeze(spectrogram[:, 0])
    return spectrogram

def get_db_resp(el, channel):
    idx, deg = el
    fir  = torch.from_numpy(CipicDatabase.subjects[SUBJECT].getHRIRFromIndex(idx, channel)).float()
    num_samples = len(fir)
    t = np.linspace(0, 1000.0*num_samples/SAMPLE_RATE, num_samples)
     # y is now a simple impulse
    y = np.copy(t)
    y[0] = 1.0

    pureTone = torch.from_numpy(y).float()
    response = fir
    stft = torchaudio.transforms.Spectrogram(
        n_fft=NFFT,
        win_length=NFFT,
        hop_length=NFFT,
        power=None,
    )
    spec_clean = stft(pureTone)
    spec_filter = stft(response)
    return get_db_response(spec_filter, spec_clean)

def get_heatmap(startIdx, channel, tones):
    posits = CipicDatabase.subjects[SUBJECT].getCartesianPositions()
    yval = round(posits[startIdx][1], 3)
    elevations = []
    for i in range(50):
        index = startIdx + i
        degree = -45 + (5.625*i)
        elevations.append((index, degree))
    heatmap = np.zeros((len(elevations), len(tones)))
    for i in range(len(elevations)):
        el = elevations[i]
        db_resp = get_db_resp(el, channel)
        for j in range(len(tones)):
            heatmap[i, j] = db_resp[j]
    return gaussian_filter(np.transpose(heatmap), sigma=0.75)

fig, axs = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(20, 10))
fig.suptitle("Subject " + str(SUBJECT) + "\nHorizontal Azimuth (°)", fontweight="bold", fontsize=16, y=0.95)
freq_map = librosa.fft_frequencies(sr=44100, n_fft=NFFT)
tones = np.linspace(0, freq_map[-1], int(NFFT/2)+1)
vLow = 0
vHigh = 0
heatmaps = [get_heatmap(1200, 1,tones)]
print("Done 0 0")
heatmaps.append(get_heatmap( 900, 1,tones))
print("Done 0 1")
heatmaps.append(get_heatmap( 600, 1,tones))
print("Done 0 2")
heatmaps.append(get_heatmap( 300, 1,tones))
print("Done 0 3")
heatmaps.append(get_heatmap(   0, 1,tones))
print("Done 0 4")
heatmaps.append(get_heatmap(1200, 0,tones))
print("Done 1 0")
heatmaps.append(get_heatmap( 900, 0,tones))
print("Done 1 1")
heatmaps.append(get_heatmap( 600, 0,tones))
print("Done 1 2")
heatmaps.append(get_heatmap( 300, 0,tones))
print("Done 1 3")
heatmaps.append(get_heatmap(   0, 0,tones))
print("Done 1 4")
for h in heatmaps:
    vLow = min(vLow, np.min(h))
    vHigh = max(vHigh, np.max(h))
im = axs[0,0].imshow(heatmaps[0], vmin=vLow, vmax=vHigh, cmap='autumn', interpolation='nearest', aspect='auto', origin='lower')
im = axs[0,1].imshow(heatmaps[1], vmin=vLow, vmax=vHigh, cmap='autumn', interpolation='nearest', aspect='auto', origin='lower')
im = axs[0,2].imshow(heatmaps[2], vmin=vLow, vmax=vHigh, cmap='autumn', interpolation='nearest', aspect='auto', origin='lower')
im = axs[0,3].imshow(heatmaps[3], vmin=vLow, vmax=vHigh, cmap='autumn', interpolation='nearest', aspect='auto', origin='lower')
im = axs[0,4].imshow(heatmaps[4], vmin=vLow, vmax=vHigh, cmap='autumn', interpolation='nearest', aspect='auto', origin='lower')
im = axs[1,0].imshow(heatmaps[5], vmin=vLow, vmax=vHigh, cmap='autumn', interpolation='nearest', aspect='auto', origin='lower')
im = axs[1,1].imshow(heatmaps[6], vmin=vLow, vmax=vHigh, cmap='autumn', interpolation='nearest', aspect='auto', origin='lower')
im = axs[1,2].imshow(heatmaps[7], vmin=vLow, vmax=vHigh, cmap='autumn', interpolation='nearest', aspect='auto', origin='lower')
im = axs[1,3].imshow(heatmaps[8], vmin=vLow, vmax=vHigh, cmap='autumn', interpolation='nearest', aspect='auto', origin='lower')
im = axs[1,4].imshow(heatmaps[9], vmin=vLow, vmax=vHigh, cmap='autumn', interpolation='nearest', aspect='auto', origin='lower')
axs[1,2].set_xlabel("Vertical Elevation (°)", fontweight="bold", fontsize=16, y=0.10)
axs[0,0].set_ylabel("Channel 1\nInput Frequency (kHz)", fontweight="bold", fontsize=16)
axs[1,0].set_ylabel("Channel 0\nInput Frequency (kHz)", fontweight="bold", fontsize=16)
axs[1,0].set_xticks([0,8,16,24,32,40,48]) #,49])
axs[1,0].set_xticklabels(["-45°\nFront","0°","45°", "90°","45°","0°","-45°\nBack"])#, "-50.625"])
myxticks = list(range(0, len(tones), len(tones)//4))
axs[1,0].set_yticks(myxticks)
axs[1,0].set_yticklabels([str(round(tones[i]/1000.0, 1)) for i in myxticks])
posits = CipicDatabase.subjects[SUBJECT].getCartesianPositions()

azimuth = round(CipicDatabase.subjects[SUBJECT].getSphericalPositionsFromIndex(1208)[0], 1)
axs[0,0].set_title(format_azimuth_angle(azimuth)) 

azimuth = round(CipicDatabase.subjects[SUBJECT].getSphericalPositionsFromIndex(908)[0], 1)
axs[0,1].set_title(format_azimuth_angle(azimuth)) 

azimuth = round(CipicDatabase.subjects[SUBJECT].getSphericalPositionsFromIndex(608)[0], 1)
axs[0,2].set_title(format_azimuth_angle(azimuth)) 

azimuth = round(CipicDatabase.subjects[SUBJECT].getSphericalPositionsFromIndex(308)[0], 1)
axs[0,3].set_title(format_azimuth_angle(azimuth))

azimuth = round(CipicDatabase.subjects[SUBJECT].getSphericalPositionsFromIndex(8)[0], 1)
axs[0,4].set_title(format_azimuth_angle(azimuth))

azimuth = round(CipicDatabase.subjects[SUBJECT].getSphericalPositionsFromIndex(1208)[0], 1)
axs[1,0].set_title(format_azimuth_angle(azimuth))

azimuth = round(CipicDatabase.subjects[SUBJECT].getSphericalPositionsFromIndex(908)[0], 1)
axs[1,1].set_title(format_azimuth_angle(azimuth))

azimuth = round(CipicDatabase.subjects[SUBJECT].getSphericalPositionsFromIndex(608)[0], 1)
axs[1,2].set_title(format_azimuth_angle(azimuth)) 

azimuth = round(CipicDatabase.subjects[SUBJECT].getSphericalPositionsFromIndex(308)[0], 1)
axs[1,3].set_title(format_azimuth_angle(azimuth)) 

azimuth = round(CipicDatabase.subjects[SUBJECT].getSphericalPositionsFromIndex(8)[0], 1)
axs[1,4].set_title(format_azimuth_angle(azimuth))

fig.subplots_adjust(bottom=0.225)
cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.01])
cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Transfer dB", fontweight="bold", fontsize=16)
plt.savefig("magnitude.pdf", bbox_inches='tight', format="pdf")
plt.close()

