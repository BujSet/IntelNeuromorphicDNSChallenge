from cipic_db import CipicDatabase
import matplotlib.pyplot as plt
import torch
import torchaudio
import numpy as np

def get_db_response(filter_freq, input_freq):
    filter_mag = filter_freq.abs() + 1e-8
    input_mag = input_freq.abs() + 1e-8

    spectrogram = 20 * torch.log10(torch.div(filter_mag, input_mag)).numpy()
    return np.average(spectrogram)

def get_db_resp(el, tone, channel):
    t = np.linspace(0, 30, 16000 * 30)
    y = np.sin(tone * 2 * np.pi * t)  #  Has frequency of 440Hz
    pureTone = torch.from_numpy(y).float()
    idx, deg = el
    fir  = torch.from_numpy(CipicDatabase.subjects[12].getHRIRFromIndex(idx, channel)).float()
    response = torchaudio.functional.convolve(pureTone, fir, "same")
    stft = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        hop_length=256,
        power=None,
    )
    spec_clean = stft(pureTone)
    spec_filter = stft(response)
    return get_db_response(spec_filter, spec_clean)

def get_heatmap(startIdx, channel, tones):
    posits = CipicDatabase.subjects[12].getCartesianPositions()
    yval = round(posits[startIdx][1], 3)
    elevations = []
    for i in range(50):
        index = startIdx + i
        degree = -45 + (5.625*i)
        elevations.append((index, degree))
    heatmap = np.zeros((len(elevations), len(tones)))
    for i in range(len(elevations)):
        el = elevations[i]
        for j in range(len(tones)):
            tone = tones[j]
            db_resp = get_db_resp(el, tone, channel)
            heatmap[i, j] = db_resp
    return heatmap

fig, axs = plt.subplots(2, 7, sharex=True, sharey=True, figsize=(40, 10))
tones = [20] + list(range(500,20000,500))
im = axs[0,0].imshow(get_heatmap(1200, 1,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 0 0")
im = axs[0,1].imshow(get_heatmap(1000, 1,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 0 1")
im = axs[0,2].imshow(get_heatmap( 800, 1,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 0 2")
im = axs[0,3].imshow(get_heatmap( 600, 1,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 0 3")
im = axs[0,4].imshow(get_heatmap( 400, 1,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 0 4")
im = axs[0,5].imshow(get_heatmap( 200, 1,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 0 5")
im = axs[0,6].imshow(get_heatmap(   0, 1,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 0 6")
im = axs[1,0].imshow(get_heatmap(1200, 0,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 1 0")
im = axs[1,1].imshow(get_heatmap(1000, 0,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 1 1")
im = axs[1,2].imshow(get_heatmap( 800, 0,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 1 2")
im = axs[1,3].imshow(get_heatmap( 600, 0,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 1 3")
im = axs[1,4].imshow(get_heatmap( 400, 0,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 1 4")
im = axs[1,5].imshow(get_heatmap( 200, 0,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 1 5")
im = axs[1,6].imshow(get_heatmap(   0, 0,tones), vmin=-20, vmax=20, cmap='autumn', interpolation='nearest', aspect='auto')
print("Done 1 6")
axs[1,3].set_xlabel("Input Frequency (kHz)")
axs[0,0].set_ylabel("Channel 1\nElevation (°)")
axs[1,0].set_ylabel("Channel 0\nElevation (°)")
axs[0,0].set_yticks([0,8,16,24,32, 40,48])
axs[0,0].set_yticklabels(["-45","0","45", "90","45","0","-45"])
ax2 = axs[0,6].twinx()
ax2.set_yticks([49,24,0])
ax2.set_yticklabels(["front","up","back"], rotation=90)
axs[1,0].set_yticks([0,8,16,24,32,40,48]) #,49])
axs[1,0].set_yticklabels(["-45","0","45", "90","45","0","-45"])#, "-50.625"])
ax2 = axs[1,6].twinx()
ax2.set_yticks([49,24,0])
ax2.set_yticklabels(["front","up","back"], rotation=90)
myxticks = list(range(0, len(tones), len(tones)//4))
axs[1,0].set_xticks(myxticks)
axs[1,0].set_xticklabels([str(tones[i]/1000.0) for i in myxticks])
posits = CipicDatabase.subjects[12].getCartesianPositions()
yval = round(posits[1200][1], 3)
axs[0,0].set_title("y="+str(yval)) 
yval = round(posits[1000][1], 3)
axs[0,1].set_title("y="+str(yval)) 
yval = round(posits[800][1], 3)
axs[0,2].set_title("y="+str(yval)) 
yval = round(posits[600][1], 3)
axs[0,3].set_title("y="+str(yval)) 
yval = round(posits[400][1], 3)
axs[0,4].set_title("y="+str(yval)) 
yval = round(posits[200][1], 3)
axs[0,5].set_title("y="+str(yval)) 
yval = round(posits[0][1], 3)
axs[0,6].set_title("y="+str(yval)) 
yval = round(posits[1200][1], 3)
axs[1,0].set_title("y="+str(yval)) 
yval = round(posits[1000][1], 3)
axs[1,1].set_title("y="+str(yval)) 
yval = round(posits[800][1], 3)
axs[1,2].set_title("y="+str(yval)) 
yval = round(posits[600][1], 3)
axs[1,3].set_title("y="+str(yval)) 
yval = round(posits[400][1], 3)
axs[1,4].set_title("y="+str(yval)) 
yval = round(posits[200][1], 3)
axs[1,5].set_title("y="+str(yval)) 
yval = round(posits[0][1], 3)
axs[1,6].set_title("y="+str(yval)) 
fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.01])
cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Transfer dB")
plt.savefig("magnitude.png", bbox_inches='tight')
plt.close()

