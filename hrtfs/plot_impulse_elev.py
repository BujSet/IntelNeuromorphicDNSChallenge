from cipic_db import CipicDatabase
import matplotlib.pyplot as plt
import torch, sys
import torchaudio
import numpy as np
import math
from matplotlib import cm

SUBJECT = 10
ch1_resps = [CipicDatabase.subjects[SUBJECT].getHRIRFromIndex(600 + i, 1) for i in range(50)]
minCH = np.min(ch1_resps[0])
maxCH = np.max(ch1_resps[0])
for i in range(50):
    minCH = min(minCH, np.min(ch1_resps[i]))
    maxCH = max(maxCH, np.max(ch1_resps[i]))

NFFT = 256
SAMPLE_RATE = 44100.0
num_samples = len(ch1_resps[0])
t = np.linspace(0, 1000.0*num_samples/SAMPLE_RATE, num_samples)
pure_impulse = np.copy(t)
pure_impulse[0] = 1.0

stft = torchaudio.transforms.Spectrogram(
    n_fft=NFFT,
    win_length=NFFT,
    hop_length=NFFT,
    power=None, # returns complex tensor
)
freqs = [round(i * SAMPLE_RATE / (1000.0 * NFFT),2) for i in range(NFFT//2 + 1)]
spec_impulse = stft(torch.from_numpy(pure_impulse).float())
ampl_impulse = stft(torch.from_numpy(pure_impulse).float())
ampl_impulse = torch.sqrt(torch.square(spec_impulse.real) + torch.square(spec_impulse.imag))
phas_impulse = torch.atan(torch.div(spec_impulse.imag, spec_impulse.real))

specs1 = [stft(torch.from_numpy(ch1).float()) for ch1 in ch1_resps]
ampls1 = [torch.sqrt(torch.square(spec1.real) + torch.square(spec1.imag)) for spec1 in specs1]
phases1 = [torch.atan(torch.div(spec1.imag, spec1.real)) for spec1 in specs1]
minPhase = torch.min(phases1[0])
maxPhase = torch.max(phases1[0])
minMag = torch.min(ampls1[0])
maxMag = torch.max(ampls1[0])
for i in range(50):
    minPhase = min(minPhase, torch.min(phases1[i]))
    maxPhase = max(maxPhase, torch.max(phases1[i]))
    minMag = min(minMag, torch.min(ampls1[i]))
    maxMag = max(maxMag, torch.max(ampls1[i]))



# Plot mesh grid in time domain
ax = plt.figure(figsize=(20,20)).add_subplot(projection='3d')
for i in range(50):
    ax.plot([i for _ in t], t, zs=ch1_resps[i], zdir='z', color='b', linewidth=0.1)
ax.set_title("Subject " + str(SUBJECT) + ", Channel:1")
ax.set_xlabel('Elevation')
ax.set_ylabel('Time (msec)')
ax.set_zlabel('Relative Amplitude (%)')
ax.set_xlim(0, 50)
ax.set_ylim(0, t[-1])
ax.set_zlim(minCH, maxCH)
ax.view_init(elev=29., azim=-11, roll=0)
plt.savefig("mesh_time_impulse_reponse.png", bbox_inches='tight')
plt.close()

ax = plt.figure(figsize=(20,20)).add_subplot(projection='3d')
X = np.array([i for i in range(50)])
Y = np.array(freqs)
X, Y = np.meshgrid(X, Y)
Z = np.transpose(np.squeeze(np.array([ampl.numpy() for ampl in ampls1])))
surf = ax.plot_surface(X, Y, Z,
        cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_title("Subject " + str(SUBJECT) + ", Channel:1")
ax.set_xlabel('Elevation')
ax.set_ylabel('Frequency (Hz)')
ax.set_zlabel('Amplitude')
ax.set_xlim(0, 50)
ax.set_ylim(0, freqs[-1])
ax.set_zlim(minMag, maxMag)
ax.view_init(elev=29., azim=-11, roll=0)
plt.savefig("mesh_ampl_impulse_reponse.png", bbox_inches='tight')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.set_title("Subject " + str(SUBJECT) + ", Channel:1")
for i in range(50):
    theta =  phases1[i].numpy() * 180.0 / np.pi
    r = ampls1[i].numpy()
    c = ax.scatter(theta, r, c='b', s=0.5)
plt.savefig("mesh_phase_impulse_reponse.png", bbox_inches='tight')
plt.close()

