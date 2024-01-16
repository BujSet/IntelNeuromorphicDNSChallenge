from cipic_db import CipicDatabase
import matplotlib.pyplot as plt
import torch, sys
import torchaudio
import numpy as np
import math

ch1  = torch.from_numpy(CipicDatabase.subjects[12].getHRIRFromIndex(600, 1)).float()
ch0 = torch.from_numpy(CipicDatabase.subjects[12].getHRIRFromIndex(600, 0)).float()
minCH = min(torch.min(ch1), torch.min(ch0))
maxCH = min(torch.max(ch1), torch.max(ch0))

NFFT = 256
SAMPLE_RATE = 44100.0
num_samples = len(ch1)
t = np.linspace(0, 1000.0*num_samples/SAMPLE_RATE, num_samples)

stft = torchaudio.transforms.Spectrogram(
    n_fft=NFFT,
    win_length=NFFT,
    hop_length=NFFT,
    power=None, # returns complex tensor?
)
freqs = [round(i * SAMPLE_RATE / (1000.0 * NFFT),2) for i in range(NFFT//2 + 1)]

spec1 = stft(ch1)
ampl1 = torch.sqrt(torch.square(spec1.real) + torch.square(spec1.imag))
phase1 = torch.atan(torch.div(spec1.imag, spec1.real))
spec0 = stft(ch0)
ampl0 = torch.sqrt(torch.square(spec0.real) + torch.square(spec0.imag))
phase0 = torch.atan(torch.div(spec0.imag, spec0.real))
minPhase = min(torch.min(phase1), torch.min(phase0))
maxPhase = max(torch.max(phase1), torch.max(phase0))
minMag = min(torch.min(ampl1), torch.min(ampl0))
maxMag = max(torch.max(ampl1), torch.max(ampl0))

fig, axs = plt.subplots(2,3, figsize=(30,20))
plt.suptitle("Impulse Response for Subject 12 on Midsagittal Plane", 
        fontsize='large',
        fontweight='bold',
        y=0.91)
axs[0,0].plot(t, ch1, color="orange", label="Channel 1")
axs[0,0].set_ylim(minCH, maxCH)
axs[0,0].legend()
axs[0,0].set_title("Time Domain Impluse Response")
axs[1,0].plot(t, ch0, color="blue", label="Channel 0")
axs[1,0].set_ylim(minCH, maxCH)
axs[1,0].legend()
axs[0,0].set_xlabel("Time (msec)")
axs[0,0].set_ylabel("Amplitude")
axs[1,0].set_xlabel("Time (msec)")
axs[1,0].set_ylabel("Amplitude")
axs[0,1].set_title("Frequency Domain Impluse Response (Magnitude)")
axs[0,1].plot(freqs, ampl1, color='orange')
axs[0,1].set_xlabel("Frequency (kHz)")
axs[0,1].set_ylabel("Magnitude")
axs[0,1].set_ylim(minMag, maxMag)
axs[0,2].set_title("Frequency Domain Impluse Response (Phase)")
axs[0,2].plot(freqs, phase1, color='orange')
axs[0,2].set_xlabel("Frequency (kHz)")
axs[0,2].set_ylabel("Phase")
axs[0,2].set_ylim(minPhase, maxPhase)

axs[1,1].plot(freqs, ampl0, color='blue')
axs[1,1].set_xlabel("Frequency (kHz)")
axs[1,1].set_ylabel("Magnitude")
axs[1,1].set_ylim(minMag, maxMag)
axs[1,2].plot(freqs, phase0, color='blue')
axs[1,2].set_xlabel("Frequency (kHz)")
axs[1,2].set_ylabel("Phase")
axs[1,2].set_ylim(minPhase, maxPhase)

plt.savefig("impulse_reponse.png", bbox_inches='tight')
plt.close()

