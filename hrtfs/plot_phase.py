from cipic_db import CipicDatabase
import matplotlib.pyplot as plt
import torch
import torchaudio
import numpy as np

SUBJECT = 165

FIRs1 = [None for _ in range(50)]
FIRs0 = [None for _ in range(50)]
for i in range(600, 650, 1):
    FIRs1[i - 600] = torch.from_numpy(CipicDatabase.subjects[SUBJECT].getHRIRFromIndex(i, 1)).float()
    FIRs0[i - 600] = torch.from_numpy(CipicDatabase.subjects[SUBJECT].getHRIRFromIndex(i, 0)).float()

minAmpl = min(torch.min(torch.cat(FIRs0)), torch.min(torch.cat(FIRs1)))
maxAmpl = min(torch.max(torch.cat(FIRs0)), torch.max(torch.cat(FIRs1)))
channel1Colors = [((255.0/255.0), ((128 +(2*i))/255.0), ((4*i)/255.0)) for i in range(50)]
channel0Colors = [(((4*i)/255.0), ((4*i)/255.0), (255.0/255.0)) for i in range(50)]

NFFT = 256
SAMPLE_RATE = 44100.0
num_samples = 200
t = np.linspace(0, 1000.0*num_samples/SAMPLE_RATE, num_samples)

stft = torchaudio.transforms.Spectrogram(
    n_fft=NFFT,
    win_length=NFFT,
    hop_length=NFFT,
    power=None, # returns complex tensor?
)
freqs = [round(i * SAMPLE_RATE / (1000.0 * NFFT),2) for i in range(NFFT//2 + 1)]
FFT1s = [stft(i) for i in FIRs1] 
FFT0s = [stft(i) for i in FIRs0] 
mags1 = [torch.sqrt(torch.square(FFT1s[i].real) + torch.square(FFT1s[i].imag)) for i in range(len(FFT1s))]
mags0 = [torch.sqrt(torch.square(FFT0s[i].real) + torch.square(FFT0s[i].imag)) for i in range(len(FFT0s))]
minMag = min(torch.min(torch.cat(mags0)), torch.min(torch.cat(mags1)))
maxMag = min(torch.max(torch.cat(mags0)), torch.max(torch.cat(mags1)))
phase1 = [torch.atan(torch.div(FFT1s[i].imag, FFT1s[i].real)) for i in range(len(FFT1s))]
phase0 = [torch.atan(torch.div(FFT0s[i].imag, FFT0s[i].real)) for i in range(len(FFT0s))]
minPhase = min(torch.min(torch.cat(phase0)), torch.min(torch.cat(phase1)))
maxPhase = min(torch.max(torch.cat(phase0)), torch.max(torch.cat(phase1)))
elevations = [ "-45°     (front)", 
               "-39.375° (front)", 
               "-33.75°  (front)", 
               "-28.125° (front)", 
               "-22.5°   (front)", 
               "-16.875° (front)", 
               "-11.25°  (front)", 
               "-5.625°  (front)", 
               "0°       (front)", 
               "5.625°   (front)", 
               "11.25°   (front)", 
               "16.875°  (front)", 
               "22.5°    (front)", 
               "28.125°  (front)", 
               "33.75°   (front)", 
               "39.375°  (front)", 
               "45°      (front)", 
               "50.625°  (front)", 
               "56.25°   (front)", 
               "61.875°  (front)", 
               "67.5°    (front)", 
               "73.125°  (front)", 
               "78.75°   (front)", 
               "84.375°  (front)", 
               "90°      (up)", 
               "84.375°  (back)", 
               "78.75°   (back)", 
               "73.125°  (back)", 
               "67.5°    (back)", 
               "61.875°  (back)", 
               "56.25°   (back)", 
               "50.625°  (back)", 
               "45°      (back)", 
               "39.375°  (back)", 
               "33.75°   (back)", 
               "28.125°  (back)", 
               "22.5°    (back)", 
               "16.875°  (back)", 
               "11.25°   (back)", 
               "5.625°   (back)", 
               "0°       (back)", 
               "-5.625°  (back)", 
               "-11.25°  (back)", 
               "-16.875° (back)", 
               "-22.5°   (back)", 
               "-28.125° (back)", 
               "-33.75°  (back)", 
               "-39.375° (back)", 
               "-45°     (back)", 
               "-50.625° (back)"]

fig, axs = plt.subplots(2,3, figsize=(20,10))
plt.subplots_adjust(top=0.85) 
plt.suptitle("Impulse Response for Subject "+str(SUBJECT)+" on Midsagittal Plane", 
        fontsize='large', 
        y=0.91,
        fontweight='bold')
for i in range(len(FIRs1)):
    if (i % 8 == 0):
        axs[0,0].plot(t, FIRs1[i], color=channel1Colors[i], label=elevations[i])
        axs[1,0].plot(t, FIRs0[i], color=channel0Colors[i], label=elevations[i])
    else:
        axs[0,0].plot(t, FIRs1[i], color=channel1Colors[i])
        axs[1,0].plot(t, FIRs0[i], color=channel0Colors[i])
    axs[0,1].plot(freqs, mags1[i], color=channel1Colors[i])
    axs[1,1].plot(freqs, mags0[i], color=channel0Colors[i])
    axs[0,2].plot(freqs, phase1[i], color=channel1Colors[i])
    axs[1,2].plot(freqs, phase0[i], color=channel0Colors[i])
axs[0,0].set_xlabel("Time (msec)")
axs[0,0].set_ylabel("Amplitude")
axs[0,0].set_ylim(minAmpl, maxAmpl)
axs[0,0].set_title("Time Domain Impluse Response")
axs[0,0].legend()
axs[0,1].set_title("Frequency Domain Impluse Response (Magnitude)")
axs[0,1].set_xlabel("Frequency (kHz)")
axs[0,1].set_ylabel("Magnitude")
axs[0,1].set_ylim(minMag, maxMag)
axs[0,2].set_xlabel("Frequency (kHz)")
axs[0,2].set_ylabel("Phase")
axs[0,2].set_title("Frequency Domain Impluse Response (Phase)")
axs[0,2].ticklabel_format(axis='y',style='sci', useOffset=True, scilimits=(0,0))
axs[0,2].set_ylim(minPhase, maxPhase)
axs[1,0].set_xlabel("Time (msec)")
axs[1,0].set_ylabel("Amplitude")
axs[1,0].set_ylim(minAmpl, maxAmpl)
axs[1,0].legend()
axs[1,0].set_title("Time Domain Impluse Response")
axs[1,1].set_title("Frequency Domain Impluse Response (Magnitude)")
axs[1,1].set_xlabel("Frequency (kHz)")
axs[1,1].set_ylabel("Magnitude")
axs[1,1].set_ylim(minMag, maxMag)
axs[1,2].set_title("Frequency Domain Impluse Response (Phase)")
axs[1,2].set_xlabel("Frequency (kHz)")
axs[1,2].set_ylabel("Phase")
axs[1,2].ticklabel_format(axis='y',style='sci', useOffset=True, scilimits=(0,0))
axs[1,2].set_ylim(minPhase, maxPhase)
plt.savefig("phase_ir.png", bbox_inches='tight')
plt.close()
