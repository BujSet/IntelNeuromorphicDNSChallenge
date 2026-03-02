import torch
import librosa
import matplotlib.pyplot as plt
import sys
import numpy as np
import math

freqsT = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=512))
freqsN = librosa.fft_frequencies(sr=16000, n_fft=512)
xticks = [i for i in range(0, freqsN.shape[0], 16)]
c = 343.0 # Speed of sound in air m/s
rho = 1.293 # Air density kg/m^3

def calcUnscaledConductance(w, x, theta):
    r = math.tan(theta) * x
    scalar = rho * c / (np.pi * r * r)
    numer = torch.mul(w, scalar)
    denom = torch.add(w, c/x)
    impedance = torch.div(numer, denom) 
    return impedance

def calcConductance(w, x, theta):
    r = math.tan(theta) * x
    scalar = rho * c / (np.pi * r * r)
    numer = torch.mul(w, scalar)
    denom = torch.add(w, c/x)
    impedance = torch.div(numer, denom)
    mini = torch.min(impedance)
    maxi = torch.max(impedance)
    scaled = torch.div(torch.sub(impedance, mini), maxi-mini)   
    return scaled


def calcUnscaledConeConductance(w, x, theta, length, steps):
    impedance = torch.ones(w.shape[0])
    for i in range(steps + 1):
        xd = x + (i * length / (steps + 1.0))
        r = math.tan(theta) * xd
        scalar = rho * c / (np.pi * r * r)
        numer = torch.mul(w, scalar)
        denom = torch.add(w, c/xd)
        impedanced = torch.div(numer, denom)
        impedance = torch.mul(impedance, impedanced) 
    return impedance

def calcConeConductance(w, x, theta, length, steps):
    impedance = torch.ones(w.shape[0])
    for i in range(steps + 1):
        xd = x + (i * length / (steps + 1.0))
        r = math.tan(theta) * xd
        scalar = rho * c / (np.pi * r * r)
        numer = torch.mul(w, scalar)
        denom = torch.add(w, c/xd)
        impedanced = torch.div(numer, denom)
        impedance = torch.mul(impedance, impedanced)
    mini = torch.min(impedance)
    maxi = torch.max(impedance)
    scaled = torch.div(torch.sub(impedance, mini), maxi-mini)   
    return scaled

fig, axs = plt.subplots(2, 3, figsize=(30,20))
fig.tight_layout()
# axs[0,0].set_xlabel("Frequency (kHz)")
axs[0,0].set_ylabel("Impedance (Zw)")
axs[0,0].plot(freqsN, calcUnscaledConductance(freqsT, 0.01, np.pi/4.0), label="x=1cm")
axs[0,0].plot(freqsN, calcUnscaledConductance(freqsT, 0.05, np.pi/4.0), label="x=5cm")
axs[0,0].plot(freqsN, calcUnscaledConductance(freqsT, 0.10, np.pi/4.0), label="x=10cm")
axs[0,0].plot(freqsN, calcUnscaledConductance(freqsT, 0.15, np.pi/4.0), label="x=15cm")
axs[0,0].plot(freqsN, calcUnscaledConductance(freqsT, 0.20, np.pi/4.0), label="x=20cm")
axs[0,0].plot(freqsN, calcUnscaledConductance(freqsT, 0.25, np.pi/4.0), label="x=25cm")
axs[0,0].plot(freqsN, calcUnscaledConductance(freqsT, 0.50, np.pi/4.0), label="x=50cm")
axs[0,0].set_title("θ=π/4")
axs[0,0].legend()

# axs[0,1].set_xlabel("Frequency (kHz)")
axs[0,1].set_ylabel("Normalized Impedance (Zw)")
axs[0,1].plot(freqsN, calcUnscaledConductance(freqsT, 0.05, np.pi/10.0), label="θ=π/10")
axs[0,1].plot(freqsN, calcUnscaledConductance(freqsT, 0.05, np.pi/5.0), label="θ=π/5")
axs[0,1].plot(freqsN, calcUnscaledConductance(freqsT, 0.05, 3*np.pi/10.0), label="θ=3π/10")
axs[0,1].plot(freqsN, calcUnscaledConductance(freqsT, 0.05, np.pi/5.0), label="θ=2π/5")
axs[0,1].plot(freqsN, calcUnscaledConductance(freqsT, 0.05, np.pi/2.0), label="θ=π/2")
axs[0,1].set_title("x=0.05m")
axs[0,1].legend()

# axs[0,2].set_xlabel("Frequency (kHz)")
axs[0,2].set_ylabel("Normalized Impedance (Zw)")
axs[0,2].plot(freqsN, calcUnscaledConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 0), label="steps=0")
axs[0,2].plot(freqsN, calcUnscaledConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 1), label="steps=1")
axs[0,2].plot(freqsN, calcUnscaledConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 5), label="steps=5")
axs[0,2].plot(freqsN, calcUnscaledConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 10), label="steps=10")
# axs[0,2].plot(freqsN, calcUnscaledConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 50), label="steps=50")
# axs[0,2].plot(freqsN, calcUnscaledConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 100), label="steps=100")
axs[0,2].set_title("x=0.05m, depth=0.10, θ=π/4")
axs[0,2].legend()

axs[1,0].set_xlabel("Frequency (kHz)")
axs[1,0].set_ylabel("Normalized Impedance (Zw)")
axs[1,0].plot(freqsN, calcConductance(freqsT, 0.01, np.pi/4.0), label="x=1cm")
axs[1,0].plot(freqsN, calcConductance(freqsT, 0.05, np.pi/4.0), label="x=5cm")
axs[1,0].plot(freqsN, calcConductance(freqsT, 0.10, np.pi/4.0), label="x=10cm")
axs[1,0].plot(freqsN, calcConductance(freqsT, 0.15, np.pi/4.0), label="x=15cm")
axs[1,0].plot(freqsN, calcConductance(freqsT, 0.20, np.pi/4.0), label="x=20cm")
axs[1,0].plot(freqsN, calcConductance(freqsT, 0.25, np.pi/4.0), label="x=25cm")
axs[1,0].plot(freqsN, calcConductance(freqsT, 0.50, np.pi/4.0), label="x=50cm")
axs[1,0].set_title("θ=π/4")
axs[1,0].legend()

axs[1,1].set_xlabel("Frequency (kHz)")
axs[1,1].set_ylabel("Normalized Impedance (Zw)")
axs[1,1].plot(freqsN, calcConductance(freqsT, 0.05, np.pi/10.0), label="θ=π/10")
axs[1,1].plot(freqsN, calcConductance(freqsT, 0.05, np.pi/5.0), label="θ=π/5")
axs[1,1].plot(freqsN, calcConductance(freqsT, 0.05, 3*np.pi/10.0), label="θ=3π/10")
axs[1,1].plot(freqsN, calcConductance(freqsT, 0.05, np.pi/5.0), label="θ=2π/5")
axs[1,1].plot(freqsN, calcConductance(freqsT, 0.05, np.pi/2.0), label="θ=π/2")
axs[1,1].set_title("x=0.05m")
axs[1,1].legend()

axs[1,2].set_xlabel("Frequency (kHz)")
axs[1,2].set_ylabel("Normalized Impedance (Zw)")
axs[1,2].plot(freqsN, calcConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 0), label="steps=0")
axs[1,2].plot(freqsN, calcConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 1), label="steps=1")
axs[1,2].plot(freqsN, calcConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 5), label="steps=5")
axs[1,2].plot(freqsN, calcConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 10), label="steps=10")
axs[1,2].plot(freqsN, calcConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 50), label="steps=50")
axs[1,2].plot(freqsN, calcConeConductance(freqsT, 0.05, np.pi/4.0, 0.10, 100), label="steps=100")
axs[1,2].set_title("x=0.05m, depth=0.10, θ=π/4")
axs[1,2].legend()

plt.savefig("cone_impedance.png", bbox_inches='tight')
plt.close()
