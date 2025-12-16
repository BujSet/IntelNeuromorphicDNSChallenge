import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from collect_heatmap import getScores 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

mask0 = '/research/selagamsetty/private/linear_mask.out'
mask1 = '/research/selagamsetty/private/linear_mask_temporal.out'

def readMask0():
    inRegion = False
    with open(mask0, 'r') as f:
        lines = f.readlines()
        finalStart = 9687 
        final = ''.join(lines[finalStart:]).replace("\n", "")
        final = final.replace(" ", "")
        final = final.replace("[", "")
        final = final.replace("]", "")
        final = final.replace("(", "")
        final = final.replace(")", "")
        final = final.replace("mask:tensor", "")
        final = final.replace(",device='cuda:0'", "")
        final = final.split(",")
        final = [float(f) for f in final]
        fig, axs = plt.subplots(figsize=(20,20))
        axs.plot([i for i in range(len(final))], final, color='red', label='Multiplicative Mask')
        axs.legend()
        axs.set_xlabel("FFT Bin")
        axs.set_ylabel("Mask Value (Multiplicative)")
        axs.set_title("Final SI-SNR ~ 7.9 dB after 250 Epochs")
        plt.savefig("multiplicative_mask.png", bbox_inches="tight")
        plt.close()

def readMask1():
    inRegion = False
    with open(mask1, 'r') as f:
        lines = f.readlines()
        finalStart = 202436 
        final = ''.join(lines[finalStart:]).replace("\n", "")
        final = final.replace(" ", "")
#        final = final.replace("[", "")
#        final = final.replace("]", "")
        final = final.replace("(", "")
        final = final.replace(")", "")
        final = final.replace("mask:tensor", "")
        final = final.replace(",device='cuda:0'", "")
        print(final)
#        final = final.split(",")
#        final = [float(f) for f in final]
#        fig, axs = plt.subplots(figsize=(20,20))
#        axs.plot([i for i in range(len(final))], final, color='red', label='Multiplicative Mask')
#        axs.legend()
#        axs.set_xlabel("FFT Bin")
#        axs.set_ylabel("Mask Value (Multiplicative)")
#        axs.set_title("Final SI-SNR ~ 7.9 dB after 250 Epochs")
#        plt.savefig("multiplicative_mask_temporal.png", bbox_inches="tight")
#        plt.close()

readMask0()
readMask1()
