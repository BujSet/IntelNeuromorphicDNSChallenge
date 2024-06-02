import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from collect_heatmap import getScores 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

(bmaxScore, bminScore, baseline, bdists, bflat) = getScores('/research/selagamsetty/private/baseline_ch0/')
(rmaxScore, rminScore, results, rdists, rflat) = getScores('/research/selagamsetty/private/results/')
channel0 = np.subtract(results, baseline)
minScore = 10000.0
maxScore = 0.0
for i in range(50):
    for j in range(50):
        rval = results[i,j]
        bval = baseline[i,j]
        if rval >= 0.0 and bval >= 0.0:
            div = rval - bval
            if (div < minScore):
                minScore = div
            if (div > maxScore):
                maxScore = div
        else:
            channel0[i,j] = np.nan

np.seterr(divide='ignore', invalid='ignore', over='ignore')
masked_array = np.ma.masked_where(channel0 == np.nan, channel0)
cmap = matplotlib.cm.hot
cmap.set_bad('blue')
fig, axs = plt.subplots(2, 2, figsize=(40,15))

fig = plt.figure(figsize=(20,20))
gs = GridSpec(nrows=1, ncols=1, figure=fig, wspace=0.1, hspace=0.1, height_ratios=[1], width_ratios=[1])
axs3 = fig.add_subplot(gs[0, 0])
im = axs3.imshow(masked_array, cmap=cmap, vmin=minScore, vmax=maxScore)
axs3.set_xlim(0, 49)
axs3.set_xticks(range(0, 49, 8))
axs3.set_xticklabels([str(deg)  + "°\nfront" for deg in range(-45, 89, 45)] + ["90°\nup"] +  [str(deg) + "°\nback" for deg in range(45, -90, -45)])
axs3.set_yticks(range(0, 49, 8))
axs3.set_yticklabels(["front " + str(deg)  + "°" for deg in range(-45, 89, 45)] + ["up 90°"] +  ["back " + str(deg) + "°" for deg in range(45, -90, -45)])
axs3.set_ylim(0, 49)
axs3.set_ylabel("Speech Elevation Angle")
axs3.set_xlabel("Noise Elevation Angle")
divider = make_axes_locatable(axs3)
colorbar_axes = divider.append_axes("right", 
                                    size="10%", 
                                    pad=0.1) 
fig.colorbar(im, cax=colorbar_axes, label="SI-SNR (dB)", location='right')
plt.savefig("sensitivity_relative.png", bbox_inches="tight")
