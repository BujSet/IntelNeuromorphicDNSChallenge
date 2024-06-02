import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from collect_heatmap import getScores 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

(maxScore, minScore, channel0, dists, ch0Scores) = getScores('/research/selagamsetty/private/results/')

trendline = dict()
for n in range(50):
    for s in range(50):
        diff = (s - n) 
        score = channel0[s,n]
        if score < minScore or score > maxScore:
            continue
        if diff in trendline.keys():
            trendline[diff].append(score)
        else:
            trendline[diff] = [score]
updated = dict()
medians = dict()
for k, v in trendline.items():
    dist = k * 5.625
    val = 1.0 * sum(v) / len(v)
    updated[dist] = val
    v.sort()
    if (len(v) % 2 == 1):
        medians[dist] = v[(len(v) - 1)// 2] 
    else:
        if (len(v) == 2):
            medians[dist] = (0.5*v[0])  + (0.5*v[1])
        else:
            medians[dist] = (0.5*v[len(v)// 2])  + (0.5*v[(len(v)// 2) + 1])
updated = sorted(updated.items())
medians = sorted(medians.items())
xs = [k for k,v in updated]
ys = [v for k,v in updated]
mxs = [k for k,v in medians]
mys = [v for k,v in medians]

fig, axs = plt.subplots(figsize=(20,20))
axs.scatter(dists, ch0Scores, color='blue', s=1)
axs.plot(xs, ys, color='red', label='Mean')
axs.plot(mxs, mys, color='green', label='Median')
axs.legend()
axs.set_xlabel("Elevation Difference (Speech - Noise)")
axs.set_xticks(range(-315, 316, 45))
axs.set_xticklabels([str(deg) + "°" for deg in range(-315, 316, 45)])
axs.set_ylabel("Denoised SI-SNR (dB)")
axs.set_xlim(-315, 315)
axs.set_title("Subject 12, Channel 0")
plt.savefig("ss_sensitivity.png", bbox_inches="tight")
plt.close()
row_means = np.zeros(channel0.shape[0])
row_stddev = np.zeros(channel0.shape[0])
col_means = np.zeros(channel0.shape[1])
col_stddev = np.zeros(channel0.shape[1])
for i in range(50):
    mean = 0.0
    num_valid = 0.0
    vals = []
    for j in range(50):
        val = channel0[i,j]
        if (val >= minScore and val <= maxScore):
            num_valid += 1.0
            mean += val
            vals.append(val)
    row_stddev[i] = np.std(vals)
    row_means[i] = mean / num_valid
for j in range(50):
    mean = 0.0
    num_valid = 0.0
    vals = []
    for i in range(50):
        val = channel0[i,j]
        if (val >= minScore and val <= maxScore):
            num_valid += 1.0
            mean += val
            vals.append(val)
    col_stddev[j] = np.std(vals)
    col_means[j] = mean / max(num_valid, 1.0)
#print(row_means)
#print(row_stddev)
np.seterr(divide='ignore', invalid='ignore', over='ignore')
#channel0[1, 5] = 0
masked_array = np.ma.masked_where(channel0 < minScore, channel0)
masked_array = np.ma.masked_where(channel0 > maxScore, masked_array)
cmap = matplotlib.cm.hot
cmap.set_bad('blue')
col_colors = np.zeros(50)
for i in range(50):
    val = col_means[i]
    scaled = cmap.N * (val - minScore) / (maxScore - minScore)
    col_colors[i] = int(round(scaled))
col_colors = np.array(col_colors, dtype=np.int32)
colors = cmap(col_colors) 
fig, axs = plt.subplots(2, 2, figsize=(40,15))

fig = plt.figure(figsize=(20,20))
gs = GridSpec(nrows=2, ncols=2, figure=fig, wspace=0.1, hspace=0.1, height_ratios=[0.1, 1], width_ratios=[1, 0.1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
axs3 = fig.add_subplot(gs[1, 0])
axs4 = fig.add_subplot(gs[1, 1])
fig.delaxes(ax2)
ax1.set_xlim(0, 49)
ax1.set_xticks(range(0, 49, 8))
ax1.set_xticklabels(["" for deg in range(-45, 89, 45)] + [""] +  ["" for deg in range(45, -90, -45)])
ax1.set_ylim(minScore, maxScore)
ax1.set_ylabel("SI-SNR (dB)")
ax1.plot(range(0, 50), col_means, color='blue')
ax1.plot(range(0, 50), [col_means[i] + col_stddev[i] for i in range(col_means.size)], color='orange')
ax1.plot(range(0, 50), [col_means[i] - col_stddev[i] for i in range(col_means.size)], color='orange')
im = axs3.imshow(masked_array, cmap=cmap, vmin=minScore, vmax=maxScore)
axs3.set_xlim(0, 49)
axs3.set_xticks(range(0, 49, 8))
axs3.set_xticklabels([str(deg)  + "°\nfront" for deg in range(-45, 89, 45)] + ["90°\nup"] +  [str(deg) + "°\nback" for deg in range(45, -90, -45)])
axs3.set_yticks(range(0, 49, 8))
axs3.set_yticklabels(["front " + str(deg)  + "°" for deg in range(-45, 89, 45)] + ["up 90°"] +  ["back " + str(deg) + "°" for deg in range(45, -90, -45)])
axs3.set_ylim(0, 49)
axs3.set_ylabel("Speech Elevation Angle")
axs3.set_xlabel("Noise Elevation Angle")
axs4.set_xlim(minScore, maxScore)
axs4.set_yticks(range(0, 49, 8))
axs4.set_yticklabels(["" for deg in range(-45, 89, 45)] + [""] +  ["" for deg in range(45, -90, -45)])
axs4.set_ylim(0, 49)
axs4.set_xlabel("SI-SNR (dB)")
row_colors = np.zeros(50)
for i in range(50):
    val = row_means[i]
    scaled = cmap.N * (val - minScore) / (maxScore - minScore)
    row_colors[i] = int(round(scaled))
row_colors = np.array(row_colors, dtype=np.int32)
colors = cmap(row_colors) 
axs4.plot(row_means,range(0, 50),  color='blue')
axs4.plot([row_means[i] + row_stddev[i] for i in range(row_means.size)],range(0, 50),  color='orange')
axs4.plot([row_means[i] - row_stddev[i] for i in range(row_means.size)],range(0, 50),  color='orange')
divider = make_axes_locatable(axs4)
colorbar_axes = divider.append_axes("right", 
                                    size="10%", 
                                    pad=0.1) 
fig.colorbar(im, cax=colorbar_axes, label="SI-SNR (dB)", location='right')
plt.savefig("sensitivity.png", bbox_inches="tight")
