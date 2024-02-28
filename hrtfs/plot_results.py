from os import listdir
from os.path import isfile, join
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from cipic_db import CipicDatabase 

mypath = '/home/selagamsetty/private/results/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print("Reading " + str(len(onlyfiles)) + " files")
channel0 = np.empty([50, 50], dtype=float)
channel1 = np.empty([50, 50], dtype=float)
minScore = 1000.0
maxScore = 0.0
posits = CipicDatabase.subjects[12].getCartesianPositions()
dists = np.zeros((len(onlyfiles),), dtype=float)
ch0Scores = np.zeros((len(onlyfiles),), dtype=float)
ch1Scores = np.zeros((len(onlyfiles),), dtype=float)

def cartDist(origin, dest):
    assert(origin.shape == dest.shape)
    return np.sqrt(np.dot(origin, dest))

for i in range(len(onlyfiles)):
    filename = onlyfiles[i]
#    print(filename)
    path = join(mypath, filename)
    name_tokens = filename.replace('.txt', '').split('_')

    # Make sure but speech and noise are from same channel
    assert(int(name_tokens[-1]) == int(name_tokens[-3]))
    channel = int(name_tokens[-1])

    speech = int(name_tokens[-4])
    noise = int(name_tokens[-2])
    with open(path, 'r') as F:
        lines = F.readlines()
        tokens = lines[-1].split()
#        print(tokens)
        score = float(tokens[-2])
        dists[i] = ((speech - noise) * 5.625)

        if (score > maxScore):
            maxScore = score
        if (score < minScore):
            minScore = score

        if (channel == 0):
            channel0[noise, speech] = score
            ch0Scores[i] = score
        else:
            assert(channel == 1)
            channel1[noise, speech] = score
            ch1Scores[i] = score
 
trendline = dict()
for n in range(50):
    for s in range(50):
        diff = (s - n) 
        if channel0[n, s] < minScore or channel0[n, s] > maxScore:
            continue
        if diff in trendline.keys():
            trendline[diff].append(channel0[n,s])
        else:
            trendline[diff] = [channel0[n,s]]
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

fig, axs = plt.subplots(1, 2, figsize=(40,15))
axs0, axs1 = axs
axs0.scatter(dists, ch0Scores, color='blue', s=1)
axs0.plot(xs, ys, color='red', label='Mean')
axs0.plot(mxs, mys, color='green', label='Median')
axs0.legend()
axs0.set_xlabel("Elevation Difference (Speech - Noise)")
axs0.set_xticks(range(-315, 316, 45))
axs0.set_xticklabels([str(deg) + "°" for deg in range(-315, 316, 45)])
axs0.set_ylabel("Denoised SI-SNR (dB)")
axs0.set_xlim(-315, 315)
np.seterr(divide='ignore', invalid='ignore', over='ignore')
masked_array = np.ma.masked_where(channel0 < minScore, channel0)
masked_array = np.ma.masked_where(channel0 > maxScore, masked_array)
cmap = matplotlib.cm.hot
cmap.set_bad('blue')
im = axs1.imshow(masked_array, cmap=cmap, vmin=minScore, vmax=maxScore)
axs1.set_xlim(0, 49)
axs1.set_xticks(range(0, 49, 8))
axs1.set_xticklabels([str(deg)  + "°\nfront" for deg in range(-45, 89, 45)] + ["90°\nup"] +  [str(deg) + "°\nback" for deg in range(45, -90, -45)])
axs1.set_yticks(range(0, 49, 8))
axs1.set_yticklabels(["front " + str(deg)  + "°" for deg in range(-45, 89, 45)] + ["up 90°"] +  ["back " + str(deg) + "°" for deg in range(45, -90, -45)])
axs1.set_ylim(0, 49)
axs1.set_ylabel("Speech Orientation")
axs1.set_xlabel("Noise Orientation")
fig.colorbar(im, ax=axs1, label="SI-SNR (dB)")
plt.savefig("sensitivity.png", bbox_inches="tight")
