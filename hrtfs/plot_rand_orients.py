from os import listdir
from os.path import isfile, join
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from cipic_db import CipicDatabase 
from scipy.interpolate import Rbf
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

mypath = '/research/selagamsetty/private/acc_16573485.txt'
raw = dict()
with open(mypath, 'r') as f:
    lines = f.readlines()
    # ignore header and footer information
    data = lines[10:37510]
    xs = np.zeros(len(data), dtype=int)
    ys = np.zeros(len(data), dtype=int)
    zs = np.zeros(len(data), dtype=float)
    for i in range(len(data)):
        line = data[i]
        tokens = line.split(",")
        speech = int(tokens[0])
        noise = int(tokens[1])
        snr = float(tokens[2])
        raw.setdefault((speech, noise), [])
        raw[(speech, noise)].append(snr)

# 'raw' may contain duplicates, if which case we average the vaules
for k,v in raw.items():
    if len(v) > 1:
        raw[k] = sum(v)/len(v)
    else:
        assert(len(v) == 1)
        raw[k] = v[0]

diffs = dict()
for k, v in raw.items():
    speech, noise = k
    speech_elev = speech % 50
    noise_elev = noise % 50 
    elev_diff = speech_elev - noise_elev
    speech_azimuth = speech // 50 
    noise_azimuth = noise // 50
    azimuth_diff = speech_azimuth - noise_azimuth
    diffs.setdefault((elev_diff, azimuth_diff), [])
    diffs[(elev_diff, azimuth_diff)].append(v)

# Again, 'diffs' may contain duplicates, in which case we average the vaules
for k,v in diffs.items():
    if len(v) > 1:
        diffs[k] = sum(v)/len(v)
    else:
        assert(len(v) == 1)
        diffs[k] = v[0]

xs = np.zeros(len(diffs), dtype=int)
ys = np.zeros(len(diffs), dtype=int)
zs = np.zeros(len(diffs), dtype=float)
idx = 0
for k,v in diffs.items():
    xs[idx], ys[idx] = k
    zs[idx] = v
    idx += 1

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('Elevation Difference (#)')
ax.set_ylabel('Azimuthal Differnce (#)')
ax.set_zlabel('Validation SNR (db)')
surf = ax.plot_trisurf(xs, ys, zs, cmap='hot', linewidth=0)
fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

plt.savefig("ventriloquy.png", bbox_inches='tight')
plt.close()
