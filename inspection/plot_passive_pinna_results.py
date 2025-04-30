import os, sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
from hrtfs.cipic_db import CipicDatabase 
from matplotlib import cm

path = os.getcwd()
filePath = os.path.join(path, "collated_results.csv")

data = np.full((1250, 1250), -1.0)
with open(filePath, "r") as f:
    lines = f.readlines()
    headerLine = lines[0]
    headerTokens = [tok.strip() for tok in headerLine.split(",")]
    assert("Subject" in headerTokens)
    assert("Channel" in headerTokens)
    assert("Speech Orient" in headerTokens)
    assert("Noise Orient" in headerTokens)
    assert("Final Validation Score SI-SNR (dB)" in headerTokens)
    subIdx = headerTokens.index("Subject")
    chanIdx = headerTokens.index("Channel")
    speechOIdx = headerTokens.index("Speech Orient")
    noiseOIdx = headerTokens.index("Noise Orient")
    scoreIdx = headerTokens.index("Final Validation Score SI-SNR (dB)")
    lines = lines[1:]
    for line in lines:
        dataTokens = [tok.strip() for tok in line.split(",")]
        sub = int(dataTokens[subIdx])
        chan = int(dataTokens[chanIdx])
        speechO = int(dataTokens[speechOIdx])
        noiseO = int(dataTokens[noiseOIdx])
        score = float(dataTokens[scoreIdx])

        assert(0 <= speechO < 1250)
        assert(0 <= noiseO  < 1250)
        if (sub == 3 and chan == 0):
            data[speechO, noiseO] = score
            if (score > 100.0):
                print(line)
                assert(False)

def fill_col_average(dest, data):
    for j in range(1250):
        col_sum = 0.0
        col_count = 0.0
        for i in range(1250):
            if data[i, j] > 0.0:
                col_sum += data[i, j]
                col_count += 1.0
        if col_count > 0.0:
            dest[j] = col_sum / col_count
        else:
            dest[j] = -1.0
    return dest

def fill_row_average(dest, data):
    for i in range(1250):
        row_sum = 0.0
        row_count = 0.0
        for j in range(1250):
            if data[i, j] > 0.0:
                row_sum += data[i, j]
                row_count += 1.0
        if row_count > 0.0:
            dest[i] = row_sum / row_count
        else:
            dest[i] = -1.0
    return dest

def make_3D_plot(ax, data3D, azimuth, prefix):
    scatterAX = ax.scatter(x, y, z, c=data3D, cmap=cm.hot)
    ax.set_xticks([0.5, -0.5], labels=["Dorsal", "Ventral"])
    ax.set_yticks([0.5, -0.5], labels=["Right", "Left"])
    ax.set_zticks([0.5, -0.5], labels=["Below", "Above"])
    tmp_planes = ax.zaxis._PLANES
    ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
    ax.view_init(elev=ELEV, azim=azimuth)
    ax.set_title(prefix+" θ="+str(azimuth)+"°", fontweight='bold')
    return scatterAX

Subject3 = CipicDatabase.subjects[3]
cart_pos = Subject3.getCartesianPositions()
cart_pos[:,0] = -1.0 * cart_pos[:, 0]
print("Index0 = " + str(cart_pos[0]))
print("Index5 = " + str(cart_pos[5]))
print("Index49 = " + str(cart_pos[49]))
print("Index600 = " + str(cart_pos[600]))
x = cart_pos[:,0]
y = cart_pos[:,1]
z = cart_pos[:,2]

col_average = np.full((1250), 0.0)
row_average = np.full((1250), 0.0)
fill_col_average(col_average, data)
fill_row_average(row_average, data)

ELEV=25

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(25,12), subplot_kw={'projection': '3d'})
scatter1 = make_3D_plot(axes[0,0], row_average, 45, "a)")
make_3D_plot(axes[0,1], row_average, 135, "b)")
make_3D_plot(axes[0,2], row_average, 225, "c)")
make_3D_plot(axes[0,3], row_average, 315, "d)")
fig.text(0.5, 0.91, 'Speech Audio-sphere', ha='center', fontsize=14, fontweight='bold')

fig.text(0.5, 0.50, 'Noise Audio-sphere', ha='center', fontsize=14, fontweight='bold')
make_3D_plot(axes[1,0], col_average, 45, "e)")
make_3D_plot(axes[1,1], col_average, 135, "f)")
make_3D_plot(axes[1,2], col_average, 225, "g)")
make_3D_plot(axes[1,3], col_average, 315, "h)")

#fig.colorbar(scatter9)
# Create an inset axes for the colorbar
cbax = fig.add_axes([0.11, 0.1, 0.78, 0.02])  # [left, bottom, width, height]
fig.text(0.5, 0.07, 'Final SI-NSR (dB)', ha='center', fontsize=14, fontweight='bold')

# Add the colorbar to the inset axes
cbar = fig.colorbar(scatter1, cax=cbax, orientation='horizontal')
plt.savefig("passive_pinna_sub3_chan0_3D.pdf", bbox_inches="tight", format='pdf')
plt.close()

plt.figure(figsize=(25, 25))
plt.xlabel('Noise Orient')
plt.xticks(ticks=[i for i in range(0, 1250, 50)] + [1249], labels=[str(i) for i in range(0,1250,50)] + ["1249"], rotation=45)
plt.ylabel('Speech Orient')
plt.yticks(ticks=[i for i in range(0, 1250, 50)] + [1249], labels=[str(i) for i in range(0,1250,50)] + ["1249"])
plt.ylim(bottom=-0.5)
plt.xlim(left=-0.5)
plt.imshow(data, cmap='hot', aspect="equal", interpolation='none', origin='lower')
plt.colorbar(shrink=0.80)
plt.savefig("passive_pinna_sub3_chan0.png", bbox_inches="tight")
plt.close()

dataSmall = np.full((50, 50), -1.0)
for i in range(0, 1250, 25):
    for j in range(0, 1250, 25):
        if data[i,j] > 0.0:
            dataSmall[i//25,j//25] = data[i,j]
 
plt.figure(figsize=(10, 10))
plt.xlabel('Noise Orient')
plt.xticks(ticks=[125*i//25 for i in range(10)], labels=[str(125*i) for i in range(10)], rotation=45)
plt.ylabel('Speech Orient')
plt.yticks(ticks=[125*i//25 for i in range(10)], labels=[str(125*i) for i in range(10)])
plt.ylim(bottom=-0.5)
plt.ylim(top=49.5)
plt.xlim(left=-0.5)
plt.xlim(right=49.5)
plt.imshow(dataSmall, cmap='hot', aspect="equal", interpolation='nearest', origin='lower')
plt.colorbar(shrink=0.80)
plt.savefig("passive_pinna_sub3_chan0_small.png", bbox_inches="tight")
plt.close()
