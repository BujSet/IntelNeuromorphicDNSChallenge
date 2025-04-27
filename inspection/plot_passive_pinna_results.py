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

fig = plt.figure(figsize=(20,12))
ax1 = fig.add_subplot(2,4,1, projection='3d')
scatter1 = ax1.scatter(x, y, z, c=col_average, cmap=cm.hot)

# Set labels
ax1.set_xticks([0.5, -0.5], labels=["Dorsal", "Ventral"])
ax1.set_yticks([0.5, -0.5], labels=["Right", "Left"])
ax1.set_zticks([0.5, -0.5], labels=["Below", "Above"])
tmp_planes = ax1.zaxis._PLANES
ax1.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
ax1.view_init(elev=ELEV, azim=45)
ax1.set_title("θ=45°")

ax2 = fig.add_subplot(2,4,2, projection='3d')
scatter2 = ax2.scatter(x, y, z, c=col_average, cmap=cm.hot)
ax2.set_xticks([0.5, -0.5], labels=["Dorsal", "Ventral"])
ax2.set_yticks([0.5, -0.5], labels=["Right", "Left"])
ax2.set_zticks([0.5, -0.5], labels=["Below", "Above"])
tmp_planes = ax2.zaxis._PLANES
ax2.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
ax2.view_init(elev=ELEV, azim=135)
ax2.set_title("θ=135°")

ax3 = fig.add_subplot(2,4,3, projection='3d')
scatter3 = ax3.scatter(x, y, z, c=col_average, cmap=cm.hot)
ax3.set_xticks([0.5, -0.5], labels=["Dorsal", "Ventral"])
ax3.set_yticks([0.5, -0.5], labels=["Right", "Left"])
ax3.set_zticks([0.5, -0.5], labels=["Below", "Above"])
tmp_planes = ax3.zaxis._PLANES
ax3.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
ax3.view_init(elev=ELEV, azim=225)
ax3.set_title("θ=225°")

ax4 = fig.add_subplot(2,4,4, projection='3d')
ax4.scatter(x, y, z, c=col_average, cmap=cm.hot)
ax4.set_xticks([0.5, -0.5], labels=["Dorsal", "Ventral"])
ax4.set_yticks([0.5, -0.5], labels=["Right", "Left"])
ax4.set_zticks([0.5, -0.5], labels=["Below", "Above"])
tmp_planes = ax4.zaxis._PLANES
ax4.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
ax4.view_init(elev=ELEV, azim=315)
ax4.set_title("θ=315°")

#fig.text(0.5, 0.95, 'Noise AudioSphere', ha='center', fontsize=14)

ax5 = fig.add_subplot(2,4,5, projection='3d')
scatter5 = ax5.scatter(x, y, z, c=row_average, cmap=cm.hot)
ax5.set_xticks([0.5, -0.5], labels=["Dorsal", "Ventral"])
ax5.set_yticks([0.5, -0.5], labels=["Right", "Left"])
ax5.set_zticks([0.5, -0.5], labels=["Below", "Above"])
tmp_planes = ax5.zaxis._PLANES
ax5.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
ax5.view_init(elev=ELEV, azim=45)
ax5.set_title("θ=45°")

ax6 = fig.add_subplot(2,4,6, projection='3d')
scatter6 = ax6.scatter(x, y, z, c=row_average, cmap=cm.hot)
ax6.set_xticks([0.5, -0.5], labels=["Dorsal", "Ventral"])
ax6.set_yticks([0.5, -0.5], labels=["Right", "Left"])
ax6.set_zticks([0.5, -0.5], labels=["Below", "Above"])
tmp_planes = ax6.zaxis._PLANES
ax6.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
ax6.view_init(elev=ELEV, azim=135)
ax6.set_title("θ=135°")

ax7 = fig.add_subplot(2,4,7, projection='3d')
scatter7 = ax7.scatter(x, y, z, c=row_average, cmap=cm.hot)
ax7.set_xticks([0.5, -0.5], labels=["Dorsal", "Ventral"])
ax7.set_yticks([0.5, -0.5], labels=["Right", "Left"])
ax7.set_zticks([0.5, -0.5], labels=["Below", "Above"])
tmp_planes = ax7.zaxis._PLANES
ax7.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
ax7.view_init(elev=ELEV, azim=225)
ax7.set_title("θ=225°")

ax8 = fig.add_subplot(2,4,8, projection='3d')
scatter8 = ax8.scatter(x, y, z, c=row_average, cmap=cm.hot)
ax8.set_xticks([0.5, -0.5], labels=["Dorsal", "Ventral"])
ax8.set_yticks([0.5, -0.5], labels=["Right", "Left"])
ax8.set_zticks([0.5, -0.5], labels=["Below", "Above"])
tmp_planes = ax8.zaxis._PLANES
ax8.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
ax8.view_init(elev=ELEV, azim=315)
ax8.set_title("θ=315°")

#ax5 = fig.add_subplot(3,3,5, projection='3d')
#ax5.scatter(x, y, z, c=col_average, cmap=cm.hot)
#ax5.set_xlabel('X')
#ax5.set_ylabel('Y')
#ax5.set_zlabel('Z')
#ax5.view_init(elev=30, azim=200)
#ax5.set_title("Elev 30, Azim 200")

#ax6 = fig.add_subplot(3,3,6, projection='3d')
#ax6.scatter(x, y, z, c=col_average, cmap=cm.hot)
#ax6.set_xlabel('X')
#ax6.set_ylabel('Y')
#ax6.set_zlabel('Z')
#ax6.view_init(elev=30, azim=240)
#ax6.set_title("Elev 30, Azim 240")

#ax7 = fig.add_subplot(3,3,7, projection='3d')
#ax7.scatter(x, y, z, c=col_average, cmap=cm.hot)
#ax7.set_xlabel('X')
#ax7.set_ylabel('Y')
#ax7.set_zlabel('Z')
#ax7.view_init(elev=30, azim=280)
#ax7.set_title("Elev 30, Azim 280")

#ax8 = fig.add_subplot(3,3,8, projection='3d')
#ax8.scatter(x, y, z, c=col_average, cmap=cm.hot)
#ax8.set_xlabel('X')
#ax8.set_ylabel('Y')
#ax8.set_zlabel('Z')
#ax8.view_init(elev=30, azim=320)
#ax8.set_title("Elev 30, Azim 320")

#ax9 = fig.add_subplot(3,3,9, projection='3d')
#scatter9 = ax9.scatter(x, y, z, c=col_average, cmap=cm.hot)
#ax9.set_xlabel('X')
#ax9.set_ylabel('Y')
#ax9.set_zlabel('Z')
#ax9.view_init(elev=30, azim=360)
#ax9.set_title("Elev 30, Azim 360")

#fig.colorbar(scatter9)
# Create an inset axes for the colorbar
cbax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]

# Add the colorbar to the inset axes
cbar = fig.colorbar(scatter1, cax=cbax, orientation='horizontal')
plt.savefig("passive_pinna_sub3_chan0_3D_noise.png", bbox_inches="tight")
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
