import os, sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
from hrtfs.cipic_db import CipicDatabase 
from matplotlib import cm
from matplotlib.tri import Triangulation

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
#x = -1.0 * x
X, Y = np.meshgrid(np.unique(x), np.unique(y))
from scipy.interpolate import griddata
Z = griddata((x, y), z, (X, Y), method='cubic')
tri = Triangulation(x, y)
num_points = 100
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), num_points),
                             np.linspace(y.min(), y.max(), num_points))

col_average = np.full((1250), 0.0)
for j in range(1250):
    col_sum = 0.0
    col_count = 0.0
    for i in range(1250):
        if data[i, j] > 0.0:
            col_sum += data[i, j]
            col_count += 1.0
    if col_count > 0.0:
        col_average[j] = col_sum / col_count
    else:
        col_average[j] = -1.0
from scipy.interpolate import griddata
z_grid = griddata((x, y), col_average, (x_grid, y_grid), method='linear')
      
#u = np.linspace(0, 2 * np.pi, 80)
#v = np.linspace(0, np.pi, 80)

# create the sphere surface
#x=10 * np.outer(np.cos(u), np.sin(v))
#print(x.shape)
#y=10 * np.outer(np.sin(u), np.sin(v))
#z=10 * np.outer(np.ones(np.size(u)), np.cos(v))

# simulate heat pattern (striped)
#myheatmap = np.abs(np.sin(y))

elevations = [30]
azimuths = range(0, 361, 45)

fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(3,3,1, projection='3d')
#ax.plot_surface(X, Y, Z, cstride=1, rstride=1, facecolors=cm.hot(col_average))
#surf = ax.plot_trisurf(tri, z, cmap=cm.viridis, linewidth=0.2, antialiased=True, shade=True)
scatter1 = ax1.scatter(x, y, z, c=col_average, cmap=cm.hot)
#surf = ax.plot_trisurf(tri, z, cmap=cm.hot, facecolors=cm.hot(col_average))#, linewidth=0.2, antialiased=True)
#surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.viridis)
# Add a color bar
#fig.colorbar(surf)

# Set labels
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.view_init(elev=30, azim=40)
ax1.set_title("Elev 30, Azim 40")
#for i, elev in enumerate(elevations):
#    for j, azim in enumerate(azimuths):
#        ax.view_init(elev=elev, azim=azim)
#        filename = f'3d_view_elev_{elev}_azim_{azim}.png'
#        plt.savefig(filename)

ax2 = fig.add_subplot(3,3,2, projection='3d')
scatter2 = ax2.scatter(x, y, z, c=col_average, cmap=cm.hot)
#ax2.colorbar(scatter2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.view_init(elev=30, azim=80)
ax2.set_title("Elev 30, Azim 80")

ax3 = fig.add_subplot(3,3,3, projection='3d')
scatter3 = ax3.scatter(x, y, z, c=col_average, cmap=cm.hot)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.view_init(elev=30, azim=120)
ax3.set_title("Elev 30, Azim 120")

ax4 = fig.add_subplot(3,3,4, projection='3d')
ax4.scatter(x, y, z, c=col_average, cmap=cm.hot)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.view_init(elev=30, azim=160)
ax4.set_title("Elev 30, Azim 160")

ax5 = fig.add_subplot(3,3,5, projection='3d')
ax5.scatter(x, y, z, c=col_average, cmap=cm.hot)
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.view_init(elev=30, azim=200)
ax5.set_title("Elev 30, Azim 200")

ax6 = fig.add_subplot(3,3,6, projection='3d')
ax6.scatter(x, y, z, c=col_average, cmap=cm.hot)
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Z')
ax6.view_init(elev=30, azim=240)
ax6.set_title("Elev 30, Azim 240")

ax7 = fig.add_subplot(3,3,7, projection='3d')
ax7.scatter(x, y, z, c=col_average, cmap=cm.hot)
ax7.set_xlabel('X')
ax7.set_ylabel('Y')
ax7.set_zlabel('Z')
ax7.view_init(elev=30, azim=280)
ax7.set_title("Elev 30, Azim 280")

ax8 = fig.add_subplot(3,3,8, projection='3d')
ax8.scatter(x, y, z, c=col_average, cmap=cm.hot)
ax8.set_xlabel('X')
ax8.set_ylabel('Y')
ax8.set_zlabel('Z')
ax8.view_init(elev=30, azim=320)
ax8.set_title("Elev 30, Azim 320")

ax9 = fig.add_subplot(3,3,9, projection='3d')
scatter9 = ax9.scatter(x, y, z, c=col_average, cmap=cm.hot)
ax9.set_xlabel('X')
ax9.set_ylabel('Y')
ax9.set_zlabel('Z')
ax9.view_init(elev=30, azim=360)
ax9.set_title("Elev 30, Azim 360")

fig.colorbar(scatter9)
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


