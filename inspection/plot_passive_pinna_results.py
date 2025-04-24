import os, sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
#from hrtfs.cipic_db import CipicDatabase 
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

def index_to_cart_pos(index):

row_average = np.full((1250), 0.0)
for i in range(1250):
    row_sum = 0.0
    row_count = 0.0
    for j in range(1250):
        if data[i, j] > 0.0:
            row_sum += data[i, j]
            row_count += 1.0
    if row_count > 0.0:
        row_average[i] = row_sum / row_count
    else:
        row_average[i] = -1.0
      
#u = np.linspace(0, 2 * np.pi, 80)
#v = np.linspace(0, np.pi, 80)

# create the sphere surface
#x=10 * np.outer(np.cos(u), np.sin(v))
#y=10 * np.outer(np.sin(u), np.sin(v))
#z=10 * np.outer(np.ones(np.size(u)), np.cos(v))

# simulate heat pattern (striped)
#myheatmap = np.abs(np.sin(y))


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cm.hot(myheatmap))
#plt.savefig("passive_pinna_sub3_chan0_3D_noise.png", bbox_inches="tight")
#plt.close()

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


