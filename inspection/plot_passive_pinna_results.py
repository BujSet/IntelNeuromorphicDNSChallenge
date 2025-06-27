import os, sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
from hrtfs.cipic_db import CipicDatabase 
from matplotlib import cm
import math

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
print("Index0 = " + str(cart_pos[0]) + " - " + str(Subject3.getSphericalPositionsFromIndex(0)))
print("Index1 = " + str(cart_pos[1]) + " - " + str(Subject3.getSphericalPositionsFromIndex(1)))
print("Index2 = " + str(cart_pos[2]) + " - " + str(Subject3.getSphericalPositionsFromIndex(2)))
print("Index3 = " + str(cart_pos[3]) + " - " + str(Subject3.getSphericalPositionsFromIndex(3)))
print("Index5 = " + str(cart_pos[5]) + " - " + str(Subject3.getSphericalPositionsFromIndex(5)))
print("Index8 = " + str(cart_pos[8]) + " - " + str(Subject3.getSphericalPositionsFromIndex(8)))
print("Index24 = " + str(cart_pos[24]) + " - " + str(Subject3.getSphericalPositionsFromIndex(24)))
print("Index49 = " + str(cart_pos[49]) + " - " + str(Subject3.getSphericalPositionsFromIndex(49)))
print("Index300 = " + str(cart_pos[300]) + " - " + str(Subject3.getSphericalPositionsFromIndex(300)))
print("Index308 = " + str(cart_pos[300]) + " - " + str(Subject3.getSphericalPositionsFromIndex(308)))
print("Index350 = " + str(cart_pos[350]) + " - " + str(Subject3.getSphericalPositionsFromIndex(350)))
print("Index358 = " + str(cart_pos[358]) + " - " + str(Subject3.getSphericalPositionsFromIndex(358)))
print("Index374 = " + str(cart_pos[374]) + " - " + str(Subject3.getSphericalPositionsFromIndex(374)))
print("Index382 = " + str(cart_pos[382]) + " - " + str(Subject3.getSphericalPositionsFromIndex(382)))
print("Index600 = " + str(cart_pos[600]) + " - " + str(Subject3.getSphericalPositionsFromIndex(600)))
print("Index608 = " + str(cart_pos[608]) + " - " + str(Subject3.getSphericalPositionsFromIndex(608)))
print("Index1200 = " + str(cart_pos[1200]) + " - " + str(Subject3.getSphericalPositionsFromIndex(1200)))
x = cart_pos[:,0]
y = cart_pos[:,1]
z = cart_pos[:,2]

col_average = np.full((1250), 0.0)
row_average = np.full((1250), 0.0)
fill_col_average(col_average, data)
fill_row_average(row_average, data)
col_min_idx = np.argmin(col_average)
col_max_idx = np.argmax(col_average)
row_min_idx = np.argmin(row_average)
row_max_idx = np.argmax(row_average)
print("Col min = " + str(col_min_idx))
print("Col max = " + str(np.argmax(col_average)))
print("Row min = " + str(np.argmin(row_average)))
print("Row max = " + str(row_max_idx))
print("Col Min = " + str(cart_pos[col_min_idx]) + " - " + str(Subject3.getSphericalPositionsFromIndex(col_min_idx)))
print("Col Max = " + str(cart_pos[col_max_idx]) + " - " + str(Subject3.getSphericalPositionsFromIndex(col_max_idx)))
print("Row Min = " + str(cart_pos[row_min_idx]) + " - " + str(Subject3.getSphericalPositionsFromIndex(row_min_idx)))
print("Row Max = " + str(cart_pos[row_max_idx]) + " - " + str(Subject3.getSphericalPositionsFromIndex(row_max_idx)))
#print("Col min = " + str(Subject3.getSphericalPositionsFromIndex(np.argmin(col_average))))
#print("Col max = " + str(Subject3.getSphericalPositionsFromIndex(np.argmax(col_average))))
#print("Row min = " + str(Subject3.getSphericalPositionsFromIndex(np.argmin(row_average))))
#print("Row max = " + str(Subject3.getSphericalPositionsFromIndex(np.argmax(row_average))))
#print(Subject3.printAnthroData())

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

def moving_average(data, window_size):
    """
    Computes the moving average of a 1D array.

    Args:
        data (numpy.ndarray): The input array.
        window_size (int): The number of data points to include in each average.

    Returns:
        numpy.ndarray: An array containing the moving averages.
                       Returns an empty array if window_size > len(data).
    """
    if window_size > len(data):
        return np.array([])
    
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def sortBy3DDistFromMax(data):
    maxIdx = np.argmax(data)
    print(maxIdx)
    print(data[maxIdx])
    sortedData = []
    maxX = x[maxIdx]
    maxY = y[maxIdx]
    maxZ = z[maxIdx]
    maxV1 = np.array([maxX, maxY, maxZ])
    normV1 = np.linalg.norm(maxV1)

    dists =  dict()
    for i in range(1250):
        v2 = np.array( [x[i], y[i], z[i]] )
        normV2 = np.linalg.norm(v2)
        cos_theta = np.dot(maxV1, v2) / (normV1 * normV2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure value is within valid range

        angle_rad = np.arccos(cos_theta)
        # angular distance
        dist = np.degrees(angle_rad)
        sortedData.append( (dist, data[i])  )
    #     dist = round(dist, 1)
    #     if str(dist) in dists.keys():
    #         dists[str(dist)].append(data[i])
    #     else:
    #         dists[str(dist)] = [data[i]]


    # for key,val in dists.items():
    #     dist = float(key)
    #     score = sum(val) / (1.0*len(val))
    #     sortedData.append( (dist, score)  )



    sortedData.sort(key=lambda x: x[0])
    return sortedData

sortedNoiseByDist = sortBy3DDistFromMax(col_average)
#print(sortedNoiseByDist)
# uniqueAngularDists = set()
# for ad,si in sortedNoiseByDist:
#     uniqueAngularDists.add(round(ad,3))
# print("Unique angular dists: " + str(len(uniqueAngularDists)))
print("Len of sortedNoiseByDist: " + str(len(sortedNoiseByDist)))
sortedSpeechByDist = sortBy3DDistFromMax(row_average)


# Analysis on averages
sorted_noise = np.sort(col_average)
x_data = [i for i in range(1250)]

def polynomial_regression(x, y, degree):
    # Calculate polynomial coefficients using numpy.polyfit
    coefficients = np.polyfit(x, y, degree)
    
    # Create a polynomial function
    polynomial = np.poly1d(coefficients)
    
    # Calculate predicted y values
    y_predicted = polynomial(x)
    
    # Calculate the sum of squared errors (SSE)
    sse = np.sum((y_predicted - y)**2)
    
    return polynomial, sse

# Find the best-fit polynomial by iterating through degrees
#best_degree = 1
#min_error = float('inf')
#best_polynomial = None

#errors = []
#NUM_DEGREES = 25
#for degree in range(1, NUM_DEGREES): #Trying polynomials from degree 1 to 3
#    polynomial, error = polynomial_regression(x_data, sorted_noise, degree)
#    errors.append(error)
#    if error < min_error:
#        min_error = error
#        best_polynomial = polynomial
#        best_degree = degree
#
#plt.figure(figsize=(10, 10))
#fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25,12))
#win_size = 50
#win_x_data = [i for i in range(1250 - win_size + 1)]
#
#coefficients = np.polyfit(x_data, sorted_noise, 4)
#    
# Create a polynomial function
#polynomial = np.poly1d(coefficients)
#    
# Calculate predicted y values
#y_predicted = polynomial(x_data)
#rolling = moving_average(sorted_noise, win_size)
#axes[1,0].set_xlabel('Sorted Speech Orients')
#axes[1,0].set_ylabel('SI-SNRs(dB)')
#axes[1,0].plot(x_data, sorted_noise, color="blue", label="data")
#axes[1,0].plot(win_x_data, rolling, color="orange", label="rolling average")
#axes[1,0].plot(x_data, y_predicted, color="green", label="polyfitDegree3")
#axes[1,0].legend()
#
#dy = np.gradient(sorted_noise)
#dypred = np.gradient(y_predicted) + 0.02
#rollingdy = moving_average(dy, win_size)
#drollingy = np.gradient(rolling) + 0.03
#axes[1,1].set_xlabel('Sorted Speech Orients')
#axes[1,1].set_ylabel('SI-SNRs(dB)')
#axes[1,1].plot(x_data, dy, color="blue", label="dx")
#axes[1,1].plot(win_x_data, rollingdy, color="orange", label="rolling average(dy)")
#axes[1,1].plot(win_x_data,drollingy, color="green", label="d'rolling average(y)")
#axes[1,1].plot(x_data, dypred, color="red", label="d'polyfit")
#axes[1,1].legend()
#
#dy2 = np.gradient(dy)
#dy2pred = np.gradient(dypred)
#rollingdy2 = moving_average(dy2, win_size)
#drollingdy = np.gradient(rollingdy) + 0.005
#axes[1,2].set_xlabel('Sorted Speech Orients')
#axes[1,2].set_ylabel('SI-SNRs(dB)')
#axes[1,2].plot(x_data, dy2, color="blue", label="dx2")
#axes[1,2].plot(win_x_data, rollingdy2, color="orange", label="rolling average(dy2)")
#axes[1,2].plot(win_x_data, drollingdy, color="green", label="d'rolling average(dy)")
#axes[1,2].plot(x_data, dy2pred, color="red", label="d2'polyfit")
#axes[1,2].legend()
#
#eightyDecay = (0.2*(max(errors) - min(errors))) + min(errors)
#axes[0,2].set_xlabel('Polyfit Degree')
#axes[0,2].set_ylabel('SSE')
#axes[0,2].plot([i for i in range(1,NUM_DEGREES)], errors, color="blue")
#axes[0,2].plot([i for i in range(1,NUM_DEGREES)], [eightyDecay for _ in range(1, NUM_DEGREES)], color="orange")
#
#win_size = 3
#distX = [dist for dist,score in sortedNoiseByDist]
#scoreY = [score for dist,score in sortedNoiseByDist]
#rollingX = moving_average(distX, win_size)
#rollingY = moving_average(scoreY, win_size)
#axes[0,0].set_xlabel('Angular Dist from Max (degrees)')
#axes[0,0].set_ylabel('SI-SNR (dB)')
#axes[0,0].set_title('Noise')
#axes[0,0].scatter(distX, scoreY, color="blue", s=2)
#axes[0,0].plot(rollingX, rollingY, color="orange", label="rolling average")
#
#distX = [dist for dist,score in sortedSpeechByDist]
#scoreY = [score for dist,score in sortedSpeechByDist]
#axes[0,1].set_xlabel('Angular Dist from Max (degrees)')
#axes[0,1].set_ylabel('SI-SNR (dB)')
#axes[0,1].set_title('Speech')
#axes[0,1].scatter(distX, scoreY, color="blue", s=2)
#
#best = np.argmin(np.abs(np.array(errors) - np.array([eightyDecay for _ in range(1, NUM_DEGREES)])))
#print("Best degree polyfit: " + str(best))
#
#plt.savefig("passive_pinna_sorted_noise.png", bbox_inches="tight")
#plt.close()

def calcR2(x, y, y_pred):
    avgy = sum(y)/len(y)
    ssreg = sum( [(y_pred[i] - avgy)**2 for i in range(len(y))])
    sstot = sum( [(y[i] - avgy)**2 for i in range(len(y))] )
    r2Val = ssreg / sstot
    return r2Val

def calcMSE(y, y_pred):
    tot = sum([ (y[i] - y_pred[i]) **2 for i in range(len(y))])
    return tot / (1.0*len(y))

def findZones(x, y, y_pred, threshold):
    remainingStart = 0
    remainingEnd = len(x)
    zones = []
    while(remainingStart < remainingEnd):
        zoneStart = remainingStart
        zoneEnd = remainingStart + 2
        if zoneEnd >= remainingEnd:
            break

        mse = calcMSE(y[zoneStart:zoneEnd],
            y_pred[zoneStart:zoneEnd])
        # print("zoneStart:" + str(zoneStart))
        # print("zoneEnd:" + str(zoneEnd))
        # print("mse:" + str(mse))
        # print()
        while (mse < threshold and zoneEnd < remainingEnd):
            zoneEnd += 1
            mse = calcMSE(y[zoneStart:zoneEnd],
                y_pred[zoneStart:zoneEnd])
        #     print("zoneStart:" + str(zoneStart))
        #     print("zoneEnd:" + str(zoneEnd))
        #     print("mse:" + str(mse))
        #     print()
        # sys.exit(0)
        zones.append( (zoneStart, zoneEnd - 1, mse) )
        remainingStart = zoneEnd
    return zones






fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
win_size = 3
distX = [dist for dist,score in sortedNoiseByDist]
scoreY = [score for dist,score in sortedNoiseByDist]
rollingX = moving_average(distX, win_size)
rollingY = moving_average(scoreY, win_size)
coefficients = np.polyfit(distX, scoreY, 1)
# Create a polynomial function
polynomial = np.poly1d(coefficients)
# Calculate predicted y values
y_predicted = polynomial(distX)
y_pred_as_list = [y_predicted[i] for i in range(len(y_predicted))]

errorR2 = calcR2(distX, scoreY, y_pred_as_list)

print("R2 value for linear fit is " + str(errorR2))
axes.set_xlabel('Angular Dist from Max (°)')
axes.set_ylabel('SI-SNR (dB)')
axes.set_title('Noise Audio Sphere', fontweight='bold')
axes.scatter(distX, scoreY, color="blue", s=2, label='Raw Data')
axes.plot(rollingX, rollingY, color="orange", label="SMA-3")
axes.plot(distX, y_predicted, color="green", label="Linear Fit")
axes.legend()

#distX = [dist for dist,score in sortedSpeechByDist]
#scoreY = [score for dist,score in sortedSpeechByDist]
#rollingX = moving_average(distX, win_size)
#rollingY = moving_average(scoreY, win_size)
#axes[0].set_xlabel('Angular Dist from Max (degrees)')
#axes[0].set_ylabel('SI-SNR (dB)')
#axes[0].set_title('a) Speech', fontweight='bold')
#axes[0].scatter(distX, scoreY, color="blue", s=2, label="Raw Data")
#axes[0].plot(rollingX, rollingY, color="orange", label="Rolling Average")
#axes[0].legend()
plt.savefig("passive_pinna_angular_dist_noise.pdf", bbox_inches="tight")
plt.close()

#print(np.roots(np.gradient(y_predicted)))

# print()
# zones = findZones(distX[:], scoreY[:], y_pred_as_list[:], 0.015)
# print(zones)
# print("Zone 1: [0,12] = [" +str(distX[0]) + "," + str(distX[11]) + "] mse=" +str(calcMSE(scoreY[0:12],y_predicted[0:12])))
# print("Zone 2: [12,190] = [" +str(distX[12]) + "," + str(distX[190]) + "] mse=" +str(calcMSE(scoreY[12:190],y_predicted[12:190])))
# print("Zone 2: [12,355] = [" +str(distX[12]) + "," + str(distX[355]) + "] mse=" +str(calcMSE(scoreY[12:355],y_predicted[12:355])))
# print("Zone 2: [12,555] = [" +str(distX[12]) + "," + str(distX[555]) + "] mse=" +str(calcMSE(scoreY[12:555],y_predicted[12:555])))
# print("Zone 3: [555,571] = [" +str(distX[555]) + "," + str(distX[571]) + "] mse=" +str(calcMSE(scoreY[555:571],y_predicted[555:571])))
# print("Zone 4: [571,649] = [" +str(distX[571]) + "," + str(distX[648]) + "] mse=" +str(calcMSE(scoreY[571:649],y_predicted[571:649])))

drollingY = [abs(rollingY[i+1] - rollingY[i]) for i in range(len(rollingY) - 1)]
win_size = 150
padded_data = np.pad(drollingY, win_size//2, mode='reflect')
rolldry = moving_average(padded_data, win_size)
paddedRollDRY = rolldry #[0 for i in range(win_size//2)] + [rolldry[i] for i in range(len(rolldry))] + [0 for i in range(win_size//2)]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,6))

axes.set_xlabel('Data Point Sorted by Angular Dist from Max')
axes.set_ylabel('ΔSI-SNR (dB)')
axes.set_title('Conic Boundaries for Auditory Acuity', fontweight='bold')
axes.plot([i for i in range(len(drollingY))], drollingY, color="blue", label="|SMA-3'|")
axes.plot([i for i in range(len(paddedRollDRY))], paddedRollDRY, color="orange", label="SMA-100(|SMA-3'|)")
axes.plot([i for i in range(len(drollingY))], [0.125 for _ in range(len(drollingY))], color="green", label="Threshold")
axes.legend()
plt.savefig("passive_pinna_deriv_sma.pdf", bbox_inches="tight")
plt.close()

crossings = []
for i in range(len(rolldry)-1):
    if rolldry[i] < 0.125 and rolldry[i+1] > 0.125:
        crossings.append(i)
    if rolldry[i] > 0.125 and rolldry[i+1] < 0.125:
        crossings.append(i)

print(crossings)
print("Zone 1: [0,84] = [" +str(distX[0]) + "," + str(distX[84]) + "] mse=" +str(calcMSE(scoreY[0:84],y_predicted[0:84])))
print("Zone 2: [84,398] = [" +str(distX[84]) + "," + str(distX[398]) + "] mse=" +str(calcMSE(scoreY[84:398],y_predicted[84:398])))
print("Zone 3: [398,705] = [" +str(distX[398]) + "," + str(distX[705]) + "] mse=" +str(calcMSE(scoreY[398:705],y_predicted[398:705])))
print("Zone 4: [705,:] = [" +str(distX[705]) + "," + str(distX[-1]) + "] mse=" +str(calcMSE(scoreY[705:],y_predicted[705:])))
