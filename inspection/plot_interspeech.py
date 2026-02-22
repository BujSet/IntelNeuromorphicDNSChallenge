import os, sys
sys.path.append('./')
import glob
import pandas as pd
from hrtfs.cipic_db import CipicDatabase 
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

csv_files = glob.glob('*.csv')
out_files = sorted(glob.glob('*.out'))


def get_plot_theta_r(sub, index, r_offset=0):
    modulo = index % 50
    r = (index // 50 ) +r_offset
    pos = sub.getSphericalPositionsFromIndex(600 + modulo)
    angle = pos[1]
    if modulo < 8:
        angle = angle + 360.0
    elif modulo > 24 and modulo <= 48:
        angle = 180.0 - angle
    elif modulo > 48:
        angle = 180.0 - angle
    return math.radians(angle), r

def read_collated_results_csv():
    df = pd.read_csv("collated_results_2026_02_18.csv")
    df.columns = df.columns.str.strip()
    df['Subject'] = df['Subject'].astype(int)
    df['Channel'] = df['Channel'].astype(int)
    df['UsesFullAudioDataset'] = True
    df['AudioDataSubsetSize'] = 60000
    df['AudioDataSubsetSeed'] = -1
    return df

def read_out_file(out_file):
    df = pd.read_csv(out_file)
    df.columns = df.columns.str.strip()
    df['Subject'] = df['Subject'].astype(int)
    df['Channel'] = df['Channel'].astype(int)
    df = df.drop(
        columns=[
            'ExecTime',
            'BatchSize',
            'Dataloader Num Workers', 
            'Dataloader Prefetch Factor',
            'Sample Size (MB)', 
            'CUDA Peak Mem Allocated (MB)',
            'CUDA Peak Mem Cached (MB)',
            'CUDA Peak Mem Reserved (MB)'],
        errors='ignore')
    df['UsesFullAudioDataset'] = False
    df['AudioDataSubsetSize'] = 120
    df['AudioDataSubsetSeed'] = 419572083
    return df

# First, read all data and concat into a single df
all_data = read_collated_results_csv()
print(f"All data conatains {len(all_data)} rows, pre merge")
for i, out_file in enumerate(out_files):
    print(out_file)
    out_data = read_out_file(out_file)
    all_data = pd.concat([all_data, out_data], ignore_index=True)
    duplicate_rows_boolean = all_data.duplicated()
    num_duplicates = duplicate_rows_boolean.sum()
#    if num_duplicates:
#        print(f"All data conatains {len(all_data)} rows after merge {i}, {num_duplicates} duplicates")
print(f"All data conatains {len(all_data)} rows, post merge")

sub_3_chan_0_full_dataset = all_data[(all_data['Subject'] == 3) &
        (all_data['Channel'] == 0) &
        (all_data['UsesFullAudioDataset']) &
        (all_data['AudioDataSubsetSize'] == 60000)]

def getPlotDataForAudioSphere(df, subject, channel, dataSubsetSize=60000, speechAudioSphere=True):
    filtered = all_data[(all_data['Subject'] == subject) &
        (all_data['Channel'] == channel)]
    if dataSubsetSize == 60000:
        filtered = filtered[(filtered['UsesFullAudioDataset'] == True) &
        (filtered['AudioDataSubsetSize'] == 60000)]
    else:
        filtered = filtered[(filtered['UsesFullAudioDataset'] == False) &
        (filtered['AudioDataSubsetSize'] == dataSubsetSize)]
    searchString = 'Speech Orient'
    if not speechAudioSphere:
        searchString = 'Noise Orient'

    sphereData = []
    for i in range(1250):
        if ((sub_3_chan_0_full_dataset[searchString] == i).any()):
            row = filtered[filtered[searchString] == i]
            theta, r = get_plot_theta_r(CipicDatabase.subjects[subject], i)
            sphereData.append({
                'CipicIndex':i, 
                'Final Validation Score SI-SNR (dB)':row['Final Validation Score SI-SNR (dB)'].mean(),
                'PlotPolarThetaRadians':theta,
                'PlotPolarR':r})
    return pd.DataFrame(sphereData)


def plotAudioSpheres(df, subject, channel, dataSubsetSize=60000):
    speech = getPlotDataForAudioSphere(df, subject, channel, dataSubsetSize, True)
    noise = getPlotDataForAudioSphere(df, subject, channel, dataSubsetSize, False)
    fig, axs = plt.subplots(nrows=1, ncols=2, 
            figsize=(12, 6), subplot_kw={'projection': 'polar'},
                               gridspec_kw={'wspace': -0.0}, layout="constrained")
    def plotSphereOnAxis(ax, df, axTitle):
        ax.set_axisbelow(True)
        scatter = ax.scatter(df['PlotPolarThetaRadians'],
                     df['PlotPolarR'], 
                     c=df['Final Validation Score SI-SNR (dB)'],
                     cmap='hot', alpha=0.75, zorder=2)
        rticks = [0, 12.5, 25]
        rlabels = ['Right', 'Middle', 'Left']
        rlines, rlabels = ax.set_rgrids(rticks, rlabels, angle=-90)
        for i, label in enumerate(rlabels):
            label.set_horizontalalignment('center') 
            if i == 0:
                label.set_verticalalignment('top') 
            elif i == 2:
                label.set_verticalalignment('bottom') 
            else:
                label.set_verticalalignment('center') 
        ax.tick_params(axis='y', labelsize=10, rotation=0)
        ax.tick_params(axis='x', labelsize=10, pad=11)
        custom_ticks_rad = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180.0
        custom_labels = ['Front', 'Antero-\nSuperior', 'Up', 'Postero-\nSuperior', 'Back', 'Postero-\nInferior', '', 'Antero-\nInferior'] # Note: 360/0 overlap

        ax.set_xticks(custom_ticks_rad)
        ax.set_xticklabels(custom_labels)
        ax.set_title(axTitle, fontweight="bold")
        ax.grid(True)
        ax.set_rorigin(-10)
        return scatter
    scatter = plotSphereOnAxis(axs[0], speech, "a) Speech Audiosphere")
    scatter = plotSphereOnAxis(axs[1], noise, "a) Noise Audiosphere")
    titleString = f"Subject {subject}'s Audio Spheres ("
    if channel == 0:
        titleString += "Right Ear"
    else: 
        titleString += "Left Ear"
    titleString += ")"
    fig.suptitle(titleString, fontsize=16, fontweight='bold')
    fig.colorbar(scatter, ax=axs, label='Validation Score SI-SNR (dB)', orientation='horizontal', shrink=0.9, aspect=50)
    plt.savefig(f'sub_{subject}_chan_{channel}_audio_spheres.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(f'sub_{subject}_chan_{channel}_audio_spheres.png', bbox_inches='tight', transparent=True)
    plt.close()


sub3 = CipicDatabase.subjects[3]
speechAudioSphere = getPlotDataForAudioSphere(all_data, 3, 0, 60000, True)
noiseAudioSphere = getPlotDataForAudioSphere(all_data, 3, 0, 60000, False)
plotAudioSpheres(all_data, 3, 0, 60000)
availableSubjects = all_data['Subject'].unique()
print(availableSubjects)
subjectsWithBothChannels = set()
for sub in availableSubjects:
    subRows = all_data.loc[all_data['Subject'] == sub]
    numChannels = len(subRows['Channel'].unique())
    if numChannels == 2:
        subjectsWithBothChannels.add(sub)
print(subjectsWithBothChannels)
inspectSub = 8
sub_3_chan_0 = getPlotDataForAudioSphere(all_data, inspectSub, 0, 120, True)
sub_3_chan_1 = getPlotDataForAudioSphere(all_data, inspectSub, 1, 120, True)
print(sub_3_chan_0.head())
print(len(sub_3_chan_0))
has_nan_A = sub_3_chan_0['Final Validation Score SI-SNR (dB)'].isna().any()
print(f"Does sub_{inspectSub}_chan_0 have any NaNs? {has_nan_A}")
if has_nan_A:
    rows_with_nan = sub_3_chan_0[sub_3_chan_0['Final Validation Score SI-SNR (dB)'].isna()]
    with pd.option_context('display.max_rows', None):
        print(rows_with_nan)

print(sub_3_chan_1.head())
print(len(sub_3_chan_1))
has_nan_B = sub_3_chan_1['Final Validation Score SI-SNR (dB)'].isna().any()
print(f"Does sub_{inspectSub}_chan_1 have any NaNs? {has_nan_B}")
if has_nan_B:
    rows_with_nan = sub_3_chan_1[sub_3_chan_1['Final Validation Score SI-SNR (dB)'].isna()]
    with pd.option_context('display.max_rows', None):
        print(rows_with_nan)

def plotContourOnAxis(ax, subject, channel, axTitle, cmapMin, cmapMax, num_levels):
    global all_data
    num_points = 500
    speech = getPlotDataForAudioSphere(all_data, subject, channel, 120, True)
    ax.set_axisbelow(True)

    grid_r = np.linspace(speech['PlotPolarR'].min(), speech['PlotPolarR'].max(), num_points)
    grid_t = np.linspace(0, 360, num_points) * np.pi / 180
    R, T = np.meshgrid(grid_r, grid_t)
    points = speech[['PlotPolarR', 'PlotPolarThetaRadians']].values
    values = speech['Final Validation Score SI-SNR (dB)'].values
    low_theta = points[points[:, 1] < 0.1].copy()
    low_theta[:, 1] += 2 * np.pi

    high_theta = points[points[:, 1] > (2 * np.pi - 0.1)].copy()
    high_theta[:, 1] -= 2 * np.pi

    # Combine original data with the "wrapped" phantom points
    points_wrapped = np.vstack([points, low_theta, high_theta])
    values_wrapped = np.concatenate([values, values[points[:, 1] < 0.1], values[points[:, 1] > (2 * np.pi - 0.1)]])

    Z = griddata(points_wrapped, values_wrapped, (R,T), method='cubic')
    t_min_hide = 225 * np.pi / 180
    t_max_hide = 315 * np.pi / 180
    mask = (T > t_min_hide) & (T <= t_max_hide)

    # Set the Z values in that slice to NaN
    Z[mask] = np.nan

    #levels = np.linspace(Z.min(), Z.max(), num_levels + 1)

    # --- 3. Create a discrete colormap from the 'hot' colormap ---
    # Get the 'hot' colormap
    #cmap_hot = plt.get_cmap('hot')
    # Select 6 colors evenly spaced from the continuous colormap
    #colors = [cmap_hot(i) for i in np.linspace(0, 1, num_levels)]
    # Create the listed colormap
    #cmap = mcolors.ListedColormap(colors)
    #cmap = plt.get_cmap('hot', num_levels) #len(levels) - 1)

    # --- 4. Create a BoundaryNorm normalization object ---
    # This maps the data values to the colors based on the levels
    #norm = mcolors.BoundaryNorm(levels, cmap.N)
    #CS = ax.contourf(T, R, Z, levels=num_levels, cmap=cmap, norm=norm)
    #print(f"Making contour subplot with levels={num_levels}, vmin={cmapMin}, vmax={cmapMax}")
    #CS = ax.contourf(T, R, Z, levels=num_levels, cmap='hot') #, vmin=cmapMin, vmax=cmapMax)
    level_list = np.linspace(Z[(T<= t_min_hide) | (T > t_max_hide)].min(), Z[T <= t_min_hide) | (T > t_max_hide)].max(), num_levels + 1)
    CS = ax.contourf(T, R, Z, levels=level_list, cmap='hot') #, vmin=cmapMin, vmax=cmapMax)
    rticks = [0, 12.5, 24]
    rlabels = ['Right', 'Middle', 'Left']
    rlines, rlabels = ax.set_rgrids(rticks, rlabels, angle=-90)
    for i, label in enumerate(rlabels):
        label.set_horizontalalignment('center') 
        if i == 0:
            label.set_verticalalignment('top') 
        elif i == 2:
            label.set_verticalalignment('bottom') 
        else:
            label.set_verticalalignment('center') 
    ax.tick_params(axis='y', labelsize=10, rotation=0)
    ax.tick_params(axis='x', labelsize=10, pad=9)
    custom_ticks_rad = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180.0
    custom_labels = ['Front', 'Antero-\nSuperior', 'Up', 'Postero-\nSuperior', 'Back', 'Postero-\nInferior', '', 'Antero-\nInferior'] # Note: 360/0 overlap

    ax.set_xticks(custom_ticks_rad)
    ax.set_xticklabels(custom_labels)
    ax.set_title(axTitle, fontweight="bold")
    ax.grid(True)
    ax.set_rorigin(-10)
    return CS

def plotMonauralContourMaps(df, subjectSet, numRows=3, numCols=8):
    subList = sorted(list(subjectSet))
    if len(subList) > (numRows*numCols):
        subLst = subList[0:numRows*numCols]
    print(len(subList))

    fig, axs = plt.subplots(nrows=numRows, ncols=numCols, 
            figsize=(20, 10), subplot_kw={'projection': 'polar'},
                               gridspec_kw={'wspace': -0.0}, layout="constrained")
    titles = [
            f"a) Subject {subList[0]} (Left Ear)",
            f"b) Subject {subList[0]} (Right Ear)",
            f"c) Subject {subList[1]} (Left Ear)",
            f"d) Subject {subList[1]} (Right Ear)",
            f"f) Subject {subList[2]} (Left Ear)",
            f"g) Subject {subList[2]} (Right Ear)",
            f"h) Subject {subList[3]} (Left Ear)",
            f"i) Subject {subList[3]} (Right Ear)",
            f"j) Subject {subList[4]} (Left Ear)",
            f"k) Subject {subList[4]} (Right Ear)",
            f"l) Subject {subList[5]} (Left Ear)",
            f"m) Subject {subList[5]} (Right Ear)",
            f"n) Subject {subList[6]} (Left Ear)",
            f"o) Subject {subList[6]} (Right Ear)",
            f"p) Subject {subList[7]} (Left Ear)",
            f"q) Subject {subList[7]} (Right Ear)",
            f"r) Subject {subList[8]} (Left Ear)",
            f"s) Subject {subList[8]} (Right Ear)", 
            f"t) Subject {subList[9]} (Left Ear)",
            f"u) Subject {subList[9]} (Right Ear)",
            f"v) Subject {subList[10]} (Left Ear)",
            f"w) Subject {subList[10]} (Right Ear)",
            f"x) Subject {subList[11]} (Left Ear)",
            f"y) Subject {subList[11]} (Right Ear)"]
    contoursMin = None
    contoursMax = None
    for r in range(numRows):
        for c in range(numCols):
            idx = (r*numCols)+c
            subIdx = (idx//2)
            subject = subList[subIdx]
            channelVal = (idx+1)%2
            filtered = all_data[(all_data['Subject'] == subject) & (all_data['Channel'] == channelVal)]
            fmin = filtered['Final Validation Score SI-SNR (dB)'].min()
            fmax = filtered['Final Validation Score SI-SNR (dB)'].max()
            print(f"axs[{r}, {c}] for subject={subject}, channel={channelVal}, fmin={fmin}, fmax={fmax}")
            if contoursMin == None or fmin < contoursMin:
                contoursMin = fmin
            if contoursMax == None or fmax > contoursMax:
                contoursMax = fmax

    for r in range(numRows):
        for c in range(numCols):
            idx = (r*numCols)+c
            subIdx = (idx//2)
            print(f"Plotting axs[{r},{c}], subList[{subIdx}]={subList[subIdx]}, channel={(idx+1)%2}, titles[{idx}]={titles[idx]}")
            cf = plotContourOnAxis(axs[r,c], subList[subIdx], (idx+1)%2, titles[idx], contoursMin, contoursMax, 6)
    #sm = ScalarMappable(cmap='hot', norm=plt.Normalize(contoursMin, contoursMax))
    #fig.colorbar(sm, ax=axs, label='Validation Score SI-SNR (dB)', orientation='horizontal', shrink=0.9, aspect=50)
    fig.colorbar(cf, ax=axs, label='Validation Score SI-SNR (dB)', orientation='horizontal', shrink=0.9, aspect=50) #, vmin=contoursMin, vmax=contoursMax)
    plt.savefig(f'contours.png', bbox_inches='tight', transparent=True)
    plt.close()


plotMonauralContourMaps(all_data, subjectsWithBothChannels, numRows=2, numCols=4)

def plotSingletonContourMap(subject, channel):
    fig, axs = plt.subplots(nrows=1, 
        ncols=1, 
        figsize=(12, 12),
        subplot_kw={'projection': 'polar'})
    titleString = f"Subject {subject} "
    if (channel == 0):
        titleString += "(Right Ear)"
    else:
        titleString += "(Left Ear)"

    filtered = all_data[(all_data['Subject'] == subject) & (all_data['Channel'] == channel)]
    fmin = filtered['Final Validation Score SI-SNR (dB)'].min()
    fmax = filtered['Final Validation Score SI-SNR (dB)'].max()
    num_levels = 6
    #levels = np.linspace(fmin, fmax, num_levels+1)
    #cmap_name = 'hot'
    # Create a ListedColormap with the desired number of colors (N) from the base colormap
    #cmap = plt.get_cmap(cmap_name, len(levels) - 1)
    # Create a BoundaryNorm to map data values to discrete color indices
    #norm = mcolors.BoundaryNorm(levels, cmap.N)

    print(f"Plotting singleton for subject={subject}, channel={channel}, fmin={fmin}, fmax={fmax}")
    #print(levels)
    cf = plotContourOnAxis(axs, subject, channel, titleString, fmin, fmax, num_levels) 
    #sm = ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    fig.colorbar(cf, ax=axs,
            label='Validation Score SI-SNR (dB)',
            orientation='horizontal',
            shrink=0.8, pad=0.01) #, ticks=levels)
    plt.savefig(f'sub_{subject}_chan_{channel}_speech_contour.png', bbox_inches='tight', transparent=True)
    plt.close()

plotSingletonContourMap(3, 0)
plotSingletonContourMap(3, 1)

def get_cart_pos(row, sub, cart_axis):
    cipicIndex = int(row['CipicIndex'])
    carts = sub.getCartesianPositions()
    pos = carts[cipicIndex]
    # Y axis is flipped for some reason?
    if cart_axis == 1:
        return -1.0 * pos[cart_axis]
    return pos[cart_axis]

speechAudioSphere['PlotCartX'] = speechAudioSphere.apply(
        get_cart_pos,
        axis=1,
        args=(sub3, 0)
        )
speechAudioSphere['PlotCartY'] = speechAudioSphere.apply(
        get_cart_pos,
        axis=1,
        args=(sub3, 1)
        )
speechAudioSphere['PlotCartZ'] = speechAudioSphere.apply(
        get_cart_pos,
        axis=1,
        args=(sub3, 2)
        )

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(speechAudioSphere['PlotCartX'], 
        speechAudioSphere['PlotCartY'], 
        speechAudioSphere['PlotCartZ'], 
        c=speechAudioSphere['Final Validation Score SI-SNR (dB)'],
        cmap='plasma', s=50, alpha=0.8)


ax.set_xticks([-1.0, 0.0, 1.0])
ax.set_xticklabels(["Back", "Mid-\nCoronal", "Front"])
ax.set_yticks([-1.0, 0.0, 1.0])
ax.set_yticklabels(["Right", "Middle", "Left"])
ax.set_zticks([-1.0, 0.0, 1.0])
ax.set_zticklabels(["Below", "Eye\nLevel", "Above"])
ax.set_title("Speech Audio Sphere\n(Subject 3, Right Ear)")
fig.colorbar(scatter, ax=ax, pad=0.1, label='Final Validation Score SI-SNR (dB)')
def update(frame):
    # Rotate the view (azim parameter controls the horizontal rotation)
    ax.view_init(elev=20., azim=frame)
    return fig,

# Create the animation object
# frames: iterates from 0 to 360 (degrees)
# interval: delay between frames in milliseconds
# blit=True means only things that have changed are drawn (can speed up animation)
anim = FuncAnimation(fig, update, frames=np.arange(0, 361, 2), interval=50, blit=False)

# 4. Save the animation as a GIF
# Requires Pillow (or ImageMagick) as a writer
print("Saving GIF... This might take a moment.")
try:
    anim.save('sub_3_chan_0_rotating_speech_audiosphere.gif', writer='pillow', fps=20)
    print("GIF saved successfully as '3d_plot_rotation.gif'")
except Exception as e:
    print(f"An error occurred while saving the GIF: {e}")
    print("Make sure you have Pillow installed (pip install Pillow).")
plt.close()

###############################
# Clustering data with DBSCAN #
###############################

def cluster_data(df, dbscanEPS=0.38, dbscan_min_samples=12):
    selected_columns = df[['PlotCartX', 'PlotCartY', 'PlotCartZ', 'Final Validation Score SI-SNR (dB)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(selected_columns)
    db = DBSCAN(eps=dbscanEPS, min_samples=dbscan_min_samples).fit(X_scaled)
    selected_columns['DBSCANCluster'] = db.labels_
    return selected_columns



selected_columns = cluster_data(speechAudioSphere, dbscanEPS=0.38, dbscan_min_samples=12)
print(selected_columns.head())
print(selected_columns['DBSCANCluster'].unique())
def dist_from_max_cart(row, sub, maxX, maxY, maxZ):
    srcX = float(row['PlotCartX'])
    srcY = float(row['PlotCartY'])
    srcZ = float(row['PlotCartZ'])
    v1 = np.array([srcX, srcY, srcZ])
    v2 = np.array([maxX, maxY, maxZ])

    # Calculate dot product and magnitudes
    dot_product = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    # Calculate cosine of the angle and clip for stability
    cos_theta = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)

    # Get angle in radians and convert to degrees
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)

maxSpeechIdx = selected_columns['Final Validation Score SI-SNR (dB)'].idxmax()
maxRow = selected_columns.loc[maxSpeechIdx]
print(maxRow)
selected_columns['DistFromMax'] = selected_columns.apply(
        dist_from_max_cart,
        axis=1,
        args=(sub3,
            float(maxRow['PlotCartX']),
            float(maxRow['PlotCartY']),
            float(maxRow['PlotCartZ']))
        )
print(selected_columns.head())
def angle_proj_plane(row, maxX, maxY, maxZ, planeIdxs):
    srcX = float(row['PlotCartX'])
    srcY = float(row['PlotCartY'])
    srcZ = float(row['PlotCartZ'])
    v1 = np.array([srcX, srcY, srcZ])
    v2 = np.array([maxX, maxY, maxZ])
    proj1 = v1[planeIdxs]
    proj2 = v2[planeIdxs]
    angle_rad = np.arctan2(proj2[1], proj2[0]) - np.arctan2(proj1[1], proj1[0])
    # Normalize to [-180, 180]
    angle_deg  = np.degrees(angle_rad)
    angle_deg = (angle_deg + 180) % 360 - 180
    return angle_deg

selected_columns['XYProjAngleDegrees'] = selected_columns.apply(
        angle_proj_plane,
        axis=1,
        args=(
            float(maxRow['PlotCartX']),
            float(maxRow['PlotCartY']),
            float(maxRow['PlotCartZ']), 
            [0,1])
        )
selected_columns['XZProjAngleDegrees'] = selected_columns.apply(
        angle_proj_plane,
        axis=1,
        args=(
            float(maxRow['PlotCartX']),
            float(maxRow['PlotCartY']),
            float(maxRow['PlotCartZ']), 
            [0,2])
        )
selected_columns['YZProjAngleDegrees'] = selected_columns.apply(
        angle_proj_plane,
        axis=1,
        args=(
            float(maxRow['PlotCartX']),
            float(maxRow['PlotCartY']),
            float(maxRow['PlotCartZ']), 
            [1,2])
        )
maxSpeechIdx = selected_columns['Final Validation Score SI-SNR (dB)'].idxmax()
maxRow = selected_columns.loc[maxSpeechIdx]
print(maxRow)
print(selected_columns.head())
XYpos = selected_columns[selected_columns['XYProjAngleDegrees'] > 0.0]
XYneg = selected_columns[selected_columns['XYProjAngleDegrees'] < 0.0]
XZpos = selected_columns[selected_columns['XZProjAngleDegrees'] > 0.0]
XZneg = selected_columns[selected_columns['XZProjAngleDegrees'] < 0.0]
YZpos = selected_columns[selected_columns['YZProjAngleDegrees'] > 0.0]
YZneg = selected_columns[selected_columns['YZProjAngleDegrees'] < 0.0]
print(f"Num points + to XY proj={len(XYpos)}, mean SI-SNR={XYpos['Final Validation Score SI-SNR (dB)'].mean()}, clusters={sorted(XYpos['DBSCANCluster'].unique())}")
print(f"Num points - to XY proj={len(XYneg)}, mean SI-SNR={XYneg['Final Validation Score SI-SNR (dB)'].mean()}, clusters={sorted(XYneg['DBSCANCluster'].unique())}")
print(f"Num points + to XZ proj={len(XZpos)}, mean SI-SNR={XZpos['Final Validation Score SI-SNR (dB)'].mean()}, clusters={sorted(XZpos['DBSCANCluster'].unique())}")
print(f"Num points - to XZ proj={len(XZneg)}, mean SI-SNR={XZneg['Final Validation Score SI-SNR (dB)'].mean()}, clusters={sorted(XZneg['DBSCANCluster'].unique())}")
print(f"Num points + to YZ proj={len(YZpos)}, mean SI-SNR={YZpos['Final Validation Score SI-SNR (dB)'].mean()}, clusters={sorted(YZpos['DBSCANCluster'].unique())}")
print(f"Num points - to YZ proj={len(YZneg)}, mean SI-SNR={YZneg['Final Validation Score SI-SNR (dB)'].mean()}, clusters={sorted(YZneg['DBSCANCluster'].unique())}")
def plot_dbscan_singleton(df):
    fig, axs = plt.subplots(1, 1, figsize=(6, 8))
    noise = df[df['DBSCANCluster'] == -1]
    clusters = df[df['DBSCANCluster'] != -1]

    base_cmap = cm.get_cmap('tab20')
    # Get 11 colors equally spaced from the 20
    subset_colors = base_cmap(np.linspace(0, 0.55, len(clusters['DBSCANCluster'].unique()) - 1)) 
    custom_cmap = ListedColormap(subset_colors)
    # 2. Plot valid clusters (colored by label)
    scatter = axs.scatter(
        clusters['DistFromMax'],
        clusters['Final Validation Score SI-SNR (dB)'],
        c=clusters['DBSCANCluster'],
        cmap=custom_cmap,
        label='Clusters',
        alpha=0.6,
        edgecolors='none'
    )

    # 3. Plot noise as black points
    axs.scatter(
        noise['DistFromMax'],
        noise['Final Validation Score SI-SNR (dB)'],
        c='black',
        marker='x',
        label='Noise',
        alpha=0.5,
        s=20 # Smaller size for noise
    )

    # Formatting
    axs.set_title('DBSCAN Clustering')
    axs.set_xlabel('Angular Distance from Optimal (°)')
    axs.set_ylabel('Validation Score SI-SNR (dB)')
    fig.colorbar(scatter, ax=axs, pad=0.1, label='ClusterId', orientation='horizontal', fraction=0.15, aspect=80)
    axs.legend()
    axs.grid(True, linestyle='--', alpha=0.5)

    plt.savefig("dbscan_singleton.pdf", bbox_inches="tight")
    plt.savefig("dbscan_singleton.png", bbox_inches="tight")
    plt.close()

plot_dbscan_singleton(selected_columns)

def plot_dbscan_results(df):
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))

    # 1. Separate noise and clusters
    noise = df[df['DBSCANCluster'] == -1]
    clusters = df[df['DBSCANCluster'] != -1]
    base_cmap = cm.get_cmap('tab20')
    # Get 11 colors equally spaced from the 20
    subset_colors = base_cmap(np.linspace(0, 0.55, len(clusters['DBSCANCluster'].unique()) - 1)) 
    custom_cmap = ListedColormap(subset_colors)
    def plotDBSCANOnAxis(ax, projection, axTitle, invertAxis):
        scatter = ax.scatter(
            clusters[projection] * invertAxis,
            clusters['Final Validation Score SI-SNR (dB)'],
            c=clusters['DBSCANCluster'],
            cmap=custom_cmap,
            label='Clusters',
            alpha=0.6,
            edgecolors='none'
        )

        ax.scatter(
            noise[projection] * invertAxis,
            noise['Final Validation Score SI-SNR (dB)'],
            c='black',
            marker='x',
            label='Noise',
            alpha=0.5,
            s=20 # Smaller size for noise
        )

        # Formatting
        ax.set_title(axTitle, fontweight='bold')
        ax.set_xlabel('Planar Angular Distance from Optimal (°)')
        ax.set_ylabel('Validation Score SI-SNR (dB)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        return scatter

    scatter = plotDBSCANOnAxis(axs[0], 'XYProjAngleDegrees','a) Horizontal Plane Projection', 1.0)
    scatter = plotDBSCANOnAxis(axs[1], 'XZProjAngleDegrees','b) Sagittal Plane Projection', 1.0)
    scatter = plotDBSCANOnAxis(axs[2], 'YZProjAngleDegrees','c) Coronal Plane Projection', -1.0)

    # Add a colorbar for the clusters and a legend
    fig.colorbar(scatter, ax=axs, pad=0.1, label='ClusterId', orientation='horizontal', fraction=0.15, aspect=80)

    plt.savefig("dbscan.pdf", bbox_inches="tight")
    plt.savefig("dbscan.png", bbox_inches="tight")
    plt.close()

plot_dbscan_results(selected_columns)
