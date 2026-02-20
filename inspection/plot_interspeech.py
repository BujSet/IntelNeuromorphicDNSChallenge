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

csv_files = glob.glob('*.csv')
out_files = glob.glob('*.out')


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
    out_data = read_out_file(out_file)
    all_data = pd.concat([all_data, out_data], ignore_index=True)
    duplicate_rows_boolean = all_data.duplicated()
    num_duplicates = duplicate_rows_boolean.sum()
    if num_duplicates:
        print(f"All data conatains {len(all_data)} rows after merge {i}, {num_duplicates} duplicates")
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

def plotMonauralContourMaps(df, subjectSet, numRows=3, numCols=8):
    subList = sorted(list(subjectSet))
    if len(subList) > (numRows*numCols):
        subLst = subList[0:numRows*numCols]
    print(len(subList))
    def plotContourOnAxis(ax, subject, channel, axTitle):
        num_points = 500
        # TODO when we have more data on the subset data, samples=120, replace
        # 60000 with 120
        speech = getPlotDataForAudioSphere(df, subject, channel, 60000, True)
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
        CS = ax.contourf(T, R, Z, levels=5, cmap='hot')
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
        custom_labels = ['Front', 'Antero-\nSuperior', '\nUp', 'Postero-\nSuperior', 'Back', 'Postero-\nInferior', '', 'Antero-\nInferior'] # Note: 360/0 overlap

        ax.set_xticks(custom_ticks_rad)
        ax.set_xticklabels(custom_labels)
        ax.set_title(axTitle)
        ax.grid(True)
        ax.set_rorigin(-10)
        return CS

    fig, axs = plt.subplots(nrows=numRows, ncols=numCols, 
            figsize=(20, 10), subplot_kw={'projection': 'polar'},
                               gridspec_kw={'wspace': -0.0}, layout="constrained")
    titles = [
            "a) Subject 3\n(Left)",
            "b) Subject 3\n(Right)",
            "c) Subject 3\n(Left)",
            "d) Subject 3\n(Right)",
            "f) Subject 3\n(Left)",
            "g) Subject 3\n(Rightt)",
            "h) Subject 3\n(Left)",
            "i) Subject 3\n(Right)",
            "j) Subject 3\n(Left)",
            "k) Subject 3\n(Right)",
            "l) Subject 3\n(Left)",
            "m) Subject 3\n(Right)",
            "n) Subject 3\n(Left)",
            "o) Subject 3\n(Right)",
            "p) Subject 3\n(Left)",
            "q) Subject 3\n(Right)",
            "r) Subject 3\n(Left)",
            "s) Subject 3\n(Right)", 
            "t) Subject 3\n(Left)",
            "u) Subject 3\n(Right)",
            "v) Subject 3\n(Left)",
            "w) Subject 3\n(Right)",
            "x) Subject 3\n(Left)",
            "y) Subject 3\n(Right)"]
    for r in range(numRows):
        for c in range(numCols):
            idx = (r*numCols)+c
            cf = plotContourOnAxis(axs[r,c], subList[idx], (idx+1)%2, titles[idx])
    # TODO need to collect more data, will remove line below once I do
    cf = plotContourOnAxis(axs[0,1], 3, 0, "b) Subject 3\n(Right)")
    fig.colorbar(cf, ax=axs, label='Validation Score SI-SNR (dB)', orientation='horizontal', shrink=0.9, aspect=50)
    plt.savefig(f'contours.png', bbox_inches='tight')
    plt.close()


plotMonauralContourMaps(all_data, subjectsWithBothChannels, numRows=2, numCols=4)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 12), subplot_kw={'projection': 'polar'},
                               gridspec_kw={'wspace': -0.0})
axs.set_axisbelow(True)
#scatter = axs[0].scatter(speechAudioSphere['PlotPolarThetaRadians'],
#                     speechAudioSphere['PlotPolarR'], 
#                     c=speechAudioSphere['Final Validation Score SI-SNR (dB)'],
#                     cmap='hot', alpha=0.75, zorder=2)
rticks = [0, 12.5, 25]
rlabels = ['Right', 'Middle', 'Left']
rlines, rlabels = axs[0].set_rgrids(rticks, rlabels, angle=-90)
for i, label in enumerate(rlabels):
    label.set_horizontalalignment('center') 
    if i == 0:
        label.set_verticalalignment('top') 
    elif i == 2:
        label.set_verticalalignment('bottom') 
    else:
        label.set_verticalalignment('center') 


axs.tick_params(axis='y', labelsize=10, rotation=0)
axs.tick_params(axis='x', labelsize=10, pad=11)
custom_ticks_rad = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180.0
custom_labels = ['Front', 'Antero-\nSuperior', 'Up', 'Postero-\nSuperior', 'Back', 'Postero-\nInferior', '', 'Antero-\nInferior'] # Note: 360/0 overlap

axs.set_xticks(custom_ticks_rad)
axs.set_xticklabels(custom_labels)
axs.set_title('Subject 3 (Left)')
axs.grid(True)


#axs[1].set_axisbelow(True)
#scatter = axs[1].scatter(noiseAudioSphere['PlotPolarThetaRadians'],
#                     noiseAudioSphere['PlotPolarR'], 
#                     c=noiseAudioSphere['Final Validation Score SI-SNR (dB)'],
#                     cmap='hot', alpha=0.75, zorder=2)
axs.set_rorigin(-10)
#rticks = [0, 12.5, 25]
#rlabels = ['Right', 'Middle', 'Left']
#axs[1].set_rgrids(rticks, rlabels, angle=-91)
#
#custom_ticks_rad = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180.0
#custom_labels = ['Front', '', 'Up', '', 'Back', '', '', ''] # Note: 360/0 overlap
#axs[1].set_xticks(custom_ticks_rad)
#axs[1].set_xticklabels(custom_labels)
#axs[1].set_title('b) Noise Audio Sphere')
#axs[1].grid(True)

num_points = 500
grid_r = np.linspace(speechAudioSphere['PlotPolarR'].min(), speechAudioSphere['PlotPolarR'].max(), num_points)
grid_t = np.linspace(0, 360, num_points) * np.pi / 180
R, T = np.meshgrid(grid_r, grid_t)
points = speechAudioSphere[['PlotPolarR', 'PlotPolarThetaRadians']].values
values = speechAudioSphere['Final Validation Score SI-SNR (dB)'].values
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
CS = axs.contourf(T, R, Z, levels=5, cmap='hot')
axs.set_rorigin(-10)
fig.colorbar(scatter, ax=axs, label='Validation Score SI-SNR (dB)', orientation='horizontal', shrink=0.8)
plt.savefig('sub_3_chan_0_speech_contour.png', bbox_inches='tight')
plt.close()

sys.exit(0)

def get_dist_from_max(row, sub, maxIdx):
    destIdx = row['CipicIndex']
    return sub.chordDistBetweenIndices(maxIdx, int(destIdx))

maxSpeechIdx = speechAudioSphere['Final Validation Score SI-SNR (dB)'].idxmax()
maxRow = speechAudioSphere.loc[maxSpeechIdx]
speechAudioSphere['DistFromMax'] = speechAudioSphere.apply(
        get_dist_from_max,
        axis=1,
        args=(sub3, int(maxRow['CipicIndex']))
        )
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
axs.scatter(x=speechAudioSphere['DistFromMax'], y=speechAudioSphere['Final Validation Score SI-SNR (dB)'], zorder=3, s=2)

# Optional: Add customizations
axs.set_title('Scatter Plot using Matplotlib')
axs.set_xlabel('X-axis Label')
axs.set_ylabel('Y-axis Label')
axs.grid(True)
axs.set_axisbelow(True)

# Display the plot
plt.savefig("sub_3_chan_0_speech_sorted.pdf", bbox_inches='tight')
plt.close()

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
#fig.tight_layout(pad=0)
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

# DBSCAN stuff
selected_columns = speechAudioSphere[['PlotCartX', 'PlotCartY', 'PlotCartZ', 'Final Validation Score SI-SNR (dB)']]
print(selected_columns.head())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(selected_columns)
db = DBSCAN(eps=0.38, min_samples=12).fit(X_scaled)

selected_columns['DBSCANCluster'] = db.labels_

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

maxSpeechIdx = selected_columns['Final Validation Score SI-SNR (dB)'].idxmax()
maxRow = selected_columns.loc[maxSpeechIdx]
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
print(selected_columns.head())
def plot_dbscan_results(df):
    fig, axs = plt.subplots(1, 4, figsize=(30, 8))

    # 1. Separate noise and clusters
    noise = df[df['DBSCANCluster'] == -1]
    clusters = df[df['DBSCANCluster'] != -1]

    # 2. Plot valid clusters (colored by label)
    scatter = axs[0].scatter(
        clusters['DistFromMax'],
        clusters['Final Validation Score SI-SNR (dB)'],
        c=clusters['DBSCANCluster'],
        cmap='turbo',
        label='Clusters',
        alpha=0.6,
        edgecolors='none'
    )

    # 3. Plot noise as black points
    axs[0].scatter(
        noise['DistFromMax'],
        noise['Final Validation Score SI-SNR (dB)'],
        c='black',
        marker='x',
        label='Noise (-1)',
        alpha=0.5,
        s=20 # Smaller size for noise
    )

    # Formatting
    axs[0].set_title('DBSCAN Clustering')
    axs[0].set_xlabel('Angular Distance from Maximum (degree)')
    axs[0].set_ylabel('Validation Score SI-SNR (dB)')

    scatter = axs[1].scatter(
        clusters['XYProjAngleDegrees'],
        clusters['Final Validation Score SI-SNR (dB)'],
        c=clusters['DBSCANCluster'],
        cmap='turbo',
        label='Clusters',
        alpha=0.6,
        edgecolors='none'
    )

    # 3. Plot noise as black points
    axs[1].scatter(
        noise['XYProjAngleDegrees'],
        noise['Final Validation Score SI-SNR (dB)'],
        c='black',
        marker='x',
        label='Noise (-1)',
        alpha=0.5,
        s=20 # Smaller size for noise
    )

    # Formatting
    axs[1].set_title('DBSCAN Clustering XY Projection')
    axs[1].set_xlabel('XY Planar Angular Distance from Maximum (degree)')
    axs[1].set_ylabel('Validation Score SI-SNR (dB)')
    scatter = axs[2].scatter(
        clusters['XZProjAngleDegrees'],
        clusters['Final Validation Score SI-SNR (dB)'],
        c=clusters['DBSCANCluster'],
        cmap='turbo',
        label='Clusters',
        alpha=0.6,
        edgecolors='none'
    )

    # 3. Plot noise as black points
    axs[2].scatter(
        noise['XZProjAngleDegrees'],
        noise['Final Validation Score SI-SNR (dB)'],
        c='black',
        marker='x',
        label='Noise (-1)',
        alpha=0.5,
        s=20 # Smaller size for noise
    )

    # Formatting
    axs[2].set_title('DBSCAN Clustering XZ Projection')
    axs[2].set_xlabel('XZ Planar Angular Distance from Maximum (degree)')
    axs[2].set_ylabel('Validation Score SI-SNR (dB)')
    scatter = axs[3].scatter(
        clusters['YZProjAngleDegrees'],
        clusters['Final Validation Score SI-SNR (dB)'],
        c=clusters['DBSCANCluster'],
        cmap='turbo',
        label='Clusters',
        alpha=0.6,
        edgecolors='none'
    )

    # 3. Plot noise as black points
    axs[3].scatter(
        noise['YZProjAngleDegrees'],
        noise['Final Validation Score SI-SNR (dB)'],
        c='black',
        marker='x',
        label='Noise (-1)',
        alpha=0.5,
        s=20 # Smaller size for noise
    )

    # Formatting
    axs[3].set_title('DBSCAN Clustering YZ Projection')
    axs[3].set_xlabel('YZ Planar Angular Distance from Maximum (degree)')
    axs[3].set_ylabel('Validation Score SI-SNR (dB)')

    # Add a colorbar for the clusters and a legend
    fig.colorbar(scatter, ax=axs, pad=0.1, label='ClusterId', orientation='horizontal', fraction=0.15, aspect=80)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)

    plt.savefig("dbscan.pdf", bbox_inches="tight")
    plt.close()

# Run the plot
plot_dbscan_results(selected_columns)
print("Min XY: " + str(selected_columns['XYProjAngleDegrees'].min()))
print("Min XZ: " + str(selected_columns['XZProjAngleDegrees'].min()))
print("Min YZ: " + str(selected_columns['YZProjAngleDegrees'].min()))
