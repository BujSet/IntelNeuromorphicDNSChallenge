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

# Load the CSV file into a DataFrame
sub_3_chan_0_full_dataset = None
for csv_file in csv_files:
    if ("collated_results" in csv_file):
        sub_3_chan_0_full_dataset = pd.read_csv(csv_file)

sub3 = CipicDatabase.subjects[3]

speechAudioSphere = []
for i in range(1250):
    if ((sub_3_chan_0_full_dataset['Speech Orient'] == i).any()):
        filtered = sub_3_chan_0_full_dataset[sub_3_chan_0_full_dataset['Speech Orient'] == i]
        theta, r = get_plot_theta_r(sub3, i)
        speechAudioSphere.append({
            'CipicIndex':str(i), 
            'Final Validation Score SI-SNR (dB)':filtered['Final Validation Score SI-SNR (dB)'].mean(),
            'PlotPolarThetaRadians':theta,
            'PlotPolarR':r})
speechAudioSphere = pd.DataFrame(speechAudioSphere)
speechAudioSphere['CipicIndex'] = speechAudioSphere['CipicIndex'].astype(int)

noiseAudioSphere = []
for i in range(1250):
    if ((sub_3_chan_0_full_dataset['Noise Orient'] == i).any()):
        filtered = sub_3_chan_0_full_dataset[sub_3_chan_0_full_dataset['Noise Orient'] == i]
        theta, r = get_plot_theta_r(sub3, i)
        noiseAudioSphere.append({
            'CipicIndex':str(i), 
            'Final Validation Score SI-SNR (dB)':filtered['Final Validation Score SI-SNR (dB)'].mean(),
            'PlotPolarThetaRadians':theta,
            'PlotPolarR':r})
noiseAudioSphere = pd.DataFrame(noiseAudioSphere)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), subplot_kw={'projection': 'polar'},
                               gridspec_kw={'wspace': -0.2})
axs[0].set_axisbelow(True)
scatter = axs[0].scatter(speechAudioSphere['PlotPolarThetaRadians'],
                     speechAudioSphere['PlotPolarR'], 
                     c=speechAudioSphere['Final Validation Score SI-SNR (dB)'],
                     cmap='hot', alpha=0.75, zorder=2)
r_offset = 0
rticks = [0+r_offset, 12.5+r_offset, 25+r_offset]
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


axs[0].tick_params(axis='y', labelsize=10, rotation=0)
axs[0].tick_params(axis='x', labelsize=10, pad=11)
custom_ticks_rad = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180.0
custom_labels = ['Front', 'Antero-\nSuperior', 'Up', 'Postero-\nSuperior', 'Back', 'Postero-\nInferior', '', 'Antero-\nInferior'] # Note: 360/0 overlap

axs[0].set_xticks(custom_ticks_rad)
axs[0].set_xticklabels(custom_labels)
axs[0].set_title('a) Speech Audio Sphere')
axs[0].grid(True)


axs[1].set_axisbelow(True)
scatter = axs[1].scatter(noiseAudioSphere['PlotPolarThetaRadians'],
                     noiseAudioSphere['PlotPolarR'], 
                     c=noiseAudioSphere['Final Validation Score SI-SNR (dB)'],
                     cmap='hot', alpha=0.75, zorder=2)
axs[0].set_rorigin(-10)
rticks = [0, 12.5, 25]
rlabels = ['Right', 'Middle', 'Left']
axs[1].set_rgrids(rticks, rlabels, angle=-91)

custom_ticks_rad = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180.0
custom_labels = ['Front', '', 'Up', '', 'Back', '', '', ''] # Note: 360/0 overlap
axs[1].set_xticks(custom_ticks_rad)
axs[1].set_xticklabels(custom_labels)
axs[1].set_title('b) Noise Audio Sphere')
axs[1].grid(True)

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
CS = axs[2].contourf(T, R, Z, levels=5, cmap='hot')
axs[2].set_rorigin(-10)
fig.colorbar(scatter, ax=axs, label='Final Validation Score SI-SNR (dB)', orientation='horizontal', shrink=0.8)
plt.savefig('sub_3_chan_0_speech_audio_sphere.pdf', bbox_inches='tight')
plt.close()

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

print(speechAudioSphere.head())
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(speechAudioSphere['PlotCartX'], 
        speechAudioSphere['PlotCartY'], 
        speechAudioSphere['PlotCartZ'], 
        c=speechAudioSphere['Final Validation Score SI-SNR (dB)'], cmap='viridis', s=50, alpha=0.8)

# Set labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Rotating 3D Scatter Plot')
fig.colorbar(scatter, ax=ax, pad=0.1)
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
    anim.save('3d_plot_rotation.gif', writer='pillow', fps=20)
    print("GIF saved successfully as '3d_plot_rotation.gif'")
except Exception as e:
    print(f"An error occurred while saving the GIF: {e}")
    print("Make sure you have Pillow installed (pip install Pillow).")
plt.close()
