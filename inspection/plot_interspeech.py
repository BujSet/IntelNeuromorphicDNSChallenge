import os, sys
sys.path.append('./')
import glob
import pandas as pd
from hrtfs.cipic_db import CipicDatabase 
import math
import matplotlib.pyplot as plt
import numpy as np

csv_files = glob.glob('*.csv')
out_files = glob.glob('*.out')


def get_plot_theta_r(sub, index):
    modulo = index % 50
    pos = sub.getSphericalPositionsFromIndex(600 + modulo)
    angle = pos[1]
    if modulo < 8:
        angle = angle + 360.0
    elif modulo > 24 and modulo <= 48:
        angle = 180.0 - angle
    elif modulo > 48:
        angle = 180.0 - angle
    return math.radians(angle), index //50


# Load the CSV file into a DataFrame
sub_3_chan_0_full_dataset = None
for csv_file in csv_files:
    if ("collated_results" in csv_file):
        sub_3_chan_0_full_dataset = pd.read_csv(csv_file)

sub3 = CipicDatabase.subjects[3]

print(sub_3_chan_0_full_dataset.head())
print(sub_3_chan_0_full_dataset.tail())
print("Initial rows: " + str(len(sub_3_chan_0_full_dataset)))

filtered = sub_3_chan_0_full_dataset[sub_3_chan_0_full_dataset['Speech Orient'] == 0]
print("Filtered rows: " + str(len(filtered)))
print(filtered.head())
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
print(speechAudioSphere.head())
print(speechAudioSphere.tail())
print(len(speechAudioSphere))

        

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), subplot_kw={'projection': 'polar'})
ax.set_axisbelow(True)
scatter = ax.scatter(speechAudioSphere['PlotPolarThetaRadians'],
                     speechAudioSphere['PlotPolarR'], 
                     c=speechAudioSphere['Final Validation Score SI-SNR (dB)'],
                     cmap='hot', alpha=0.75, zorder=2)
rticks = [5, 12.5, 25]
rlabels = ['Right', 'Middle', 'Left']
ax.set_rgrids(rticks, rlabels, angle=-91)
custom_ticks_rad = np.array([0, 90, 180, 270]) * np.pi / 180.0
custom_labels = ['Front', 'Up', 'Back', ''] # Note: 360/0 overlap

# 3. Set the tick locations
ax.set_xticks(custom_ticks_rad)

# 4. Set the tick labels
ax.set_xticklabels(custom_labels)
ax.set_title('Speech Audio Sphere')
ax.grid(True)
plt.colorbar(scatter, ax=ax, label='Final Validation Score SI-SNR (dB)', orientation='horizontal')
plt.savefig('sub_3_chan_0_speech_audio_sphere.pdf', bbox_inches='tight')
plt.close()
