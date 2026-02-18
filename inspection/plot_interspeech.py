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

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), subplot_kw={'projection': 'polar'},
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
r_offset = 0
axs[0].set_rorigin(-10)
rticks = [0+r_offset, 12.5+r_offset, 25+r_offset]
rlabels = ['Right', 'Middle', 'Left']
axs[1].set_rgrids(rticks, rlabels, angle=-91)

custom_ticks_rad = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180.0
custom_labels = ['Front', '', 'Up', '', 'Back', '', '', ''] # Note: 360/0 overlap
axs[1].set_xticks(custom_ticks_rad)
axs[1].set_xticklabels(custom_labels)
axs[1].set_title('b) Noise Audio Sphere')
axs[1].grid(True)
fig.colorbar(scatter, ax=axs, label='Final Validation Score SI-SNR (dB)', orientation='horizontal', shrink=0.8)
plt.savefig('sub_3_chan_0_speech_audio_sphere.pdf', bbox_inches='tight')
plt.close()
