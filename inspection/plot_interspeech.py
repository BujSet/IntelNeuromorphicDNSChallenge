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
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

csv_files = glob.glob('*.csv')
out_files = sorted(glob.glob('*.out'))

left_ear_img_data = mpimg.imread('stock_images/left_ear.jpg')
right_ear_img_data = mpimg.imread('stock_images/right_ear.jpg')

def get_plot_theta_r(sub, index, channel=0):
    modulo = index % 50
    r = (index // 50 )
    #if channel == 1:
    #    r = 25 - r
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
            #if dataSubsetSize != 60000:
            #    print(row.head())
            #    sys.exit(0)
            theta, r = get_plot_theta_r(CipicDatabase.subjects[subject], i, channel)
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
    def plotSphereOnAxis(ax, df, axTitle, channel=0):
        ax.set_axisbelow(True)
        scatter = ax.scatter(df['PlotPolarThetaRadians'],
                     df['PlotPolarR'], 
                     c=df['Final Validation Score SI-SNR (dB)'],
                     cmap='hot', alpha=0.75, zorder=2)
        rticks = [0, 12.5, 25]
        rlabels = ['Right', 'Middle', 'Left']
        #if (channel == 1):
        #    rlabels = ['Left', 'Middle', 'Right']
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
    scatter = plotSphereOnAxis(axs[0], speech, "a) Speech Audiosphere", channel)
    scatter = plotSphereOnAxis(axs[1], noise, "a) Noise Audiosphere", channel)
    titleString = f"Subject {subject}'s Audiospheres ("
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


def check_for_nans(df, subject, channel, samples, isSpeech=True):
    filtered = getPlotDataForAudioSphere(df, subject, channel, samples, isSpeech)
    hasNan = filtered['Final Validation Score SI-SNR (dB)'].isna().any()
    if hasNan:
        rows_with_nan = filtered[filtered['Final Validation Score SI-SNR (dB)'].isna()]
        with pd.option_context('display.max_rows', None):
            print(rows_with_nan)
    else:
        print(f"Subject:{subject} channel:{channel} samples:{samples} speech:{isSpeech} has no Nans")

def check_for_missing_data(df, subject, channel, samples):
    filtered = df[(df['Subject'] == subject) & (df['Channel'] == channel)]
    if samples == 60000:
        filtered = filtered[(filtered['UsesFullAudioDataset'] == True) &
        (filtered['AudioDataSubsetSize'] == 60000)]
    else:
        filtered = filtered[(filtered['UsesFullAudioDataset'] == False) &
        (filtered['AudioDataSubsetSize'] == samples)]

    num_rows = len(filtered)
    print(f"Subject:{subject}, channel:{channel}, samples:{samples} has {num_rows} rows of data")
    num_missing = 0
    # for subject 3, chan 0, samples 120, skipp the first 400 because we know
    # these are already complete
    for so in range(400,1250):
        row_data = filtered[filtered['Speech Orient'] == so]
        print(f"Dataframe from speech orient:{so} containes {len(row_data)} rows")
        for no in range(1250):
            matches = len(filtered[(filtered['Speech Orient'] == so) & (filtered['Noise Orient'] == no)])
            if (matches == 0):
                print(f"({so},{no})")
                num_missing += 1
    if num_missing > 0:
        print(f"Subject:{subject}, channel:{channel}, samples:{samples} missing {num_missing} data points")


print("Checking for missing data...")
check_for_missing_data(all_data, 3, 0, 120)
sys.exit(0)


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

check_for_nans(all_data, 8, 0, 120, True)
check_for_nans(all_data, 8, 1, 120, True)

inspectSub = 12
sub_3_chan_0 = getPlotDataForAudioSphere(all_data, inspectSub, 0, 120, True)
sub_3_chan_1 = getPlotDataForAudioSphere(all_data, inspectSub, 1, 120, True)
print(sub_3_chan_0.head())
print(len(sub_3_chan_0))
print(sub_3_chan_1.head())
print(len(sub_3_chan_1))

def getZMinMax(subject, channel):
    global all_data
    num_points = 500
    speech = getPlotDataForAudioSphere(all_data, subject, channel, 120, True)

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
    Zmin = Z[(T <= t_min_hide) | (T > t_max_hide)].min()
    Zmax = Z[(T <= t_min_hide) | (T > t_max_hide)].max()
    return Zmin, Zmax

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

    level_list = np.linspace(cmapMin, cmapMax, num_levels + 1)
    CS = ax.contourf(T, R, Z, levels=level_list, cmap='hot', zorder=2)

    rticks = [0, 12.5, 24]
    rlabels = ['Right', 'Middle', 'Left']
    #if (channel == 1):
    #    rlabels = ['Left', 'Middle', 'Right']
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

    imagebox = OffsetImage(right_ear_img_data, zoom=0.15)
    #if channel == 1:
    #    imagebox = OffsetImage(left_ear_img_data, zoom=0.15)
    ab = AnnotationBbox(imagebox, (0.5, 0.5), xycoords='axes fraction',
                    boxcoords="axes fraction", box_alignment=(0.5, 0.5), frameon=False)
    ab.set_zorder(0) 
    ax.add_artist(ab)
    return CS

def plotMonauralContourMaps(df, subjectSet, numRows=3, numCols=8):
    subList = sorted(list(subjectSet))
    if len(subList) > (numRows*numCols):
        subLst = subList[0:numRows*numCols]
    print(len(subList))

    fig, axs = plt.subplots(nrows=numRows, ncols=numCols, 
            figsize=(20, 8), 
            subplot_kw={'projection': 'polar'}, 
            layout="constrained")
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
            fmin,fmax = getZMinMax(subject, channelVal)
            if contoursMin == None or fmin < contoursMin:
                contoursMin = fmin
            if contoursMax == None or fmax > contoursMax:
                contoursMax = fmax

    for r in range(numRows):
        for c in range(numCols):
            idx = (r*numCols)+c
            subIdx = (idx//2)
            cf = plotContourOnAxis(axs[r,c], subList[subIdx], (idx+1)%2, titles[idx], contoursMin, contoursMax, 6)
    fig.colorbar(cf, ax=axs, label='Validation Score SI-SNR (dB)', orientation='horizontal', shrink=0.99, aspect=50)
    plt.savefig('contours.png', bbox_inches='tight', transparent=True)
    plt.savefig('contours.pdf', bbox_inches='tight', transparent=True)
    plt.close()

plotMonauralContourMaps(all_data, subjectsWithBothChannels, numRows=2, numCols=6)

def plotSingletonContourMap(subject, channel, num_levels=6):
    fig, axs = plt.subplots(nrows=1, 
        ncols=1, 
        figsize=(12, 12),
        subplot_kw={'projection': 'polar'})
    titleString = f"Subject {subject} "
    if (channel == 0):
        titleString += "(Right Ear)"
    else:
        titleString += "(Left Ear)"


    fmin,fmax = getZMinMax(subject, channel)
    cf = plotContourOnAxis(axs, subject, channel, titleString, fmin, fmax, num_levels) 
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

def plot_3d_audiosphere_static():
    fig, axs = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
    scatter = axs[0].scatter(speechAudioSphere['PlotCartX'], 
        speechAudioSphere['PlotCartY'], 
        speechAudioSphere['PlotCartZ'], 
        c=speechAudioSphere['Final Validation Score SI-SNR (dB)'],
        cmap='plasma', s=50, alpha=0.8)

    axs[0].set_xticks([-0.5, 0.5])
    axs[0].set_xticklabels(["Back", "Front"])
    axs[0].set_yticks([-0.5, 0.5])
    axs[0].set_yticklabels(["Right", "Left"])
    axs[0].set_zticks([-0.5, 0.5])
    axs[0].set_zticklabels(["Below", "Above"])
    axs[0].set_title("a) Viewing \u0398=315\u00b0", fontweight='bold')
    axs[0].view_init(elev=20., azim=315)
    scatter = axs[1].scatter(speechAudioSphere['PlotCartX'], 
        speechAudioSphere['PlotCartY'], 
        speechAudioSphere['PlotCartZ'], 
        c=speechAudioSphere['Final Validation Score SI-SNR (dB)'],
        cmap='plasma', s=50, alpha=0.8)

    axs[1].set_xticks([-0.5, 0.5])
    axs[1].set_xticklabels(["Back", "Front"])
    axs[1].set_yticks([-0.5, 0.5])
    axs[1].set_yticklabels(["Right", "Left"])
    axs[1].set_zticks([-0.5, 0.5])
    axs[1].set_zticklabels([])
    axs[1].set_title("b) Viewing \u0398=45\u00b0", fontweight='bold')
    axs[1].view_init(elev=20., azim=45)
    fig.suptitle('Speech Audiosphere', fontsize=16, fontweight='bold', y=0.90)
    fig.colorbar(scatter, ax=axs, pad=0.1, label='Validation Score SI-SNR (dB)', orientation='horizontal')
    plt.savefig("passive_pinna_sub3_chan0_3D.pdf", bbox_inches="tight")
    plt.close()

plot_3d_audiosphere_static()

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
ax.set_title("Speech Audiosphere\n(Subject 3, Right Ear)")
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
selected_columns['DistFromMax'] = selected_columns.apply(
        dist_from_max_cart,
        axis=1,
        args=(sub3,
            float(maxRow['PlotCartX']),
            float(maxRow['PlotCartY']),
            float(maxRow['PlotCartZ']))
        )
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
corr_angle_score = selected_columns['Final Validation Score SI-SNR (dB)'].corr(selected_columns['DistFromMax'])
corr_XYangle_score = selected_columns['Final Validation Score SI-SNR (dB)'].corr(selected_columns['XYProjAngleDegrees'])
corr_XZangle_score = selected_columns['Final Validation Score SI-SNR (dB)'].corr(selected_columns['XZProjAngleDegrees'])
corr_YZangle_score = selected_columns['Final Validation Score SI-SNR (dB)'].corr(selected_columns['YZProjAngleDegrees'])
print(f"Pearson correlation between SI-SNR and angular dist={corr_angle_score}")
print(f"Pearson correlation between SI-SNR and XYangular dist={corr_XYangle_score}")
print(f"Pearson correlation between SI-SNR and XZangular dist={corr_XZangle_score}")
print(f"Pearson correlation between SI-SNR and YZangular dist={corr_YZangle_score}")
maxSpeechIdx = selected_columns['Final Validation Score SI-SNR (dB)'].idxmax()
maxRow = selected_columns.loc[maxSpeechIdx]
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
