import os, sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

def get_memory_trace(df, percentile=0.5):
    trace = [
        df['STFTPreAllocatedMB'].quantile(percentile),
        df['STFTPeakAllocatedMB'].quantile(percentile),
        df['STFTPostAllocatedMB'].quantile(percentile),
        df['CenteringPreAllocatedMB'].quantile(percentile),
        df['CenteringPeakAllocatedMB'].quantile(percentile),
        df['CenteringPostAllocatedMB'].quantile(percentile),
        df['Dense1PreAllocatedMB'].quantile(percentile),
        df['Dense1PeakAllocatedMB'].quantile(percentile),
        df['Dense1PostAllocatedMB'].quantile(percentile),
        df['Dense2PreAllocatedMB'].quantile(percentile),
        df['Dense2PeakAllocatedMB'].quantile(percentile),
        df['Dense2PostAllocatedMB'].quantile(percentile),
        df['OutputPreAllocatedMB'].quantile(percentile),
        df['OutputPeakAllocatedMB'].quantile(percentile),
        df['OutputPostAllocatedMB'].quantile(percentile),
        df['BackwardsPreAllocatedMB'].quantile(percentile),
        df['BackwardsPeakAllocatedMB'].quantile(percentile),
        df['BackwardsPostAllocatedMB'].quantile(percentile),
        ]
    scaled = [x / 1024.0 for x in trace]
    return scaled

def get_traces(df):
    m50 = get_memory_trace(df, percentile=0.50)

    time_trace = [
        0.0,
        df['STFTMS'].median() / 2.0,
        df['STFTMS'].median(),
        df['STFTMS'].median() + 0.01,
        df['STFTMS'].median() + 0.01 + (df['CenteringMS'].median()/2.0),
        df['STFTMS'].median() + 0.01 + df['CenteringMS'].median(),
        df['STFTMS'].median() + 0.02 + df['CenteringMS'].median(),
        df['STFTMS'].median() + 0.02 + df['CenteringMS'].median() + (df['Dense1MS'].median()/2.0),
        df['STFTMS'].median() + 0.02 + df['CenteringMS'].median() + df['Dense1MS'].median(),
        df['STFTMS'].median() + 0.03 + df['CenteringMS'].median() + df['Dense1MS'].median(),
        df['STFTMS'].median() + 0.03 + df['CenteringMS'].median() + df['Dense1MS'].median() + (df['Dense2MS'].median()/2.0),
        df['STFTMS'].median() + 0.03 + df['CenteringMS'].median() + df['Dense1MS'].median() + df['Dense2MS'].median(),
        df['STFTMS'].median() + 0.04 + df['CenteringMS'].median() + df['Dense1MS'].median() + df['Dense2MS'].median(),
        df['STFTMS'].median() + 0.04 + df['CenteringMS'].median() + df['Dense1MS'].median() + df['Dense2MS'].median() + (df['OutputMS'].median()/2.0),
        df['STFTMS'].median() + 0.04 + df['CenteringMS'].median() + df['Dense1MS'].median() + df['Dense2MS'].median() + df['OutputMS'].median(),
        df['STFTMS'].median() + 0.05 + df['CenteringMS'].median() + df['Dense1MS'].median() + df['Dense2MS'].median() + df['OutputMS'].median(),
        df['STFTMS'].median() + 0.05 + df['CenteringMS'].median() + df['Dense1MS'].median() + df['Dense2MS'].median() + df['OutputMS'].median() + (df['BackwardsMS'].median()/2.0),
        df['STFTMS'].median() + 0.05 + df['CenteringMS'].median() + df['Dense1MS'].median() + df['Dense2MS'].median() + df['OutputMS'].median() + df['BackwardsMS'].median()
        ]
    print(f"STFT: {df['STFTMS'].median()} msec")
    print(f"Input: {df['CenteringMS'].median()} msec")
    print(f"Dense1: {df['Dense1MS'].median()} msec")
    print(f"Dense2: {df['Dense2MS'].median()} msec")
    print(f"Output: {df['OutputMS'].median()} msec")
    print(f"Backwards: {df['BackwardsMS'].median()} msec")
    return (time_trace, m50)

def axis_artistry(ax, t, m, title):
    ax.plot(t, m, color='black', linewidth=2, label='Median Value', zorder=10)

    handles, labels = ax.get_legend_handles_labels()

    ax.axvspan(t[0], t[2], color='red', alpha=0.3, hatch='//')
    vspan_proxy = mpatches.Patch(color='red', alpha=0.3)
    handles.append(vspan_proxy)
    labels.append("STFT")

    ax.axvspan(t[3], t[5], color='brown', alpha=0.3, hatch='oo')
    vspan_proxy = mpatches.Patch(color='brown', alpha=0.3)
    handles.append(vspan_proxy)
    labels.append("Input")

    ax.axvspan(t[6], t[8], color='blue', alpha=0.3, hatch='\\\\')
    vspan_proxy = mpatches.Patch(color='blue', alpha=0.3)
    handles.append(vspan_proxy)
    labels.append("Dense1")

    ax.axvspan(t[9], t[11], color='green', alpha=0.3, hatch='||')
    vspan_proxy = mpatches.Patch(color='green', alpha=0.3)
    handles.append(vspan_proxy)
    labels.append("Dense2")

    ax.axvspan(t[12], t[14], color='yellow', alpha=0.3, hatch='--')
    vspan_proxy = mpatches.Patch(color='yellow', alpha=0.3)
    handles.append(vspan_proxy)
    labels.append("Output")

    ax.axvspan(t[15], t[17], color='purple', alpha=0.3, hatch='xx')
    vspan_proxy = mpatches.Patch(color='purple', alpha=0.3)
    handles.append(vspan_proxy)
    labels.append("Backwards")

    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('Time (msec)', fontsize=12)
    ax.set_ylabel('GPU Memory Usage (GB)', fontsize=12)
    ax.legend(handles=handles, labels=labels, loc='upper left')
    ax.set_xlim(t[0], t[-1])

df = pd.read_csv('ANN_data.csv')
filtered = df[df['Epoch'] == 2]
ann_t, ann_m50,  = get_traces(filtered)
print('ANN stats:')
print(f'\ttime:{ann_t}')
print(f'\tmem:{ann_m50}')
df = pd.read_csv('SNN_data_detailed.csv')
filtered = df[df['Epoch'] == 1]
snn_t, snn_m50,  = get_traces(filtered)
print('SNN stats:')
print(f'\ttime:{snn_t}')
print(f'\tmem:{snn_m50}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))
axis_artistry(ax1, ann_t, ann_m50, 'a) ANN Training Epoch ')
axis_artistry(ax2, snn_t, snn_m50, 'b) SNN Training Epoch ')

plt.savefig("ann_vs_snn.pdf", bbox_inches='tight')
