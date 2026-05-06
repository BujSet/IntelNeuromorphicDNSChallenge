import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    ax.set_xlabel("Time (sec)")

def plot_spectrogram(specgram, title=None, ylabel="Frequency (kHz)", ax=None, yticks=[], yticklabels=[]):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (FFT Frame)")
    img = ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    return img

# #  scat = ax.scatter(1, 0)
# x = np.linspace(0, 10)


def animate(scat, x, i):
    # img.set_offsets((x[i], 0)) 
    scat.set_offsets((x[i], 0))
    return scat
    # return img

# ani = animation.FuncAnimation(fig, animate, repeat=True,
#                                     frames=len(x) - 1, interval=50)

# # To save the animation using Pillow as a gif
# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('spec_noisy.gif', writer=writer)

def update_plot(i, data, scat):
    scat.set_array(data[i])
    return scat,

def plot_phase(specgram, title=None, ylabel="Frequency (kHz)", yticks=[], yticklabels=[], filename="dummy", fmap=[]):
    ampls = torch.sqrt(torch.square(specgram.real) + torch.square(specgram.imag))
    phase = torch.atan(torch.div(specgram.imag, specgram.real))
    print(specgram.shape)
    print(fmap.shape)
    print(phase.shape)
    print(len(phase))
    print(phase.shape[1])
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='polar')
    numframes = phase.shape[1]
    numpoints = phase.shape[0]
    # low frequencies are red, high frequencies are blue
    rgbs = [(1.0-(1.0 * i / (numpoints - 1))), 0.0, (1.0 * i / (numpoints - 1)) for i in range(numpoints)]
    # fft_frames = np.linspace(0, phase.shape[1])
    # ani = animation.FuncAnimation(fig, animate, repeat=True,
                                    # frames=len(x) - 1, interval=50)
    for i in range(phase.shape[0]):
        w = round(fmap[i] / 1000.0,2)
        color  = (1.0 * i / (phase.shape[0] - 1))
        rgb = [(1.0-color), 0.0, color]
        theta =  phase[i].numpy() * 180.0 / np.pi
        # print(theta.shape)
        r = ampls[i].numpy()
        # if (i % 16 == 0):
        #     c = ax.scatter(theta[0], r[0], color=rgb, s=2.0, label=str(w)+" kHz")
        # else: 
        #     c = ax.scatter(theta[0], r[0], color=rgb, s=2.0)
        c = ax.scatter(theta[0], r[0], color=rgb, s=2.0)

    ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(color_data, scat))
    writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
    ani.save('spec_noisy.gif', writer=writer)
    plt.legend()
    plt.savefig(filename + "_frame_0.png", bbox_inches='tight')
    plt.close()


NOISY_FILE = './training_set/noisy/SLR61_es_ar_female_arf_06592_00774535859_48kHz-l_ile_mysterieuse_2_06_f000145-book_00394_chp_0008_re_f872gDrfumM-fan_Freesound_validated_437341_3-door_Freesound_validated_349356_0_snr16_tl-23_fileid_1542.wav'
CLEAN_FILE = './training_set/clean/clean_fileid_1542.wav'
NOISE_FILE = './training_set/noise/noise_fileid_1542.wav'
METADATA = {'snr': 16, 'target_level': -23}


noisy, nysr = torchaudio.load(NOISY_FILE)
clean, clsr = torchaudio.load(CLEAN_FILE)
noise, nesr = torchaudio.load(NOISE_FILE)

# Define transform
spectrogram = T.Spectrogram(
        n_fft=512,
        power=None)

# Perform transform
spec_noisy = spectrogram(noisy).squeeze()
print(spec_noisy.shape)
spec_clean = spectrogram(clean)
spec_noise = spectrogram(noise)

freq_map = librosa.fft_frequencies(sr=16000, n_fft=512)
yticks = [i for i in range(0, len(freq_map), int(len(freq_map)/5))]
yticklabels = [round(freq_map[i]/1000.0, 2) for i in yticks]

#fig, axs = plt.subplots(2, 1)
# plot_waveform(noisy, nysr, title="Original waveform", ax=axs[0])
# plot_spectrogram(spec_noisy, title="Power Spectrogram", ax=axs[1], yticks=yticks, yticklabels=yticklabels)
# fig.tight_layout()
# plt.savefig("spec_noisy.png")
# plt.close()
# plot_phase(spec_noisy, title="Phase Spectrogram", yticks=yticks, yticklabels=yticklabels, filename="spec_noisy_phase", fmap=freq_map)

# fig, axs = plt.subplots(2, 1)
# plot_waveform(clean, clsr, title="Original waveform", ax=axs[0])
# plot_spectrogram(spec_clean[0], title="spectrogram", ax=axs[1], yticks=yticks, yticklabels=yticklabels)
# fig.tight_layout()
# plt.savefig("spec_clean.png")
# plt.close()
# plot_phase(spec_clean, title="Phase Spectrogram", yticks=yticks, yticklabels=yticklabels, filename="spec_clean_phase.png")

# fig, axs = plt.subplots(2, 1)
# plot_waveform(noise, nesr, title="Original waveform", ax=axs[0])
# plot_spectrogram(spec_noise[0], title="spectrogram", ax=axs[1], yticks=yticks, yticklabels=yticklabels)
# fig.tight_layout()
# plt.savefig("spec_noise.png")
# plt.close()
# plot_phase(spec_noise, title="Phase Spectrogram", yticks=yticks, yticklabels=yticklabels, filename="spec_noise_phase.png")



# img = plot_spectrogram(spec_noisy, title="Power Spectrogram", ax=axs[1], yticks=yticks, yticklabels=yticklabels)
# # fig, ax = plt.subplots()
# # ax.set_xlim([0, 10])




# fps = 30
# nSeconds = 5
# snapshots = [ np.random.rand(5,5) for _ in range( nSeconds * fps ) ]

# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure( figsize=(8,8) )

# a = snapshots[0]
# im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

# def animate_func(i):
#     if i % fps == 0:
#         print( '.', end ='' )

#     im.set_array(snapshots[i])
#     return [im]

# anim = animation.FuncAnimation(
#                                fig, 
#                                animate_func, 
#                                frames = nSeconds * fps,
#                                interval = 1000 / fps, # in ms
#                                )

# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# anim.save('spec_noisy.gif', writer=writer)
# # anim.save('test_anim.mp4', fps=fps)#, extra_args=['-vcodec', 'libx264'])

# print('Done!')
# # plt.show()

def main():
    numframes = 100
    numpoints = 10
    color_data = np.random.random((numframes, numpoints))
    x, y, c = np.random.random((3, numpoints))

    fig = plt.figure()
    scat = plt.scatter(x, y, c=c, s=100)

    ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(color_data, scat))
    writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
    ani.save('spec_noisy.gif', writer=writer)
    # plt.show()



main()