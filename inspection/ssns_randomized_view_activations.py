# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os, sys, math
sys.path.append('./')
from audio_dataloader import DNSAudio
from audio_dataloader import DNSAudioNoNoisy
from audio_dataloader import DNSAudioNoNoise
from hrtfs.cipic_db import CipicDatabase 
import h5py
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf

from lava.lib.dl import slayer
from snr import si_snr
import torchaudio
from noisyspeech_synthesizer import segmental_snr_mixer
import random
import librosa
import re

def stft_splitter(audio, n_fft=512, method=None):
    with torch.no_grad():
        if (method == None):
            audio_stft = torch.stft(audio,
                                n_fft=n_fft,
                                onesided=True,
                                return_complex=True)
            return audio_stft.abs(), audio_stft.angle()
        spec = method(audio)
        return spec.abs(), spec.angle()    

def stft_mixer(stft_abs, stft_angle, n_fft=512, method=None):
    spec = torch.complex(stft_abs * torch.cos(stft_angle),
                                        stft_abs * torch.sin(stft_angle))
    if (method == None):
        return torch.istft(spec, n_fft=n_fft, onesided=True)
    if (type(method) == int):
        print("Perform inver mel scale transform")
        sys.exit(0)

    return method(spec)

class Network(torch.nn.Module):
    def __init__(self, 
            threshold=0.1, 
            tau_grad=0.1, 
            scale_grad=0.8, 
            max_delay=64, 
            out_delay=0,
            hiddenLayerWidths=512,
            n_fft=512):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay
        self.EPS = 2.220446049250313e-16
        self.hiddenLayerWidths = hiddenLayerWidths

        sigma_params = { # sigma-delta neuron parameters
            'threshold'     : threshold,   # delta unit threshold
            'tau_grad'      : tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,  # trainable threshold
            'shared_param'  : True,   # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : F.relu, # activation function
        }

        self.input_quantizer = lambda x: slayer.utils.quantize(x, step=1 / 64)

        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Dense(sdnn_params, n_fft//2 + 1, hiddenLayerWidths, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Dense(sdnn_params, hiddenLayerWidths, hiddenLayerWidths, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Output(sdnn_params, hiddenLayerWidths, n_fft//2 + 1, weight_norm=False),
        ])

        self.blocks[0].pre_hook_fx = self.input_quantizer

        self.blocks[1].delay.max_delay = max_delay
        self.blocks[2].delay.max_delay = max_delay

    def _segmental_snr_mixer(self, clean, noise, snr,
                        target_level, 
                        target_level_lower,
                        target_level_higher,
                        clipping_threshold=0.99,
                        ):
        '''Function to mix clean speech and noise at various segmental SNR levels'''
        clean_div = torch.max(torch.abs(clean)) + self.EPS
        noise_div = torch.max(torch.abs(noise)) + self.EPS
        ssl_clean = torch.div(clean, clean_div.item())
        ssl_noise = torch.div(noise, noise_div.item())
        # TODO should only calculate the RMS of the 'active' windows, but
        # for now we just use the whole audio sample
        clean_rms = torch.sqrt(torch.mean(torch.square(ssl_clean))).item()
        noise_rms = torch.sqrt(torch.mean(torch.square(ssl_noise))).item()
        clean_scalar = 10 ** (target_level / 20) / (clean_rms + self.EPS)
        noise_scalar = 10 ** (target_level / 20) / (noise_rms + self.EPS)
        ssl_clean = torch.mul(ssl_clean, clean_scalar)
        ssl_noise = torch.mul(ssl_noise, noise_scalar)
        # Adjust noise to SNR level
        noise_scalar = clean_rms / (10**(snr/20)) / (noise_rms+self.EPS)
        ssl_noise = torch.mul(ssl_noise, noise_scalar)
        ssl_noisy = torch.add(ssl_clean, ssl_noise)
        noisy_rms_level = torch.randint(
                target_level_lower,
                target_level_higher,
                (1,))
        noisy_rmsT = torch.sqrt(torch.mean(torch.square(ssl_noisy)))
        noisy_rms = torch.sqrt(torch.mean(torch.square(ssl_noisy))).item()
        noisy_scalar = 10 ** (noisy_rms_level / 20) / (noisy_rms + self.EPS)
        ssl_noisy = torch.mul(ssl_noisy, noisy_scalar.item())
        ssl_clean = torch.mul(ssl_clean, noisy_scalar.item())
        ssl_noise = torch.mul(ssl_noise, noisy_scalar.item())
        # check if any clipping happened
        needToClip = torch.gt(torch.abs(ssl_noisy), 0.99).any() # 0.99 is the clipping threshold 
        if (needToClip):
            noisyspeech_maxamplevel = torch.max(torch.abs(ssl_noisy)).item() / (0.99 - self.EPS)
            ssl_noisy = torch.div(ssl_noisy, noisyspeech_maxamplevel)
            ssl_noise = torch.div(ssl_noise, noisyspeech_maxamplevel)
            ssl_clean = torch.div(ssl_clean, noisyspeech_maxamplevel)
            noisy_rms_level = int(20 * np.log10(noisy_scalar/noisyspeech_maxamplevel * (noisy_rms + self.EPS)))
        return ssl_clean, ssl_noise, ssl_noisy, noisy_rms_level

    def synthesizeNoisySpeech(self, clean, noise, batchSize, 
            snr,
            targetLevel,
            targetLevelLower,
            targetLevelHigher):
        ssl_noisy = torch.zeros(batchSize, 480000).to(device)
        ssl_noise = torch.zeros(batchSize, 480000).to(device)
        ssl_clean = torch.zeros(batchSize, 480000).to(device)
        for i in range(batchSize):
            ssl_clean[i, :], ssl_noise[i,:], ssl_noisy[i,:], rms = self._segmental_snr_mixer(clean[i,:], noise[i,:], 
                snr[i].item(), 
                targetLevel[i].item(),
                targetLevelLower,
                targetLevelHigher)
                       
        return ssl_noisy, ssl_clean, ssl_noise

    def forward(self, noisy):
        x = noisy - self.stft_mean
        for block in self.blocks:
            x = block(x)

        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask

    def _plot_spectrogram(self, data, filename, freq_map, total_frames):
        plt.figure(figsize=(40,5))
        mappable = plt.imshow(data, cmap='plasma', interpolation='nearest', origin="lower")
        cbar = plt.colorbar(mappable, aspect=100, pad=0.02)#, orientation='horizontal')
        cbar.ax.set_aspect('auto')
        num_freq_bins, num_fft_frames = data.shape
        time_axis = np.arange(0, num_fft_frames) * (30.0 / total_frames)
        yticks = [i for i in range(0, len(freq_map), int(len(freq_map)/10))]
        yticklabels = [round(freq_map[i]/1000.0, 2) for i in yticks]
        xticks = [i for i in range(0, len(time_axis), int(len(time_axis)/10))]
        xticklabels = [round(time_axis[i]) for i in xticks]
        plt.xlabel("Time (sec)")
        plt.ylabel("Frequency (kHz)")
        plt.yticks(ticks=yticks, labels=yticklabels)
        plt.xticks(ticks=xticks, labels=xticklabels)
        plt.savefig(filename + ".png", bbox_inches="tight")
        plt.close()

    def _plot_spectrogram_on_ax(self, fig, ax, data, freq_map, total_frames, hideYTickLabels=False):
        mappable = ax.imshow(data, cmap='BuPu', interpolation='nearest', origin="lower", aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.ax.tick_params(labelsize=16)
        num_freq_bins, num_fft_frames = data.shape
        time_axis = np.arange(0, num_fft_frames) * (30.0 / total_frames)
        yticks = [i for i in range(0, len(freq_map), int(len(freq_map)/10))]
        yticklabels = [round(freq_map[i]/1000.0, 2) for i in yticks]
        xticks = [i for i in range(0, len(time_axis), int(len(time_axis)/10))]
        xticklabels = [round(time_axis[i]) for i in xticks]
        ax.set_xlabel("Time (sec)", fontweight="bold", fontsize=16)
        if hideYTickLabels:
            ax.set_yticks(ticks=yticks, labels=["" for i in yticks])
        else: 
            ax.set_yticks(ticks=yticks, labels=yticklabels, fontsize=16)
            ax.set_ylabel("Frequency (kHz)", fontweight="bold", fontsize=16)
        ax.set_xticks(ticks=xticks, labels=xticklabels, fontsize=16)

    def view_activations(self, noisy, noisy_file_name, freq_map, model):
        model = model.replace("./Trained/", "")
        model = model.replace("_cosineLR/", "")
        model = model.replace("/network_200.pt", "")
        model = model.replace("network_200.pt", "")
        print(model)
        fig = plt.figure(figsize=(70,15))
        fig.suptitle(model.title(), fontweight="bold", fontsize=40, y=0.95)
        fig.tight_layout()
        gs = GridSpec(1, 3, figure=fig, wspace=0.1)
        axs = [None, None, None]
        axInit = fig.add_subplot(gs[0,0])
        axMask = fig.add_subplot(gs[0,1])
        axFinal = fig.add_subplot(gs[0,2])

        name0 = noisy_file_name.replace("./validation_set/noisy/", "")
        name0 = name0.replace(".wav", "")

        x = noisy - self.stft_mean
        data = x.clone().detach().cpu().numpy()
        mini_batch_size, freq_bins, frames = data.shape
        self._plot_spectrogram_on_ax(fig, axInit, data[0,:,(frames//3)+20:2*(frames//3)+30], freq_map, frames, hideYTickLabels=False)
        # axInit.set_xticklabels([])
        # axInit.set_xlabel("")

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            x = block(x)
            # data = x.clone().detach().cpu().numpy()
            # mini_batch_size, num_neurons_per_layer, total_frames = data.shape
            # data0 = data[0,:,(total_frames//3)+20:2*(total_frames//3)+30]
            # num_neurons_per_layer, frames = data0.shape
            # time_axis = np.arange(0, frames) * (30.0 / total_frames)
            # xticks = [i for i in range(0, len(time_axis), int(len(time_axis)/10))]
            # xticklabels = [round(time_axis[i]) for i in xticks]
            # yticks = [i for i in range(0, num_neurons_per_layer, num_neurons_per_layer//5)]
            # yticklabels = [round(range(0,num_neurons_per_layer)[i]) for i in yticks]
            # abs_max = max(  abs(data0.min()) ,  abs(data0.max())  ) + 1 # Center around 1.0?
            # mappable = axs[i].imshow(data0, cmap='seismic', interpolation='nearest', origin="lower", vmin=(-1*abs_max) + 1, vmax=abs_max + 1, aspect='auto')
            # divider = make_axes_locatable(axs[i])
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # fig.colorbar(mappable, cax=cax)#, pad=0.02)#ax=axs[i])#, orientation='horizontal', aspect=60)
            # axs[i].set_xticks(xticks)
            # if (i == len(self.blocks) - 1):
            #     axs[i].set_xlabel("Time (sec)")
            #     axs[i].set_xticklabels(xticklabels)
            # else:
            #     axs[i].set_xticklabels([])
            # axs[i].set_yticks(yticks)
            # axs[i].set_yticklabels(yticklabels)
            # axs[i].set_ylabel("Neuron in Layer " + str(i) + " (ID)")
        
        print(model)
        if model == "baseline":
            axInit.set_title("Noisy Input Audio", fontweight="bold", fontsize=25)
            # axs[0].set_title("a) Input Layer")
            # axs[1].set_title("b) Hidden Layer 1")
            # axs[2].set_title("c) Hidden Layer 2")
            # axs[3].set_title("d) Ouput Layer 1")
            axMask.set_title("Network Mask", fontweight="bold", fontsize=25)
            axFinal.set_title("Denoised Output Audio", fontweight="bold", fontsize=25)
        if model == "speech1_null_island_trial_1":
            axInit.set_title("Noisy Input Audio", fontweight="bold", fontsize=25)
            # axs[0].set_title("h) Input Layer")
            # axs[1].set_title("i) Hidden Layer 1")
            # axs[2].set_title("j) Hidden Layer 2")
            # axs[3].set_title("k) Ouput Layer 1")
            axMask.set_title("Network Mask", fontweight="bold", fontsize=25)
            axFinal.set_title("Denoised Output Audio", fontweight="bold", fontsize=25)
        
        mask = torch.relu(x + 1)
        data = mask.clone().detach().cpu().numpy()
        mini_batch_size, freq_bins, total_frames = data.shape
        
        mini_batch_size, freq_bins, frames = data.shape
        data0 = data[0,:,(frames//3)+20:2*(frames//3)+30]
        abs_max = max(  abs(data0.min()) ,  abs(data0.max())  )

        freq_bins, num_fft_frames = data0.shape

        time_axis = np.arange(0, num_fft_frames) * (30.0 / total_frames)
        xticks = [i for i in range(0, len(time_axis), int(len(time_axis)/10))]
        xticklabels = [round(time_axis[i]) for i in xticks]

        nsteps = 128
        blues = plt.cm.Blues(np.linspace(1, 0., nsteps ))
        reds = plt.cm.Reds(np.linspace(0., 1, math.ceil(nsteps * (abs_max-1.0))) )
        colors = np.vstack((blues, reds))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        lower = np.linspace(0., 1, 128)
        upper = np.linspace(1., abs_max, 128)
        bounds = np.concatenate([lower, upper])
        norm = mpl.colors.BoundaryNorm(bounds, mymap.N)

        mappable = axMask.imshow(data0, cmap=mymap, interpolation='nearest', origin="lower", vmin=0, vmax=abs_max+1, aspect='auto')

        divider = make_axes_locatable(axMask)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, norm=norm, cmap=mymap, cax=cax)
        cbar.mappable.set_clim(0, abs_max)
        cbar.ax.tick_params(labelsize=16)
        # axMask.set_xticklabels([])
        axMask.set_xticks(ticks=xticks, labels=xticklabels, fontsize=16)
        yticks = [i for i in range(0, len(freq_map), int(len(freq_map)/10))]
        yticklabels = ["" for i in yticks]
        axMask.set_yticks(ticks=yticks, labels=yticklabels)
        axMask.set_xlabel("Time (sec)", fontweight="bold", fontsize=16)
        # axMask.set_ylabel("Frequency (Hz)")

        ret = slayer.axon.delay(noisy, self.out_delay) * mask
        data = ret.clone().detach().cpu().numpy()
        mini_batch_size, freq_bins, frames = data.shape
        self._plot_spectrogram_on_ax(fig, axFinal, data[0,:,(frames//3)+20:2*(frames//3)+30], freq_map, frames, hideYTickLabels=True)

        plt.savefig(model + "_activations.png", bbox_inches="tight")
        plt.close()

    def view_mask(self, noisy, clean, noisy_file_name, freq_map, model):
        model = model.replace("./Trained/", "")
        model = model.replace("_cosineLR/", "")
        model = model.replace("/network_200.pt", "")
        model = model.replace("network_200.pt", "")
        print(model)
        fig = plt.figure(figsize=(90,15))
        fig.tight_layout()
        gs = GridSpec(1, 3, figure=fig, wspace=0.1)
        axInit = fig.add_subplot(gs[0,0])
        axMask = fig.add_subplot(gs[0,1])
        # axIdealMask = fig.add_subplot(gs[1,0])
        axFinal = fig.add_subplot(gs[0,2])
        # axIdealFinal = fig.add_subplot(gs[1,1])
        # axClean = fig.add_subplot(gs[1,2])

        name0 = noisy_file_name.replace("./validation_set/noisy/", "")
        name0 = name0.replace(".wav", "")

        x = noisy - self.stft_mean
        data = x.clone().detach().cpu().numpy()
        mini_batch_size, freq_bins, frames = data.shape
        self._plot_spectrogram_on_ax(fig, axInit, data[0,:,(frames//3)+20:2*(frames//3)+30], freq_map, frames)
        axInit.set_xticklabels([])
        axInit.set_yticklabels([])
        axInit.set_xlabel("")
        axInit.set_ylabel("")

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            x = block(x)
            data = x.clone().detach().cpu().numpy()
        
        # if model == "baseline":
        #     axInit.set_title("a) Noisy Input Audio")
        #     axMask.set_title("b) Network Mask")
        #     axFinal.set_title("c) Denoised Output Audio")
        # if model == "speech1_null_island":
        #     axInit.set_title("e) Noisy Input Audio")
        #     axMask.set_title("f) Network Mask")
        #     axFinal.set_title("g) Denoised Output Audio")
        
        mask = torch.relu(x + 1)
        data = mask.clone().detach().cpu().numpy()
        mini_batch_size, freq_bins, total_frames = data.shape
        data0 = data[0,:,(total_frames//3)+20:2*(total_frames//3)+30]
        freq_bins, num_fft_frames = data0.shape
        abs_max = max(  abs(data0.min()) ,  abs(data0.max())  )

        nsteps = 128
        blues = plt.cm.Blues(np.linspace(1, 0., nsteps ))
        reds = plt.cm.Reds(np.linspace(0., 1, math.ceil(nsteps * (abs_max-1.0))) )
        colors = np.vstack((blues, reds))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        lower = np.linspace(0., 1, 128)
        upper = np.linspace(1., abs_max, 128)
        bounds = np.concatenate([lower, upper])
        norm = mpl.colors.BoundaryNorm(bounds, mymap.N)

        mappable = axMask.imshow(data0, cmap=mymap, interpolation='nearest', origin="lower", vmin=0, vmax=abs_max+1, aspect='auto')

        divider = make_axes_locatable(axMask)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, norm=norm, cmap=mymap, cax=cax)
        cbar.mappable.set_clim(0, abs_max)
        


        time_axis = np.arange(0, num_fft_frames) * (30.0 / total_frames)
        xticks = [i for i in range(0, len(time_axis), int(len(time_axis)/10))]
        xticklabels = [round(time_axis[i]) for i in xticks]
        axMask.set_xticks(ticks=xticks, labels=[])#xticklabels)
        axMask.set_xlabel("")


        yticks = [i for i in range(0, len(freq_map), int(len(freq_map)/10))]
        yticklabels = [round(freq_map[i]/1000.0, 2) for i in yticks]
        axMask.set_yticks(ticks=yticks, labels=[])#yticklabels)
        axMask.set_ylabel("")#Frequency (Hz)")



        idealMask = torch.div(clean, noisy)
        # if torch.isnan(idealMask).any():
        #     idealMask[torch.isnan(idealMask)] = 0.0
        # data = idealMask.clone().detach().cpu().numpy()
        # mini_batch_size, freq_bins, frames = data.shape
        # data0 = data[0,:,(frames//3)+20:2*(frames//3)+30]
        # abs_max = max(  abs(data0.min()) ,  abs(data0.max())  )

        # nsteps = 128
        # blues = plt.cm.Blues(np.linspace(1, 0., nsteps ))
        # reds = plt.cm.Reds(np.linspace(0., 1, math.ceil(nsteps * (abs_max-1.0))) )
        # colors = np.vstack((blues, reds))
        # mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        # lower = np.linspace(0., 1, 128)
        # upper = np.linspace(1., abs_max, 128)
        # bounds = np.concatenate([lower, upper])
        # norm = mpl.colors.BoundaryNorm(bounds, mymap.N)

        # mappable = axIdealMask.imshow(data0, cmap=mymap, interpolation='nearest', origin="lower", vmin=0, vmax=abs_max+1, aspect='auto')

        # divider = make_axes_locatable(axIdealMask)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = fig.colorbar(mappable, norm=norm, cmap=mymap, cax=cax)
        # cbar.mappable.set_clim(0, abs_max)
        # axIdealMask.set_xticklabels([])
        # yticks = [i for i in range(0, len(freq_map), int(len(freq_map)/10))]
        # yticklabels = [round(freq_map[i]/1000.0, 2) for i in yticks]
        # axIdealMask.set_yticks(ticks=yticks, labels=yticklabels)
        # axIdealMask.set_ylabel("Frequency (Hz)")




        ret = slayer.axon.delay(noisy, self.out_delay) * mask
        data = ret.clone().detach().cpu().numpy()
        mini_batch_size, freq_bins, frames = data.shape
        self._plot_spectrogram_on_ax(fig, axFinal, data[0,:,(frames//3)+20:2*(frames//3)+30], freq_map, frames)
        axFinal.set_xticklabels([])
        axFinal.set_yticklabels([])
        axFinal.set_xlabel("")
        axFinal.set_ylabel("")


        idealRet = slayer.axon.delay(noisy, self.out_delay) * idealMask
        # data = idealRet.clone().detach().cpu().numpy()
        # mini_batch_size, freq_bins, frames = data.shape
        # self._plot_spectrogram_on_ax(fig, axIdealFinal, data[0,:,(frames//3)+20:2*(frames//3)+30], freq_map, frames)


        data = clean.clone().detach().cpu().numpy()
        # mini_batch_size, freq_bins, frames = data.shape
        # self._plot_spectrogram_on_ax(fig, axClean, data[0,:,(frames//3)+20:2*(frames//3)+30], freq_map, frames)

        denoisedLoss = F.mse_loss(ret, clean)
        print("Denoised Loss = " + str(denoisedLoss))

        denoisedIdealLoss = F.mse_loss(idealRet, clean)
        print("Denoised Ideal Loss = " + str(denoisedIdealLoss))


        cleanLoss = F.mse_loss(clean, clean)
        print("Clean Loss = " + str(cleanLoss))


        plt.savefig(model + ".png", format="png", bbox_inches="tight")
        plt.close()


    def get_axon_delay_weights(self, layer):
        assert(hasattr(layer, "delay") and layer.delay != None)
        assert("delay" in layer.delay._parameters.keys())
        weights  = layer.delay._parameters['delay'].cpu().numpy()
        return weights

    def plot_axon_delays(self, model):
        model = model.replace("./Trained/", "")
        model = model.replace("_cosineLR/", "")
        model = model.replace("/network_200.pt", "")
        model = model.replace("network_200.pt", "")
        print(model)

        L1 = self.blocks[1]
        L2 = self.blocks[2]

        l1_weights = self.get_axon_delay_weights(self.blocks[1])
        l2_weights = self.get_axon_delay_weights(self.blocks[2])
        plt.hist(l1_weights, bins=50, alpha=0.5, label='Hidden Layer 1')
        plt.hist(l2_weights, bins=50, alpha=0.5, label='Hidden Layer 2')
        plt.legend()
        plt.savefig(model + "_axon_delays.png", format="png", bbox_inches="tight")
        plt.close()

        data = [l1_weights, l2_weights]
        plt.boxplot(data)
        plt.xticks(ticks=[1,2], labels=["Hidden Layer 1", "Hidden Layer 2"])
        plt.savefig(model + "_axon_delays_box.png", format="png", bbox_inches="tight")
        plt.close()

        sorted_L1 = np.sort(l1_weights)
        sorted_L2 = np.sort(l2_weights)
        plt.scatter([i for i in range(l1_weights.size)], sorted_L1, label="Hidden Layer 1", color="orange", marker='s', s=2, zorder=5)
        plt.scatter([i for i in range(l2_weights.size)], sorted_L2, label="Hidden Layer 2", color="blue", marker='^', s=2, zorder=6)
        mean_L1 = np.mean(sorted_L1)
        std_L1 = np.std(sorted_L1)
        mean_L1_index = np.argmin(np.abs(sorted_L1 - mean_L1))
        plt.vlines(x=mean_L1_index, ymin=0.0, ymax=sorted_L1[mean_L1_index], color='orange', linestyle='--', label="L1 Mean (μ=" + str(round(mean_L1, 2))+",σ=" + str(round(std_L1, 2)) + ")", zorder=3)
        plt.hlines(y=sorted_L1[mean_L1_index], xmin=0.0, xmax=mean_L1_index, color='orange', linestyle='--', zorder=1)
        mean_L2 = np.mean(sorted_L2)
        std_L2 = np.std(sorted_L2)
        mean_L2_index = np.argmin(np.abs(sorted_L2 - mean_L2))
        plt.vlines(x=mean_L2_index, ymin=0.0, ymax=sorted_L2[mean_L2_index], color='blue', linestyle='--', label="L2 Mean (μ=" + str(round(mean_L2, 2)) +",σ=" + str(round(std_L2, 2)) +")", zorder=4)
        plt.hlines(y=sorted_L2[mean_L2_index], xmin=0.0, xmax=mean_L2_index, color='blue', linestyle='--', zorder=2)
        if "baseline" in model:
            plt.title("a) Baseline", fontweight='bold', fontsize=20)
        if "trial_1" in model:
            plt.title("b) Trial 1", fontweight='bold', fontsize=20)
        if "trial_2" in model:
            plt.title("c) Trial 2", fontweight='bold', fontsize=20)
        if "trial_3" in model:
            plt.title("d) Trial 3", fontweight='bold', fontsize=20)
        if "trial_4" in model:
            plt.title("e) Trial 4", fontweight='bold', fontsize=20)
        if "trial_5" in model:
            plt.title("f) Trial 5", fontweight='bold', fontsize=20)
        if "trial_6" in model:
            plt.title("g) Trial 6", fontweight='bold', fontsize=20)
        if "trial_7" in model:
            plt.title("h) Trial 7", fontweight='bold', fontsize=20)
        if "trial_8" in model:
            plt.title("i) Trial 8", fontweight='bold', fontsize=20)

        plt.legend(fontsize=14)
        plt.xlabel("Sorted Neuron", fontsize=18)
        plt.ylabel("Axon Delay Weight Value", fontsize=18)
        plt.savefig(model + "_axon_delays_scatter.svg", format="svg", bbox_inches="tight")
        plt.close()


        return

    def validate_gradients(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any()
                                       or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            self.zero_grad()

def plot_weights(data):
    for name in data.keys():
        num_epochs = max(data[name].keys()) + 1
        num_neurons = data[name][0].size()[0]
        matrix = np.zeros(shape=(num_epochs, num_neurons))
        for i in range(num_epochs):
            for j in range(num_neurons):
                matrix[i,j] = data[name][i][j]

        plt.figure(figsize=(20,20))
        plt.imshow(np.transpose(matrix), cmap='hot', interpolation='nearest')
        plt.xlabel("Training Epochs")
        plt.ylabel("Axons")
        plt.savefig(name + ".png", bbox_inches="tight")
        plt.close()

def run_warm_up_training_cipic(args, net, optimizer, scheduler, train_loader, orientList):
    for i, (clean, noise, idx) in enumerate(train_loader):
        net.train()
        speechFilterOrient, noiseFilterOrient = random.choice(orientList)
        speechFilter  = torch.from_numpy(CIPICSubject.getHRIRFromIndex(speechFilterOrient, args.filterChannel)).float()
        speechFilter  = speechFilter.to(device)
        noiseFilter   = torch.from_numpy(CIPICSubject.getHRIRFromIndex(noiseFilterOrient, args.filterChannel)).float()
        noiseFilter   = noiseFilter.to(device)
        noise = noise.to(device)
        clean = clean.to(device)
        ssl_noise = torch.zeros(args.b, 480000).to(device)
        ssl_clean = torch.zeros(args.b, 480000).to(device)
        ssl_snrs  = torch.zeros(args.b, 1).to(device)
        ssl_targlvls= torch.zeros(args.b, 1).to(device)
        for batch_idx in range(args.b):
            ssl_noise[batch_idx,:] = conv_transform(noise[batch_idx,:], downsampler(noiseFilter))
            ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], downsampler(speechFilter))
            clean_file, noise_file, metadata = train_set._get_filenames(idx[batch_idx])
            ssl_snrs[batch_idx] = metadata['snr']
            ssl_targlvls[batch_idx] = metadata['target_level']

        ssl_noisy, ssl_clean, ssl_noise = module.synthesizeNoisySpeech(
            ssl_clean, 
            ssl_noise, 
            args.b, 
            ssl_snrs,
            ssl_targlvls,
            -35,
            -15)

        noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, None)
        clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, None)

        denoised_abs = net(noisy_abs)
        noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
        clean_abs = slayer.axon.delay(clean_abs, out_delay)
        clean = slayer.axon.delay(ssl_clean, args.n_fft // 4 * out_delay)

        clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft, None)

        score = si_snr(clean_rec, clean)
        loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))

        if torch.isnan(loss).any():
            loss[torch.isnan(loss)] = 0
        assert torch.isnan(loss) == False

        optimizer.zero_grad()
        loss.backward()
        module.validate_gradients()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        return

def run_warm_up_training(args, net, optimizer, scheduler, train_loader):
    net.train()
    # Run single epoch just to set the network dimensions (Weird that this is necessary)?
    for i, (clean, noisy, idx) in enumerate(train_loader):
        ssl_noisy = noisy.to(device)
        ssl_clean = clean.to(device)

        noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, None)
        clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, None)

        denoised_abs = net(noisy_abs)
        noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
        clean_abs = slayer.axon.delay(clean_abs, out_delay)
        clean = slayer.axon.delay(ssl_clean, args.n_fft // 4 * out_delay)

        clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft, None)

        score = si_snr(clean_rec, clean)
        loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))

        if torch.isnan(loss).any():
            loss[torch.isnan(loss)] = 0
        assert torch.isnan(loss) == False

        optimizer.zero_grad()
        loss.backward()
        module.validate_gradients()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu',
                        type=int,
                        default=[0],
                        help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',
                        type=int,
                        default=32,
                        help='batch size for dataloader')
    parser.add_argument('-lr',
                        type=float,
                        default=0.001,
                        help='initial learning rate')
    parser.add_argument('-lam',
                        type=float,
                        default=0.001,
                        help='lagrangian factor')
    parser.add_argument('-threshold',
                        type=float,
                        default=0.1,
                        help='neuron threshold')
    parser.add_argument('-tau_grad',
                        type=float,
                        default=0.1,
                        help='surrogate gradient time constant')
    parser.add_argument('-scale_grad',
                        type=float,
                        default=0.8,
                        help='surrogate gradient scale')
    parser.add_argument('-n_fft',
                        type=int,
                        default=512,
                        help='number of FFT specturm, hop is n_fft // 4')
    parser.add_argument('-dmax',
                        type=int,
                        default=64,
                        help='maximum axonal delay')
    parser.add_argument('-out_delay',
                        type=int,
                        default=0,
                        help='prediction output delay (multiple of 128)')
    parser.add_argument('-clip',
                        type=float,
                        default=10,
                        help='gradient clipping limit')
    parser.add_argument('-exp',
                        type=str,
                        default='',
                        help='experiment differentiater string')
    parser.add_argument('-seed',
                        type=int,
                        default=None,
                        help='random seed of the experiment')
    parser.add_argument('-spectrogram',
                        type=int,
                        default=0,
                        help='What type of FT to use, 0: torch.stft, 1: torchaudio.Transforms.Spectrogram, 2:numpy melspec')
    parser.add_argument('-path',
                        type=str,
                        default='../../',
                        help='dataset path')
    parser.add_argument('-training_samples',
                        type=int,
                        default=32,
                        help='Number of samples training should use, supports small dataset subset')
    parser.add_argument('-useCheckpoint',
                        type=str,
                        default='',
                        help='Checkpoint to continue training from')

    # CIPIC Filter Parameters
    # ID:21 ==> Mannequin with large pinna
    # ID 165 ==> Mannequin with small pinna
    # The rest are real subjects
    parser.add_argument('-useCipic',
                        dest='useCipic', 
                        action='store_true',
                        help='Switch flag to toggle using the CIPIC pre-filter')
    parser.add_argument('-fixedOrients',
                        dest='fixedOrients', 
                        action='store_true',
                        help='Switch flag to manually configure which orients will be selected')
    parser.add_argument('-numFixedOrients',
                        type=int,
                        default=1,
                        help='Number of manually set orientation pairs to use if fixedOrients switch is set')
    parser.add_argument('-cipicSubject',
                        type=int,
                        default=12,
                        help='Cipic subject ID for pinna filters')
    parser.add_argument('-filterChannel',
                        type=int,
                        default=0,
                        help='Channel used for speech and noise separation')
    parser.add_argument('-hiddenLayerWidths',
                        type=int,
                        default=512,
                        help='# of nuerons in hidden layers')
    parser.add_argument('-numOrients',
                        type=int,
                        default=8,
                        help='Number of additional orientations, must be >= 8')

    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}{}'.format(args.optim, args.seed)

    assert(args.spectrogram == 0 or args.spectrogram == 1 or args.spectrogram == 2)
    trained_folder = 'Trained' + identifier
    logs_folder = 'Logs' + identifier
    writer = SummaryWriter('runs/' + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    lam = args.lam

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    out_delay = args.out_delay
    net = torch.nn.DataParallel(Network(
                args.threshold,
                args.tau_grad,
                args.scale_grad,
                args.dmax,
                args.out_delay,
                args.hiddenLayerWidths,
                args.n_fft).to(device),
                    device_ids=args.gpu)
    module = net.module
    stft_transform =torchaudio.transforms.Spectrogram(
                n_fft=args.n_fft,
                onesided=True, 
                power=None,
                hop_length=math.floor(args.n_fft//4)).to(device)
    inv_stft_transform =torchaudio.transforms.InverseSpectrogram(
                n_fft=args.n_fft,
                onesided=True, 
                hop_length=math.floor(args.n_fft//4)).to(device)
    mel_transform =torchaudio.transforms.MelSpectrogram(
                n_fft=4*args.n_fft,
                n_mels=257,
                power=2,
                hop_length=math.floor(args.n_fft//4)).to(device)
    # Seems like we cannot use inverseMelScale
    # https://stackoverflow.com/questions/74447735/why-is-the-inversemelscale-torchaudio-function-so-slow
    conv_transform = torchaudio.transforms.Convolve("same").to(device)

    # Input audio is recorded at 16 kHz, but CIPIC HRTFs are at 44.1 kHz
    downsampler= torchaudio.transforms.Resample(44100, 16000, dtype=torch.float32).to(device)

    # Define optimizer module.
    optimizer = torch.optim.RAdam(net.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    if (not args.useCipic):
        print("NOT using CIPIC subject to preprocess audio")

    orientList = []
    if (args.useCipic):
        CIPICSubject = CipicDatabase.subjects[args.cipicSubject]
        print("Using Subject " + str(args.cipicSubject) + " for spatial sound separation...")
        if not args.fixedOrients:
            orientSet = set()
            orientSet.add(316) # center of front upper  right hemisphere
            orientSet.add(300) # center of front bottom right hemisphere
            orientSet.add(332) # center of back  upper  right hemisphere
            orientSet.add(348) # center of back  bottom right hemisphere
            orientSet.add(916) # center of front upper  left  hemisphere
            orientSet.add(900) # center of front bottom left  hemisphere
            orientSet.add(932) # center of back  upper  left  hemisphere
            orientSet.add(948) # center of back  bottom left  hemisphere
            allPossibleOrients = set(range(0, 1250)).difference(orientSet)
            for _ in range(8, args.numOrients):
                randOrient = list(allPossibleOrients)[random.randint(0, len(allPossibleOrients) - 1)]
                orientSet.add(randOrient)
                allPossibleOrients.remove(randOrient)
            selectedOrients = list(orientSet)
            orientPairSet = set()
            for o1 in selectedOrients:
                for o2 in selectedOrients:
                    orientPairSet.add( (o1, o2) )
            orientList = list(orientPairSet)
        else:
            assert(args.fixedOrients and args.numFixedOrients >= 1)
            if args.numFixedOrients == 1:
                orientList.append( (608, 640) ) # speech in front, noise in back, medial plane
            if args.numFixedOrients == 2:
                orientList.append( (608, 640) ) # speech in front, noise in back, medial plane
                orientList.append( (640, 608) ) # speech in back, noise in front, medial plane
            if args.numFixedOrients == 4:
                orientList.append( (316, 948) )
                orientList.append( (300, 932) )
                orientList.append( (916, 348) )
                orientList.append( (900, 332) )

    print("Orient list contains " + str(len(orientList)) + " orientation pairs")

    if (args.useCipic):
        train_set = DNSAudioNoNoisy(root=args.path + 'training_set/', maxFiles=args.training_samples)
    else:
        train_set = DNSAudioNoNoise(root=args.path + 'training_set/', maxFiles=args.training_samples)
    
    train_loader = DataLoader(train_set,
                          batch_size=args.b,
                          shuffle=True,
                          collate_fn=train_set.collate_fn,
                          num_workers=4,
                          pin_memory=True)

    startingEpoch = 0
    trackingInfo = dict()
    assert(args.useCheckpoint != "")
    if args.useCipic:
        run_warm_up_training_cipic(args, net, optimizer, scheduler, train_loader, orientList)
    else:
        run_warm_up_training(args, net, optimizer, scheduler, train_loader)
    checkpoint = torch.load(args.useCheckpoint, weights_only=True)
    module.load_state_dict(checkpoint['module_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    startingEpoch = checkpoint['epochs_completed']
    trackingInfo = checkpoint['tracking_info']
    startingTrainingLoss = trackingInfo[startingEpoch]['training_loss']
    startingTrainingScore = trackingInfo[startingEpoch]['training_score']
    startingValidationLoss = trackingInfo[startingEpoch]['validation_loss']
    startingValidationScore = trackingInfo[startingEpoch]['validation_score']
    statusString  = "Resuming from checkpoint [epochs_completed:" 
    statusString += str(startingEpoch) + ", training loss=" 
    statusString += str(startingTrainingLoss) + ", training si-snr:" 
    statusString += str(startingTrainingScore) + ", validation loss="
    statusString += str(startingValidationLoss) + ", validation si-snr:" 
    statusString += str(startingValidationScore) + "]"
    print(statusString)

    noisy_file = "book_07187_chp_0023_reader_11967_24_seg_0-zweiplaneten_08_lasswitz_f000055-sous_les_mers_2_17_f00004_foNpH3TLA2s-Zfc2hL97mC4-HEaVl_grIjk-Sl5ZVMclaq4_snr-1_tl-20_fileid_11916"
    clean_file = "clean_fileid_11916"
    noise_file = "noise_fileid_11916"

    noisy_full_path = os.path.join(os.getcwd(), "validation_set")
    noisy_full_path = os.path.join(noisy_full_path, "noisy")
    noisy_full_path = os.path.join(noisy_full_path, noisy_file + ".wav")
    noisy_wav, noisy_wav_fs = sf.read(noisy_full_path)
    noise_full_path = os.path.join(os.getcwd(), "validation_set")
    noise_full_path = os.path.join(noise_full_path, "noise")
    noise_full_path = os.path.join(noise_full_path, noise_file + ".wav")
    clean_full_path = os.path.join(os.getcwd(), "validation_set")
    clean_full_path = os.path.join(clean_full_path, "clean")
    clean_full_path = os.path.join(clean_full_path, clean_file + ".wav")

    noisy_wav, noisy_wav_fs = sf.read(noisy_full_path)
    noise_wav, noise_wav_fs = sf.read(noise_full_path)
    clean_wav, clean_wav_fs = sf.read(clean_full_path)

    num_samples = 30 * noisy_wav_fs  # 30 sec data

    if len(noisy_wav) > num_samples:
        noisy_wav = noisy_wav[:num_samples]
    else:
        noisy_wav = np.concatenate([noisy_wav,
                                      np.zeros(num_samples
                                               - len(noisy_wav))])
    if len(noise_wav) > num_samples:
        noise_wav = noise_wav[:num_samples]
    else:
        noise_wav = np.concatenate([noise_wav,
                                      np.zeros(num_samples
                                               - len(noise_wav))])
    if len(clean_wav) > num_samples:
        clean_wav = clean_wav[:num_samples]
    else:
        clean_wav = np.concatenate([clean_wav,
                                      np.zeros(num_samples
                                               - len(clean_wav))])

    noisy_wav = np.repeat(noisy_wav[np.newaxis,:], args.b, axis=0)
    noise_wav = np.repeat(noise_wav[np.newaxis,:], args.b, axis=0)
    clean_wav = np.repeat(clean_wav[np.newaxis,:], args.b, axis=0)

    freq_map = librosa.fft_frequencies(sr=16000, n_fft=args.n_fft)
    CIPICSubject = CipicDatabase.subjects[args.cipicSubject]
    with torch.no_grad():
        noisy_wav = torch.from_numpy(noisy_wav).float()
        noisy_wav = noisy_wav.to(device)

        noise_wav = torch.from_numpy(noise_wav).float()
        noise_wav = noise_wav.to(device)

        clean_wav = torch.from_numpy(clean_wav).float()
        clean_wav = clean_wav.to(device)

        speechFilterOrient, noiseFilterOrient = (608, 640)
        speechFilter  = torch.from_numpy(CIPICSubject.getHRIRFromIndex(speechFilterOrient, args.filterChannel)).float()
        speechFilter  = speechFilter.to(device)
        noiseFilter   = torch.from_numpy(CIPICSubject.getHRIRFromIndex(noiseFilterOrient, args.filterChannel)).float()
        noiseFilter   = noiseFilter.to(device)

        ssl_noise = torch.zeros(args.b, 480000).to(device)
        ssl_clean = torch.zeros(args.b, 480000).to(device)
        ssl_snrs  = torch.zeros(args.b, 1).to(device)
        ssl_targlvls= torch.zeros(args.b, 1).to(device)
        file_id_from_name = re.compile('fileid_(\d+)')
        snr_from_name = re.compile('snr(-?\d+)')
        target_level_from_name = re.compile('tl(-?\d+)')
        source_info_from_name = re.compile('^(.*?)_snr')
        snr = int(snr_from_name.findall(noisy_file)[0])
        target_level = int(target_level_from_name.findall(noisy_file)[0])
        for batch_idx in range(args.b):
            ssl_noise[batch_idx,:] = conv_transform(noise_wav[batch_idx,:], downsampler(noiseFilter))
            ssl_clean[batch_idx,:] = conv_transform(clean_wav[batch_idx,:], downsampler(speechFilter))
            ssl_snrs[batch_idx] = snr
            ssl_targlvls[batch_idx] = target_level

        ssl_noisy, ssl_clean, ssl_noise = module.synthesizeNoisySpeech(
            ssl_clean, 
            ssl_noise, 
            args.b, 
            ssl_snrs,
            ssl_targlvls,
            -35,
            -15)

        if (args.useCheckpoint == "./Trained/speech1_null_island/network_200.pt"):
            noisy_wav = ssl_noisy
            clean_wav = ssl_clean

        if (args.spectrogram == 0):
            noisy_abs, noisy_arg = stft_splitter(noisy_wav, args.n_fft, None)
            clean_abs, clean_arg = stft_splitter(clean_wav, args.n_fft, None)
        elif(args.spectrogram == 1):
            noisy_abs, noisy_arg = stft_splitter(noisy_wav, args.n_fft, stft_transform)
            clean_abs, clean_arg = stft_splitter(clean_wav, args.n_fft, stft_transform)
        else:
            noisy_abs, noisy_arg = stft_splitter(noisy_wav, args.n_fft, mel_transform)
            clean_abs, clean_arg = stft_splitter(clean_wav, args.n_fft, mel_transform)
        net.module.view_activations(noisy_abs, noisy_file, freq_map, args.useCheckpoint)

        # net.module.view_mask(noisy_abs, clean_abs, noisy_file, freq_map, args.useCheckpoint)

        # net.module.plot_axon_delays(args.useCheckpoint)