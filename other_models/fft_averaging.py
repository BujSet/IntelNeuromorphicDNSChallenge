# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os, sys, math
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf

from lava.lib.dl import slayer
sys.path.append('./')
from audio_dataloader import DNSAudio
from hrtfs.cipic_db import CipicDatabase 
from snr import si_snr
import torchaudio
import librosa

def collate_fn(batch):
    noisy, clean, noise = [], [], []

    indices = torch.IntTensor([s[4] for s in batch])

    for sample in batch:
        noisy += [torch.FloatTensor(sample[0])]
        clean += [torch.FloatTensor(sample[1])]
        noise += [torch.FloatTensor(sample[2])]

    return torch.stack(noisy), torch.stack(clean), torch.stack(noise), indices

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

def min_max_rescale(inputT):
    with torch.no_grad():
        mini = torch.min(inputT)
        maxi = torch.max(inputT)
        return torch.div(torch.sub(inputT, mini), maxi - mini)

class Network(torch.nn.Module):
    def __init__(self, 
            threshold=0.1, 
            tau_grad=0.1, 
            scale_grad=0.8, 
            max_delay=64, 
            out_delay=0):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay
        self.EPS = 2.220446049250313e-16

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
            slayer.block.sigma_delta.Dense(sdnn_params, 257, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Dense(sdnn_params, 512, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Output(sdnn_params, 512, 257, weight_norm=False),
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('-path',
                        type=str,
                        default='./',
                        help='dataset path')
    # CIPIC Filter Parameters

    # ID:21 ==> Mannequin with large pinna
    # ID 165 ==> Mannequin with small pinna
    # The rest are real subjects
    parser.add_argument('-cipicSubject',
                        type=int,
                        default=12,
                        help='Cipic subject ID for pinna filters')
    # Spatially distribute the sound sources, (azimuth, elevation)
    # index = 624 ==> (0,  90)
    # index = 600 ==> (0, -45)
    # We choose filters in the midsagittal plane, so either selecting
    # which channels is read from 'should' be irrelevant
    parser.add_argument('-speechFilterOrient',
                        type=int,
                        default=608,
                        help='Index into CIPIC source directions to orient the speech to ')
    parser.add_argument('-speechFilterChannel',
                        type=int,
                        default=0,
                        help='Channel to orient the speech to ')
    parser.add_argument('-noiseFilterOrient',
                        type=int,
                        default=608,
                        help='Index into CIPIC source directions to orient the noise to ')
    parser.add_argument('-noiseFilterChannel',
                        type=int,
                        default=0,
                        help='Channel to orient the noise to ')

    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}{}'.format(args.optim, args.seed)

    trained_folder = 'Trained' + identifier
    logs_folder = 'Logs' + identifier

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    lam = args.lam

    device = torch.device('cuda:0')

    out_delay = args.out_delay
    net = Network(
                args.threshold,
                args.tau_grad,
                args.scale_grad,
                args.dmax,
                args.out_delay)
    net = torch.nn.DataParallel(net.to(device), device_ids=[0])
    module = net.module
    conv_transform = torchaudio.transforms.Convolve("same").to(device)

    # Input audio is recorded at 16 kHz, but CIPIC HRTFs are at 44.1 kHz
    downsampler= torchaudio.transforms.Resample(44100, 16000, dtype=torch.float32).to(device)

    # Define optimizer module.
    optimizer = torch.optim.RAdam(net.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-5)

    train_set = DNSAudio(root=args.path + 'training_set/')
    validation_set = DNSAudio(root=args.path + 'validation_set/')

    train_loader = DataLoader(train_set,
                              batch_size=args.b,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=4,
                              pin_memory=True)
    validation_loader = DataLoader(validation_set,
                                   batch_size=args.b,
                                   shuffle=True,
                                   collate_fn=collate_fn,
                                   num_workers=4,
                                   pin_memory=True)

    print("Loading CIPIC HRTF filters...")
    CIPICSubject = CipicDatabase.subjects[args.cipicSubject]
    speechFilter  = torch.from_numpy(CIPICSubject.getHRIRFromIndex(args.speechFilterOrient, args.speechFilterChannel)).float()
    speechFilter  = speechFilter.to(device)
    noiseFilter   = torch.from_numpy(CIPICSubject.getHRIRFromIndex(args.noiseFilterOrient, args.noiseFilterChannel)).float()
    noiseFilter   = noiseFilter.to(device)
    print("\tSet speech to " + args.speechFilterOrient + ":" + args.speechFilterChannel)
    print("\tSet noise to " + args.noiseFilterOrient + ":" + args.noiseFilterChannel)

    freqs = librosa.fft_frequencies(sr=16000, n_fft=512)
    yticks = [i for i in range(0, 257, 16)]

    # run it for training set

    sum_noise = torch.zeros(257, 3751).to(device)
    sum_noisy = torch.zeros(257, 3751).to(device)
    sum_clean = torch.zeros(257, 3751).to(device)

    ssl_sum_noise = torch.zeros(257, 3751).to(device)
    ssl_sum_noisy = torch.zeros(257, 3751).to(device)
    ssl_sum_clean = torch.zeros(257, 3751).to(device)
    net.eval()
    num_iters = 0
    for i, (noisy, clean, noise, idx) in enumerate(train_loader):
        with torch.no_grad():

            noisy = noisy.to(device)
            noise = noise.to(device)
            clean = clean.to(device)

            ssl_noise = torch.zeros(args.b, 480000).to(device)
            ssl_clean = torch.zeros(args.b, 480000).to(device)
            ssl_snrs  = torch.zeros(args.b, 1).to(device)
            ssl_targlvls= torch.zeros(args.b, 1).to(device)
            speechOrients = torch.zeros(args.b, 1)
            noiseOrients = torch.zeros(args.b, 1)
            for batch_idx in range(args.b):
                ssl_noise[batch_idx,:] = conv_transform(noise[batch_idx,:], downsampler(noiseFilter))
                ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], downsampler(speechFilter))
                noisy_file, clean_file, noise_file, metadata = train_set._get_filenames(idx[batch_idx])
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

        noise_abs, noise_arg = stft_splitter(noise, args.n_fft, None) 
        noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft, None)
        clean_abs, clean_arg = stft_splitter(clean, args.n_fft, None)

        ssl_noise_abs, ssl_noise_arg = stft_splitter(ssl_noise, args.n_fft, None) 
        ssl_noisy_abs, ssl_noisy_arg = stft_splitter(ssl_noisy, args.n_fft, None)
        ssl_clean_abs, ssl_clean_arg = stft_splitter(ssl_clean, args.n_fft, None)
        
        with torch.no_grad():
            sum_noise += torch.sum(noise_abs, dim=0)
            sum_noisy += torch.sum(noisy_abs, dim=0)
            sum_clean += torch.sum(clean_abs, dim=0)

            ssl_sum_noise += torch.sum(ssl_noise_abs, dim=0)
            ssl_sum_noisy += torch.sum(ssl_noisy_abs, dim=0)
            ssl_sum_clean += torch.sum(ssl_clean_abs, dim=0)

        num_iters += 1.0

    sum_noise = min_max_rescale(sum_noise)
    sum_noisy = min_max_rescale(sum_noisy)
    sum_clean = min_max_rescale(sum_clean)

    ssl_sum_noise = min_max_rescale(ssl_sum_noise)
    ssl_sum_noisy = min_max_rescale(ssl_sum_noisy)
    ssl_sum_clean = min_max_rescale(ssl_sum_clean)
    with torch.no_grad():
        sum_noise = torch.div(sum_noise, num_iters).detach().cpu()
        sum_noisy = torch.div(sum_noisy, num_iters).detach().cpu()
        sum_clean = torch.div(sum_clean, num_iters).detach().cpu()

        ssl_sum_noise = torch.div(ssl_sum_noise, num_iters).detach().cpu()
        ssl_sum_noisy = torch.div(ssl_sum_noisy, num_iters).detach().cpu()
        ssl_sum_clean = torch.div(ssl_sum_clean, num_iters).detach().cpu()
  
    fig, axs = plt.subplots(3, 2, figsize=(30,10))
    fig.tight_layout()
    axs[0,0].imshow(sum_noise, aspect='auto')
    axs[0,0].title.set_text("Noise")
    axs[0,0].set_ylabel("Frequency (kHz)")
    axs[0,0].set_yticks(yticks)
    axs[0,0].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[1,0].imshow(sum_clean, aspect='auto')
    axs[1,0].title.set_text("Clean")
    axs[1,0].set_ylabel("Frequency (kHz)")
    axs[1,0].set_yticks(yticks)
    axs[1,0].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[2,0].imshow(sum_noisy, aspect='auto')
    axs[2,0].title.set_text("Noisy")
    axs[2,0].set_ylabel("Frequncy (kHz)")
    axs[2,0].set_yticks(yticks)
    axs[2,0].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[2,0].set_xlabel("FFT Frame")
    axs[0,1].imshow(ssl_sum_noise, aspect='auto')
    axs[0,1].title.set_text("Pinna Noise")
    axs[0,1].set_yticks(yticks)
    axs[0,1].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[1,1].imshow(ssl_sum_clean, aspect='auto')
    axs[1,1].title.set_text("Pinna Clean")
    axs[1,1].set_yticks(yticks)
    axs[1,1].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[2,1].imshow(ssl_sum_noisy, aspect='auto')
    axs[2,1].title.set_text("Pinna Noisy")
    axs[2,1].set_yticks(yticks)
    axs[2,1].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[2,1].set_xlabel("FFT Frame")
    plt.savefig("training_fft_" + str(args.speechFilterOrient) + "_" + str(args.speechFilterChannel) + "_" + str(args.noiseFilterOrient) + "_" + str(args.noiseFilterChannel) + ".png", bbox_inches='tight')
    plt.close()
    
    print("Finished training set, looking at validation now...")
    # Now run it for validation set

    sum_noise = torch.zeros(257, 3751).to(device)
    sum_noisy = torch.zeros(257, 3751).to(device)
    sum_clean = torch.zeros(257, 3751).to(device)

    ssl_sum_noise = torch.zeros(257, 3751).to(device)
    ssl_sum_noisy = torch.zeros(257, 3751).to(device)
    ssl_sum_clean = torch.zeros(257, 3751).to(device)
    net.eval()
    num_iters = 0
    for i, (noisy, clean, noise, idx) in enumerate(validation_loader):
        with torch.no_grad():

            noisy = noisy.to(device)
            noise = noise.to(device)
            clean = clean.to(device)

            ssl_noise = torch.zeros(args.b, 480000).to(device)
            ssl_clean = torch.zeros(args.b, 480000).to(device)
            ssl_snrs  = torch.zeros(args.b, 1).to(device)
            ssl_targlvls= torch.zeros(args.b, 1).to(device)
            speechOrients = torch.zeros(args.b, 1)
            noiseOrients = torch.zeros(args.b, 1)
            for batch_idx in range(args.b):
                ssl_noise[batch_idx,:] = conv_transform(noise[batch_idx,:], downsampler(noiseFilter))
                ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], downsampler(speechFilter))
                noisy_file, clean_file, noise_file, metadata = validation_set._get_filenames(idx[batch_idx])
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

        noise_abs, noise_arg = stft_splitter(noise, args.n_fft, None) 
        noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft, None)
        clean_abs, clean_arg = stft_splitter(clean, args.n_fft, None)

        ssl_noise_abs, ssl_noise_arg = stft_splitter(ssl_noise, args.n_fft, None) 
        ssl_noisy_abs, ssl_noisy_arg = stft_splitter(ssl_noisy, args.n_fft, None)
        ssl_clean_abs, ssl_clean_arg = stft_splitter(ssl_clean, args.n_fft, None)
        
        with torch.no_grad():
            sum_noise += torch.sum(noise_abs, dim=0)
            sum_noisy += torch.sum(noisy_abs, dim=0)
            sum_clean += torch.sum(clean_abs, dim=0)

            ssl_sum_noise += torch.sum(ssl_noise_abs, dim=0)
            ssl_sum_noisy += torch.sum(ssl_noisy_abs, dim=0)
            ssl_sum_clean += torch.sum(ssl_clean_abs, dim=0)

        num_iters += 1.0
    sum_noise = min_max_rescale(sum_noise)
    sum_noisy = min_max_rescale(sum_noisy)
    sum_clean = min_max_rescale(sum_clean)

    ssl_sum_noise = min_max_rescale(ssl_sum_noise)
    ssl_sum_noisy = min_max_rescale(ssl_sum_noisy)
    ssl_sum_clean = min_max_rescale(ssl_sum_clean)
    with torch.no_grad():
        sum_noise = torch.div(sum_noise, num_iters).detach().cpu()
        sum_noisy = torch.div(sum_noisy, num_iters).detach().cpu()
        sum_clean = torch.div(sum_clean, num_iters).detach().cpu()

        ssl_sum_noise = torch.div(ssl_sum_noise, num_iters).detach().cpu()
        ssl_sum_noisy = torch.div(ssl_sum_noisy, num_iters).detach().cpu()
        ssl_sum_clean = torch.div(ssl_sum_clean, num_iters).detach().cpu()

    fig, axs = plt.subplots(3, 2, figsize=(30,10))
    fig.tight_layout()
    axs[0,0].imshow(sum_noise, aspect='auto')
    axs[0,0].title.set_text("Noise")
    axs[0,0].set_ylabel("Frequency (kHz)")
    axs[0,0].set_yticks(yticks)
    axs[0,0].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[1,0].imshow(sum_clean, aspect='auto')
    axs[1,0].title.set_text("Clean")
    axs[1,0].set_ylabel("Frequency (kHz)")
    axs[1,0].set_yticks(yticks)
    axs[1,0].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[2,0].imshow(sum_noisy, aspect='auto')
    axs[2,0].title.set_text("Noisy")
    axs[2,0].set_ylabel("Frequncy (kHz)")
    axs[2,0].set_yticks(yticks)
    axs[2,0].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[2,0].set_xlabel("FFT Frame")
    axs[0,1].imshow(ssl_sum_noise, aspect='auto')
    axs[0,1].title.set_text("Pinna Noise")
    axs[0,1].set_yticks(yticks)
    axs[0,1].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[1,1].imshow(ssl_sum_clean, aspect='auto')
    axs[1,1].title.set_text("Pinna Clean")
    axs[1,1].set_yticks(yticks)
    axs[1,1].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[2,1].imshow(ssl_sum_noisy, aspect='auto')
    axs[2,1].title.set_text("Pinna Noisy")
    axs[2,1].set_yticks(yticks)
    axs[2,1].set_yticklabels([str(round(freqs[y] / 1000.0, 2)) for y in yticks])
    axs[2,1].set_xlabel("FFT Frame")
    plt.savefig("validation_fft_" + str(args.speechFilterOrient) + "_" + str(args.speechFilterChannel) + "_" + str(args.noiseFilterOrient) + "_" + str(args.noiseFilterChannel) + ".png", bbox_inches='tight')
    plt.close()

    