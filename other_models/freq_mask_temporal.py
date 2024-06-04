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
from lava.lib.dl.slayer.block.base import AbstractDense
import sys
sys.path.append('./')
from audio_dataloader import DNSAudio
from hrtfs.cipic_db import CipicDatabase 
from snr import si_snr
import torchaudio
from noisyspeech_synthesizer import segmental_snr_mixer
import random
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

class FreqMask(torch.nn.Module):
    def __init__(self, 
        n_fft=512, 
        sample_rate=44100.0):
        super().__init__()
        freq_map = torch.from_numpy(librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)).float()
        self.mask = torch.nn.Parameter(torch.randn( (freq_map.size()[0], 3751) ))

    def forward(self, noisy):
        return torch.mul(noisy, self.mask)

    def print_parameters(self):
        print("mask: " + str(self.mask.detach().item()))

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            FreqMask()
        ])

    def forward(self, noisy):
        linearMask = self.blocks[0]
        return linearMask(noisy)


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
    parser.add_argument('-n_fft',
                        type=int,
                        default=512,
                        help='number of FFT specturm, hop is n_fft // 4')
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
    parser.add_argument('-training_epoch',
                        type=int,
                        default=50,
                        help='number of training epochs to run')
    parser.add_argument('-spectrogram',
                        type=int,
                        default=0,
                        help='What type of FT to use, 0: torch.stft, 1: torchaudio.Transforms.Spectrogram, 2:numpy melspec')
    parser.add_argument('-path',
                        type=str,
                        default='../../',
                        help='dataset path')

    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}{}'.format(args.optim, args.seed)

    assert(args.spectrogram == 0 or args.spectrogram == 1 or args.spectrogram == 2)
    trained_folder = 'Trained' + identifier
    logs_folder = 'Logs' + identifier

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    lam = args.lam

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    net = Network()
    net = torch.nn.DataParallel(net.to(device), device_ids=args.gpu)
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

    base_stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                            accuracy_unit='dB')

    stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                       accuracy_unit='dB')

    print("Training for " + str(args.training_epoch) + " epochs")
    training_st = datetime.now()
    for epoch in range(args.training_epoch):
        t_st = datetime.now()
        batch_st = datetime.now()
        batch_tot = 0.0
        conv_tot = 0.0
        synth_tot = 0.0
        for i, (noisy, clean, noise, idx) in enumerate(train_loader):
            batch_tot += (datetime.now() - batch_st).total_seconds()
            net.train()

            noisy = noisy.to(device)
            clean = clean.to(device)

            if (args.spectrogram == 0):
                noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft, None)
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, None)
            elif(args.spectrogram == 1):
                noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft, stft_transform)
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, stft_transform)
            else:
                noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft, mel_transform)
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, mel_transform)

            denoised_abs = net(noisy_abs)

            if (args.spectrogram == 0):
                clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft, None)
            elif (args.spectrogram == 1):
                clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft, inv_stft_transform)
            else:
                clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft, 2)

            score = si_snr(clean_rec, clean)
            loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))

            if torch.isnan(loss).any():
                loss[torch.isnan(loss)] = 0
            assert torch.isnan(loss) == False

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            if torch.isnan(score).any():
                score[torch.isnan(score)] = 0

            stats.training.correct_samples += torch.sum(score).item()
            stats.training.loss_sum += loss.item()
            stats.training.num_samples += noisy.shape[0]

            processed = i * train_loader.batch_size
            total = len(train_loader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            samples_sec = time_elapsed / (i + 1) / train_loader.batch_size
            header_list = [f'Train: [{processed}/{total} '
                          f'({100.0 * processed / total:.0f}%)]']
            stats.print(epoch, i, samples_sec, header=header_list)
            batch_st = datetime.now()

        stats.update()

    t_st = datetime.now()
    for i, (noisy, clean, noise, idx) in enumerate(validation_loader):
        net.eval()
        with torch.no_grad():

            noisy = noisy.to(device)
            clean = clean.to(device)
            
            if (args.spectrogram == 0):
                noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft, None)
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, None)
            elif(args.spectrogram == 1):
                noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft, stft_transform)
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, stft_transform)
            else:
                noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft, mel_transform)
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, mel_transform)

            denoised_abs = net(noisy_abs)
            if (args.spectrogram == 0):
                clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft, None)
            elif(args.spectrogram == 1):
                clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft, inv_stft_transform)
            else:
                clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft, 2)

            score = si_snr(clean_rec, clean)
            if torch.isnan(score).any():
                score[torch.isnan(score)] = 0
            loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))
            if torch.isnan(loss).any():
                loss[torch.isnan(loss)] = 0
            stats.validation.correct_samples += torch.sum(score).item()
            stats.validation.loss_sum += loss.item()
            stats.validation.num_samples += noisy.shape[0]

            processed = i * validation_loader.batch_size
            total = len(validation_loader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            samples_sec = time_elapsed / (i + 1) / validation_loader.batch_size
            header_list = [f'Valid: [{processed}/{total} ' f'({100.0 * processed / total:.0f}%)]']
            stats.print(epoch, i, samples_sec, header=header_list)

