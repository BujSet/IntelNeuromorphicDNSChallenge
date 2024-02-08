# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lava.lib.dl import slayer
from audio_dataloader import DNSAudio
from models.pinna import Network
from snr import si_snr
from hrtfs.cipic_db import CipicDatabase
from snr import si_snr
import torchaudio
from noisyspeech_synthesizer import segmental_snr_mixer


def collate_fn(batch):
    noisy, clean, noise = [], [], []

    indices = torch.IntTensor([s[4] for s in batch])

    for sample in batch:
        noisy += [torch.FloatTensor(sample[0])]
        clean += [torch.FloatTensor(sample[1])]
        noise += [torch.FloatTensor(sample[2])]

    return torch.stack(noisy), torch.stack(clean), torch.stack(noise), indices


def stft_splitter(audio, n_fft=512):
    with torch.no_grad():
        audio_stft = torch.stft(audio,
                                n_fft=n_fft,
                                onesided=True,
                                return_complex=True)
        return audio_stft.abs(), audio_stft.angle()


def stft_mixer(stft_abs, stft_angle, n_fft=512):
    return torch.istft(torch.complex(stft_abs * torch.cos(stft_angle),
                                        stft_abs * torch.sin(stft_angle)),
                        n_fft=n_fft, onesided=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',
                        type=int,
                        default=32,
                        help='batch size for dataloader')
    parser.add_argument('-lr',
                        type=float,
                        default=0.01,
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
    parser.add_argument('-epoch',
                        type=int,
                        default=50,
                        help='number of epochs to run')
    parser.add_argument('-spectrogram',
                        type=int,
                        default=0,
                        help='What type of FT to use, 0: torch.stft, 1: torchaudio.Transforms.Spectrogram, 2:numpy melspec')
    parser.add_argument('-path',
                        type=str,
                        default='./',
                        help='dataset path')
    parser.add_argument('-ssnns',
                        dest="ssnns",
                        action="store_true",
                        help='Flag to turn on Spatial Separation of Noise and Speech')
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
                        default=624,
                        help='Index into CIPIC source directions to orient the speech to ')
    parser.add_argument('-speechFilterChannel',
                        type=int,
                        default=0,
                        help='Channel to orient the speech to ')
    parser.add_argument('-noiseFilterOrient',
                        type=int,
                        default=600,
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
    writer = SummaryWriter('runs/' + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    lam = args.lam

    device = torch.device('cuda:0')
    out_delay = args.out_delay
    net = torch.nn.DataParallel(Network(args.threshold,
                                            args.tau_grad,
                                            args.scale_grad,
                                            args.dmax,
                                            args.out_delay).to(device),
                                    device_ids=[0])
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
    conv_transform = torchaudio.transforms.Convolve("same").to(device)

    # Define optimizer module.
    optimizer = torch.optim.RAdam(net.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-5)

    train_set = DNSAudio(root=args.path + 'training_set/')
    train_loader = DataLoader(train_set,
                              batch_size=args.b,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=4,
                              pin_memory=True)

    stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                       accuracy_unit='dB')

    if (args.ssnns):
        CIPICSubject = CipicDatabase.subjects[args.cipicSubject]
        speechFilter  = torch.from_numpy(CIPICSubject.getHRIRFromIndex(args.speechFilterOrient, args.speechFilterChannel)).float()
        speechFilter  = speechFilter.to(device)
        noiseFilter   = torch.from_numpy(CIPICSubject.getHRIRFromIndex(args.noiseFilterOrient, args.noiseFilterChannel)).float()
        noiseFilter   = noiseFilter.to(device)
        print("Using Subject " + str(args.cipicSubject) + " for spatial sound separation...")
        print("\tPlacing speech at orient " + str(args.speechFilterOrient) + " from channel " + str(args.speechFilterChannel))
        print("\tPlacing noise at  orient " + str(args.noiseFilterOrient) + " from channel " + str(args.noiseFilterChannel))
    for epoch in range(args.epoch):
        t_st = datetime.now()
        for i, (noisy, clean, noise, idx) in enumerate(train_loader):
            net.train()
            if (args.ssnns):
                noise = noise.to(device)
                clean = clean.to(device)
                ssl_noise = torch.zeros(args.b, 480000).to(device)
                ssl_clean = torch.zeros(args.b, 480000).to(device)
                ssl_snrs  = torch.zeros(args.b, 1).to(device)
                ssl_targlvls= torch.zeros(args.b, 1).to(device)
                for batch_idx in range(args.b):
                    ssl_noise[batch_idx,:] = conv_transform(noise[batch_idx,:], noiseFilter)
                    ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], speechFilter)
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

            else:
                ssl_noise = noise.to(device)
                ssl_noisy = noisy.to(device)
                ssl_clean = clean.to(device)

            if (args.spectrogram == 0):
                noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, None)
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, None)
            elif(args.spectrogram == 1):
                noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, stft_transform)
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, stft_transform)
            else:
                noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, stft_transform)
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, stft_transform)
#                noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, mel_transform)
#                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, mel_transform)

            noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft)
            clean_abs, clean_arg = stft_splitter(clean, args.n_fft)

            denoised_abs = net(noisy_abs)
            noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
            clean_abs = slayer.axon.delay(clean_abs, out_delay)
            clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

            clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft)

            score = si_snr(clean_rec, clean)
            loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))

            assert torch.isnan(loss) == False

            optimizer.zero_grad()
            loss.backward()
            net.validate_gradients()
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

        writer.add_scalar('Loss/train', stats.training.loss, epoch)
        writer.add_scalar('SI-SNR/train', stats.training.accuracy, epoch)

        stats.update()
    torch.save(module.state_dict(), trained_folder + '/network.pt')
    stats.save(trained_folder + '/')

    writer.flush()
    writer.close()
