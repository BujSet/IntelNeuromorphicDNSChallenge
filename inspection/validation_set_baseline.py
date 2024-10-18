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
import sys
sys.path.append('./')
from audio_dataloader import DNSAudio
from snr import si_snr
import torchaudio

def collate_fn(batch):
    noisy, clean, noise = [], [], []

    indices = torch.IntTensor([s[4] for s in batch])

    for sample in batch:
        noisy += [torch.FloatTensor(sample[0])]
        clean += [torch.FloatTensor(sample[1])]
        noise += [torch.FloatTensor(sample[2])]

    return torch.stack(noisy), torch.stack(clean), torch.stack(noise), indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',
                        type=int,
                        default=32,
                        help='batch size for dataloader')
    parser.add_argument('-seed',
                        type=int,
                        default=None,
                        help='random seed of the experiment')
    parser.add_argument('-path',
                        type=str,
                        default='./',
                        help='dataset path')
    parser.add_argument('-validation_samples',
                        type=int,
                        default=60000,
                        help='Number of samples validation should use, supports small dataset subset')
    parser.add_argument('-print_output_while_validation',
                        dest='printOutputWhileValidation', 
                        action='store_true',
                        help='Switch flag to print score after every mini-batch during validation')

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device('cuda:0')

    conv_transform = torchaudio.transforms.Convolve("same").to(device)

    # Input audio is recorded at 16 kHz, but CIPIC HRTFs are at 44.1 kHz
    downsampler= torchaudio.transforms.Resample(44100, 16000, dtype=torch.float32).to(device)

    validation_set = DNSAudio(root=args.path + 'validation_set/', maxFiles=args.validation_samples)
    validation_loader = DataLoader(validation_set,
                               batch_size=args.b,
                               shuffle=True,
                               collate_fn=collate_fn,
                               num_workers=4,
                               pin_memory=True)
 
    validationScores = []
    for i, (noisy, clean, noise, idx) in enumerate(validation_loader):
        ssl_noise = noise.to(device)
        ssl_clean = clean.to(device)
        ssl_noisy = noisy.to(device)

        with torch.no_grad():
            score = si_snr(ssl_noisy, ssl_clean)
            if torch.isnan(score).any():
                score[torch.isnan(score)] = 0

            validationScores.append(torch.mean(score).item())
            if args.printOutputWhileValidation:
                statString = "Valid [" + str(i) + "] -> "
                statString += str(torch.mean(score).item()) + " SI-SNR dB"
                print(statString)

    averageValidationScore = sum(validationScores) / (1.0 * len(validationScores))
    print("Final validation score: " + str(averageValidationScore) + " SI-SNR (dB)")
