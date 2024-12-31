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
from audio_dataloader import DNSAudioNoNoisy
from hrtfs.cipic_db import CipicDatabase 
from snr import si_snr
import torchaudio
from noisyspeech_synthesizer import segmental_snr_mixer
import random
import time
from torch.profiler import profile, record_function, ProfilerActivity

def calc_rms(x):
    return torch.sqrt(torch.mean(torch.square(x)))

def bias_normalize(x, bias):
    div = torch.add(torch.max(torch.abs(x)), bias) 
    return torch.div(x, div)

def scale_by_target_level(x, targ, rms, bias):
    norm = torch.pow(10, torch.div(targ, 20.0))
    scalar = torch.div(norm, torch.add(rms, bias))
    return torch.mul(x, scalar)

def _segmental_snr_mixer(clean, noise, snr,
                        target_level,
                        noise_stream,
                        clean_stream, 
                        target_level_lower=-35,
                        target_level_higher=-15,
                        clipping_threshold=0.99,
                        EPS = 2.220446049250313e-16
                        ):
    '''Function to mix clean speech and noise at various segmental SNR levels'''
    epsT = torch.tensor([EPS], device="cuda")
    clipT = torch.tensor([clipping_threshold], device="cuda")
    normSNRT = torch.pow(10, torch.div(snr, 20.0))

    # TODO should only calculate the RMS of the 'active' windows, but
    # for now we just use the whole audio sample

    with torch.cuda.stream(clean_stream):
        ssl_clean = bias_normalize(clean, epsT)
        clean_rms = calc_rms(ssl_clean)
        ssl_clean = scale_by_target_level(ssl_clean, target_level, clean_rms, epsT)

    with torch.cuda.stream(noise_stream):
        ssl_noise = bias_normalize(noise, epsT)
        noise_rms = calc_rms(ssl_noise)
        ssl_noise = scale_by_target_level(ssl_noise, target_level, noise_rms, epsT)

    torch.cuda.synchronize()
    # Adjust noise to SNR level
    noise_scalar = torch.div(torch.div(clean_rms, normSNRT), torch.add(noise_rms, epsT))
    ssl_noise = torch.mul(ssl_noise, noise_scalar)
    ssl_noisy = torch.add(ssl_clean, ssl_noise)
    noisy_rms_level = torch.randint(
            target_level_lower,
            target_level_higher,
            (1,), device="cuda")
    noisy_rms = calc_rms(ssl_noisy)
    noisy_scalar = torch.div(torch.pow(10, torch.div(noisy_rms_level, 20.0)), torch.add(noisy_rms, epsT))
    ssl_noisy = torch.mul(ssl_noisy, noisy_scalar)
    ssl_clean = torch.mul(ssl_clean, noisy_scalar)
    ssl_noise = torch.mul(ssl_noise, noisy_scalar)
#    # check if any clipping happened
#    needToClip = torch.gt(torch.abs(ssl_noisy), 0.99).any() # 0.99 is the clipping threshold 
#    if (needToClip):
#        noisyspeech_maxamplevel = torch.div(torch.max(torch.abs(ssl_noisy)), torch.sub(clipT, epsT))
#        ssl_noisy = torch.div(ssl_noisy, noisyspeech_maxamplevel)
#        ssl_noise = torch.div(ssl_noise, noisyspeech_maxamplevel)
#        ssl_clean = torch.div(ssl_clean, noisyspeech_maxamplevel)
    # Checking for clipping requires a GPU-CPU synchronization via the .any()
    # function call. Rather than check if clipping occured, always rescale
    noisyspeech_maxamplevel = torch.div(torch.max(torch.abs(ssl_noisy)), torch.sub(clipT, epsT))
    ssl_noisy = torch.div(ssl_noisy, noisyspeech_maxamplevel)
    ssl_noise = torch.div(ssl_noise, noisyspeech_maxamplevel)
    ssl_clean = torch.div(ssl_clean, noisyspeech_maxamplevel)
    return ssl_clean, ssl_noise, ssl_noisy

def synthesizeNoisySpeech(clean, noise, noisy, batchSize, 
            snr,
            targetLevel, noise_stream, clean_stream):
    for i in range(batchSize):
        clean[i, :], noise[i,:], noisy[i,:] = _segmental_snr_mixer(clean[i,:], noise[i,:], 
            snr[i], 
            targetLevel[i], noise_stream, clean_stream)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',
                        type=int,
                        default=32,
                        help='batch size for dataloader')
    parser.add_argument('-dataloader_workers',
                        type=int,
                        default=4,
                        help='batch size for dataloader')
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
    parser.add_argument('-validation_samples',
                        type=int,
                        default=60000,
                        help='Number of samples validation should use, supports small dataset subset')
    parser.add_argument('-print_output_while_validation',
                        dest='printOutputWhileValidation', 
                        action='store_true',
                        help='Switch flag to print score after every mini-batch during validation')
    parser.add_argument('-hiddenLayerWidths',
                        type=int,
                        default=512,
                        help='# of nuerons in hidden layers')
    # CIPIC Filter Parameters

    # ID:21 ==> Mannequin with large pinna
    # ID 165 ==> Mannequin with small pinna
    # The rest are real subjects
    parser.add_argument('-cipicSubject',
                        type=int,
                        default=12,
                        help='CIPIC subject ID for pinna filters')
    # 0 ==> right ear?
    # 1 ==> left ear?
    parser.add_argument('-cipicChannel',
                        type=int,
                        default=0,
                        help='Channel to use for pinna filter choice')
    # Spatially distribute the sound sources, (azimuth, elevation)
    # index = 624 ==> (0,  90)
    # index = 600 ==> (0, -45)
    # We choose filters in the midsagittal plane, so either selecting
    # which channels is read from 'should' be irrelevant
    parser.add_argument('-speechFilterOrient',
                        type=int,
                        default=608,
                        help='Index into CIPIC source directions to orient the speech to ')
    parser.add_argument('-noiseFilterOrient',
                        type=int,
                        default=608,
                        help='Index into CIPIC source directions to orient the noise to ')

    parser.add_argument('-print_validation_results_header',
                        dest='printValidationResultsHeader', 
                        action='store_true',
                        help='Switch flag to print validation results header (useful for CHTC)')

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device('cuda:0')

    conv_transform = torchaudio.transforms.Convolve("same").to(device)

    # Input audio is recorded at 16 kHz, but CIPIC HRTFs are at 44.1 kHz
    downsampler= torchaudio.transforms.Resample(44100, 16000, dtype=torch.float32).to(device)

    validation_set = DNSAudioNoNoisy(root=args.path + 'validation_set/', maxFiles=args.validation_samples)
    validation_loader = DataLoader(validation_set,
                               batch_size=args.b,
                               shuffle=False,
                               collate_fn=validation_set.collate_fn,
                               num_workers=args.dataloader_workers,
                               pin_memory=True)
    numIters = round(args.validation_samples / args.b)
    CIPICSubject = CipicDatabase.subjects[args.cipicSubject]
    with torch.no_grad():
        speechFilter = CIPICSubject.getHRIRFromIndex(args.speechFilterOrient, args.cipicChannel)
        speechFilter = torch.from_numpy(speechFilter).float().to(device)
        speechFilter = downsampler(speechFilter) 
        noiseFilter = CIPICSubject.getHRIRFromIndex(args.noiseFilterOrient, args.cipicChannel)
        noiseFilter  = torch.from_numpy(noiseFilter).float().to(device)
        noiseFilter = downsampler(noiseFilter) 
        ssl_noise = torch.zeros(args.b, 480000, device="cuda")
        ssl_clean = torch.zeros(args.b, 480000, device="cuda")
        ssl_noisy = torch.zeros(args.b, 480000, device="cuda")
        ssl_snrs  = torch.zeros(args.b, 1, device="cuda")
        ssl_targlvls= torch.zeros(args.b, 1, device="cuda")
        runningScore = torch.zeros(1, device="cuda")
 
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    noise_stream = torch.cuda.Stream()
    clean_stream = torch.cuda.Stream()
    synth_time = 0
    score_time = 0
    load_time = 0
    start_time = time.time()
#    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
#        with record_function("passive_pinna_validation_score"):
    with torch.no_grad():
        for i, (clean, noise, idx) in enumerate(validation_loader):
            load_start_time = time.time()
            with torch.cuda.stream(noise_stream):
                noise = noise.to(device, non_blocking=True)
                for batch_idx in range(args.b):
                    ssl_noise[batch_idx,:] = conv_transform(noise[batch_idx,:], noiseFilter)
            with torch.cuda.stream(clean_stream):
                clean = clean.to(device, non_blocking=True)
                for batch_idx in range(args.b):
                    ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], speechFilter)
       
            for batch_idx in range(args.b):
                clean_file, noise_file, metadata = validation_set._get_filenames(idx[batch_idx])
                ssl_snrs[batch_idx] = metadata['snr']
                ssl_targlvls[batch_idx] = metadata['target_level']
                       
            torch.cuda.synchronize()
            load_end_time = time.time()
       
            synth_start_time = time.time()
            synthesizeNoisySpeech(
                               ssl_clean, 
                               ssl_noise, 
                               ssl_noisy,
                               args.b, 
                               ssl_snrs,
                               ssl_targlvls,
                               noise_stream,
                               clean_stream
                               )
            synth_end_time = time.time()
               
            score_start_time = time.time()
            score = si_snr(ssl_noisy, ssl_clean)
            score = torch.nan_to_num(score, nan=0.0)
            runningScore = torch.add(runningScore, torch.mean(score), alpha=1.0/float(numIters))
            if args.printOutputWhileValidation:
                statString = "Valid [" + str(i) + "] -> "
                statString += str(torch.mean(score).item()) + " SI-SNR dB"
                print(statString)
            score_end_time = time.time()
       
            synth_time += synth_end_time - synth_start_time
            score_time += score_end_time - score_start_time
            load_time += load_end_time - load_start_time

    end_time = time.time()
#    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    averageValidationScore = runningScore.item()
    if args.printValidationResultsHeader:
        headerString = "Subject, Channel, Speech Orient, "
        headerString += "Noise Orient, "
        headerString += "Final Validation Score SI-SNR (dB), "
        headerString += "ExecTime, "
        headerString += "BatchSize, "
        headerString += "Dataloader Num Workers, "
        headerString += "Synthesis Time, "
        headerString += "Score Compute Time, "
        headerString += "Batch Load Time"
        print(headerString)
    resultString  = str(args.cipicSubject) + "," + str(args.cipicChannel) + "," 
    resultString += str(args.speechFilterOrient) + "," + str(args.noiseFilterOrient) + "," 
    resultString += str(averageValidationScore) + "," + str(end_time - start_time) + ","
    resultString += str(args.b) + "," + str(args.dataloader_workers) + ","
    resultString += str(synth_time) + "," + str(score_time) + "," + str(load_time)
    print(resultString)
