# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os, sys, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append('./')
from audio_dataloader import DNSAudioCleanOnly
from audio_dataloader import DNSAudioAndCrepeCleanOnly
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import soundfile as sf

from lava.lib.dl import slayer
import torchaudio
import random
import parselmouth, librosa, time
from scipy.io import wavfile
from chtc_files.htchirp_utils import *

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

def crepe_detect_fundamental_frequency(filePath, wavLength=30.0, stepSizeMsec=8, threshold=0.5):
    sr, audio = wavfile.read(filePath)
    timesarray, freq, conf, activation = crepe.predict(audio, sr, model_capacity='full', step_size=stepSizeMsec, center=True, viterbi=True)
    endIdx = np.searchsorted(timesarray, 30.0, side="left") + 1
    timesarray = torch.from_numpy(timesarray[:endIdx]).float()
    freq = torch.from_numpy(freq[:endIdx]).float()
    conf = torch.from_numpy(conf[:endIdx]).float()
    clippedFreq = torch.where(conf >= threshold, freq, 0.0)
    return clippedFreq

def crepe_collate_pitch_estimation(step, times, values, confs, threshold=-1.0):
    pass
'''
    freq = torch.from_numpy(freq[:endIdx]).float()
    conf = torch.from_numpy(conf[:endIdx]).float()

    if threshold >= 0.0:
        clippedFreq = torch.where(conf >= threshold, freq, 0.0)
        return clippedFreq
    return freqs
    '''

def freq_to_one_hot(value, freq_bins):
    one_hot = torch.zeros(len(freq_bins))
    abs_diff = torch.abs(freq_bins - value)
    min_index = torch.argmin(abs_diff)
    one_hot[min_index] = 1.0
    return one_hot

def run_validation_loop(args, validation_loader, validation_set):
    validationLosses = []
    freq_map = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=args.n_fft)).to(device)
    for i, (clean, crepe_times, crepe_values, crepe_confs, idx) in enumerate(validation_loader):
        with torch.no_grad():
            clean = clean.to(device)

            if (args.spectrogram == 0):
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, None)
            elif(args.spectrogram == 1):
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, stft_transform)
            else:
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, mel_transform)

            num_fft_frames = clean_abs.size()[-1]
            period = (480000.0 / num_fft_frames) / 16000.0
            one_hot_clean_pitch = torch.zeros(clean_abs.size()).to(device)
            one_hot_crepe_pitch = torch.zeros(clean_abs.size()).to(device)
            print("Period: " + str(period))
            for batch_idx in range(args.b):
                clean_file = validation_set._get_filenames(idx[batch_idx])
                # Compute praat pitch prediction for ground-truth in time domain
                praatSound = parselmouth.Sound(clean_file)
                praatTimeStep = 1.0*(args.n_fft//4)/praatSound.sampling_frequency
                print("PraatTimeStep: " + str(praatTimeStep))
                clean_pitch = parselmouth.Sound(clean_file).to_pitch(time_step=praatTimeStep, pitch_floor=50.0, pitch_ceiling=1000.0)
                # Subsample prediction to only look at FFT frames the network also looks at
                clean_pitch_freq = [clean_pitch.get_value_at_time((i * period) + (period/2)) for i in range(0, num_fft_frames)]

                # Final clean up to deal with off-by-one and error vals
                clean_pitch_freq = torch.FloatTensor(clean_pitch_freq).to(device)
                if torch.isnan(clean_pitch_freq).any():
                    clean_pitch_freq[torch.isnan(clean_pitch_freq)] = 0

                # Now compute other comparative models, first we look at crepe
                print("Need to implement reading from crepe files rather than using crepe package here")
                print(crepe_times)
                print(crepe_values)
                print(crepe_confs)
                crepe_collate_pitch_estimation(praatTimeStep, crepe_times, crepe_values, crepe_confs, args.crepeThreshold):

                sys.exit(0)
#                crepe_pitch_freq = crepe_detect_fundamental_frequency(clean_file, wavLength=30.0, stepSizeMsec=praatTimeStep*1000.0, threshold=args.crepeThreshold)
#                crepe_pitch_freq = crepe_pitch_freq.to(device)
                # Convert to 1-hot vector for easier-to-learn loss function, i.e. network does not need to 
                # learn to perform ISTFT
#                for frame in range(num_fft_frames):
#                    one_hot_clean_pitch[batch_idx,:, frame] = freq_to_one_hot(clean_pitch_freq[frame], freq_map) 
#                    one_hot_crepe_pitch[batch_idx,:, frame] = freq_to_one_hot(crepe_pitch_freq[frame], freq_map) 
#            one_hot_clean_pitch.to(device)
#            one_hot_crepe_pitch.to(device)
            
#            loss = F.cross_entropy(one_hot_clean_pitch, one_hot_crepe_pitch)
             
#            if torch.isnan(loss).any():
#                loss[torch.isnan(loss)] = 0
#            assert torch.isnan(loss) == False

#            validationLosses.append(torch.mean(loss).item())

#            if args.printOutputWhileValidation or args.isCHTCJob:
#                statString = "Validation Loss [DataLoaderIdx=" + str(i) + "] -> "
#                statString += str(loss.item())
#                if args.isCHTCJob:
#                    send_log_msg(statString)
#                if args.printOutputWhileValidation:
#                    print(statString)
#    averageValidationLoss = sum(validationLosses) / (1.0 * len(validationLosses))
#    return averageValidationLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-script_name',
                        type=str,
                        default='other_models/pitch_validate_comparisons',
                        help='name of this file')
    parser.add_argument('-gpu',
                        type=int,
                        default=[0],
                        help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',
                        type=int,
                        default=32,
                        help='batch size for dataloader')
    parser.add_argument('-dataloader_workers',
                        type=int,
                        default=4,
                        help='batch size for dataloader')
    parser.add_argument('-dataloader_prefetch_factor',
                        type=int,
                        default=2,
                        help='prefetch factor for dataloader')
    parser.add_argument('-n_fft',
                        type=int,
                        default=512,
                        help='number of FFT specturm, hop is n_fft // 4')
    parser.add_argument('-exp',
                        type=str,
                        default='',
                        help='experiment differentiater string')
    parser.add_argument('-seed',
                        type=int,
                        default=1231241234,
                        help='random seed of the experiment')
    parser.add_argument('-spectrogram',
                        type=int,
                        default=0,
                        help='What type of FT to use, 0: torch.stft, 1: torchaudio.Transforms.Spectrogram, 2:numpy melspec')
    parser.add_argument('-path',
                        type=str,
                        default='../../',
                        help='dataset path')
    parser.add_argument('-use_validation_set',
                        dest='useValidationSet', 
                        action='store_true',
                        help='Use the validation set for comparison')
    parser.add_argument('-validation_samples',
                        type=int,
                        default=60000,
                        help='Number of samples validation should use from validation dataset, supports small dataset subset')
    parser.add_argument('-use_training_set',
                        dest='useTrainingSet', 
                        action='store_true',
                        help='Use the training set for comparison')
    parser.add_argument('-training_samples',
                        type=int,
                        default=60000,
                        help='Number of samples validation should use from training datastet, supports small dataset subset')
    parser.add_argument('-print_output_while_validation',
                        dest='printOutputWhileValidation', 
                        action='store_true',
                        help='Switch flag to print score after every mini-batch during validation')
    parser.add_argument('-crepeThreshold',
                        type=float,
                        default=-1.0,
                        help='Threshold for CREPE comparison')
    parser.add_argument('-is_CHTC_job',
                        dest='isCHTCJob', 
                        action='store_true',
                        help='Switch flag to indicate if this job was run on CHTC')

    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}'.format(args.seed)

    assert(args.spectrogram == 0 or args.spectrogram == 1 or args.spectrogram == 2)
    device = torch.device('cuda:{}'.format(args.gpu[0]))
    # Get CUDA capability
    device_cap = torch.cuda.get_device_capability()
    torch_compile_capable = False
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        torch_compile_capable = True
        print("Detected device capable of using torch.compile, will attempt to use for torch operations")

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

    chosenDataset = None
    if args.useValidationSet:
        if args.isCHTCJob:
            send_log_msg("Running validation on validation set")
        else:
            print("Running validation on validation set")
        chosenDataset = DNSAudioCleanOnly(root=args.path + 'validation_set/', maxFiles=args.validation_samples)
    elif args.useTrainingSet:
        if args.isCHTCJob:
            send_log_msg("Running validation on training set")
        else:
            print("Running validation on training set")
        chosenDataset = DNSAudioAndCrepeCleanOnly(root=args.path + 'training_set/', maxFiles=args.training_samples)
    else:
        if args.isCHTCJob:
            send_log_msg("Dataset for validation not chosen! Must specify training or validation set to be used")
        else:
            print("Dataset for validation not chosen!")
            print("    Must specify training or validation set to be used")

    assert(chosenDataset != None)
    chosenDataloader = DataLoader(chosenDataset,
                               batch_size=args.b,
                               shuffle=False,
                               collate_fn=chosenDataset.collate_fn,
                               num_workers=args.dataloader_workers,
                               prefetch_factor=args.dataloader_prefetch_factor,
                               pin_memory=True)
    finalValidationLoss = run_validation_loop(args, chosenDataloader, chosenDataset)
    statusString  = "Completed validation on " 
    statusString += "validation" if args.useValidationSet else ""
    statusString += "training" if args.useTrainingSet else ""
    statusString += " set [loss=" 
    statusString += str(finalValidationLoss) + "]"
    if args.isCHTCJob:
        send_log_msg("Created trainind set data loader")
    else:
        print(statusString)
