# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os, sys, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append('./')
from audio_dataloader import DNSAudioCleanAndPitch, DNSAudioCleanOnly
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
import torchyin
from scipy.io import wavfile
from chtc_files.htchirp_utils import *

def chtc_print(args, string):
    if args.isCHTCJob:
        send_log_msg(string)
    else:
        print(string)

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

def freq_to_one_hot(value, freq_bins):
    one_hot = torch.zeros(len(freq_bins))
    abs_diff = torch.abs(freq_bins - value)
    min_index = torch.argmin(abs_diff)
    one_hot[min_index] = 1.0
    return one_hot

def freq_to_index(value, freq_bins):
    result = torch.zeros(1, dtype=torch.int)
    abs_diff = torch.abs(freq_bins - value)
    min_index = torch.argmin(abs_diff)
    result[0] = int(min_index)
    return result

def torchyin_pitch_estimate(args, device, speech):
    pitch = torchyin.estimate(speech, 
            sample_rate=16000, 
            pitch_min=50, 
            pitch_max=1000, 
            frame_stride=0.008, 
            threshold=args.validation_method_threshold)
    pitch = pitch.to(device)
    p1d = (3,2)
    pitch = torch.nn.functional.pad(pitch, p1d, "constant", 0)
    return pitch

def torchaudio_pitch_estimate(args, device, speech):
    torchaudio_pitch = torchaudio.functional.detect_pitch_frequency(speech,
            16000, 
            frame_time=0.008, 
            win_length=3, 
            freq_low=50, 
            freq_high=1000).squeeze()
    torchaudio_pitch = torchaudio_pitch.to(device)
    p1d = (1,1)
    torchaudio_pitch = torch.nn.functional.pad(torchaudio_pitch, p1d, "constant", 0)
    return torchaudio_pitch

def crepe_pitch_estimate(args, device, speech, files):
    num_fft_frames = int((30 * 16000 / (args.n_fft // 4)) + 1)
    crepeFreqs = torch.zeros( (args.b, num_fft_frames) )
    crepeConfs = torch.zeros( (args.b, num_fft_frames) )
    for batch_idx in range(args.b):
        file = files[batch_idx]
        pathTokens = file.split("/")
        fullPath = "/".join(pathTokens[:-2])
        crepeFilePath = os.path.join(fullPath, "crepe_pitch_annotations")
        crepeFilePath = os.path.join(crepeFilePath, "clean")
        crepeFileName = pathTokens[-1].replace(".wav", ".f0.csv")
        crepeFile = os.path.join(crepeFilePath, crepeFileName)
        with open(crepeFile) as f:
            lines = f.readlines()
            # Skip header line
            freqs = []
            confs = []
            for i in range(1, len(lines)):
                values = lines[i].split(",")
                assert(len(values) == 3)
                freqs.append(float(values[1]))
                confs.append(float(values[2]))

        freqs = torch.tensor(freqs[0:num_fft_frames])
        confs = torch.tensor(confs[0:num_fft_frames])
        crepeFreqs[batch_idx,:] = freqs
        crepeConfs[batch_idx,:] = confs
    crepeFreqs = torch.where(crepeConfs > args.validation_method_threshold, crepeFreqs, 0.0)
    crepeFreqs = crepeFreqs.to(device)
    return crepeFreqs

def predict_pitch(args, device, clean_speech, clean_files):
    if args.validation_method == "yin":
        return torchyin_pitch_estimate(args, device, clean_speech)
    elif args.validation_method == "torchaudio":
        return torchaudio_pitch_estimate(args, device, clean_speech)
    elif args.validation_method == "crepe":
        return crepe_pitch_estimate(args, device, clean_speech, clean_files)
    else:
        errorString = "[ERROR] Validation method " 
        errorString += str(args.validation_method) + " not implemented!"
        chtc_print(errorString)
        sys.exit(0)

# Based on this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6739213
# we can define raw pich accuracy and raw chroma accuracy in torch friendly formats
def calc_rpa(pred, targ):
    valid = (pred > 0) & (targ > 0)
    valid = valid & (~pred.isnan()) & (~targ.isnan()) 
    valid = valid & (~pred.isinf()) & (~targ.isinf()) 
    pred = pred[valid]
    targ = targ[valid]
    # Use 700Hz as the corner frequency since we assume pitchs will be less 1000Hz
    # https://en.wikipedia.org/wiki/Mel_scale#:~:text=In%201976%2C%20Makhoul%20and%20Cosell%20published%20the,the%20700%20Hz%20version%20again%20fits%20better.
    mel_pred = 12.0 * 100.0 * torch.log2(torch.div(pred, 700))
    mel_targ = 12.0 * 100.0 * torch.log2(torch.div(targ, 700))
    diff = torch.abs(torch.sub(mel_pred, mel_targ))
    threshold = torch.where(diff > 50,1, 0 )
    acc = torch.sum(threshold) / torch.numel(threshold)
    return acc

def calc_rca(pred, targ):
    valid = (pred > 0) & (targ > 0)
    valid = valid & (~pred.isnan()) & (~targ.isnan()) 
    valid = valid & (~pred.isinf()) & (~targ.isinf()) 
    pred = pred[valid]
    targ = targ[valid]
    # Use 700Hz as the corner frequency since we assume pitchs will be less 1000Hz
    # https://en.wikipedia.org/wiki/Mel_scale#:~:text=In%201976%2C%20Makhoul%20and%20Cosell%20published%20the,the%20700%20Hz%20version%20again%20fits%20better.
    mel_pred = 12.0 * 100.0 * torch.log2(torch.div(pred, 700))
    mel_targ = 12.0 * 100.0 * torch.log2(torch.div(targ, 700))
    diff = torch.abs(torch.sub(mel_pred, mel_targ))
    octave = diff % 1200 #diff - 12 * torch.floor(torch.div(diff, 12) +50 )
    #octave = torch.minimum(octave, 1200 - octave)
    threshold = torch.where(octave > 50,1, 0 )
    acc = torch.sum(threshold) / torch.numel(threshold)
    return acc

def run_validation_loop(args, device, validation_loader, validation_set):
    crossEntropyLosses = []
    L1Losses = []
    MSELosses = []
    RPALosses = []
    RCALosses = []
    freq_map = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=args.n_fft)).to(device)
    num_batches = int(len(validation_set) / args.b)
    for i, (clean, idx) in enumerate(validation_loader):
        with torch.no_grad():
            clean = clean.to(device)

            if (args.spectrogram == 0):
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, None)
            elif(args.spectrogram == 1):
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, stft_transform)
            else:
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, mel_transform)

            num_fft_frames = clean_abs.size()[-1] # should be set to 3751
            period = (480000.0 / num_fft_frames) / 16000.0 # ~ 0.008 == 8 msec
            fft_centers = [(i * period) + (period/2) for i in range(0, num_fft_frames)]

            clean_pitch_batched = torch.zeros( (args.b, num_fft_frames) ).to(device)
            one_hot_predicted_pitch = torch.zeros( (args.b, num_fft_frames, 257) ).to(device)
            clean_files = [None for _ in range(args.b)]
            for batch_idx in range(args.b):
                clean_file = validation_set._get_filenames(idx[batch_idx])
                clean_files[batch_idx] = clean_file
                # Compute praat pitch prediction for ground-truth in time domain
                praatSound = parselmouth.Sound(clean_file)
                praatTimeStep = 1.0*(args.n_fft//4)/praatSound.sampling_frequency
                clean_pitch = parselmouth.Sound(clean_file).to_pitch(time_step=praatTimeStep, pitch_floor=50.0, pitch_ceiling=1000.0)
                # Subsample prediction to only look at FFT frames the network also looks at
                clean_pitch_freq = [clean_pitch.get_value_at_time(center_time) for center_time in fft_centers]

                # Final clean up to deal with off-by-one and error vals
                clean_pitch_freq = torch.FloatTensor(clean_pitch_freq).to(device)
                if torch.isnan(clean_pitch_freq).any():
                    clean_pitch_freq[torch.isnan(clean_pitch_freq)] = 0
                clean_pitch_batched[batch_idx] = clean_pitch_freq

            # Now compute the selected comparative model
            predicted = predict_pitch(args, device, clean, clean_files)
                
            rpa = calc_rpa(predicted, clean_pitch_batched)
            rca = calc_rca(predicted, clean_pitch_batched)
            l1 = torch.nn.L1Loss()(predicted, clean_pitch_batched)
            if torch.isnan(l1).any():
                l1[torch.isnan(l1)] = 0
            mse = torch.nn.MSELoss()(predicted, clean_pitch_batched)
            if torch.isnan(mse).any():
                mse[torch.isnan(mse)] = 0
            for batch_idx in range(args.b):
                for frame in range(num_fft_frames):
                    one_hot_predicted_pitch[batch_idx,frame, :] = freq_to_one_hot(predicted[batch_idx, frame], freq_map)

            # next we convert the frequencies to indices so we can use cross
            # entropy loss later with praat as the correct labels
            for batch_idx in range(args.b):
                for frame in range(num_fft_frames):
                    clean_pitch_batched[batch_idx,frame] = freq_to_index(clean_pitch_batched[batch_idx, frame], freq_map)
            
            crossEntropyFrameLosses = torch.zeros( (num_fft_frames) ).to(device)
            for frame in range(num_fft_frames):
                clean_pitch_frame = clean_pitch_batched[:,frame].squeeze()
                predicted_pitch_frame = one_hot_predicted_pitch[:,frame,:].squeeze()
                frameLoss = F.cross_entropy(predicted_pitch_frame, clean_pitch_frame.long())
                crossEntropyFrameLosses[frame] = frameLoss

            if torch.isnan(crossEntropyFrameLosses).any():
                crossEntropyFrameLosses[torch.isnan(crossEntropyrameLosses)] = 0
            crossEntropyLoss = torch.mean(crossEntropyFrameLosses)

            RPALosses.append(torch.mean(rpa).item())
            RCALosses.append(torch.mean(rca).item())
            crossEntropyLosses.append(torch.mean(crossEntropyLoss).item())
            L1Losses.append(torch.mean(l1).item())
            MSELosses.append(torch.mean(mse).item())

            if args.printOutputWhileValidation or args.isCHTCJob:
                statString = "Losses [i="
                statString += str(i) + "/" + str(num_batches) + "] -> "
                statString += "RPA=" + str(torch.mean(rpa).item()) + ","
                statString += "RCA=" + str(torch.mean(rca).item()) + ","
                statString += "L1L=" + str(torch.mean(l1).item()) + ","
                statString += "MSE=" + str(torch.mean(mse).item()) + ","
                statString += "CEL=" + str(torch.mean(crossEntropyLoss).item())
                chtc_print(args, statString)
    avgRCA = sum(RCALosses) / (1.0 * len(RCALosses))
    avgRPA = sum(RPALosses) / (1.0 * len(RPALosses))
    avgL1L = sum(L1Losses) / (1.0 * len(L1Losses))
    avgMSE = sum(MSELosses) / (1.0 * len(MSELosses))
    avgCEL = sum(crossEntropyLosses) / (1.0 * len(crossEntropyLosses))
    return avgRPA, avgRCA, avgL1L, avgMSE, avgCEL

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
    parser.add_argument('-validation_method',
                        type=str,
                        default='',
                        help='What type of validation method to run: [crepe, yin, torchaudio]')
    parser.add_argument('-validation_method_threshold',
                        type=float,
                        default=-1.0,
                        help='Threshold for comparison if needed')
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

    assert(args.validation_method != "")
    validation_methods = {"crepe", "yin", "torchaudio"}
    if not args.validation_method in validation_methods:
        errorString = "[ERROR] Unknown validation method: " 
        errorString += str(args.validation_method) + "\n"
        errorString += "Must be one of:" +str(validation_methods)
        chtc_print(args, errorString)
        assert(False)
    valid_method_info_string = "[INFO] Running validation with method:"
    valid_method_info_string += str(args.validation_method)
    chtc_print(args, valid_method_info_string)

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
        chtc_print(args, "[INFO] Running validation on validation set")
        chosenDataset = DNSAudioCleanOnly(root=args.path + 'validation_set/', maxFiles=args.validation_samples)
    elif args.useTrainingSet:
        chtc_print(args, "[INFO] Running validation on training set")
        chosenDataset = DNSAudioCleanOnly(root=args.path + 'training_set/', maxFiles=args.training_samples)
    else:
        errorString = "[ERROR] Dataset for validation not chosen! "
        errorString += "Must specify one of [training, validation]"
        chtc_print(args, errorString)
        assert(False)

    assert(chosenDataset != None)
    chosenDataloader = DataLoader(chosenDataset,
                               batch_size=args.b,
                               shuffle=False,
                               collate_fn=chosenDataset.collate_fn,
                               num_workers=args.dataloader_workers,
                               prefetch_factor=args.dataloader_prefetch_factor,
                               pin_memory=True)
    if args.validation_method == 'crepe' or args.validation_method == "yin":
        if args.validation_method_threshold >= 0.0:
            thresholdString = "[INFO] Running validation with threshold set to "
            thresholdString += str(args.validation_method_threshold)
            chtc_print(args, thresholdString)
    finalRPA, finalRCA, finalL1L, finalMSE, finalCEL = run_validation_loop(args, device, chosenDataloader, chosenDataset)
    statusString  = "Completed validation on " 
    statusString += "validation" if args.useValidationSet else ""
    statusString += "training" if args.useTrainingSet else ""
    statusString += " set ["
    statusString += "RPA=" + str(finalRPA) + ","
    statusString += "RCA=" + str(finalRCA) + ","
    statusString += "L1L=" + str(finalL1L) + ","
    statusString += "MSE=" + str(finalMSE) + ","
    statusString += "CEL=" + str(finalCEL) 
    statusString += "]"
    chtc_print(args, statusString)
