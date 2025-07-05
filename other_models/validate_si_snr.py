import os, sys, math
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
from chtc_files.htchirp_utils import *

def chtc_print(args, string):
    if args.isCHTCJob:
        send_log_msg(string)
    else:
        print(string)

def calc_rms(x):
    return torch.sqrt(torch.mean(torch.square(x)))

def bias_normalize(x, bias):
    div = torch.add(torch.max(torch.abs(x)), bias) 
    return torch.div(x, div)

def scale_by_target_level(x, targ, rms, bias):
    norm = torch.pow(10, torch.div(targ, 20.0))
    scalar = torch.div(norm, torch.add(rms, bias))
    return torch.mul(x, scalar)

def _segmental_snr_mixer(device, clean, noise, snr,
                        target_level,
                        target_level_lower=-35,
                        target_level_higher=-15,
                        clipping_threshold=0.99,
                        EPS = 2.220446049250313e-16
                        ):
    '''Function to mix clean speech and noise at various segmental SNR levels'''
    epsT = torch.tensor([EPS]).to(device)
    clipT = torch.tensor([clipping_threshold]).to(device)
    normSNRT = torch.pow(10, torch.div(snr, 20.0))

    # TODO should only calculate the RMS of the 'active' windows, but
    # for now we just use the whole audio sample

    ssl_clean = bias_normalize(clean, epsT)
    clean_rms = calc_rms(ssl_clean)
    ssl_clean = scale_by_target_level(ssl_clean, target_level, clean_rms, epsT)

    ssl_noise = bias_normalize(noise, epsT)
    noise_rms = calc_rms(ssl_noise)
    ssl_noise = scale_by_target_level(ssl_noise, target_level, noise_rms, epsT)

    # Adjust noise to SNR level
    noise_scalar = torch.div(torch.div(clean_rms, normSNRT), torch.add(noise_rms, epsT))
    ssl_noise = torch.mul(ssl_noise, noise_scalar)
    ssl_noisy = torch.add(ssl_clean, ssl_noise)
    noisy_rms_level = torch.randint(
            target_level_lower,
            target_level_higher,
            (1,)).to(device)
    noisy_rms = calc_rms(ssl_noisy)
    noisy_scalar = torch.div(torch.pow(10, torch.div(noisy_rms_level, 20.0)), torch.add(noisy_rms, epsT))
    ssl_noisy = torch.mul(ssl_noisy, noisy_scalar)
    ssl_clean = torch.mul(ssl_clean, noisy_scalar)
    ssl_noise = torch.mul(ssl_noise, noisy_scalar)
    # Checking for clipping requires a GPU-CPU synchronization via the .any()
    # function call. Rather than check if clipping occured, always rescale
    noisyspeech_maxamplevel = torch.div(torch.max(torch.abs(ssl_noisy)), torch.sub(clipT, epsT))
    ssl_noisy = torch.div(ssl_noisy, noisyspeech_maxamplevel)
    ssl_noise = torch.div(ssl_noise, noisyspeech_maxamplevel)
    ssl_clean = torch.div(ssl_clean, noisyspeech_maxamplevel)
    return ssl_clean, ssl_noise, ssl_noisy

def synthesizeNoisySpeech(device, clean, noise, noisy, batchSize, 
            snr,
            targetLevel
            ):
    for i in range(batchSize):
        clean[i, :], noise[i,:], noisy[i,:] = _segmental_snr_mixer(device, 
            clean[i,:], noise[i,:], 
            snr[i], 
            targetLevel[i])

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
    parser.add_argument('-dataloader_prefetch_factor',
                        type=int,
                        default=2,
                        help='prefetch factor for dataloader')
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
    parser.add_argument('-speechFilterOrientStart',
                        type=int,
                        default=608,
                        help='Index into CIPIC source directions to orient the speech to ')
    parser.add_argument('-speechFilterOrientEnd',
                        type=int,
                        default=-1,
                        help='Last index into CIPIC source directions to orient the speech to ')
    parser.add_argument('-speechFilterOrientStep',
                        type=int,
                        default=1,
                        help='Sampling step size for speech orient')
    parser.add_argument('-noiseFilterOrientStart',
                        type=int,
                        default=608,
                        help='First index (inclusive) into CIPIC source directions to orient the noise to ')
    parser.add_argument('-noiseFilterOrientStep',
                        type=int,
                        default=1,
                        help='Sampling step size for noise orient')
    parser.add_argument('-noiseFilterOrientEnd',
                        type=int,
                        default=1250,
                        help='Last index (exclusive) into CIPIC source directions to orient the noise to ')
    parser.add_argument('-print_validation_results_header',
                        dest='printValidationResultsHeader', 
                        action='store_true',
                        help='Switch flag to print validation results header (useful for CHTC)')
    parser.add_argument('-is_CHTC_job',
                        dest='isCHTCJob', 
                        action='store_true',
                        help='Switch flag to indicate if this job was run on CHTC')
    parser.add_argument('-epochsEarlyEndBuffer',
                        type=int,
                        default=3,
                        help='When early stopping is supporting, this value indicates how early to stop')
    parser.add_argument('-enableTorchCompile',
                        dest='enableTorchCompile', 
                        action='store_true',
                        help='Switch flag to indicate whether to optimize with torch compile')

    args = parser.parse_args()
    if args.speechFilterOrientEnd == -1:
        args.speechFilterOrientEnd = args.speechFilterOrientStart + 1

    if args.seed is not None:
        torch.manual_seed(args.seed)

    chtc_print(args, "[INFO] args.b set to: " + str(args.b))

    deviceString = "cuda:0" if torch.cuda.is_available() else "cpu"
    chtc_print(args, "[INFO] Device string set to " + str(deviceString))
    device = torch.device(deviceString)
    torch_compile_capable = False
    if deviceString == "cuda:0":
        # Get CUDA capability
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            torch_compile_capable = True
            infoString = "[INFO] Detected device capable of using "
            infoString += "torch.compile" 
            if args.enableTorchCompile:
                infoString += ", will attempt to use for " 
                infoString += "torch operations."
                optSynthesizeNoisySpeech = torch.compile(synthesizeNoisySpeech)
            else:
                infoString += ", but config says not to use"
            chtc_print(args, infoString)

        cudaDeviceName = torch.cuda.get_device_name(0)
        chtc_print(args, "[INFO] Running on GPU: " + str(torch.cuda.get_device_name(0)))
        cudaDeviceMemory = torch.cuda.get_device_properties(0).total_memory
        chtc_print(args, f"[INFO] Available GPU memory (estimated): {cudaDeviceMemory / (1024**3):.2f} GB")
        # On NVIDIA L40 44GB memory, see these infra param perfs
        # ExecTime, BatchSize, Dataloader Num Workers, Dataloader Prefetch Factor, Sample Size (MB)
        # 331.4012989997864,16,8,4,58.59430694580078
        # 316.9886281490326,32,4,2,117.18805694580078
        # 317.27633690834045,60,4,2,219.7
        # 317.5880286693573,75,4,2,274.6587600708008
        # 318.55631279945374,75,4,4,274.6587600708008
        # 347.68118691444397,75,8,4,274.6587600708008
        # 337.07224130630493,100,4,2,366.2114944458008
        #
        # On NVIDIA A100-SXM4-40GB 39GB memory, see these infra param perfs
        # ExecTime, BatchSize, Dataloader Num Workers, Dataloader Prefetch Factor, Sample Size (MB)
        # 889.7344930171967,32,4,2,117.18805694580078
        # 955.3944482803345,60,4,2,219.72711944580078

    conv_transform = torchaudio.transforms.Convolve("same").to(device)

    # Input audio is recorded at 16 kHz, but CIPIC HRTFs are at 44.1 kHz
    downsampler= torchaudio.transforms.Resample(44100, 16000, dtype=torch.float32).to(device)

    validation_set = DNSAudioNoNoisy(root=args.path + 'validation_set/', maxFiles=args.validation_samples)
    validation_loader = DataLoader(validation_set,
                               batch_size=args.b,
                               shuffle=False,
                               collate_fn=validation_set.collate_fn,
                               num_workers=args.dataloader_workers,
                               prefetch_factor=args.dataloader_prefetch_factor,
                               pin_memory=True)
    numIters = round(args.validation_samples / args.b)
    if args.validation_samples % args.b != 0:
        chtc_print(args, "[WARN] batch size to non-perfect multiple, likely to crash...")

    CIPICSubject = CipicDatabase.subjects[args.cipicSubject]
    with torch.no_grad():
        ssl_noise = torch.zeros(args.b, 480000).to(device)
        ssl_clean = torch.zeros(args.b, 480000).to(device)
        ssl_noisy = torch.zeros(args.b, 480000).to(device)
        ssl_snrs  = torch.zeros(args.b, 1).to(device)
        ssl_targlvls= torch.zeros(args.b, 1).to(device)
        runningScore = torch.zeros(1).to(device)
 
    if args.isCHTCJob:
        infoString = "[INFO] Detected that this instance in running in a CHTC Job"
        if "cuda" in deviceString:
            infoString += " with " + get_gpu_time_remaining()
            infoString += " time remaining."
            chtc_print(args, infoString)
        if "cpu" in deviceString:
            infoString += " with " + get_cpu_time_remaining()
            infoString += " time remaining."
            chtc_print(args, infoString)
    iterationLatencies = []
    orientationPairs = []
    for so in range(args.speechFilterOrientStart, args.speechFilterOrientEnd, args.speechFilterOrientStep):
        for no in range(args.noiseFilterOrientStart, args.noiseFilterOrientEnd, args.noiseFilterOrientStep):
            orientationPairs.append( (so, no) )

    infoString = "[INFO] looking at " + str(len(orientationPairs)) +" orientation pairs varying"
    infoString += " speech: (" + str(args.speechFilterOrientStart) + "," + str(args.speechFilterOrientEnd)
    infoString += "," + str(args.speechFilterOrientStep) + ") and noise: (" + str(args.noiseFilterOrientStart)
    infoString += "," + str(args.noiseFilterOrientEnd) + "," + str(args.noiseFilterOrientStep) + ")"
    chtc_print(args, infoString)
    
    resultsFile = os.path.join(os.getcwd(), "collated_results.csv")
    completedResults = set()
    if os.path.isfile(resultsFile):
        infoString = "[INFO] Detected collated_results.csv file."
        chtc_print(args, infoString)
        with open(resultsFile, 'r') as rf:
            lines = rf.readlines()
            headerLine = lines[0]
            assert("Speech Orient" in headerLine and 
                   "Noise Orient" in headerLine and
                   "Final Validation Score SI-SNR (dB)" in headerLine)
            headerTokens = [tok.strip() for tok in headerLine.split(",")]
            speechHeaderIdx = headerTokens.index("Speech Orient")
            noiseHeaderIdx = headerTokens.index("Noise Orient")
            for line in lines[1:]:
                valueTokens = [tok.strip() for tok in line.split(",")]
                speechOComplete = int(valueTokens[speechHeaderIdx])
                noiseOComplete = int(valueTokens[noiseHeaderIdx])
                completedResults.add( (speechOComplete, noiseOComplete) )
        infoString = "[INFO] File collated_results.csv contained "
        infoString += str(len(completedResults)) + " completed results."
        chtc_print(args, infoString)

        pruned = [(suo,nuo) for (suo, nuo) in orientationPairs if (not (suo,nuo) in completedResults) ]
        infoString = "[INFO] Only " + str(len(pruned)) 
        infoString += " orientation pairs haven't already been computed."
        infoString += " Running on only those."
        chtc_print(args, infoString)
        orientationPairs = [tup for tup in pruned]
        if len(pruned) == 0:
            infoString = "[INFO] All orientation pairs complete, no work to do"
            chtc_print(args, infoString)
            sys.exit(0)
           
    orientationPairIdx = 0
    assert(len(orientationPairs) > 0)
    speechOrient, noiseOrient = orientationPairs[orientationPairIdx]
    enoughTimeForMoreWork = True
    with torch.no_grad():
        while enoughTimeForMoreWork:
            # Reset running score for current iteration
            runningScore.fill_(0)
            start_time = time.time()

            speechFilter = CIPICSubject.getHRIRFromIndex(speechOrient, args.cipicChannel)
            speechFilter = torch.from_numpy(speechFilter).float().to(device)
            speechFilter = downsampler(speechFilter) 

            noiseFilter = CIPICSubject.getHRIRFromIndex(noiseOrient, args.cipicChannel)
            noiseFilter  = torch.from_numpy(noiseFilter).float().to(device)
            noiseFilter = downsampler(noiseFilter) 
            sampleSizeInBytes = -1
            torch.cuda.reset_peak_memory_stats(0)
            for i, (clean, noise, idx) in enumerate(validation_loader):
                if sampleSizeInBytes < 0:
                    cleanBytes = clean.element_size() * clean.nelement()
                    noiseBytes = noise.element_size() * noise.nelement()
                    speechFBytes = speechFilter.element_size() * speechFilter.nelement()
                    noiseFBytes = noiseFilter.element_size() * noiseFilter.nelement()
                    sampleSizeInBytes = cleanBytes
                    sampleSizeInBytes += noiseBytes
                    sampleSizeInBytes += speechFBytes
                    sampleSizeInBytes += noiseFBytes

                noise = noise.to(device, non_blocking=True)
                for batch_idx in range(args.b):
                    ssl_noise[batch_idx,:] = conv_transform(noise[batch_idx,:], noiseFilter)
                clean = clean.to(device, non_blocking=True)
                for batch_idx in range(args.b):
                    ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], speechFilter)
                     
                for batch_idx in range(args.b):
                    clean_file, noise_file, metadata = validation_set._get_filenames(idx[batch_idx])
                    ssl_snrs[batch_idx] = metadata['snr']
                    ssl_targlvls[batch_idx] = metadata['target_level']
                                       
                if torch_compile_capable and args.enableTorchCompile: 
                    optSynthesizeNoisySpeech(
                            device,
                            ssl_clean, 
                            ssl_noise, 
                            ssl_noisy,
                            args.b, 
                            ssl_snrs,
                            ssl_targlvls
                    )
                else: 
                    synthesizeNoisySpeech(
                            device,
                            ssl_clean, 
                            ssl_noise, 
                            ssl_noisy,
                            args.b, 
                            ssl_snrs,
                            ssl_targlvls
                    )
                       
                score = si_snr(ssl_noisy, ssl_clean)
                score = torch.nan_to_num(score, nan=0.0)
                runningScore = torch.add(runningScore, torch.mean(score), alpha=1.0/float(numIters))
                if args.printOutputWhileValidation:
                    statString = "Valid [" + str(i) + "] -> "
                    statString += str(torch.mean(score).item()) + " SI-SNR dB"
                    print(statString)

            end_time = time.time()
            exec_time = end_time - start_time
            iterationLatencies.append(exec_time)
            # only keep track of the last iteration for accurate runtime estimate
            if len(iterationLatencies) > 10:
                iterationLatencies = iterationLatencies[1:]
            averageValidationScore = runningScore.item()
            if (args.printValidationResultsHeader and 
                    orientationPairIdx == 0):
                headerString = "Subject, Channel, Speech Orient, "
                headerString += "Noise Orient, "
                headerString += "Final Validation Score SI-SNR (dB), "
                headerString += "ExecTime, "
                headerString += "BatchSize, "
                headerString += "Dataloader Num Workers, "
                headerString += "Dataloader Prefetch Factor, "
                headerString += "Sample Size (MB), "
                headerString += "CUDA Peak Mem Allocated (MB)"
                print(headerString)
            resultString  = str(args.cipicSubject) + "," + str(args.cipicChannel) + "," 
            resultString += str(speechOrient) + "," + str(noiseOrient) + "," 
            resultString += str(averageValidationScore) + "," + str(exec_time) + ","
            resultString += str(args.b) + "," + str(args.dataloader_workers) + ","
            resultString += str(args.dataloader_prefetch_factor) + ","
            resultString += str(sampleSizeInBytes/ (1024.0*1024.0)) + ","
            resultString += str(torch.cuda.max_memory_allocated(0)/ (1024.0*1024.0))
            print(resultString)
            
            # Determine if ending condition is met
            orientationPairIdx += 1
            if orientationPairIdx >= len(orientationPairs):
                enoughTimeForMoreWork = False
            else:
                speechOrient, noiseOrient = orientationPairs[orientationPairIdx]

            if args.isCHTCJob:
                avgIterationLatency = 1.0 * sum(iterationLatencies)/ len(iterationLatencies)
                if "cuda" in deviceString:
                    timeLeft = 1.0 * get_gpu_time_remaining(rawValue=True)
                    if timeLeft / avgIterationLatency < args.epochsEarlyEndBuffer:
                        enoughTimeForMoreWork = False
                if "cpu" in deviceString:
                    timeLeft = 1.0 * get_cpu_time_remaining(rawValue=True)
                    if timeLeft / avgIterationLatency < args.epochsEarlyEndBuffer:
                        enoughTimeForMoreWork = False
