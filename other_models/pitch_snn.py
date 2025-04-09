# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os, sys, math
sys.path.append('./')
from audio_dataloader import DNSAudioCleanOnly
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
from chtc_files.htchirp_utils import *

def chtc_print(args, string, prefix="[INFO]"):
    if args.isCHTCJob:
        send_log_msg(prefix + " " +  string)
    else:
        print(prefix + " " + str(datetime.datetime.now()) + " " + string)

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

class Network(torch.nn.Module):
    def __init__(self, 
            threshold=0.1, 
            tau_grad=0.1, 
            scale_grad=0.8, 
            max_delay=64, 
            out_delay=0,
            hiddenLayerWidths=[512, 512],
            hiddenLayers=2,
            n_fft=512):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay
        self.EPS = 2.220446049250313e-16
        self.hiddenLayerWidths = hiddenLayerWidths
        self.hiddenLayers = hiddenLayers

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

        # Create the input layer
        self.blocks = torch.nn.ModuleList([slayer.block.sigma_delta.Input(sdnn_params)])

        # Next, create a variable number of dense layers, first layer 
        # dimensions depend on input FFT params, but the rest can be directly 
        # configured from command line parameters
        for i in range(self.hiddenLayers):
            if i == 0:
                dense_layer = slayer.block.sigma_delta.Dense(sdnn_params,
                        n_fft//2 + 1, 
                        self.hiddenLayerWidths[0], 
                        weight_norm=False, 
                        delay=True, 
                        delay_shift=True)
            else:
                dense_layer = slayer.block.sigma_delta.Dense(sdnn_params,
                        self.hiddenLayerWidths[i-1],
                        self.hiddenLayerWidths[i], 
                        weight_norm=False, 
                        delay=True, 
                        delay_shift=True)

            dense_layer.delay.max_delay = max_delay
            self.blocks.append(dense_layer)

        # Create the final output layer, for every FFT frame, the network 
        # should predict what the estimated fundamental frequency is, so the
        # output is a one-hot vector, where the 1 will indicate which the 
        # perceived fundamental frequency
        self.blocks.append(slayer.block.sigma_delta.Output(sdnn_params, 
                    self.hiddenLayerWidths[self.hiddenLayers-1], 
                    n_fft//2 + 1,
                    weight_norm=False))
        # Normally, we add a softmax layer to network since pitch predictions
        # should be mutually exclusive, but because the PyTorch implementation
        # of cross entropy loss already does this, we don't need to

        self.blocks[0].pre_hook_fx = self.input_quantizer

    def forward(self, speech):
        x = speech
        for block in self.blocks:
            x = block(x)
        return x

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

def run_training_loop(args, net, optimizer, scheduler, train_loader, train_set, startingEpoch=0):
    delay_weights = dict()
    averageTrainingLoss = 0
    freq_map = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=args.n_fft)).to(device)
    epochLatencies = []
    enoughTimeForMoreWork = True
    currentEpoch = 0
    net.train()
    while enoughTimeForMoreWork:
        trainingLosses = []
        start_time = time.time()
        for i, (clean, idx) in enumerate(train_loader):        
            clean = clean.to(device)

            if (args.spectrogram == 0):
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, None)
            elif(args.spectrogram == 1):
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, stft_transform)
            else:
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, mel_transform)
            
            pitch_prediction = net(clean_abs)

            ssl_clean_pitch = torch.zeros(pitch_prediction.size()).to(device)
            num_fft_frames = ssl_clean_pitch.size()[-1]
            period = (480000.0 / num_fft_frames) / 16000.0
            for batch_idx in range(args.b):
                clean_file = train_set._get_filenames(idx[batch_idx])
                clean_pitch = parselmouth.Sound(clean_file).to_pitch(time_step=(1.0*(args.n_fft//4)/16000), pitch_floor=50.0, pitch_ceiling=1000.0)
                clean_pitch_freq = [clean_pitch.get_value_at_time((i * period) + (period/2)) for i in range(0, ssl_clean_pitch.size()[-1] - 1)]
                clean_pitch_freq.append(np.nan)
                clean_pitch_freq = torch.FloatTensor(clean_pitch_freq).to(device)
                if torch.isnan(clean_pitch_freq).any():
                    clean_pitch_freq[torch.isnan(clean_pitch_freq)] = 0
                for frame in range(num_fft_frames):
                    ssl_clean_pitch[batch_idx,:, frame] = freq_to_one_hot(clean_pitch_freq[frame], freq_map)  
            ssl_clean_pitch.to(device)  
            
            loss = F.cross_entropy(pitch_prediction, ssl_clean_pitch)
             
            if torch.isnan(loss).any():
                loss[torch.isnan(loss)] = 0
            assert torch.isnan(loss) == False
            optimizer.zero_grad()
            loss.backward()
            module.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            trainingLosses.append(torch.mean(loss).item())
            if args.printOutputWhileTraining:
                statString = "Train [" + str(currentEpoch + startingEpoch)
                statString += " | " + str(i) + "] - > "
                statString += str(loss.item())
                print(statString)
        scheduler.step()
        averageTrainingLoss = sum(trainingLosses) / (1.0 * len(trainingLosses))
        end_time = time.time()
        currentEpochLatency = end_time - start_time
        epochLatencies.append(currentEpochLatency)
        # only keep track of the last 10 iterations for accurate runtime estimate
        if len(epochLatencies) > 10:
            epochLatencies = epochLatencies[-10:]
        currentEpoch += 1

        # Now determine if end condition is met
        if args.isCHTCJob:
            avgEpochLatency = 1.0 * sum(epochLatencies) / len(epochLatencies)
            updateString = "Epoch " + str(currentEpoch) + " took " 
            updateString += str(currentEpochLatency)
            updateString += " secs. Avg = "
            updateString += str(avgEpochLatency) + " secs/epoch. Time left = "
            timeLeft = 1.0 * get_gpu_time_remaining(rawValue=True)
            updateString += str(timeLeft) + " secs. Train Loss = "
            updateString += str(trainingLosses[-1]) + " SI-SNR db" 
            # Add a buffer of two epochs before job end to allow 
            # validation loop to occur
            if timeLeft / avgEpochLatency < 2.0:
                enoughTimeForMoreWork = False
            send_log_msg(updateString)
        if currentEpoch == args.epochs:
            enoughTimeForMoreWork = False
            if args.isCHTCJob:
                send_log_msg("Finished training all " + str(args.epochs) + " epochs.")
            else:
                print("Finished training all " + str(args.epochs) + " epochs.")
    return delay_weights, averageTrainingLoss, currentEpoch+startingEpoch

def run_warm_up_training(args, net, optimizer, scheduler, train_loader, train_set):
    net.train()
    # Run single epoch just to set the network dimensions (Weird that this is necessary)?
    freq_map = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=args.n_fft)).to(device)
    for i, (clean, idx) in enumerate(train_loader):
        ssl_clean = clean.to(device)
        clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, None)
        pitch_prediction = net(clean_abs)
        ssl_clean_pitch = torch.zeros(pitch_prediction.size()).to(device)
        num_fft_frames = ssl_clean_pitch.size()[-1]
        period = (480000.0 / num_fft_frames) / 16000.0
        for batch_idx in range(args.b):
            clean_file = train_set._get_filenames(idx[batch_idx])
            clean_pitch = parselmouth.Sound(clean_file).to_pitch(time_step=(1.0*(args.n_fft//4)/16000), pitch_floor=50.0, pitch_ceiling=1000.0)
            clean_pitch_freq = [clean_pitch.get_value_at_time((i * period) + (period/2)) for i in range(0, ssl_clean_pitch.size()[-1] - 1)]
            clean_pitch_freq.append(np.nan)
            clean_pitch_freq = torch.FloatTensor(clean_pitch_freq).to(device)
            if torch.isnan(clean_pitch_freq).any():
                clean_pitch_freq[torch.isnan(clean_pitch_freq)] = 0
            for frame in range(num_fft_frames):
                ssl_clean_pitch[batch_idx,:, frame] = freq_to_one_hot(clean_pitch_freq[frame], freq_map)
        ssl_clean_pitch.to(device) 
        
        loss = F.cross_entropy(pitch_prediction, ssl_clean_pitch)

        if torch.isnan(loss).any():
            loss[torch.isnan(loss)] = 0
        assert torch.isnan(loss) == False

        optimizer.zero_grad()
        loss.backward()
        module.validate_gradients()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        return

def run_validation_loop(args, net, validation_loader, validation_set):
    net.eval()
    validationLosses = []
    freq_map = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=args.n_fft)).to(device)
    for i, (clean, idx) in enumerate(validation_loader):
        with torch.no_grad():
            clean = clean.to(device)

            if (args.spectrogram == 0):
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, None)
            elif(args.spectrogram == 1):
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, stft_transform)
            else:
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft, mel_transform)

            pitch_prediction = net(clean_abs)

            ssl_clean_pitch = torch.zeros(pitch_prediction.size()).to(device)
            num_fft_frames = ssl_clean_pitch.size()[-1]
            period = (480000.0 / num_fft_frames) / 16000.0
            for batch_idx in range(args.b):
                clean_file = validation_set._get_filenames(idx[batch_idx])
                clean_pitch = parselmouth.Sound(clean_file).to_pitch(time_step=(1.0*(args.n_fft//4)/16000), pitch_floor=50.0, pitch_ceiling=1000.0)
                clean_pitch_freq = [clean_pitch.get_value_at_time((i * period) + (period/2)) for i in range(0, ssl_clean_pitch.size()[-1] - 1)]
                clean_pitch_freq.append(np.nan)
                clean_pitch_freq = torch.FloatTensor(clean_pitch_freq).to(device)
                if torch.isnan(clean_pitch_freq).any():
                    clean_pitch_freq[torch.isnan(clean_pitch_freq)] = 0
                for frame in range(num_fft_frames):
                    ssl_clean_pitch[batch_idx,:, frame] = freq_to_one_hot(clean_pitch_freq[frame], freq_map) 
            ssl_clean_pitch.to(device)
            
            loss = F.cross_entropy(pitch_prediction, ssl_clean_pitch)
             
            if torch.isnan(loss).any():
                loss[torch.isnan(loss)] = 0
            assert torch.isnan(loss) == False

            validationLosses.append(torch.mean(loss).item())
            if args.printOutputWhileValidation:
                statString = "Valid [" + str(i) + "] -> "
                statString += str(loss.item())
                print(statString)
    averageValidationLoss = sum(validationLosses) / (1.0 * len(validationLosses))
    return averageValidationLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-script_name',
                        type=str,
                        default='other_models/pitch_snn',
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
    parser.add_argument('-lr',
                        type=float,
                        default=0.001,
                        help='initial learning rate')
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
    parser.add_argument('-epochs',
                        type=int,
                        default=50,
                        help='Number of training epochs to run')
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
                        default=60000,
                        help='Number of samples training should use, supports small dataset subset')
    parser.add_argument('-print_output_while_training',
                        dest='printOutputWhileTraining', 
                        action='store_true',
                        help='Switch flag to print score after every mini-batch during training')
    parser.add_argument('-validation_samples',
                        type=int,
                        default=60000,
                        help='Number of samples validation should use, supports small dataset subset')
    parser.add_argument('-print_output_while_validation',
                        dest='printOutputWhileValidation', 
                        action='store_true',
                        help='Switch flag to print score after every mini-batch during validation')
    parser.add_argument('-useCheckpoint',
                        type=str,
                        default='',
                        help='Checkpoint to continue training from')
    parser.add_argument('-saveCheckpoint',
                        dest='saveCheckpoint', 
                        action='store_true',
                        help='Switch flag to enable saving a chekpoint after training')
    parser.add_argument('-hiddenLayerWidths',
                        type=int,
                        nargs="+",
                        default=[512,512],
                        help='# of nuerons in hidden layers')
    parser.add_argument('-hiddenLayers',
                        type=int,
                        default=2,
                        help='# of hidden layers')
    parser.add_argument('-is_CHTC_job',
                        dest='isCHTCJob', 
                        action='store_true',
                        help='Switch flag to indicate if this job was run on CHTC')

    args = parser.parse_args()

    if args.seed is not None:
        chtc_print(args, "Setting seed to " + str(args.seed)) 
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if len(args.exp) == 0:
            args.exp = "seed" + str(args.seed)
        else:
            if (args.exp[-1] == '_'):
                args.exp += "seed" + str(args.seed)
            else:
                args.exp += "_seed" + str(args.seed)

    assert(args.spectrogram == 0 or args.spectrogram == 1 or args.spectrogram == 2)
    trained_folder = 'Trained'
    logs_folder = 'Logs'

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    device = torch.device('cuda:{}'.format(args.gpu[0]))
    # Get CUDA capability
    device_cap = torch.cuda.get_device_capability()
    torch_compile_capable = False
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        torch_compile_capable = True
        chtc_print(args, "Detected device capable of using torch.compile, will attempt to use for torch operations")
        # TODO should try torch compile on the network somehow

    out_delay = args.out_delay
    if len(args.hiddenLayerWidths) != args.hiddenLayers:
        args.hiddenLayerWidths = [args.n_fft for _ in range(args.hiddenLayers)]
    assert(len(args.hiddenLayerWidths) == args.hiddenLayers)
    net = torch.nn.DataParallel(Network(
                args.threshold,
                args.tau_grad,
                args.scale_grad,
                args.dmax,
                args.out_delay,
                args.hiddenLayerWidths,
                args.hiddenLayers,
                args.n_fft).to(device),
                    device_ids=args.gpu)
    # For some reason this breaks on some of the CHTC machines
    # if torch_compile_capable:
    #    net = torch.compile(net)
    module = net.module
    chtc_print(args, "Creating " + str(len(module.blocks)) + "-layer network with hidden layer widths=" + str(module.hiddenLayerWidths))
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

    # Define optimizer module.
    optimizer = torch.optim.RAdam(net.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    train_set = DNSAudioCleanOnly(root=args.path + 'training_set/', maxFiles=args.training_samples)
    train_loader = DataLoader(train_set,
                          batch_size=args.b,
                          shuffle=True,
                          collate_fn=train_set.collate_fn,
                          num_workers=args.dataloader_workers,
                          prefetch_factor=args.dataloader_prefetch_factor,
                          pin_memory=True)

    chtc_print(args, "Created training DataSet and DataLoader")
    startingEpoch = 0
    trackingInfo = dict()
    if args.useCheckpoint != "":
        run_warm_up_training(args, net, optimizer, scheduler, train_loader, train_set)
        checkpoint = torch.load(args.useCheckpoint, weights_only=False)
        module.load_state_dict(checkpoint['module_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        startingEpoch = checkpoint['epochs_completed']
        trackingInfo = checkpoint['tracking_info']
        cmd_line_args = checkpoint['command_line_args']
        # TODO should verify that every value in current args matches the args
        # from the checkpoint we're loading.
        print("Current Tracking info:")
        print("Epoch | Training Loss | Validation Loss ")
        for i in range(0, startingEpoch+1):
            if i in trackingInfo.keys():
                tloss = trackingInfo[i]['training_loss']
                vLoss = trackingInfo[i]['validation_loss']
                checkpointStr  = str(i) + " | "
                checkpointStr += str(tloss) + " | "
                checkpointStr += str(vLoss)
                print(checkpointStr)
        startingTrainingLoss = trackingInfo[startingEpoch]['training_loss']
        startingValidationLoss = trackingInfo[startingEpoch]['validation_loss']
        statusString  = "Resuming from checkpoint [epochs_completed:" 
        statusString += str(startingEpoch) + ", training loss=" 
        statusString += str(startingTrainingLoss) + ", validation loss="
        statusString += str(startingValidationLoss) + "]"
        chtc_print(args, statusString)

    if args.isCHTCJob:
        infoString = "[INFO] Detected that this instance in running in a CHTC Job "
        infoString += " with " + get_gpu_time_remaining()
        infoString += " time remaining."
        print(infoString)

    chtc_print(args, "Beginning training loop")
    delay_weights, lastTrainingLoss, epochsCompleted = run_training_loop(args, net, optimizer, scheduler, train_loader, train_set, startingEpoch=startingEpoch)

    chtc_print(args, "Completed training loop [epochs_completed:" + str(epochsCompleted) + ", training loss=" + str(lastTrainingLoss) + "]")

    validation_set = DNSAudioCleanOnly(root=args.path + 'validation_set/', maxFiles=args.validation_samples)
    validation_loader = DataLoader(validation_set,
                               batch_size=args.b,
                               shuffle=True,
                               collate_fn=validation_set.collate_fn,
                               num_workers=args.dataloader_workers,
                               prefetch_factor=args.dataloader_prefetch_factor,
                               pin_memory=True)
    chtc_print(args, "Created validation DataSet and DataLoader")
    chtc_print(args, "Beginning validation")
    finalValidationLoss = run_validation_loop(args, net, validation_loader, validation_set)
    statusString  = "Completed training and validation [epochs_completed:" 
    statusString += str(epochsCompleted) + ", training loss=" 
    statusString += str(lastTrainingLoss) + ", validation loss="
    statusString += str(finalValidationLoss) + "]"
    chtc_print(args, statusString)
    if (args.saveCheckpoint):
        trackingInfo[epochsCompleted] = dict()
        currEpochStats = trackingInfo[epochsCompleted]
        currEpochStats['training_loss'] = lastTrainingLoss
        currEpochStats['validation_loss'] = finalValidationLoss
        saveFileName = trained_folder + '/pitch_snn_' + args.exp + '_' + str(epochsCompleted) + '.pt'
        chtc_print(args, "Attempting to save model as " + saveFileName)
        torch.save({
                'epochs_completed': epochsCompleted,
                'module_state_dict': module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'tracking_info': trackingInfo,
                'command_line_args': args,
                }, saveFileName)
