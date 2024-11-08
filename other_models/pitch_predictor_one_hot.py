# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os, sys, math
sys.path.append('./')
from audio_dataloader import DNSAudioCleanOnly
from hrtfs.cipic_db import CipicDatabase 
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
import parselmouth, librosa

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

def run_training_loop(args, net, optimizer, scheduler, train_loader, orientList=[], startingEpoch=0):
    delay_weights = dict()
    averageTrainingLoss = 0
    freq_map = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=args.n_fft)).to(device)
    if len(orientList) == 0:
        assert(not args.useCipic)

    for epoch in range(args.epochs):
        trainingLosses = []
        for i, (clean, idx) in enumerate(train_loader):
            net.train()
            if (len(orientList) > 0):
                clean = clean.to(device)

                speechFilterOrient = random.choice(orientList)
                speechFilter  = torch.from_numpy(CIPICSubject.getHRIRFromIndex(speechFilterOrient, args.filterChannel)).float()
                speechFilter  = speechFilter.to(device)
                
                ssl_clean = torch.zeros(args.b, 480000).to(device)
                for batch_idx in range(args.b):
                    ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], downsampler(speechFilter))
            else:
                ssl_clean = clean.to(device)

            if (args.spectrogram == 0):
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, None)
            elif(args.spectrogram == 1):
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, stft_transform)
            else:
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, mel_transform)
            
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
                statString = "Train [" + str(epoch + startingEpoch + 1) + " | " + str(i) + "]"
                if (args.useCipic):
                    statString += " (s)=("
                    statString += str(speechFilterOrient) + ") -> "
                else:
                    statString += " -> "
                statString += str(loss.item())
                print(statString)
        scheduler.step()
        if args.trackDelayWhileTraining:
            for param_tensor in net.state_dict():
                if ("delay.delay" in param_tensor):
                    if not param_tensor in delay_weights.keys():
                        delay_weights[param_tensor] = dict()
                    delay_weights[param_tensor][epoch] = net.state_dict()[param_tensor].clone().detach().cpu()
                    #print(param_tensor + "," + str(epoch) + "," + str(delay_weights[param_tensor][epoch]))
        # Updates only the last training epoch's loss is kept
        averageTrainingLoss = sum(trainingLosses) / (1.0 * len(trainingLosses))
    return delay_weights, averageTrainingLoss

def run_warm_up_training(args, net, optimizer, scheduler, train_loader):
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
        
        loss = F.mse_loss(pitch_prediction, ssl_clean_pitch)

        if torch.isnan(loss).any():
            loss[torch.isnan(loss)] = 0
        assert torch.isnan(loss) == False

        optimizer.zero_grad()
        loss.backward()
        module.validate_gradients()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        return

def run_validation_loop(args, net, validation_loader, orientList=[]):
    net.eval()
    validationLosses = []
    freq_map = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=args.n_fft)).to(device)
    if len(orientList) == 0:
        assert(not args.useCipic)
    for i, (clean, idx) in enumerate(validation_loader):
        with torch.no_grad():
            if len(orientList) > 0:
                speechFilterOrient = random.choice(orientList)
                speechFilter  = torch.from_numpy(CIPICSubject.getHRIRFromIndex(speechFilterOrient, args.filterChannel)).float()
                speechFilter  = speechFilter.to(device)
                clean = clean.to(device)
                ssl_clean = torch.zeros(args.b, 480000).to(device)
                for batch_idx in range(args.b):
                    ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], downsampler(speechFilter))
            else:
                ssl_clean = clean.to(device)

            if (args.spectrogram == 0):
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, None)
            elif(args.spectrogram == 1):
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, stft_transform)
            else:
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, mel_transform)

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
            
            loss = F.mse_loss(pitch_prediction, ssl_clean_pitch)
             
            if torch.isnan(loss).any():
                loss[torch.isnan(loss)] = 0
            assert torch.isnan(loss) == False

            validationLosses.append(torch.mean(loss).item())
            if args.printOutputWhileValidation:
                statString = "Valid [" + str(i) + "]"
                if (args.useCipic):
                    statString += " (s)=("
                    statString += str(speechFilterOrient) + ") -> "
                else:
                    statString += " -> "
                statString += str(loss.item())
                print(statString)
    averageValidationLoss = sum(validationLosses) / (1.0 * len(validationLosses))
    return averageValidationLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-script_name',
                        type=str,
                        default='other_models/pitch_predictor',
                        help='name of this file')
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
    parser.add_argument('-trackDelayWhileTraining',
                        dest='trackDelayWhileTraining', 
                        action='store_true',
                        help='Switch flag to track updates to delay weights while training')
    parser.add_argument('-useCheckpoint',
                        type=str,
                        default='',
                        help='Checkpoint to continue training from')
    parser.add_argument('-saveCheckpoint',
                        dest='saveCheckpoint', 
                        action='store_true',
                        help='Switch flag to enable saving a chekpoint after training')

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
                        help='When using randomized orients, number of additional orientations, must be >= 8')

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
            orientList = list(orientSet)
        else:
            assert(args.fixedOrients and args.numFixedOrients >= 1)
            if args.numFixedOrients == 1:
                orientList.append( 608 ) # speech in front, noise in back, medial plane
            if args.numFixedOrients == 2:
                orientList.append( 608 ) # speech in front, noise in back, medial plane
                orientList.append( 640 ) # speech in back, noise in front, medial plane
            if args.numFixedOrients == 4:
                orientList.append( 316 )
                orientList.append( 300 )
                orientList.append( 916 )
                orientList.append( 900 )

    print("Orient list contains " + str(len(orientList)) + " orientations")

    train_set = DNSAudioCleanOnly(root=args.path + 'training_set/', maxFiles=args.training_samples)
    train_loader = DataLoader(train_set,
                          batch_size=args.b,
                          shuffle=True,
                          collate_fn=train_set.collate_fn,
                          num_workers=4,
                          pin_memory=True)

    startingEpoch = 0
    trackingInfo = dict()
    if args.useCheckpoint != "":
        run_warm_up_training(args, net, optimizer, scheduler, train_loader)
        checkpoint = torch.load(args.useCheckpoint, weights_only=True)
        module.load_state_dict(checkpoint['module_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        startingEpoch = checkpoint['epochs_completed']
        trackingInfo = checkpoint['tracking_info']
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
        print(statusString)

    delay_weights, lastTrainingLoss = run_training_loop(args, net, optimizer, scheduler, train_loader, orientList=orientList, startingEpoch=startingEpoch)
    if args.trackDelayWhileTraining:
    	plot_weights(delay_weights)

    print("Completed training loop [epochs_completed:" + str(args.epochs) + ", training loss=" + str(lastTrainingLoss) + "]")

    validation_set = DNSAudioCleanOnly(root=args.path + 'validation_set/', maxFiles=args.validation_samples)
    validation_loader = DataLoader(validation_set,
                               batch_size=args.b,
                               shuffle=True,
                               collate_fn=validation_set.collate_fn,
                               num_workers=4,
                               pin_memory=True)
    finalValidationLoss = run_validation_loop(args, net, validation_loader, orientList=orientList)
    statusString  = "Completed training and validation [epochs_completed:" 
    statusString += str(startingEpoch+args.epochs) + ", training loss=" 
    statusString += str(lastTrainingLoss) + ", validation loss="
    statusString += str(finalValidationLoss) + "]"
    print(statusString)
    if (args.saveCheckpoint):
        trackingInfo[startingEpoch+args.epochs] = dict()
        currEpochStats = trackingInfo[startingEpoch+args.epochs]
        currEpochStats['training_loss'] = lastTrainingLoss
        currEpochStats['validation_loss'] = finalValidationLoss
        torch.save({
                'epochs_completed': startingEpoch + args.epochs,
                'module_state_dict': module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'tracking_info': trackingInfo,
                }, trained_folder + '/network.pt')
    print("Final validation loss: " + str(finalValidationLoss))
