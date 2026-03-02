# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os, sys, math
sys.path.append('./')
from audio_dataloader import DNSAudioNoNoise
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
from snr import si_snr
import torchaudio
import random
import pandas as pd
import time
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

def stft_mixer(stft_abs, stft_angle, n_fft=512, method=None):
    spec = torch.complex(stft_abs * torch.cos(stft_angle),
                                        stft_abs * torch.sin(stft_angle))
    if (method == None):
        return torch.istft(spec, n_fft=n_fft, onesided=True)
    if (type(method) == int):
        print("Perform inver mel scale transform")
        sys.exit(0)

    return method(spec)

class LIFCUBANetwork(torch.nn.Module):
    """LIF network.

    A network consisting of the following topology:

    Layer
    ===============
    - BlockCubaInput
    - BlockCubaDense
    - BlockCubaDense
    - BlockCubaAffine

    """

    def __init__(self):
        """Initialize network."""
        super(Network, self).__init__()

        cuba_params = {
            "threshold": 0.1,
            "current_decay": 0.9,
            "voltage_decay": 0.9,
            "tau_grad": 1,
            "scale_grad": 1,
            "scale": 1 << 6,
            "norm": None,
            "dropout": None,
            "shared_param": True,
            "persistent_state": False,
            "requires_grad": False,
            "graded_spike": False,
        }

        width = 32

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Input(
                    neuron_params=cuba_params, count_log=False
                ),
                slayer.block.cuba.Dense(
                    neuron_params=cuba_params,
                    in_neurons=2,
                    out_neurons=width,
                    count_log=False,
                ),
                slayer.block.cuba.Dense(
                    neuron_params=cuba_params,
                    in_neurons=width,
                    out_neurons=width,
                    count_log=False,
                ),
                slayer.block.cuba.Affine(
                    neuron_params=cuba_params,
                    in_neurons=width,
                    out_neurons=1,
                    dynamics=False,
                    count_log=False,
                ),
            ]
        )

    def forward(self, x):
        """Forward pass."""
        count = []
        for block in self.blocks:
            x = block(x)
            count.append(torch.mean(x).item())

        return x, torch.as_tensor(count)

class LinearANNNetwork(torch.nn.Module):
    def __init__(self, 
            hiddenLayerWidths=512,
            n_fft=512): 
        super().__init__()
        self.hiddenLayerWidths = hiddenLayerWidths
        self.n_fft = n_fft
        self.stft_mean = 0.2

        self.linear1 = torch.nn.Linear(n_fft//2 + 1, hiddenLayerWidths)
        self.lrelu1  = torch.nn.LeakyReLU(negative_slope=1e-5)
        self.linear2 = torch.nn.Linear(hiddenLayerWidths, hiddenLayerWidths)
        self.lrelu2  = torch.nn.LeakyReLU(negative_slope=1e-5)
        self.linear3 = torch.nn.Linear(hiddenLayerWidths, n_fft//2+1)
        self.lrelu3  = torch.nn.LeakyReLU(negative_slope=1e-5)
        self.forward_pass_stats = dict()

    def name(self):
        return f"LinearANN_depth2_widths{self.hiddenLayerWidths}_nfft{self.n_fft}"

    def forward(self, noisy):
        self.forward_pass_stats = dict()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(device)
        start.record()

        x = noisy - self.stft_mean
        x = x.transpose(1,2)
        x = x.reshape(-1, self.n_fft//2 + 1)

        end.record()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device)
        mem_peak = torch.cuda.max_memory_allocated()
        self.forward_pass_stats['CenteringMS'] = start.elapsed_time(end)
        self.forward_pass_stats['CenteringPreAllocatedMB'] = mem_before / 1024**2
        self.forward_pass_stats['CenteringPeakAllocatedMB'] = mem_peak / 1024**2
        self.forward_pass_stats['CenteringPostAllocatedMB'] = mem_after / 1024**2

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(device)
        start.record()
        x = self.linear1(x)
        x = self.lrelu1(x)
        end.record()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device)
        mem_peak = torch.cuda.max_memory_allocated()
        self.forward_pass_stats['Dense1MS'] = start.elapsed_time(end)
        self.forward_pass_stats['Dense1PreAllocatedMB'] = mem_before / 1024**2
        self.forward_pass_stats['Dense1PeakAllocatedMB'] = mem_peak / 1024**2
        self.forward_pass_stats['Dense1PostAllocatedMB'] = mem_after / 1024**2

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(device)
        start.record()
        x = self.linear2(x)
        x = self.lrelu2(x)
        end.record()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device)
        mem_peak = torch.cuda.max_memory_allocated()
        self.forward_pass_stats['Dense2MS'] = start.elapsed_time(end)
        self.forward_pass_stats['Dense2PreAllocatedMB'] = mem_before / 1024**2
        self.forward_pass_stats['Dense2PeakAllocatedMB'] = mem_peak / 1024**2
        self.forward_pass_stats['Dense2PostAllocatedMB'] = mem_after / 1024**2

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(device)
        start.record()
        x = self.linear3(x)
        x = self.lrelu3(x)
        result = x.view(32, 3751, 257).transpose(1,2)
        end.record()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device)
        mem_peak = torch.cuda.max_memory_allocated()
        self.forward_pass_stats['OutputMS'] = start.elapsed_time(end)
        self.forward_pass_stats['OutputPreAllocatedMB'] = mem_before / 1024**2
        self.forward_pass_stats['OutputPeakAllocatedMB'] = mem_peak / 1024**2
        self.forward_pass_stats['OutputPostAllocatedMB'] = mem_after / 1024**2

        return result

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

class Network(torch.nn.Module):
    def __init__(self, 
            threshold=0.1, 
            tau_grad=0.1, 
            scale_grad=0.8, 
            max_delay=64, 
            out_delay=0,
            hiddenLayerWidths=512,
            n_fft=512):
            #profilingMemory=False):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay
        self.EPS = 2.220446049250313e-16
        self.hiddenLayerWidths = hiddenLayerWidths
        self.n_fft = n_fft
        #self.profilingMemory = profilingMemory

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

        self.sd_input = slayer.block.sigma_delta.Input(sdnn_params)
        self.sd_dense_1 = slayer.block.sigma_delta.Dense(sdnn_params, n_fft//2 + 1, hiddenLayerWidths, weight_norm=False, delay=True, delay_shift=True)
        self.sd_dense_2 = slayer.block.sigma_delta.Dense(sdnn_params, hiddenLayerWidths, hiddenLayerWidths, weight_norm=False, delay=True, delay_shift=True)
        self.sd_output = slayer.block.sigma_delta.Output(sdnn_params, hiddenLayerWidths, n_fft//2 + 1, weight_norm=False)

        self.sd_input.pre_hook_fx = self.input_quantizer

        self.sd_dense_1.delay.max_delay = max_delay
        self.sd_dense_2.delay.max_delay = max_delay
        self.forward_pass_stats = dict()

    def name(self):
        return f"SigmaDelta_depth2_widths{self.hiddenLayerWidths}_nfft{self.n_fft}"

    def forward(self, noisy):
        self.forward_pass_stats = dict()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(device)
        start.record()
        x = noisy - self.stft_mean
        x = self.sd_input(x)
        end.record()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device)
        mem_peak = torch.cuda.max_memory_allocated()
        self.forward_pass_stats['CenteringMS'] = start.elapsed_time(end)
        self.forward_pass_stats['CenteringPreAllocatedMB'] = mem_before / 1024**2
        self.forward_pass_stats['CenteringPeakAllocatedMB'] = mem_peak / 1024**2
        self.forward_pass_stats['CenteringPostAllocatedMB'] = mem_after / 1024**2

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(device)
        start.record()
        x = self.sd_dense_1(x)
        end.record()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device)
        mem_peak = torch.cuda.max_memory_allocated()
        self.forward_pass_stats['Dense1MS'] = start.elapsed_time(end)
        self.forward_pass_stats['Dense1PreAllocatedMB'] = mem_before / 1024**2
        self.forward_pass_stats['Dense1PeakAllocatedMB'] = mem_peak / 1024**2
        self.forward_pass_stats['Dense1PostAllocatedMB'] = mem_after / 1024**2

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(device)
        start.record()
        x = self.sd_dense_2(x)
        end.record()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device)
        mem_peak = torch.cuda.max_memory_allocated()
        self.forward_pass_stats['Dense2MS'] = start.elapsed_time(end)
        self.forward_pass_stats['Dense2PreAllocatedMB'] = mem_before / 1024**2
        self.forward_pass_stats['Dense2PeakAllocatedMB'] = mem_peak / 1024**2
        self.forward_pass_stats['Dense2PostAllocatedMB'] = mem_after / 1024**2

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(device)
        start.record()
        x = self.sd_output(x)
        mask = torch.relu(x + 1)
        result =  slayer.axon.delay(noisy, self.out_delay) * mask
        end.record()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device)
        mem_peak = torch.cuda.max_memory_allocated()
        self.forward_pass_stats['OutputMS'] = start.elapsed_time(end)
        self.forward_pass_stats['OutputPreAllocatedMB'] = mem_before / 1024**2
        self.forward_pass_stats['OutputPeakAllocatedMB'] = mem_peak / 1024**2
        self.forward_pass_stats['OutputPostAllocatedMB'] = mem_after / 1024**2

        return result

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

def run_training_loop(args, net, optimizer, scheduler, train_loader):
    net.train()
    data_list = []
    for epoch in range(args.epochs):
        print(f"Beginning epoch: {epoch}")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i, (clean, noisy, idx) in enumerate(train_loader):
            batch_stats = dict()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated(device)
            start.record()
            ssl_noisy = noisy.to(device)
            ssl_clean = clean.to(device)

            if (args.spectrogram == 0):
                noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, None)
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, None)
            elif(args.spectrogram == 1):
                noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, stft_transform)
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, stft_transform)
            else:
                noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, mel_transform)
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, mel_transform)
            end.record()
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated(device)
            mem_peak = torch.cuda.max_memory_allocated()
            batch_stats['STFTMS'] = start.elapsed_time(end)
            batch_stats['STFTPreAllocatedMB'] = mem_before / 1024**2
            batch_stats['STFTPeakAllocatedMB'] = mem_peak / 1024**2
            batch_stats['STFTPostAllocatedMB'] = mem_after / 1024**2

            denoised_abs = net(noisy_abs)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated(device)
            start.record()

            noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
            clean_abs = slayer.axon.delay(clean_abs, out_delay)
            clean = slayer.axon.delay(ssl_clean, args.n_fft // 4 * out_delay)

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
            net.module.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            if torch.isnan(score).any():
                score[torch.isnan(score)] = 0

            end.record()
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated(device)
            mem_peak = torch.cuda.max_memory_allocated()
            batch_stats['BackwardsMS'] = start.elapsed_time(end)
            batch_stats['BackwardsPreAllocatedMB'] = mem_before / 1024**2
            batch_stats['BackwardsPeakAllocatedMB'] = mem_peak / 1024**2
            batch_stats['BackwardsPostAllocatedMB'] = mem_after / 1024**2

            new_row_data = {'Model': net.module.name(),
                            'BatchSize':args.b, 
                            'Epoch': epoch, 
                            'Batch':i,
                            'DeviceName':torch.cuda.get_device_name(0),
                            'MachineName':get_machine_attr_name_0()}
            data_list.append(new_row_data | batch_stats | net.module.forward_pass_stats)
            print(data_list[-1])
        scheduler.step()
    df = pd.DataFrame(data_list)
    return df

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
    parser.add_argument('-hiddenLayerWidths',
                        type=int,
                        default=512,
                        help='# of nuerons in hidden layers')

    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}{}'.format(args.optim, args.seed)

    assert(args.spectrogram == 0 or args.spectrogram == 1 or args.spectrogram == 2)
    trained_folder = 'Trained' + identifier
    logs_folder = 'Logs' + identifier
    writer = SummaryWriter('runs/' + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    lam = args.lam

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    out_delay = args.out_delay
    '''
    my_network = LinearANNNetwork(
        args.hiddenLayerWidths,
        args.n_fft)
    '''
    my_network = Network(
                args.threshold,
                args.tau_grad,
                args.scale_grad,
                args.dmax,
                args.out_delay,
                args.hiddenLayerWidths,
                args.n_fft)
    total_params = sum(p.numel() for p in my_network.parameters())
    print(f"Total parameters: {total_params}")
    net = torch.nn.DataParallel(my_network.to(device),device_ids=args.gpu)

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

    train_set = DNSAudioNoNoise(root=args.path + 'training_set/', maxFiles=args.training_samples)
    
    train_loader = DataLoader(train_set,
                          batch_size=args.b,
                          shuffle=True,
                          collate_fn=train_set.collate_fn,
                          num_workers=4,
                          pin_memory=True)

    device_name = torch.cuda.get_device_name(0) # or use torch.cuda.current_device(
    print(f"Beginning training on {device_name}")
    stats = run_training_loop(args, net, optimizer, scheduler, train_loader)
    stats.to_csv("SNN_data.csv", index=False)
    print("Completed training loop [epochs_completed:" + str(args.epochs) + "]")
