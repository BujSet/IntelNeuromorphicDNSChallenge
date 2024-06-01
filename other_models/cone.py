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

class ConeFilter(torch.nn.Module):
    def __init__(self, 
        n_fft=512, 
        sample_rate=44100.0,
        steps=100):
        super().__init__()
        self.x0 = torch.nn.Parameter(torch.randn(1))
        self.angle = torch.nn.Parameter(torch.randn(1))
        self.depth = torch.nn.Parameter(torch.randn(1))
        freq_map = torch.from_numpy(librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)).float()
        self.register_buffer('freq_map', freq_map, persistent=False)
        self.c = 343 # Speed of sound in air m/s
        self.rho = 1.293 # Air density kg/m^3
        self.steps = 100
        self.pi = 3.1415927

    def forward(self, noisy):
        scalar = self.c*self.rho
        result = noisy
        tot_impedance = None
        myx0 = torch.relu(self.x0) # x0 must be non-negative
        mydepth = torch.relu(self.depth) # depth must be non-negative
        myangle = ((torch.sigmoid(self.angle) + 1.0)/2.0) * (self.pi/2.0)

        for i in range(self.steps):
            x = (myx0 + (i * mydepth/self.steps)).float()
            r = torch.tan(myangle) * x
            numer = torch.mul(self.freq_map, x)
            denom = torch.add(numer, self.c)
            frac = torch.div(numer,denom)
            
            impedance = frac * self.rho * self.c / (self.pi * r * r)         

            if (tot_impedance == None):
                tot_impedance = impedance
            else:
                tot_impedance = torch.mul(tot_impedance, impedance)
        tot_impedance = tot_impedance.unsqueeze(1)
        tot_impedance = tot_impedance.unsqueeze(0)
        result = torch.mul(result, tot_impedance)
        self.print_parameters()
        return result

    def print_parameters(self):
        print("x0: " + str(self.x0.detach().item()))
        print("angle: " + str(self.angle.detach().item() * self.pi / 180.0))
        print("depth: " + str(self.depth.detach().item()))

class Network(torch.nn.Module):
    def __init__(self, 
            threshold=0.1, 
            tau_grad=0.1, 
            scale_grad=0.8, 
            max_delay=64, 
            out_delay=0):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay
        self.EPS = 2.220446049250313e-16

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
            ConeFilter(),
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Dense(sdnn_params, 257, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Dense(sdnn_params, 512, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Output(sdnn_params, 512, 257, weight_norm=False),
        ])

        self.blocks[1].pre_hook_fx = self.input_quantizer

        self.blocks[2].delay.max_delay = max_delay
        self.blocks[3].delay.max_delay = max_delay

    def _display_weights(self, filename=None):
        layerIndices = []
        # First determine which layers have actual weights
        vmin =  100000.0
        vmax = -100000.0
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            if (hasattr(block, 'synapse')):
                if (hasattr(block.synapse, 'weight')):
                    layerIndices.append(i)
                    mini = block.synapse.weight.cpu().detach().min()
                    maxi = block.synapse.weight.cpu().detach().max()
                    if (mini < vmin):
                        vmin = mini
                    if (maxi > vmax):
                        vmax = maxi
            if (hasattr(block, 'print_parameters')):
                block.print_parameters()

        if (filename == None):
            return

        fig, axs = plt.subplots(1, len(layerIndices), figsize=(30,20))

        for i in range(len(layerIndices)):
            idx = layerIndices[i]
            block = self.blocks[idx]
            assert(hasattr(block, 'synapse') and hasattr(block.synapse, 'weight'))
            w = block.synapse.weight.cpu().squeeze().detach().numpy().transpose()
            pad = np.zeros((max(w.shape), max(w.shape)))
            m = w.reshape((w.shape[0], -1))
            pad[0:m.shape[0], 0:m.shape[1]] = m
            im = axs[i].imshow(pad, cmap='hot', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
            axs[i].set_title('Layer ' + str(i) + ' Weights')
        axs[1].set_xlabel("Output Neurons")
        axs[0].set_ylabel("Input Neurons")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def forward(self, noisy):

        cone = self.blocks[0]
        noisy = cone(noisy)

        x = noisy - self.stft_mean

        for i in range(1, len(self.blocks)):
            block = self.blocks[i]
            x = block(x)

        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

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

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

        
def nop_stats(dataloader, stats, sub_stats, print=True):
    t_st = datetime.now()
    for i, (noisy, clean, noise) in enumerate(dataloader):
        with torch.no_grad():
            noisy = noisy
            clean = clean

            score = si_snr(noisy, clean)
            sub_stats.correct_samples += torch.sum(score).item()
            sub_stats.num_samples += noisy.shape[0]

            processed = i * dataloader.batch_size
            total = len(dataloader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            samples_sec = time_elapsed / (i + 1) / dataloader.batch_size
            header_list = [f'Train: [{processed}/{total} '
                           f'({100.0 * processed / total:.0f}%)]']
            if print:
                stats.print(0, i, samples_sec, header=header_list)


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
    # print(trained_folder)
#    writer = SummaryWriter('runs/' + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    lam = args.lam

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    out_delay = args.out_delay
    # if len(args.gpu) == 0:
    #     print("Building CPU network")
    #     net = Network(args.threshold,
    #                   args.tau_grad,
    #                   args.scale_grad,
    #                   args.dmax,
    #                   args.out_delay).to(device)
    #     module = net
    # else:
    print("Building GPU network")
    net = Network(
                args.threshold,
                args.tau_grad,
                args.scale_grad,
                args.dmax,
                args.out_delay)
    net = torch.nn.DataParallel(net.to(device), device_ids=args.gpu)
    module = net.module
    # module._display_weights("initial_weights.png")
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
                noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, mel_transform)
                clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, mel_transform)

            denoised_abs = net(noisy_abs)
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
            module.validate_gradients()
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

#        writer.add_scalar('Loss/train', stats.training.loss, epoch)
#        writer.add_scalar('SI-SNR/train', stats.training.accuracy, epoch)
        stats.update()
    print("")
    training_time = (datetime.now() - training_st).total_seconds()
    print("Training time: " + str(training_time))
    print("Mini batch fill time: " + str(batch_tot))
    print("Convolution time: " + str(conv_tot))
    print("Synth time: " + str(synth_tot))
    print("")
    # module._display_weights()
    with open("validation_orients.txt", 'w') as vof:
        print("Speech Orientation, Noise Orientation, Validation Accuracy", file=vof)
    validation_st = datetime.now()

    t_st = datetime.now()
    for i, (noisy, clean, noise, idx) in enumerate(validation_loader):
        net.eval()
        with torch.no_grad():

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

            denoised_abs = net(noisy_abs)
            noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
            clean_abs = slayer.axon.delay(clean_abs, out_delay)
            clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)
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
#        writer.add_scalar('Loss/valid', stats.validation.loss, epoch)
#        writer.add_scalar('SI-SNR/valid', stats.validation.accuracy, epoch)
#        stats.testing.update()
#        stats.testing.reset()

    # print("")
    # print("")
    # print("")
    # validation_time = (datetime.now() - validation_st).total_seconds()
    # print("Validation time: " + str(validation_time))
    # print("Mini batch fill time: " + str(batch_tot))
    # print("Convolution time: " + str(conv_tot))
    # print("Synth time: " + str(synth_tot))
    # module._display_weights("final_weights.png")
#        stats.plot(path=trained_folder + '/')
#        if stats.validation.best_accuracy is True:
#            torch.save(module.state_dict(), trained_folder + '/network.pt')
#        stats.save(trained_folder + '/')

    # torch.save(module.state_dict(), trained_folder + '/network.pt')
    # module.load_state_dict(torch.load(trained_folder + '/network.pt'))
#    module.export_hdf5(trained_folder + '/network.net')

#    params_dict = {}
#    for key, val in args._get_kwargs():
#        params_dict[key] = str(val)
#    writer.add_hparams(params_dict, {'SI-SNR': stats.validation.max_accuracy})
#    writer.flush()
#    writer.close()
