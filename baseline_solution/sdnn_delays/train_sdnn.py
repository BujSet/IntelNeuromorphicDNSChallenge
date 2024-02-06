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
from hrtfs.cipic_db import CipicDatabase 
from snr import si_snr
import torchaudio
from noisyspeech_synthesizer import segmental_snr_mixer


def collate_fn(batch):
    noisy, clean, noise = [], [], []

    for sample in batch:
        noisy += [torch.FloatTensor(sample[0])]
        clean += [torch.FloatTensor(sample[1])]
        noise += [torch.FloatTensor(sample[2])]

    return torch.stack(noisy), torch.stack(clean), torch.stack(noise)


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


class Network(torch.nn.Module):
    def __init__(self, 
            threshold=0.1, 
            tau_grad=0.1, 
            scale_grad=0.8, 
            max_delay=64, 
            out_delay=0,
            subjectID=12, 
            speechFilterOrient=624, speechFilterChannel=0,
            noiseFilterOrient=600,noiseFilterChannel=0):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay
        self.cipicSubject = CipicDatabase.subjects[subjectID]
        self.speechFilter  = self.cipicSubject.getHRIRFromIndex(
                speechFilterOrient, speechFilterChannel)
        self.noiseFilter  = self.cipicSubject.getHRIRFromIndex(
                noiseFilterOrient, noiseFilterChannel)
        self.speechFilter  = torch.from_numpy(self.speechFilter).float()
        self.noiseFilter   = torch.from_numpy(self.noiseFilter).float()
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
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Dense(sdnn_params, 257, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Dense(sdnn_params, 512, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Output(sdnn_params, 512, 257, weight_norm=False),
        ])

        self.blocks[0].pre_hook_fx = self.input_quantizer

        self.blocks[1].delay.max_delay = max_delay
        self.blocks[2].delay.max_delay = max_delay

    def _segmental_snr_mixer(self, clean, noise, snr,
                        target_level, 
                        target_level_lower,
                        target_level_higher,
                        clipping_threshold=0.99,
                        ):
        '''Function to mix clean speech and noise at various segmental SNR levels'''
#        print(clean.shape)
#        print(noise.shape)
#        print(clean.size())
#        print(noise.size())
        clean_div = torch.max(torch.abs(clean)) + self.EPS
        noise_div = torch.max(torch.abs(noise)) + self.EPS
#        print("GPU: Clean divisor="+ str(clean_div.item()))
#        print("GPU: Noise divisor="+ str(noise_div.item()))
        ssl_clean = torch.div(clean, clean_div.item())
        ssl_noise = torch.div(noise, noise_div.item())
        # TODO should only calculate the RMS of the 'active' windows, but
        # for now we just use the whole audio sample
        clean_rms = torch.sqrt(torch.mean(torch.square(ssl_clean))).item()
        noise_rms = torch.sqrt(torch.mean(torch.square(ssl_noise))).item()
#        print("GPU: Clean rms="+ str(clean_rms))
#        print("GPU: Noise rms=" + str(noise_rms))
        clean_scalar = 10 ** (target_level / 20) / (clean_rms + self.EPS)
        noise_scalar = 10 ** (target_level / 20) / (noise_rms + self.EPS)
#        print("GPU: Clean scalar="+ str(clean_scalar) + " from target_level=" + str(target_level) + ",rms="+str(clean_rms)+",EPS="+str(self.EPS))
#        print("GPU: Noise scalar" + str(noise_scalar)+" from target_level="+str(target_level)+",rms="+str(noise_rms)+",EPS="+str(self.EPS))
        ssl_clean = torch.mul(ssl_clean, clean_scalar)
        ssl_noise = torch.mul(ssl_noise, noise_scalar)
        # Adjust noise to SNR level
        noise_scalar = clean_rms / (10**(snr/20)) / (noise_rms+self.EPS)
#        print("GPU: newnoiselevel=" + str(noise_scalar))
        ssl_noise = torch.mul(ssl_noise, noise_scalar)
#        print("ssl clean is located on " + str(ssl_clean.get_device()))
#        print("ssl noise is located on " + str(ssl_noise.get_device()))
        ssl_noisy = torch.add(ssl_clean, ssl_noise)
#        print("ssl noisy is located on " + str(ssl_noisy.get_device()))
        noisy_rms_level = torch.randint(
                target_level_lower,
                target_level_higher,
                (1,))
        noisy_rmsT = torch.sqrt(torch.mean(torch.square(ssl_noisy)))
#        print("noisy_rmsT on " + str(noisy_rmsT.get_device()))
        noisy_rms = torch.sqrt(torch.mean(torch.square(ssl_noisy))).item()
#        print("noisy_rmsT is " + str(noisy_rmsT))
        noisy_scalar = 10 ** (noisy_rms_level / 20) / (noisy_rms + self.EPS)
#        print("GPU: "+str(noisy_scalar))
#        print("GPU: "+str(noisy_scalar.get_device()))
#        print("GPU: "+str(noisy_scalar.item()))
        ssl_noisy = torch.mul(ssl_noisy, noisy_scalar.item())
        ssl_clean = torch.mul(ssl_clean, noisy_scalar.item())
        ssl_noise = torch.mul(ssl_noise, noisy_scalar.item())
        # check if any clipping happened
        needToClip = torch.gt(torch.abs(ssl_noisy), 0.99).any() # 0.99 is the clipping threshold 
        if (needToClip):
            noisyspeech_maxamplevel = torch.max(torch.abs(ssl_noisy)).item() / (0.99 - self.EPS)
            ssl_noisy = torch.div(ssl_noisy, noisyspeech_maxamplevel)
            ssl_noise = torch.div(ssl_noise, noisyspeech_maxamplevel)
            ssl_clean = torch.div(ssl_clean, noisyspeech_maxamplevel)
            noisy_rms_level = int(20 * np.log10(noisy_scalar/noisyspeech_maxamplevel * (noisy_rms + self.EPS)))
        return ssl_clean, ssl_noise, ssl_noisy, noisy_rms_level



    def synthesizeNoisySpeech(self, clean, noise, batchSize, snr,
            targetLevel,
            targetLevelLower,
            targetLevelHigher):
        ssl_noisy = torch.zeros(batchSize, 480000).to(device)
        ssl_noise = torch.zeros(batchSize, 480000).to(device)
        ssl_clean = torch.zeros(batchSize, 480000).to(device)
        for i in range(batchSize):
            ssl_clean[i, :], ssl_noise[i,:], ssl_noisy[i,:], rms = self._segmental_snr_mixer(clean[0,:], noise[0,:], snr, 
                targetLevel,
                targetLevelLower,
                targetLevelHigher)
                       
        return ssl_noisy, ssl_clean, ssl_noise

    def forward(self, noisy):
        x = noisy - self.stft_mean

        for block in self.blocks:
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
    parser.add_argument('-path',
                        type=str,
                        default='../../',
                        help='dataset path')
    # CIPIC Filter Parameters
    parser.add_argument('-cipicSubject',
                        type=int,
                        default=12,
                        help='cipic subject ID for pinna filters')
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
    print(trained_folder)
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
    if len(args.gpu) == 0:
        print("Building CPU network")
        net = Network(args.threshold,
                      args.tau_grad,
                      args.scale_grad,
                      args.dmax,
                      args.out_delay,
                      args.cipicSubject,
                      args.speechFilterOrient,
                      args.speechFilterChannel,
                      args.noiseFilterOrient,
                      args.noiseFilterChannel).to(device)
        module = net
    else:
        print("Building GPU network")
        net = torch.nn.DataParallel(Network(
                    args.threshold,
                    args.tau_grad,
                    args.scale_grad,
                    args.dmax,
                    args.out_delay,
                    args.cipicSubject,
                    args.speechFilterOrient,
                    args.speechFilterChannel,
                    args.noiseFilterOrient,
                    args.noiseFilterChannel).to(device),
                        device_ids=args.gpu)
        module = net.module
        stft_transform =torchaudio.transforms.Spectrogram(
                n_fft=args.n_fft,
                onesided=True, 
                power=None,
                hop_length=math.floor(args.n_fft//4)).to(device)
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

    # print()
    # print('Base Statistics')
    # nop_stats(train_loader, base_stats, base_stats.training)
    # nop_stats(validation_loader, base_stats, base_stats.validation)
    # print()

    stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                       accuracy_unit='dB')

    speechFilter  = torch.from_numpy(CipicDatabase.subjects[12].getHRIRFromIndex(args.speechFilterOrient, 0)).float() # .unsqueeze(0)
    speechFilter  = speechFilter.to(device) #view(1, 200)
    noiseFilter   = torch.from_numpy(CipicDatabase.subjects[12].getHRIRFromIndex(args.noiseFilterOrient, 0)).float() # .unsqueeze(0)
    noiseFilter   = noiseFilter.to(device) #view(1, 200)
    for epoch in range(args.epoch):
        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(train_loader):
            noisy_file, clean_file, noise_file, metadata = train_set._get_filenames(i)
            net.train()
            noise = noise.to(device)
            clean = clean.to(device)
            ssl_noise = torch.zeros(args.b, 480000).to(device)
            ssl_clean = torch.zeros(args.b, 480000).to(device)
            for batch_idx in range(args.b):
                ssl_noise[batch_idx,:] = conv_transform(noise[batch_idx,:], noiseFilter)
                ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], speechFilter)
                
#            ssl_noisy, ssl_clean, ssl_noise = module.synthesizeNoisySpeech(
#                    ssl_clean, ssl_noise, 
#                    args.b, 
#                    metadata['snr'],
#                    metadata['target_level'],
#                    -35,
#                    -15)
#            print(ssl_noisy.get_device())
#            print(ssl_noise.get_device())
#            print(ssl_clean.get_device())
            ssl_noisy = np.zeros((args.b, 480000))
            ssl_noise = ssl_noise.cpu().numpy() #to(device)
            ssl_clean = ssl_clean.cpu().numpy() #to(device)
            for batch_idx in range(args.b):
                cl, no, ny, nyrms = segmental_snr_mixer(
                    {'target_level_lower':-35.0,
                     'target_level_upper':-15.0},
                    ssl_clean[batch_idx,:],
                    ssl_noise[batch_idx,:], 
                    metadata['snr'],
                    target_level=metadata['target_level'])#.to(device)
                ssl_noise[batch_idx,:] = no
                ssl_clean[batch_idx,:] = cl
                ssl_noisy[batch_idx,:] = ny
            ssl_noise = torch.from_numpy(ssl_noise).float()#.to(device)
            ssl_clean = torch.from_numpy(ssl_clean).float()#.to(device)
            ssl_noisy = torch.from_numpy(ssl_noisy).float()#.to(device)
            ssl_noise = ssl_noise.to(device)
            ssl_noisy = ssl_noisy.to(device)
            ssl_clean = ssl_clean.to(device)

#            cl, no, ny, nyrms = segmental_snr_mixer(
#                 {'target_level':metadata['target_level'],
#                  'target_level_lower':-35.0,
#                  'target_level_upper':-15.0},
#                   ssl_clean[0,:].cpu().numpy(),
#                   ssl_noise[0,:].cpu().numpy(), 
#                   metadata['snr'])
#            ssl_noise = torch.from_numpy(no).float().to(device)
#            ssl_clean = torch.from_numpy(cl).float().to(device)
#            ssl_noisy = torch.from_numpy(ny).float().to(device)
#            ssl_noisy = (0.5*ssl_clean) + (0.5*ssl_noise)

#            for i in range(args.b):
#                ssl_noise[i, :] = torchaudio.functional.convolve(noise[i,:], noiseFilter, "same")
#                ssl_clean[i, :] = torchaudio.functional.convolve(clean[i,:], speechFilter, "same")
#                ssl_noisy[i, :] = (0.5*ssl_clean[i, :]) + (0.5*ssl_noise[i, :])
#            noisy_file, clean_file, noise_file, metadata = train_set._get_filenames(i)
#            ssl_clean = torchaudio.functional.convolve(clean[0,:], speechFilter, "same")
#            ssl_noise = torchaudio.functional.convolve(noise[0,:], noiseFilter, "same")
#            spec = stft_transform(ssl_noisy) # this be so broke
            noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft)
            clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft)

            denoised_abs = net(noisy_abs)
            noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
            clean_abs = slayer.axon.delay(clean_abs, out_delay)
            clean = slayer.axon.delay(ssl_clean, args.n_fft // 4 * out_delay)

            clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft)

            score = si_snr(clean_rec, ssl_clean)
            loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))

            assert torch.isnan(loss) == False

            optimizer.zero_grad()
            loss.backward()
            module.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            if i < 10:
                module.grad_flow(path=trained_folder + '/')

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

        if (epoch != args.epoch - 1):
            continue
        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(validation_loader):
            net.eval()

            with torch.no_grad():
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft)
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft)

                denoised_abs = net(noisy_abs)
                noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

                clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft)
                
                score = si_snr(clean_rec, clean)
                loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))
                stats.validation.correct_samples += torch.sum(score).item()
                stats.validation.loss_sum += loss.item()
                stats.validation.num_samples += noisy.shape[0]

                processed = i * validation_loader.batch_size
                total = len(validation_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / \
                    (i + 1) / validation_loader.batch_size
                header_list = [f'Valid: [{processed}/{total} '
                               f'({100.0 * processed / total:.0f}%)]']
                stats.print(epoch, i, samples_sec, header=header_list)

        writer.add_scalar('Loss/train', stats.training.loss, epoch)
        writer.add_scalar('Loss/valid', stats.validation.loss, epoch)
        writer.add_scalar('SI-SNR/train', stats.training.accuracy, epoch)
        writer.add_scalar('SI-SNR/valid', stats.validation.accuracy, epoch)

        stats.update()
        stats.plot(path=trained_folder + '/')
        if stats.validation.best_accuracy is True:
            torch.save(module.state_dict(), trained_folder + '/network.pt')
        stats.save(trained_folder + '/')

    module.load_state_dict(torch.load(trained_folder + '/network.pt'))
    module.export_hdf5(trained_folder + '/network.net')

    params_dict = {}
    for key, val in args._get_kwargs():
        params_dict[key] = str(val)
    writer.add_hparams(params_dict, {'SI-SNR': stats.validation.max_accuracy})
    writer.flush()
    writer.close()
