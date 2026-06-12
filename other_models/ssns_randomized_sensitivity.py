import os, sys, math
sys.path.append('./')
from audio_dataloader import DNSAudio
from audio_dataloader import DNSAudioNoNoisy
from hrtfs.cipic_db import CipicDatabase 
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
import time
from lava.lib.dl import slayer
from snr import si_snr
import torchaudio
from noisyspeech_synthesizer import segmental_snr_mixer
import random
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

def stft_mixer(stft_abs, stft_angle, n_fft=512, method=None):
    spec = torch.complex(stft_abs * torch.cos(stft_angle),
                                        stft_abs * torch.sin(stft_angle))
    if (method == None):
        return torch.istft(spec, n_fft=n_fft, onesided=True)
    if (type(method) == int):
        print("Perform inver mel scale transform")
        sys.exit(0)

    return method(spec)

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

    def _segmental_snr_mixer(self, clean, noise, snr,
                        target_level, 
                        target_level_lower,
                        target_level_higher,
                        clipping_threshold=0.99,
                        ):
        '''Function to mix clean speech and noise at various segmental SNR levels'''
        clean_div = torch.max(torch.abs(clean)) + self.EPS
        noise_div = torch.max(torch.abs(noise)) + self.EPS
        ssl_clean = torch.div(clean, clean_div.item())
        ssl_noise = torch.div(noise, noise_div.item())
        # TODO should only calculate the RMS of the 'active' windows, but
        # for now we just use the whole audio sample
        clean_rms = torch.sqrt(torch.mean(torch.square(ssl_clean))).item()
        noise_rms = torch.sqrt(torch.mean(torch.square(ssl_noise))).item()
        clean_scalar = 10 ** (target_level / 20) / (clean_rms + self.EPS)
        noise_scalar = 10 ** (target_level / 20) / (noise_rms + self.EPS)
        ssl_clean = torch.mul(ssl_clean, clean_scalar)
        ssl_noise = torch.mul(ssl_noise, noise_scalar)
        # Adjust noise to SNR level
        noise_scalar = clean_rms / (10**(snr/20)) / (noise_rms+self.EPS)
        ssl_noise = torch.mul(ssl_noise, noise_scalar)
        ssl_noisy = torch.add(ssl_clean, ssl_noise)
        noisy_rms_level = torch.randint(
                target_level_lower,
                target_level_higher,
                (1,))
        noisy_rmsT = torch.sqrt(torch.mean(torch.square(ssl_noisy)))
        noisy_rms = torch.sqrt(torch.mean(torch.square(ssl_noisy))).item()
        noisy_scalar = 10 ** (noisy_rms_level / 20) / (noisy_rms + self.EPS)
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

    def synthesizeNoisySpeech(self, clean, noise, batchSize, 
            snr,
            targetLevel,
            targetLevelLower,
            targetLevelHigher):
        ssl_noisy = torch.zeros(batchSize, 480000).to(device)
        ssl_noise = torch.zeros(batchSize, 480000).to(device)
        ssl_clean = torch.zeros(batchSize, 480000).to(device)
        for i in range(batchSize):
            ssl_clean[i, :], ssl_noise[i,:], ssl_noisy[i,:], rms = self._segmental_snr_mixer(clean[i,:], noise[i,:], 
                snr[i].item(), 
                targetLevel[i].item(),
                targetLevelLower,
                targetLevelHigher)
                       
        return ssl_noisy, ssl_clean, ssl_noise

    def forward(self, noisy):
        x = noisy - self.stft_mean

        for block in self.blocks:
            x = block(x)

        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask

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

def run_warm_up_training_cipic(args, net, optimizer, scheduler, train_loader):

    for i, (clean, noise, idx) in enumerate(train_loader):
        net.train()
        speechFilter  = torch.from_numpy(CIPICSubject.getHRIRFromIndex(args.speechFilterOrient, args.speechFilterChannel)).float()
        speechFilter  = speechFilter.to(device)
        noiseFilter   = torch.from_numpy(CIPICSubject.getHRIRFromIndex(args.noiseFilterOrient, args.noiseFilterChannel)).float()
        noiseFilter   = noiseFilter.to(device)
        noise = noise.to(device)
        clean = clean.to(device)
        ssl_noise = torch.zeros(args.b, 480000).to(device)
        ssl_clean = torch.zeros(args.b, 480000).to(device)
        ssl_snrs  = torch.zeros(args.b, 1).to(device)
        ssl_targlvls= torch.zeros(args.b, 1).to(device)
        for batch_idx in range(args.b):
            ssl_noise[batch_idx,:] = conv_transform(noise[batch_idx,:], downsampler(noiseFilter))
            ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], downsampler(speechFilter))
            clean_file, noise_file, metadata = train_set._get_filenames(idx[batch_idx])
            ssl_snrs[batch_idx] = metadata['snr']
            ssl_targlvls[batch_idx] = metadata['target_level']

        ssl_noisy, ssl_clean, ssl_noise = module.synthesizeNoisySpeech(
            ssl_clean, 
            ssl_noise, 
            args.b, 
            ssl_snrs,
            ssl_targlvls,
            -35,
            -15)

        noisy_abs, noisy_arg = stft_splitter(ssl_noisy, args.n_fft, None)
        clean_abs, clean_arg = stft_splitter(ssl_clean, args.n_fft, None)

        denoised_abs = net(noisy_abs)
        noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
        clean_abs = slayer.axon.delay(clean_abs, out_delay)
        clean = slayer.axon.delay(ssl_clean, args.n_fft // 4 * out_delay)

        clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft, None)

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
        return

def run_training_loop_with_cipic(args, net, optimizer, scheduler, train_loader, startingEpoch=0):
    delay_weights = dict()
    averageTrainingLoss = 0
    averageTrainingScore = 0
    speechFilter  = torch.from_numpy(CIPICSubject.getHRIRFromIndex(args.speechFilterOrient, args.speechFilterChannel)).float()
    speechFilter  = speechFilter.to(device)
    noiseFilter   = torch.from_numpy(CIPICSubject.getHRIRFromIndex(args.noiseFilterOrient, args.noiseFilterChannel)).float()
    noiseFilter   = noiseFilter.to(device)
    epochLatencies = []
    #chtc_print(args, "Beginning training epochs...")
    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()
        trainingLosses = []
        trainingScores = []
        mini_batch_iter = 0
        for i, (clean, noise, idx) in enumerate(train_loader):
            mini_batch_start_time = time.perf_counter()
            net.train()
            noise = noise.to(device)
            clean = clean.to(device)
            ssl_noise = torch.zeros(args.b, 480000).to(device)
            ssl_clean = torch.zeros(args.b, 480000).to(device)
            ssl_snrs  = torch.zeros(args.b, 1).to(device)
            ssl_targlvls= torch.zeros(args.b, 1).to(device)
            for batch_idx in range(args.b):
                ssl_noise[batch_idx,:] = conv_transform(noise[batch_idx,:], downsampler(noiseFilter))
                ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], downsampler(speechFilter))
                clean_file, noise_file, metadata = train_set._get_filenames(idx[batch_idx])
                ssl_snrs[batch_idx] = metadata['snr']
                ssl_targlvls[batch_idx] = metadata['target_level']

            ssl_noisy, ssl_clean, ssl_noise = module.synthesizeNoisySpeech(
                ssl_clean, 
                ssl_noise, 
                args.b, 
                ssl_snrs,
                ssl_targlvls,
                -35,
                -15)


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

            trainingLosses.append(torch.mean(loss).item())
            trainingScores.append(torch.mean(score).item())
            if args.printOutputWhileTraining:
                statString = "Train [" + str(epoch + startingEpoch + 1) + " | " + str(i) + "]"
                statString += " (s,n)=("
                statString += str(speechFilterOrient) + "," + str(noiseFilterOrient) + ") -> "
                statString += str(loss.item()) + " " 
                statString += str(torch.mean(score).item()) + " SI-SNR dB"
                chtc_print(args, statString)
            mini_batch_end_time = time.perf_counter()
            mini_batch_duration = mini_batch_end_time - mini_batch_start_time
            #chtc_print(args, f"Mini Batch Iteration {mini_batch_iter+1} completed in {mini_batch_duration:.5f} seconds")
            mini_batch_iter += 1
        scheduler.step()
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epochLatencies.append(epoch_duration)
        chtc_print(args, f"Epoch {epoch+1}:{epoch_duration:.5f}")
        # Updates only the last training epoch's loss is kept
        averageTrainingLoss = sum(trainingLosses) / (1.0 * len(trainingLosses))
        averageTrainingScore = sum(trainingScores) / (1.0 * len(trainingScores))
    return delay_weights, averageTrainingLoss, averageTrainingScore

def run_validation_loop_with_cipic(args, net, validation_loader):
    net.eval()
    validationLosses = []
    validationScores = []
    speechFilter  = torch.from_numpy(CIPICSubject.getHRIRFromIndex(args.speechFilterOrient, args.speechFilterChannel)).float()
    speechFilter  = speechFilter.to(device)
    noiseFilter   = torch.from_numpy(CIPICSubject.getHRIRFromIndex(args.noiseFilterOrient, args.noiseFilterChannel)).float()
    noiseFilter   = noiseFilter.to(device)
    #chtc_print(args, "Beginning validation pass...")
    epoch_start_time = time.perf_counter()
    for i, (clean, noise, idx) in enumerate(validation_loader):
        noise = noise.to(device)
        clean = clean.to(device)
        ssl_noise = torch.zeros(args.b, 480000).to(device)
        ssl_clean = torch.zeros(args.b, 480000).to(device)
        ssl_snrs  = torch.zeros(args.b, 1).to(device)
        ssl_targlvls= torch.zeros(args.b, 1).to(device)
        for batch_idx in range(args.b):
            ssl_noise[batch_idx,:] = conv_transform(noise[batch_idx,:], downsampler(noiseFilter))
            ssl_clean[batch_idx,:] = conv_transform(clean[batch_idx,:], downsampler(speechFilter))
            clean_file, noise_file, metadata = validation_set._get_filenames(idx[batch_idx])
            ssl_snrs[batch_idx] = metadata['snr']
            ssl_targlvls[batch_idx] = metadata['target_level']
        
        ssl_noisy, ssl_clean, ssl_noise = module.synthesizeNoisySpeech(
            ssl_clean, 
            ssl_noise, 
            args.b, 
            ssl_snrs,
            ssl_targlvls,
            -35,
            -15)

        with torch.no_grad():
            
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

            validationScores.append(torch.mean(score).item())
            validationLosses.append(torch.mean(loss).item())
            if args.printOutputWhileValidation:
                statString = "Valid [" + str(i) + "]"
                statString += " (s,n)=("
                statString += str(speechFilterOrient) + "," + str(noiseFilterOrient) + ") -> "
                statString += str(loss.item()) + " " 
                statString += str(torch.mean(score).item()) + " SI-SNR dB"
                print(statString)
    epoch_end_time = time.perf_counter()
    epoch_duration = epoch_end_time - epoch_start_time
    chtc_print(args, f"Validation pass:{epoch_duration:.5f}")
    averageValidationLoss = sum(validationLosses) / (1.0 * len(validationLosses))
    averageValidationScore = sum(validationScores) / (1.0 * len(validationScores))
    return averageValidationLoss, averageValidationScore


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
    parser.add_argument('-saveCheckpoint',
                        dest='saveCheckpoint', 
                        action='store_true',
                        help='Switch flag to enable saving a chekpoint after training')

    # CIPIC Filter Parameters
    # ID:21 ==> Mannequin with large pinna
    # ID 165 ==> Mannequin with small pinna
    # The rest are real subjects
    parser.add_argument('-cipicSubject',
                        type=int,
                        default=12,
                        help='Cipic subject ID for pinna filters')
    parser.add_argument('-hiddenLayerWidths',
                        type=int,
                        default=512,
                        help='# of nuerons in hidden layers')
    parser.add_argument('-speechFilterOrient',
                        type=int,
                        default=608,
                        help='Orientation for speech filter')
    parser.add_argument('-speechFilterChannel',
                        type=int,
                        default=0,
                        help='Channel for speech filter')
    parser.add_argument('-noiseFilterOrient',
                        type=int,
                        default=640,
                        help='Orientation for noise filter')
    parser.add_argument('-noiseFilterChannel',
                        type=int,
                        default=0,
                        help='Channel for noise filter')
    parser.add_argument('-is_CHTC_job',
                        dest='isCHTCJob',
                        action='store_true',
                        help='Switch flag to indicate if this job was run on CHTC')

    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}{}'.format(args.optim, args.seed)

    assert(args.spectrogram == 0 or args.spectrogram == 1 or args.spectrogram == 2)
    trained_folder = 'Trained'
    logs_folder = 'Logs' + identifier
    writer = SummaryWriter('runs/' + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    lam = args.lam

    chtc_print(args, f'Using GPUs {args.gpu} with name {torch.cuda.get_device_name(args.gpu)}')
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

    # 316 center of front upper  right hemisphere
    # 300 center of front bottom right hemisphere
    # 332 center of back  upper  right hemisphere
    # 348 center of back  bottom right hemisphere
    # 916 center of front upper  left  hemisphere
    # 900 center of front bottom left  hemisphere
    # 932 center of back  upper  left  hemisphere
    # 948 center of back  bottom left  hemisphere
    # 608 midsaggittal in front
    # 640 midsaggittal in back 
    CIPICSubject = CipicDatabase.subjects[args.cipicSubject]
    #chtc_print(args, "Using Subject " + str(args.cipicSubject) + " for spatial sound separation...")
    validation_set = DNSAudioNoNoisy(root=args.path + 'validation_set/', maxFiles=args.validation_samples)
    
    validation_loader = DataLoader(validation_set,
                               batch_size=args.b,
                               shuffle=True,
                               collate_fn=validation_set.collate_fn,
                               num_workers=4,
                               pin_memory=True)
    finalValidationLoss, finalValidationScore = run_validation_loop_with_cipic(args, net, validation_loader)

    train_set = DNSAudioNoNoisy(root=args.path + 'training_set/', maxFiles=args.training_samples)
    
    train_loader = DataLoader(train_set,
                          batch_size=args.b,
                          shuffle=True,
                          collate_fn=train_set.collate_fn,
                          num_workers=4,
                          pin_memory=True)

    trackingInfo = dict()
    delay_weights, lastTrainingLoss, lastTrainingScore = run_training_loop_with_cipic(args, net, optimizer, scheduler, train_loader)
    
    #chtc_print(args, "Completed training loop [epochs_completed:" + str(args.epochs) + ", training loss=" + str(lastTrainingLoss) + ", si-snr:" + str(lastTrainingScore) + "]")

    #statusString  = "Completed training and validation [epochs_completed:" 
    #statusString += str(args.epochs) + ", training loss=" 
    #statusString += str(lastTrainingLoss) + ", training si-snr:" 
    #statusString += str(lastTrainingScore) + ", validation loss="
    #statusString += str(finalValidationLoss) + ", validation si-snr:" 
    #statusString += str(finalValidationScore) + "]"
    #chtc_print(args, statusString)
    checkpoint_start_time = time.perf_counter()
    if (args.saveCheckpoint):
        trackingInfo[args.epochs] = dict()
        currEpochStats = trackingInfo[args.epochs]
        currEpochStats['training_loss'] = lastTrainingLoss
        currEpochStats['training_score'] = lastTrainingScore
        currEpochStats['validation_loss'] = finalValidationLoss
        currEpochStats['validation_score'] = finalValidationScore
        torch.save({
                'epochs_completed': args.epochs,
                'module_state_dict': module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'tracking_info': trackingInfo,
                }, trained_folder + '/network_' + args.exp + '.pt')
    checkpoint_end_time = time.perf_counter()
    checkpoint_duration = checkpoint_end_time - checkpoint_start_time
    chtc_print(args, f"Checkpoint:{checkpoint_duration:.5f}")
    #chtc_print(args, "Final validation score: " + str(finalValidationScore) + " SI-SNR (dB)")
