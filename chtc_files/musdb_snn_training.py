# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os, sys, math
sys.path.append('./')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lava.lib.dl import slayer
from snr import si_snr
from mir_eval import separation
import torchaudio
import random

# MUSDB18-HQ always stacks sources in the order they are requested. Fixing
# this order lets every stem be recovered by a simple index into the
# batch/stem dimension instead of doing a name lookup per sample.
SOURCES = ['mixture', 'vocals']
TARGET_STEMS = SOURCES[1:]
NUM_STEMS = len(TARGET_STEMS)

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

def stft_mixer(stft_abs, stft_angle, n_fft=512, method=None, length=None):
    '''`length`, when given, is passed straight to the ISTFT so the
    reconstructed waveform is forced to exactly that many samples. Without
    it, torch.istft only approximately inverts the STFT: whenever the input
    length isn't an exact multiple of the hop size, the round trip can land
    a few dozen samples short or long of the original, which then fails to
    broadcast against the target waveform in si_snr.'''
    spec = torch.complex(stft_abs * torch.cos(stft_angle),
                                        stft_abs * torch.sin(stft_angle))
    if (method == None):
        return torch.istft(spec, n_fft=n_fft, onesided=True, length=length)
    if (type(method) == int):
        print("Perform inver mel scale transform")
        sys.exit(0)

    return method(spec, length=length)

class Network(torch.nn.Module):
    def __init__(self,
            threshold=0.1,
            tau_grad=0.1,
            scale_grad=0.8,
            max_delay=64,
            out_delay=0,
            hiddenLayerWidths=512,
            n_fft=512,
            num_stems=NUM_STEMS):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay
        self.EPS = 2.220446049250313e-16
        self.hiddenLayerWidths = hiddenLayerWidths
        self.num_stems = num_stems
        self.freq_bins = n_fft // 2 + 1

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
            slayer.block.sigma_delta.Dense(sdnn_params, self.freq_bins, hiddenLayerWidths, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Dense(sdnn_params, hiddenLayerWidths, hiddenLayerWidths, weight_norm=False, delay=True, delay_shift=True),
            # Output is widened to num_stems masks (one per target stem)
            # stacked along the feature dimension: [stem0_freqs, stem1_freqs, ...]
            slayer.block.sigma_delta.Output(sdnn_params, hiddenLayerWidths, num_stems * self.freq_bins, weight_norm=False),
        ])

        self.blocks[0].pre_hook_fx = self.input_quantizer

        self.blocks[1].delay.max_delay = max_delay
        self.blocks[2].delay.max_delay = max_delay

    def forward(self, mixture_abs):
        x = mixture_abs - self.stft_mean

        for block in self.blocks:
            x = block(x)

        batch, _, time_frames = x.shape
        mask = torch.relu(x + 1).reshape(batch, self.num_stems, self.freq_bins, time_frames)
        delayed_mixture = slayer.axon.delay(mixture_abs, self.out_delay).unsqueeze(1)
        return delayed_mixture * mask

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

def compute_loss_and_score(args, net, mixture, stems, return_waveforms=False):
    '''Shared forward/loss computation for one track (b=1). `stems`
    has shape (b, NUM_STEMS, num_frames). The stem dimension is flattened
    into the batch dimension for the STFT/ISTFT/axon-delay calls, which are
    agnostic to leading dims, then reshaped back out to score each stem.
    Set return_waveforms=True to additionally get back the reconstructed
    estimate and delay-aligned target waveforms (needed for mir_eval SDR).'''
    b = mixture.size(0)

    if (args.spectrogram == 0):
        mixture_abs, mixture_arg = stft_splitter(mixture, args.n_fft, None)
    elif (args.spectrogram == 1):
        mixture_abs, mixture_arg = stft_splitter(mixture, args.n_fft, stft_transform)
    else:
        mixture_abs, mixture_arg = stft_splitter(mixture, args.n_fft, mel_transform)

    stems_flat = stems.reshape(b * NUM_STEMS, stems.size(-1))
    if (args.spectrogram == 0):
        stems_abs_flat, stems_arg_flat = stft_splitter(stems_flat, args.n_fft, None)
    elif (args.spectrogram == 1):
        stems_abs_flat, stems_arg_flat = stft_splitter(stems_flat, args.n_fft, stft_transform)
    else:
        stems_abs_flat, stems_arg_flat = stft_splitter(stems_flat, args.n_fft, mel_transform)

    freq_bins, time_frames = mixture_abs.size(1), mixture_abs.size(2)

    denoised_abs = net(mixture_abs) # (b, NUM_STEMS, freq_bins, time_frames)
    denoised_abs_flat = denoised_abs.reshape(b * NUM_STEMS, freq_bins, time_frames)

    mixture_arg_delayed = slayer.axon.delay(mixture_arg, out_delay)
    mixture_arg_delayed_flat = mixture_arg_delayed.unsqueeze(1).expand(
        b, NUM_STEMS, freq_bins, time_frames).reshape(b * NUM_STEMS, freq_bins, time_frames)

    stems_abs_delayed_flat = slayer.axon.delay(stems_abs_flat, out_delay)
    stems_waveform_delayed_flat = slayer.axon.delay(stems_flat, args.n_fft // 4 * out_delay)

    target_len = stems_waveform_delayed_flat.size(-1)
    if (args.spectrogram == 0):
        clean_rec_flat = stft_mixer(denoised_abs_flat, mixture_arg_delayed_flat, args.n_fft, None, length=target_len)
    elif (args.spectrogram == 1):
        clean_rec_flat = stft_mixer(denoised_abs_flat, mixture_arg_delayed_flat, args.n_fft, inv_stft_transform, length=target_len)
    else:
        clean_rec_flat = stft_mixer(denoised_abs_flat, mixture_arg_delayed_flat, args.n_fft, 2)

    score_flat = si_snr(clean_rec_flat, stems_waveform_delayed_flat)
    if torch.isnan(score_flat).any():
        score_flat[torch.isnan(score_flat)] = 0
    score_per_stem = score_flat.reshape(b, NUM_STEMS).mean(dim=0)

    loss = lam * F.mse_loss(denoised_abs_flat, stems_abs_delayed_flat) + (100 - torch.mean(score_flat))
    if torch.isnan(loss).any():
        loss[torch.isnan(loss)] = 0

    if return_waveforms:
        return loss, score_flat, score_per_stem, clean_rec_flat, stems_waveform_delayed_flat
    return loss, score_flat, score_per_stem

def stem_breakdown_string(score_per_stem):
    return ", ".join(
        "{}={:.2f}dB".format(TARGET_STEMS[i], score_per_stem[i].item())
        for i in range(NUM_STEMS))

def compute_sdr_per_stem(clean_rec_flat, target_waveform_flat, b):
    '''mir_eval SDR (the same metric hybrid_demucs_test.txt reports),
    computed per crop on the mono waveforms and averaged across crops, so
    the SNN's test-set CSV is directly comparable to the Hybrid Demucs
    baseline. bss_eval_sources is called with a single (1, samples) "source"
    per stem, since our pipeline works on downmixed mono audio rather than
    the stereo channels-as-sources trick the baseline script used.'''
    estimate = clean_rec_flat.reshape(b, NUM_STEMS, -1).detach().cpu().numpy()
    reference = target_waveform_flat.reshape(b, NUM_STEMS, -1).detach().cpu().numpy()
    sdr_per_stem = []
    for s in range(NUM_STEMS):
        crop_scores = []
        for c in range(b):
            try:
                sdr, _, _, _ = separation.bss_eval_sources(
                    reference[c, s][None, :], estimate[c, s][None, :])
                crop_scores.append(sdr[0])
            except ValueError:
                # bss_eval_sources rejects near-silent references (e.g. a
                # 4s crop of a stem with no signal in it); skip that crop.
                crop_scores.append(np.nan)
        sdr_per_stem.append(np.nanmean(crop_scores))
    return sdr_per_stem

def iterate_track_chunks(total_len, chunk_len, overlap_frames):
    '''Sequentially walks fixed-length, overlapping windows across a track's
    time axis, mirroring hybrid_demucs' separate_sources chunking (see
    chtc_files/hybrid_demucs_full_dataset.py) so a whole track is processed
    piece by piece instead of in one giant forward pass.'''
    if total_len <= chunk_len:
        yield 0, total_len
        return
    start = 0
    end = chunk_len
    while start < total_len - overlap_frames:
        yield start, min(end, total_len)
        if start == 0:
            start += chunk_len - overlap_frames
        else:
            start += chunk_len
        end += chunk_len

def run_warm_up_training(args, net, optimizer, train_loader):
    net.train()
    # Run a single mini-batch just to set the network dimensions (needed
    # before loading a checkpoint, same requirement as the DNS scripts).
    for i, (waveform, sr, num_frames, name) in enumerate(train_loader):
        mono = waveform.squeeze(0).to(device).mean(dim=1)
        mono = downsampler(mono)
        start, end = next(iterate_track_chunks(mono.size(-1), chunk_len, overlap_frames))
        mixture = mono[0:1, start:end]
        stems = mono[1:, start:end].unsqueeze(0)

        loss, score_flat, score_per_stem = compute_loss_and_score(args, net, mixture, stems)

        optimizer.zero_grad()
        loss.backward()
        module.validate_gradients()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        return

def run_training_loop(args, net, optimizer, train_loader, startingEpoch=0):
    '''Trains on whole tracks, but processes each one as a sequence of
    overlapping chunks (see iterate_track_chunks, mirroring hybrid_demucs'
    chunked inference) rather than one giant forward pass, since a full
    track is too long to comfortably fit in memory at once. Tracks also
    vary in length, so they can't be stacked into a real batch either:
    args.b is instead implemented as gradient accumulation, with each
    track's loss averaged over its own chunks first, then args.b tracks'
    losses summed before each optimizer step.'''
    net.train()
    delay_weights = dict()
    averageTrainingLoss = 0
    averageTrainingScore = 0
    for epoch in range(args.epochs):
        trainingLosses = []
        trainingScores = []
        optimizer.zero_grad()
        accumulated = 0
        for i, (waveform, sr, num_frames, name) in enumerate(train_loader):
            net.train()
            mono = waveform.squeeze(0).to(device).mean(dim=1)
            mono = downsampler(mono)

            chunks = list(iterate_track_chunks(mono.size(-1), chunk_len, overlap_frames))
            chunkLosses = []
            chunkScores = []
            chunkStemScores = []
            for start, end in chunks:
                mixture = mono[0:1, start:end]
                stems = mono[1:, start:end].unsqueeze(0)

                loss, score_flat, score_per_stem = compute_loss_and_score(args, net, mixture, stems)
                (loss / (len(chunks) * args.b)).backward()

                chunkLosses.append(loss.item())
                chunkScores.append(torch.mean(score_flat).item())
                chunkStemScores.append(score_per_stem.detach())

            trackLoss = sum(chunkLosses) / len(chunkLosses)
            trackScore = sum(chunkScores) / len(chunkScores)
            trackStemScore = torch.stack(chunkStemScores).mean(dim=0)

            accumulated += 1
            if accumulated == args.b:
                module.validate_gradients()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0

            trainingLosses.append(trackLoss)
            trainingScores.append(trackScore)
            if args.printOutputWhileTraining:
                statString = "Train [" + str(epoch + startingEpoch + 1) + " | " + str(i) + "] ("
                statString += name[0] + ", " + str(len(chunks)) + " chunks) -> "
                statString += str(trackLoss) + " "
                statString += str(trackScore) + " SI-SNR dB ["
                statString += stem_breakdown_string(trackStemScore) + "]"
                print(statString)
        if accumulated > 0:
            module.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
        if args.trackDelayWhileTraining:
            for param_tensor in net.state_dict():
                if ("delay.delay" in param_tensor):
                    if not param_tensor in delay_weights.keys():
                        delay_weights[param_tensor] = dict()
                    delay_weights[param_tensor][epoch] = net.state_dict()[param_tensor].clone().detach().cpu()
        # Updates only the last training epoch's loss is kept
        averageTrainingLoss = sum(trainingLosses) / (1.0 * len(trainingLosses))
        averageTrainingScore = sum(trainingScores) / (1.0 * len(trainingScores))
    return delay_weights, averageTrainingLoss, averageTrainingScore

def run_validation_loop(args, net, validation_loader, csv_path=None, subset_label='validation'):
    validationScores = []
    validationLosses = []
    perStemScores = [[] for _ in range(NUM_STEMS)]
    net.eval()

    score_file = None
    if csv_path is not None:
        score_file = open(csv_path, "w")
        score_file.write("track ID, train/test set, " + ", ".join(TARGET_STEMS))

    for i, (waveform, sr, num_frames, name) in enumerate(validation_loader):
        with torch.no_grad():
            mono = waveform.squeeze(0).to(device).mean(dim=1)
            mono = downsampler(mono)

            chunks = list(iterate_track_chunks(mono.size(-1), chunk_len, overlap_frames))
            chunkLosses = []
            chunkScores = []
            chunkStemScores = []
            for start, end in chunks:
                mixture = mono[0:1, start:end]
                stems = mono[1:, start:end].unsqueeze(0)

                loss, score_flat, score_per_stem = compute_loss_and_score(args, net, mixture, stems)
                chunkLosses.append(loss.item())
                chunkScores.append(torch.mean(score_flat).item())
                chunkStemScores.append(score_per_stem)

            trackLoss = sum(chunkLosses) / len(chunkLosses)
            trackScore = sum(chunkScores) / len(chunkScores)
            trackStemScore = torch.stack(chunkStemScores).mean(dim=0)

            validationScores.append(trackScore)
            validationLosses.append(trackLoss)
            for s in range(NUM_STEMS):
                perStemScores[s].append(trackStemScore[s].item())
            if score_file is not None:
                row = str(i) + ", " + subset_label + ", "
                row += ", ".join(str(trackStemScore[s].item()) for s in range(NUM_STEMS))
                score_file.write("\n" + row)
            if args.printOutputWhileValidation:
                statString = "Valid [" + str(i) + "] (" + name[0] + ", " + str(len(chunks)) + " chunks) -> "
                statString += str(trackLoss) + " "
                statString += str(trackScore) + " SI-SNR dB ["
                statString += stem_breakdown_string(trackStemScore) + "]"
                print(statString)

    if score_file is not None:
        score_file.close()

    averageValidationLoss = sum(validationLosses) / (1.0 * len(validationLosses))
    averageValidationScore = sum(validationScores) / (1.0 * len(validationScores))
    averagePerStemScore = [sum(scores) / (1.0 * len(scores)) for scores in perStemScores]
    return averageValidationLoss, averageValidationScore, averagePerStemScore

def run_test_loop(args, net, test_loader, csv_path=None, sisnr_csv_path=None):
    '''Evaluates on MUSDB18-HQ's real held-out "test" subset -- the same 50
    tracks hybrid_demucs_test.txt was generated from -- and writes a CSV in
    that file's exact format, reporting SDR (via mir_eval) rather than
    SI-SNR so the two are a direct, same-metric baseline comparison. Also
    writes a second CSV in the same format with the SI-SNR scores.'''
    net.eval()
    testLosses = []
    testScores = []
    perStemSiSnr = [[] for _ in range(NUM_STEMS)]
    perStemSdr = [[] for _ in range(NUM_STEMS)]

    score_file = None
    if csv_path is not None:
        score_file = open(csv_path, "w")
        score_file.write("track ID, train/test set, " + ", ".join(TARGET_STEMS))

    sisnr_score_file = None
    if sisnr_csv_path is not None:
        sisnr_score_file = open(sisnr_csv_path, "w")
        sisnr_score_file.write("track ID, train/test set, " + ", ".join(TARGET_STEMS))

    for i, (waveform, sr, num_frames, name) in enumerate(test_loader):
        with torch.no_grad():
            mono = waveform.squeeze(0).to(device).mean(dim=1)
            mono = downsampler(mono)

            chunks = list(iterate_track_chunks(mono.size(-1), chunk_len, overlap_frames))
            chunkLosses = []
            chunkScores = []
            chunkStemScores = []
            chunkSdrPerStem = []
            for start, end in chunks:
                mixture = mono[0:1, start:end]
                stems = mono[1:, start:end].unsqueeze(0)

                loss, score_flat, score_per_stem, clean_rec_flat, target_waveform_flat = \
                    compute_loss_and_score(args, net, mixture, stems, return_waveforms=True)
                sdr_per_stem = compute_sdr_per_stem(clean_rec_flat, target_waveform_flat, mixture.size(0))

                chunkLosses.append(loss.item())
                chunkScores.append(torch.mean(score_flat).item())
                chunkStemScores.append(score_per_stem)
                chunkSdrPerStem.append(sdr_per_stem)

            trackLoss = sum(chunkLosses) / len(chunkLosses)
            trackScore = sum(chunkScores) / len(chunkScores)
            trackStemScore = torch.stack(chunkStemScores).mean(dim=0)
            trackSdrPerStem = [np.nanmean([c[s] for c in chunkSdrPerStem]) for s in range(NUM_STEMS)]

            testLosses.append(trackLoss)
            testScores.append(trackScore)
            for s in range(NUM_STEMS):
                perStemSiSnr[s].append(trackStemScore[s].item())
                perStemSdr[s].append(trackSdrPerStem[s])

            if score_file is not None:
                row = str(i) + ", test, "
                row += ", ".join(str(trackSdrPerStem[s]) for s in range(NUM_STEMS))
                score_file.write("\n" + row)
            if sisnr_score_file is not None:
                row = str(i) + ", test, "
                row += ", ".join(str(trackStemScore[s].item()) for s in range(NUM_STEMS))
                sisnr_score_file.write("\n" + row)
            if args.printOutputWhileTest:
                statString = "Test [" + str(i) + "] (" + name[0] + ", " + str(len(chunks)) + " chunks) -> "
                statString += str(trackLoss) + " "
                statString += str(trackScore) + " SI-SNR dB, SDR ["
                statString += ", ".join(
                    "{}={:.2f}dB".format(TARGET_STEMS[s], trackSdrPerStem[s])
                    for s in range(NUM_STEMS))
                statString += "]"
                print(statString)

    if score_file is not None:
        score_file.close()
    if sisnr_score_file is not None:
        sisnr_score_file.close()

    averageTestLoss = sum(testLosses) / (1.0 * len(testLosses))
    averageTestScore = sum(testScores) / (1.0 * len(testScores))
    averagePerStemSiSnr = [sum(scores) / (1.0 * len(scores)) for scores in perStemSiSnr]
    averagePerStemSdr = [np.nanmean(scores) for scores in perStemSdr]
    return averageTestLoss, averageTestScore, averagePerStemSiSnr, averagePerStemSdr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu',
                        type=int,
                        default=[0],
                        help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',
                        type=int,
                        default=32,
                        help='number of whole tracks accumulated into each training batch '
                             '(via gradient accumulation, since tracks vary in length)')
    parser.add_argument('-lr',
                        type=float,
                        default=0.0001,
                        help='learning rate (fixed for the whole run, no scheduler)')
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
                        help='root directory containing the musdb18hq dataset folder')
    parser.add_argument('-sample_rate',
                        type=int,
                        default=16000,
                        help='sample rate the network operates at; MUSDB18-HQ is downsampled from 44100 Hz to this')
    parser.add_argument('-segment_seconds',
                        type=float,
                        default=10.0,
                        help='base length (in seconds) of each sequential chunk a track is split '
                             'into for train/validation/test, mirroring hybrid_demucs\' chunked '
                             'inference (see chtc_files/hybrid_demucs_full_dataset.py)')
    parser.add_argument('-overlap',
                        type=float,
                        default=0.1,
                        help='fractional overlap between consecutive chunks, as in hybrid_demucs')
    parser.add_argument('-training_samples',
                        type=int,
                        default=60000,
                        help='Number of tracks training should use, supports small dataset subset')
    parser.add_argument('-print_output_while_training',
                        dest='printOutputWhileTraining',
                        action='store_true',
                        help='Switch flag to print score after every mini-batch during training')
    parser.add_argument('-validation_samples',
                        type=int,
                        default=60000,
                        help='Number of tracks validation should use, supports small dataset subset')
    parser.add_argument('-print_output_while_validation',
                        dest='printOutputWhileValidation',
                        action='store_true',
                        help='Switch flag to print score after every mini-batch during validation')
    parser.add_argument('-test_samples',
                        type=int,
                        default=60000,
                        help='Number of tracks the held-out MUSDB18-HQ test-subset eval should use, supports small dataset subset')
    parser.add_argument('-print_output_while_test',
                        dest='printOutputWhileTest',
                        action='store_true',
                        help='Switch flag to print score after every track during the test-subset eval')
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
    parser.add_argument('-hiddenLayerWidths',
                        type=int,
                        default=512,
                        help='# of nuerons in hidden layers')

    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        identifier += '_seed{}'.format(args.seed)

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
    # As in hybrid_demucs' chunked inference (chtc_files/hybrid_demucs_full_dataset.py):
    # chunk_len is the base segment plus its overlap margin, and consecutive
    # chunks advance by chunk_len after the first step overlaps by
    # overlap_frames with the chunk before it.
    chunk_len = int(args.sample_rate * args.segment_seconds * (1 + args.overlap))
    overlap_frames = int(args.overlap * args.sample_rate)
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
    stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=args.n_fft,
                onesided=True,
                power=None,
                hop_length=math.floor(args.n_fft//4)).to(device)
    inv_stft_transform = torchaudio.transforms.InverseSpectrogram(
                n_fft=args.n_fft,
                onesided=True,
                hop_length=math.floor(args.n_fft//4)).to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
                n_fft=4*args.n_fft,
                n_mels=257,
                power=2,
                hop_length=math.floor(args.n_fft//4)).to(device)

    # MUSDB18-HQ tracks are recorded at 44100 Hz; resample down to the rate
    # the SNN's STFT front end was designed around (16 kHz, same as DNS).
    downsampler = torchaudio.transforms.Resample(44100, args.sample_rate, dtype=torch.float32).to(device)

    # Define optimizer module.
    optimizer = torch.optim.RAdam(net.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-5)

    train_set = torchaudio.datasets.MUSDB_HQ(args.path,
            subset="train",
            sources=SOURCES,
            split="train",
            download=False)
    if args.training_samples < len(train_set.names):
        train_set.names = train_set.names[:args.training_samples]

    # Train on whole, uncropped tracks: batch_size=1 loads one full track per
    # DataLoader item (tracks vary in length, so they can't be stacked into a
    # real batch), and run_training_loop accumulates gradients over args.b
    # tracks before each optimizer step, so args.b is still the effective
    # training batch size.
    train_loader = DataLoader(train_set,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)

    startingEpoch = 0
    trackingInfo = dict()
    if args.useCheckpoint != "":
        run_warm_up_training(args, net, optimizer, train_loader)
        checkpoint = torch.load(args.useCheckpoint, weights_only=True)
        module.load_state_dict(checkpoint['module_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startingEpoch = checkpoint['epochs_completed']
        trackingInfo = checkpoint['tracking_info']
        print("Current Tracking info:")
        print("Epoch | Training Loss | Training Score (dB) | Validation Loss | Validation Score (dB)")
        for i in range(0, startingEpoch+1):
            if i in trackingInfo.keys():
                tloss = trackingInfo[i]['training_loss']
                tScore = trackingInfo[i]['training_score']
                vLoss = trackingInfo[i]['validation_loss']
                vScore = trackingInfo[i]['validation_score']
                checkpointStr  = str(i) + " | "
                checkpointStr += str(tloss) + " | " + str(tScore) + " | "
                checkpointStr += str(vLoss) + " | " + str(vScore)
                print(checkpointStr)
        startingTrainingLoss = trackingInfo[startingEpoch]['training_loss']
        startingTrainingScore = trackingInfo[startingEpoch]['training_score']
        startingValidationLoss = trackingInfo[startingEpoch]['validation_loss']
        startingValidationScore = trackingInfo[startingEpoch]['validation_score']
        statusString  = "Resuming from checkpoint [epochs_completed:"
        statusString += str(startingEpoch) + ", training loss="
        statusString += str(startingTrainingLoss) + ", training si-snr:"
        statusString += str(startingTrainingScore) + ", validation loss="
        statusString += str(startingValidationLoss) + ", validation si-snr:"
        statusString += str(startingValidationScore) + "]"
        print(statusString)

    delay_weights, lastTrainingLoss, lastTrainingScore = run_training_loop(args, net, optimizer, train_loader, startingEpoch)

    if args.trackDelayWhileTraining:
        plot_weights(delay_weights)

    print("Completed training loop [epochs_completed:" + str(args.epochs) + ", training loss=" + str(lastTrainingLoss) + ", si-snr:" + str(lastTrainingScore) + "]")

    validation_set = torchaudio.datasets.MUSDB_HQ(args.path,
            subset="train",
            sources=SOURCES,
            split="validation",
            download=False)
    if args.validation_samples < len(validation_set.names):
        validation_set.names = validation_set.names[:args.validation_samples]

    validation_loader = DataLoader(validation_set,
                               batch_size=1,
                               shuffle=True,
                               num_workers=4,
                               pin_memory=True)
    csv_path = os.path.join(logs_folder, 'si_snr_scores.csv')
    finalValidationLoss, finalValidationScore, finalPerStemScore = run_validation_loop(
        args, net, validation_loader, csv_path=csv_path, subset_label='validation')
    statusString  = "Completed training and validation [epochs_completed:"
    statusString += str(startingEpoch+args.epochs) + ", training loss="
    statusString += str(lastTrainingLoss) + ", training si-snr:"
    statusString += str(lastTrainingScore) + ", validation loss="
    statusString += str(finalValidationLoss) + ", validation si-snr:"
    statusString += str(finalValidationScore) + "]"
    print(statusString)

    # Evaluate on MUSDB18-HQ's real "test" subset -- the same 50 tracks
    # hybrid_demucs_test.txt was generated from -- so this CSV is a direct,
    # same-metric (SDR) baseline comparison against that file. shuffle=False
    # keeps track ID i lined up with hybrid_demucs_test.txt's row i, since
    # MUSDB_HQ always lists tracks in the same sorted order.
    test_set = torchaudio.datasets.MUSDB_HQ(args.path,
            subset="test",
            sources=SOURCES,
            download=False)
    if args.test_samples < len(test_set.names):
        test_set.names = test_set.names[:args.test_samples]

    test_loader = DataLoader(test_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)
    test_csv_path = os.path.join(logs_folder, 'musdb_snn_test_sdr_scores.csv')
    test_sisnr_csv_path = os.path.join(logs_folder, 'musdb_snn_test_sisnr_scores.csv')
    finalTestLoss, finalTestScore, finalPerStemSiSnr, finalPerStemSdr = run_test_loop(
        args, net, test_loader, csv_path=test_csv_path, sisnr_csv_path=test_sisnr_csv_path)

    if (args.saveCheckpoint):
        trackingInfo[startingEpoch+args.epochs] = dict()
        currEpochStats = trackingInfo[startingEpoch+args.epochs]
        currEpochStats['training_loss'] = lastTrainingLoss
        currEpochStats['training_score'] = lastTrainingScore
        currEpochStats['validation_loss'] = finalValidationLoss
        currEpochStats['validation_score'] = finalValidationScore
        currEpochStats['test_loss'] = finalTestLoss
        currEpochStats['test_score'] = finalTestScore
        currEpochStats['test_sdr_per_stem'] = finalPerStemSdr
        torch.save({
                'epochs_completed': startingEpoch + args.epochs,
                'module_state_dict': module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tracking_info': trackingInfo,
                }, trained_folder + '/network.pt')
    print("Final validation score: " + str(finalValidationScore) + " SI-SNR (dB)")
    for i in range(NUM_STEMS):
        print("  " + TARGET_STEMS[i] + ": " + str(finalPerStemScore[i]) + " SI-SNR (dB)")
    print("Final test-subset score (vs. hybrid_demucs_test.txt baseline): " + str(finalTestScore) + " SI-SNR (dB)")
    for i in range(NUM_STEMS):
        statString  = "  " + TARGET_STEMS[i] + ": "
        statString += str(finalPerStemSiSnr[i]) + " SI-SNR (dB), "
        statString += str(finalPerStemSdr[i]) + " SDR (dB)"
        print(statString)
    print("Per-track SDR scores written to " + test_csv_path + " in the same format as hybrid_demucs_test.txt")
    print("Per-track SI-SNR scores written to " + test_sisnr_csv_path)
