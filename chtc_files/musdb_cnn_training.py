import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from mir_eval import separation
from snr import si_snr
from torch.utils.data import DataLoader
from torchaudio.transforms import Fade

SOURCES = ["mixture", "vocals"]
TARGET_STEMS = SOURCES[1:]
NUM_STEMS = len(TARGET_STEMS)


def stft_splitter(audio, args):
    hop_length = args.n_fft // 4
    if args.spectrogram == 2:
        spectrum = torch.stft(
            audio,
            n_fft=4 * args.n_fft,
            hop_length=hop_length,
            onesided=True,
            return_complex=True,
        )
        magnitude = mel_transform(audio)
        return magnitude, spectrum.angle()

    if args.spectrogram == 1:
        spectrum = stft_transform(audio)
    else:
        spectrum = torch.stft(
            audio,
            n_fft=args.n_fft,
            hop_length=hop_length,
            onesided=True,
            return_complex=True,
        )
    return spectrum.abs(), spectrum.angle()


def stft_mixer(magnitude, angle, args, length):
    if args.spectrogram == 2:
        magnitude = inverse_mel_transform(magnitude)
        n_fft = 4 * args.n_fft
    else:
        n_fft = args.n_fft
    spectrum = torch.polar(magnitude, angle)
    if args.spectrogram == 1:
        return inv_stft_transform(spectrum, length=length)
    return torch.istft(
        spectrum,
        n_fft=n_fft,
        hop_length=args.n_fft // 4,
        onesided=True,
        length=length,
    )


class Network(torch.nn.Module):
    """Convolutional spectrogram-mask estimator."""

    def __init__(self, numHiddenLayers=2, hiddenLayerWidths=64,
                 n_fft=512, num_stems=NUM_STEMS, spectrogram=0):
        super().__init__()
        if numHiddenLayers < 0:
            raise ValueError("numHiddenLayers must be non-negative")
        if hiddenLayerWidths < 1:
            raise ValueError("hiddenLayerWidths must be positive")
        self.num_stems = num_stems
        self.freq_bins = 257 if spectrogram == 2 else n_fft // 2 + 1

        layers = []
        channels = 1
        for _ in range(numHiddenLayers):
            layers.extend([
                torch.nn.Conv2d(channels, hiddenLayerWidths, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
            ])
            channels = hiddenLayerWidths
        self.features = torch.nn.Sequential(*layers)
        self.mask_head = torch.nn.Conv2d(channels, num_stems, kernel_size=1)

    def forward(self, mixture_abs):
        features = self.features(mixture_abs.unsqueeze(1))
        mask = torch.relu(self.mask_head(features) + 1)
        return mixture_abs.unsqueeze(1) * mask


def segmental_snr_mixer(vocals, accompaniment, target_level,
                        target_level_lower, target_level_higher,
                        clipping_threshold=0.99):
    epsilon = torch.finfo(vocals.dtype).eps
    vocals_scaled = vocals / (vocals.abs().max() + epsilon)
    accompaniment_scaled = accompaniment / (accompaniment.abs().max() + epsilon)
    vocals_rms = vocals_scaled.square().mean().sqrt()
    accompaniment_rms = accompaniment_scaled.square().mean().sqrt()
    vocals_scaled = vocals_scaled * (10 ** (target_level / 20) / (vocals_rms + epsilon))
    accompaniment_scaled = accompaniment_scaled * (
        10 ** (target_level / 20) / (accompaniment_rms + epsilon)
    )
    mixture = vocals_scaled + accompaniment_scaled
    output_level = torch.randint(target_level_lower, target_level_higher + 1, (1,)).item()
    mixture_rms = mixture.square().mean().sqrt()
    scale = 10 ** (output_level / 20) / (mixture_rms + epsilon)
    mixture = mixture * scale
    vocals_scaled = vocals_scaled * scale
    accompaniment_scaled = accompaniment_scaled * scale
    peak = mixture.abs().max()
    if peak > clipping_threshold:
        clip_scale = peak / clipping_threshold
        mixture = mixture / clip_scale
        vocals_scaled = vocals_scaled / clip_scale
        accompaniment_scaled = accompaniment_scaled / clip_scale
    return vocals_scaled, accompaniment_scaled, mixture


def synthesize_random_mixture(vocals, accompaniment, target_level,
                              target_level_lower, target_level_higher):
    return segmental_snr_mixer(
        vocals, accompaniment, target_level, target_level_lower, target_level_higher
    )


def compute_loss_and_score(args, net, mixture, stems, return_waveforms=False):
    batch_size = mixture.size(0)
    mixture_abs, mixture_angle = stft_splitter(mixture, args)
    stems_flat = stems.reshape(batch_size * NUM_STEMS, stems.size(-1))
    stems_abs, _ = stft_splitter(stems_flat, args)

    frequency_bins, time_frames = mixture_abs.shape[1:]
    estimate_abs = net(mixture_abs)
    estimate_flat = estimate_abs.reshape(
        batch_size * NUM_STEMS, frequency_bins, time_frames
    )
    phase_bins = mixture_angle.size(1)
    angle_flat = mixture_angle.unsqueeze(1).expand(
        batch_size, NUM_STEMS, phase_bins, time_frames
    ).reshape(batch_size * NUM_STEMS, phase_bins, time_frames)
    target_length = stems_flat.size(-1)
    estimate_waveform = stft_mixer(estimate_flat, angle_flat, args, target_length)

    score_flat = si_snr(stems_flat, estimate_waveform)
    score_flat = torch.nan_to_num(score_flat, nan=0.0, posinf=0.0, neginf=0.0)
    loss = args.lam * F.mse_loss(estimate_flat, stems_abs) + (100 - score_flat.mean())
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    score_per_stem = score_flat.reshape(batch_size, NUM_STEMS).mean(dim=0)
    if return_waveforms:
        return loss, score_flat, score_per_stem, estimate_waveform, stems_flat
    return loss, score_flat, score_per_stem


def stem_breakdown_string(score_per_stem):
    return ", ".join(
        "{}={:.2f}dB".format(TARGET_STEMS[index], score_per_stem[index].item())
        for index in range(NUM_STEMS)
    )


def compute_sdr_per_stem(estimate, reference):
    estimate = estimate.reshape(NUM_STEMS, -1).detach().cpu().numpy()
    reference = reference.reshape(NUM_STEMS, -1).detach().cpu().numpy()
    scores = []
    for stem_index in range(NUM_STEMS):
        try:
            sdr, _, _, _ = separation.bss_eval_sources(
                reference[stem_index][None, :], estimate[stem_index][None, :]
            )
            scores.append(sdr[0])
        except ValueError:
            scores.append(np.nan)
    return scores


def iterate_track_chunks(total_len, chunk_len, overlap_frames):
    if total_len <= chunk_len:
        yield 0, total_len
        return
    start = 0
    end = chunk_len
    while start < total_len - overlap_frames:
        yield start, min(end, total_len)
        start += chunk_len - overlap_frames if start == 0 else chunk_len
        end += chunk_len


def stitch_chunks(chunk_waveforms, chunks, total_len, overlap_frames, device):
    stitched = torch.zeros(chunk_waveforms[0].size(0), total_len, device=device)
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")
    last = len(chunks) - 1
    for index, (waveform, (start, _)) in enumerate(zip(chunk_waveforms, chunks)):
        end = min(start + waveform.size(-1), total_len)
        fade.fade_in_len = 0 if index == 0 else int(overlap_frames)
        fade.fade_out_len = 0 if index == last else int(overlap_frames)
        stitched[:, start:end] += fade(waveform[:, :end - start])
    return stitched


def prepare_track(waveform, args):
    mono = waveform.squeeze(0).to(device).mean(dim=1)
    mono = downsampler(mono)
    if not args.useCipic:
        return mono

    vocals = conv_transform(mono[1], vocals_filter)
    accompaniment = conv_transform(mono[0] - mono[1], accompaniment_filter)
    target_level = random.uniform(args.target_level_lower, args.target_level_upper)
    vocals, accompaniment, mixture = synthesize_random_mixture(
        vocals,
        accompaniment,
        target_level,
        args.target_level_lower,
        args.target_level_upper,
    )
    return torch.stack([mixture, vocals])


def process_track(args, net, mono, training=False):
    chunks = list(iterate_track_chunks(mono.size(-1), chunk_len, overlap_frames))
    chunk_losses = []
    estimates = []
    targets = []
    for start, end in chunks:
        mixture = mono[0:1, start:end]
        stems = mono[1:, start:end].unsqueeze(0)
        loss, _, _, estimate, target = compute_loss_and_score(
            args, net, mixture, stems, return_waveforms=True
        )
        if training:
            (loss / (len(chunks) * args.b)).backward()
        chunk_losses.append(loss.item())
        estimates.append(estimate.detach().reshape(NUM_STEMS, -1))
        targets.append(target.detach().reshape(NUM_STEMS, -1))

    track_loss = sum(chunk_losses) / len(chunk_losses)
    stitched_estimate = stitch_chunks(
        estimates, chunks, mono.size(-1), overlap_frames, device
    )
    stitched_target = stitch_chunks(
        targets, chunks, mono.size(-1), overlap_frames, device
    )
    stem_scores = torch.nan_to_num(
        si_snr(stitched_target, stitched_estimate), nan=0.0, posinf=0.0, neginf=0.0
    )
    return track_loss, stem_scores, stitched_estimate, stitched_target, chunks


def validate_gradients(net):
    for parameter in net.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            print("Non-finite gradients detected; resetting gradients")
            net.zero_grad()
            return


def run_training_loop(args, net, optimizer, scheduler, train_loader, starting_epoch=0):
    net.train()
    last_loss = 0.0
    last_score = 0.0
    for epoch in range(args.epochs):
        training_losses = []
        training_scores = []
        optimizer.zero_grad()
        accumulated = 0
        for index, (waveform, _, _, name) in enumerate(train_loader):
            net.train()
            mono = prepare_track(waveform, args)
            track_loss, stem_scores, _, _, chunks = process_track(
                args, net, mono, training=True
            )
            accumulated += 1
            if accumulated == args.b:
                validate_gradients(net)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0
            track_score = stem_scores.mean().item()
            training_losses.append(track_loss)
            training_scores.append(track_score)
            if args.printOutputWhileTraining:
                print(
                    "Train [{} | {}] ({}, {} chunks) -> {} {} SI-SNR dB [{}]".format(
                        epoch + starting_epoch + 1, index, name[0], len(chunks),
                        track_loss, track_score, stem_breakdown_string(stem_scores),
                    )
                )
        if accumulated:
            validate_gradients(net)
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        last_loss = sum(training_losses) / len(training_losses)
        last_score = float(np.median(training_scores))
    return last_loss, last_score


def run_evaluation_loop(args, net, loader, subset_label, csv_path=None,
                        sisnr_csv_path=None, test_mode=False):
    net.eval()
    losses = []
    track_scores = []
    per_stem_sisnr = [[] for _ in range(NUM_STEMS)]
    per_stem_sdr = [[] for _ in range(NUM_STEMS)]
    score_file = open(csv_path, "w") if csv_path else None
    sisnr_file = open(sisnr_csv_path, "w") if sisnr_csv_path else None
    header = "track ID, train/test set, " + ", ".join(TARGET_STEMS)
    if score_file:
        score_file.write(header)
    if sisnr_file:
        sisnr_file.write(header)

    try:
        with torch.no_grad():
            for index, (waveform, _, _, name) in enumerate(loader):
                mono = prepare_track(waveform, args)
                loss, stem_sisnr, estimate, target, chunks = process_track(args, net, mono)
                stem_sdr = compute_sdr_per_stem(estimate, target) if test_mode else None
                losses.append(loss)
                track_scores.append(stem_sisnr.mean().item())
                for stem_index in range(NUM_STEMS):
                    per_stem_sisnr[stem_index].append(stem_sisnr[stem_index].item())
                    if test_mode:
                        per_stem_sdr[stem_index].append(stem_sdr[stem_index])

                track_id = index + args.test_offset if test_mode else index
                if score_file:
                    scores = stem_sdr if test_mode else [stem_sisnr[i].item() for i in range(NUM_STEMS)]
                    score_file.write("\n{}, {}, {}".format(
                        track_id, "test" if test_mode else subset_label,
                        ", ".join(str(value) for value in scores),
                    ))
                if sisnr_file:
                    sisnr_file.write("\n{}, test, {}".format(
                        track_id, ", ".join(str(stem_sisnr[i].item()) for i in range(NUM_STEMS)),
                    ))
                if args.save_audio_dir and test_mode:
                    save_track_audio(
                        args.save_audio_dir, name[0], args.sample_rate,
                        mono[0:1], estimate[0:1], target[0:1],
                        stem_sisnr[0].item(), stem_sdr[0],
                    )
                if test_mode and args.printOutputWhileTest:
                    print("Test [{}] ({}, {} chunks) -> {} SI-SNR dB, SDR [{}]".format(
                        track_id, name[0], len(chunks), stem_sisnr.mean().item(),
                        ", ".join("{}={:.2f}dB".format(TARGET_STEMS[s], stem_sdr[s])
                                  for s in range(NUM_STEMS)),
                    ))
                elif not test_mode and args.printOutputWhileValidation:
                    print("Valid [{}] ({}, {} chunks) -> {} SI-SNR dB [{}]".format(
                        index, name[0], len(chunks), stem_sisnr.mean().item(),
                        stem_breakdown_string(stem_sisnr),
                    ))
    finally:
        if score_file:
            score_file.close()
        if sisnr_file:
            sisnr_file.close()

    average_loss = sum(losses) / len(losses)
    median_score = float(np.median(track_scores))
    median_sisnr = [float(np.median(scores)) for scores in per_stem_sisnr]
    median_sdr = [float(np.nanmedian(scores)) for scores in per_stem_sdr] if test_mode else None
    return average_loss, median_score, median_sisnr, median_sdr


def save_track_audio(save_audio_dir, track_name, sample_rate,
                     mixture, estimated_vocals, target_vocals, si_snr_score, sdr_score):
    track_dir = os.path.join(save_audio_dir, track_name)
    os.makedirs(track_dir, exist_ok=True)
    mixture = mixture.detach().cpu()
    estimated_vocals = estimated_vocals.detach().cpu()
    torchaudio.save(os.path.join(track_dir, "mixture.wav"), mixture, sample_rate)
    torchaudio.save(os.path.join(track_dir, "vocals_estimate.wav"), estimated_vocals, sample_rate)
    torchaudio.save(
        os.path.join(track_dir, "accompaniment_estimate.wav"),
        mixture - estimated_vocals, sample_rate,
    )
    torchaudio.save(
        os.path.join(track_dir, "vocals_target.wav"), target_vocals.detach().cpu(), sample_rate
    )
    with open(os.path.join(track_dir, "scores.txt"), "w") as scores_file:
        scores_file.write("si_snr_db: {}\n".format(si_snr_score))
        scores_file.write("sdr_db: {}\n".format(sdr_score))


def make_parser():
    parser = argparse.ArgumentParser(description="Train a CNN MUSDB18-HQ source separator")
    parser.add_argument("-gpu", type=int, default=[0], nargs="+", help="GPU device IDs; CPU is used when CUDA is unavailable")
    parser.add_argument("-b", type=int, default=32, help="tracks accumulated per optimizer step")
    parser.add_argument("-lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("-lam", type=float, default=0.001, help="magnitude-loss weight")
    parser.add_argument("-n_fft", type=int, default=512, help="FFT size; hop is n_fft // 4")
    parser.add_argument("-clip", type=float, default=10, help="gradient clipping limit")
    parser.add_argument("-exp", type=str, default="", help="experiment identifier")
    parser.add_argument("-seed", type=int, default=None, help="random seed")
    parser.add_argument("-epochs", type=int, default=50, help="training epochs")
    parser.add_argument("-spectrogram", type=int, choices=[0, 1, 2], default=0,
                        help="0: torch STFT, 1: torchaudio STFT, 2: mel spectrogram")
    parser.add_argument("-path", type=str, default="../../", help="MUSDB18-HQ root directory")
    parser.add_argument("-sample_rate", type=int, default=16000, help="network sample rate")
    parser.add_argument("-segment_seconds", type=float, default=10.0, help="base chunk duration")
    parser.add_argument("-overlap", type=float, default=0.1, help="fractional chunk overlap")
    parser.add_argument("-training_samples", type=int, default=60000, help="maximum training tracks")
    parser.add_argument("-print_output_while_training", dest="printOutputWhileTraining", action="store_true")
    parser.add_argument("-validation_samples", type=int, default=60000, help="maximum validation tracks")
    parser.add_argument("-print_output_while_validation", dest="printOutputWhileValidation", action="store_true")
    parser.add_argument("-test_samples", type=int, default=60000, help="maximum test tracks")
    parser.add_argument("-test_offset", type=int, default=0, help="first test track index")
    parser.add_argument("-print_output_while_test", dest="printOutputWhileTest", action="store_true")
    parser.add_argument("-useCheckpoint", type=str, default="", help="checkpoint to resume")
    parser.add_argument("-saveCheckpoint", dest="saveCheckpoint", action="store_true", help="save a checkpoint after evaluation")
    parser.add_argument("-saveCheckpointName", type=str, default="", help="checkpoint filename")
    parser.add_argument("-numHiddenLayers", type=int, default=4, help="number of convolutional hidden layers")
    parser.add_argument("-hiddenLayerWidths", type=int, default=64, help="channels in each hidden layer")
    parser.add_argument("-save_audio_dir", type=str, default="", help="directory for test-track WAV outputs")
    parser.add_argument("-useCipic", dest="useCipic", action="store_true",
                        help="spatialize and remix audio with CIPIC HRIRs")
    parser.add_argument("-cipicSubject", type=int, default=12, help="CIPIC subject ID")
    parser.add_argument("-filterChannel", type=int, default=0, help="CIPIC channel used for separation")
    parser.add_argument("-vocalsFilterOrient", type=int, default=None,
                        help="CIPIC orientation index for the vocals pinna (default follows filterChannel)")
    parser.add_argument("-accompanimentFilterOrient", type=int, default=None,
                        help="CIPIC orientation index for the accompaniment pinna (default follows filterChannel)")
    parser.add_argument("-target_level_lower", type=int, default=-35, help="lower randomized mix level in dB")
    parser.add_argument("-target_level_upper", type=int, default=-15, help="upper randomized mix level in dB")
    return parser


def main():
    global device, downsampler, conv_transform, vocals_filter, accompaniment_filter
    global stft_transform, inv_stft_transform, mel_transform, inverse_mel_transform
    global chunk_len, overlap_frames

    parser = make_parser()
    args = parser.parse_args()
    if args.b < 1:
        parser.error("-b must be positive")
    if args.n_fft < 4 or args.n_fft % 4:
        parser.error("-n_fft must be a positive multiple of 4")
    if args.spectrogram == 2 and args.n_fft < 128:
        parser.error("-n_fft must be at least 128 for the 257-bin mel spectrogram")
    if args.sample_rate < 1 or args.segment_seconds <= 0 or args.overlap < 0:
        parser.error("sample rate and segment duration must be positive; overlap cannot be negative")
    if args.useCipic and NUM_STEMS != 1:
        parser.error("CIPIC remixing currently supports one target stem")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda", args.gpu[0])
    else:
        device = torch.device("cpu")
    identifier = args.exp + ("_seed{}".format(args.seed) if args.seed is not None else "")
    trained_folder = os.path.join("Trained", identifier)
    logs_folder = os.path.join("Logs", identifier)
    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    with open(os.path.join(trained_folder, "args.txt"), "w") as args_file:
        for key, value in sorted(vars(args).items()):
            args_file.write("{} : {}\n".format(key, value))

    chunk_len = int(args.sample_rate * args.segment_seconds * (1 + args.overlap))
    overlap_frames = int(args.overlap * args.sample_rate)
    net = Network(
        numHiddenLayers=args.numHiddenLayers,
        hiddenLayerWidths=args.hiddenLayerWidths,
        n_fft=args.n_fft,
        spectrogram=args.spectrogram,
    ).to(device)
    if use_cuda and len(args.gpu) > 1:
        net = torch.nn.DataParallel(net, device_ids=args.gpu)
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    total_params = sum(parameter.numel() for parameter in module.parameters())
    param_bytes = sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in module.buffers())
    with open(os.path.join(trained_folder, "model_size.txt"), "w") as size_file:
        size_file.write("total_params : {}\n".format(total_params))
        size_file.write("trainable_params : {}\n".format(total_params))
        size_file.write("param_bytes : {}\n".format(param_bytes))
        size_file.write("buffer_bytes : {}\n".format(buffer_bytes))
        size_file.write("total_bytes : {}\n".format(param_bytes + buffer_bytes))
        size_file.write("total_MB : {:.4f}\n".format((param_bytes + buffer_bytes) / 1e6))
    print("Using device {}. Model size: {:,} parameters, {:.4f} MB".format(
        device, total_params, (param_bytes + buffer_bytes) / 1e6
    ))

    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=args.n_fft, onesided=True, power=None, hop_length=args.n_fft // 4
    ).to(device)
    inv_stft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=args.n_fft, onesided=True, hop_length=args.n_fft // 4
    ).to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        n_fft=4 * args.n_fft, n_mels=257, power=1.0, hop_length=args.n_fft // 4
    ).to(device)
    inverse_mel_transform = torchaudio.transforms.InverseMelScale(
        n_stft=2 * args.n_fft + 1, n_mels=257
    ).to(device)
    downsampler = torchaudio.transforms.Resample(
        44100, args.sample_rate, dtype=torch.float32
    ).to(device)
    conv_transform = torchaudio.transforms.Convolve("same").to(device)

    if args.useCipic:
        from hrtfs.cipic_db import CipicDatabase

        if args.filterChannel not in (0, 1):
            parser.error("-filterChannel must be 0 or 1")
        default_orientations = (916, 316) if args.filterChannel == 1 else (316, 916)
        vocals_orientation = args.vocalsFilterOrient
        accompaniment_orientation = args.accompanimentFilterOrient
        vocals_orientation = default_orientations[0] if vocals_orientation is None else vocals_orientation
        accompaniment_orientation = default_orientations[1] if accompaniment_orientation is None else accompaniment_orientation
        subject = CipicDatabase.subjects[args.cipicSubject]
        vocals_filter = downsampler(torch.from_numpy(
            subject.getHRIRFromIndex(vocals_orientation, args.filterChannel)
        ).float().to(device))
        accompaniment_filter = downsampler(torch.from_numpy(
            subject.getHRIRFromIndex(accompaniment_orientation, args.filterChannel)
        ).float().to(device))
        print("CIPIC subject {}, vocals orientation {}, accompaniment orientation {}".format(
            args.cipicSubject, vocals_orientation, accompaniment_orientation
        ))

    optimizer = torch.optim.RAdam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    train_set = torchaudio.datasets.MUSDB_HQ(
        args.path, subset="train", sources=SOURCES, split="train", download=False
    )
    train_set.names = train_set.names[:args.training_samples]
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=use_cuda
    )

    starting_epoch = 0
    tracking_info = {}
    if args.useCheckpoint:
        checkpoint = torch.load(args.useCheckpoint, map_location=device, weights_only=False)
        module.load_state_dict(checkpoint["module_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        starting_epoch = checkpoint["epochs_completed"]
        tracking_info = checkpoint.get("tracking_info", {})
        print("Resuming from checkpoint after {} epochs".format(starting_epoch))

    last_training_loss, last_training_score = run_training_loop(
        args, net, optimizer, scheduler, train_loader, starting_epoch
    )
    print("Completed training: loss={}, SI-SNR={} dB".format(
        last_training_loss, last_training_score
    ))

    validation_set = torchaudio.datasets.MUSDB_HQ(
        args.path, subset="train", sources=SOURCES, split="validation", download=False
    )
    validation_set.names = validation_set.names[:args.validation_samples]
    validation_loader = DataLoader(
        validation_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=use_cuda
    )
    validation_path = os.path.join(logs_folder, "si_snr_scores.csv")
    validation_loss, validation_score, validation_per_stem, _ = run_evaluation_loop(
        args, net, validation_loader, "validation", csv_path=validation_path
    )

    test_set = torchaudio.datasets.MUSDB_HQ(
        args.path, subset="test", sources=SOURCES, download=False
    )
    test_set.names = test_set.names[args.test_offset:args.test_offset + args.test_samples]
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=use_cuda
    )
    test_csv_path = os.path.join(logs_folder, "musdb_cnn_test_sdr_scores.csv")
    test_sisnr_path = os.path.join(logs_folder, "musdb_cnn_test_sisnr_scores.csv")
    test_loss, test_score, test_per_stem_sisnr, test_per_stem_sdr = run_evaluation_loop(
        args, net, test_loader, "test", csv_path=test_csv_path,
        sisnr_csv_path=test_sisnr_path, test_mode=True,
    )

    if args.saveCheckpoint:
        checkpoint_name = args.saveCheckpointName
        if not checkpoint_name:
            cipic_prefix = "sub{}_chan{}_".format(args.cipicSubject, args.filterChannel) if args.useCipic else ""
            checkpoint_name = "{}b{}_depth{}_width{}_nfft{}_epochs{}.pt".format(
                cipic_prefix, args.b, args.numHiddenLayers, args.hiddenLayerWidths,
                args.n_fft, starting_epoch + args.epochs,
            )
        completed_epoch = starting_epoch + args.epochs
        tracking_info[completed_epoch] = {
            "training_loss": last_training_loss,
            "training_score": last_training_score,
            "validation_loss": validation_loss,
            "validation_score": validation_score,
            "test_loss": test_loss,
            "test_score": test_score,
            "test_sdr_per_stem": test_per_stem_sdr,
        }
        torch.save({
            "epochs_completed": completed_epoch,
            "module_state_dict": module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "tracking_info": tracking_info,
            "model_config": {
                "numHiddenLayers": args.numHiddenLayers,
                "hiddenLayerWidths": args.hiddenLayerWidths,
                "n_fft": args.n_fft,
                "spectrogram": args.spectrogram,
            },
        }, os.path.join(trained_folder, checkpoint_name))

    print("Final validation score: {} SI-SNR (dB)".format(validation_score))
    for index, stem in enumerate(TARGET_STEMS):
        print("  {}: {} SI-SNR (dB)".format(stem, validation_per_stem[index]))
    print("Final test score: {} SI-SNR (dB)".format(test_score))
    for index, stem in enumerate(TARGET_STEMS):
        print("  {}: {} SI-SNR (dB), {} SDR (dB)".format(
            stem, test_per_stem_sisnr[index], test_per_stem_sdr[index]
        ))
    print("Per-track SDR scores written to {}".format(test_csv_path))
    print("Per-track SI-SNR scores written to {}".format(test_sisnr_path))


if __name__ == "__main__":
    main()
