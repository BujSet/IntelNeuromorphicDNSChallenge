import os
import torch
from torch.utils.data import Dataset
import torchaudio

class NoiseDataset(Dataset):
    def __init__(self, file_paths, base_path):
        """
        Args:
            file_paths (list): List of file paths to the audio files.
        """
        self.file_paths = file_paths
        self.base_path = base_path

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(path)
        n_fft = 2048
        hop_length = 512  # Number of samples between successive frames
        window = torch.hann_window(n_fft)

        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        magnitude = torch.abs(stft)
        phase_angle = torch.angle(stft)
        # spectrogram = torchaudio.transforms.Spectrogram(n_fft=2048, power=None)(waveform)
        # mag, phase = torchaudio.functional.magphase(spectrogram)
        # phase_angle = torch.angle(phase)
        clean_phase_tensor = torch.zeros_like(phase_angle)
        
        return phase_angle, clean_phase_tensor
        # # Preprocess the audio - STFT, magnitude, and phase
        # n_fft = 2048  # Number of FFT components
        # hop_length = 512  # Number of samples between successive frames
        # window = torch.hann_window(n_fft)

        # stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        # magnitude = torch.abs(stft)
        # phase = torch.angle(stft)

        # phase_path = path + "_phase.pt"
        # torch.save(phase, phase_path)
        # # Concatenate magnitude and phase along the last dimension
        # input_data = torch.cat((magnitude.unsqueeze(-1), phase.unsqueeze(-1)), dim=-1)

        # return input_data
