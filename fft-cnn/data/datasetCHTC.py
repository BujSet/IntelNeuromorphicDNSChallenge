import os
import glob
import re
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
from scipy.io.wavfile import read, write

class NoiseDataset(Dataset):
    def __init__(self, root: str = './'):
        """Aduio dataset loader for fft-cnn.

        Parameters
        ----------
        root : str, optional
            Path of the dataset location, by default './'.
        """
        self.root = root
        self.noisy_files = glob.glob(root + 'noisy/**.wav')
        self.file_id_from_name = re.compile('fileid_(\d+)')
    
    def _get_filenames(self, n: int) -> Tuple[str, str]:
        noisy_file = self.noisy_files[n % self.__len__()]
        filename = noisy_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        noise_file = self.root + f'noise/noise_fileid_{file_id}.wav'
       
        return noisy_file, noise_file

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_file, noise_file = self._get_filenames(idx)

        fs, audio = read(noise_file)
        audio = audio.astype(np.float32)  # Ensure float32 type for processing
        signal = audio.astype(np.float32)  # Ensure float32 type for processing
        t = np.linspace(0, len(audio) / fs, len(audio), endpoint=False)

        fft_result = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_result)
        fft_phase = np.angle(fft_result)
        
        phase_tensor = torch.tensor(fft_phase.reshape(-1, 1), dtype=torch.float32)
        magnitude_tensor = torch.tensor(fft_magnitude, dtype=torch.float32)
        
        # Preprocess the audio - STFT, magnitude, and phase
        # n_fft = 2048  # Number of FFT components
        # hop_length = 512  # Number of samples between successive frames
        # window = torch.hann_window(n_fft)

        # stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        # magnitude = torch.abs(stft)
        # phase = torch.angle(stft)

        # # phase_path = path + "_phase.pt"
        # # torch.save(phase, phase_path)
        # # Concatenate magnitude and phase along the last dimension
        # input_data = torch.cat((magnitude.unsqueeze(-1), phase.unsqueeze(-1)), dim=-1)

        return magnitude_tensor, phase_tensor, signal
   
if __name__ == '__main__':
    train_set = NoiseDataset(
        root='../../data/MicrosoftDNS_4_ICASSP/training_set/')
    validation_set = NoiseDataset(
        root='../../data/MicrosoftDNS_4_ICASSP/validation_set/')