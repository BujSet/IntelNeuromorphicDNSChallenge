import os
import glob
import re
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Tuple

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

        waveform, sample_rate = torchaudio.load(noise_file)

        # Preprocess the audio - STFT, magnitude, and phase
        n_fft = 2048  # Number of FFT components
        hop_length = 512  # Number of samples between successive frames
        window = torch.hann_window(n_fft)

        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        # phase_path = path + "_phase.pt"
        # torch.save(phase, phase_path)
        # Concatenate magnitude and phase along the last dimension
        input_data = torch.cat((magnitude.unsqueeze(-1), phase.unsqueeze(-1)), dim=-1)

        return input_data
   
if __name__ == '__main__':
    train_set = NoiseDataset(
        root='../../data/MicrosoftDNS_4_ICASSP/training_set/')
    validation_set = NoiseDataset(
        root='../../data/MicrosoftDNS_4_ICASSP/validation_set/')