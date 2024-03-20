import torch
import torchaudio

def load_and_preprocess_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    original_length = waveform.size(1)
    # Perform STFT
    n_fft = 2048 # Number of FFT components
    hop_length = 512 # Number of samples between successive frames
    window = torch.hann_window(window_length=n_fft)
    
    # torchaudio's stft returns a tensor with the last dimension being complex (real + imaginary parts)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    
    # Separate magnitude and phase
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)

    phase_path = path + "_phase.pt"
    torch.save(phase, phase_path)
    
    input_data = torch.cat((magnitude.unsqueeze(-1), phase.unsqueeze(-1)), dim=-1)
    
    return input_data, sample_rate, original_length

# path = 'training/iQD6mIwQkbw.wav'
# input_data = load_and_preprocess_audio(path)
# print(f"Input Data Shape: {input_data.shape}")