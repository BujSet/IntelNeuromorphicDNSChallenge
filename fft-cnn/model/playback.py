import torch
import torchaudio
import matplotlib.pyplot as plt
from network import PhaseShiftCNN 
import sys
import numpy as np
sys.path.append('../data')
from preprocess import load_and_preprocess_audio

def get_min_max_values(spectrogram):
    # Convert to dB
    spectrogram_db = 10 * np.log10(np.maximum(spectrogram, 1e-10))
    vmin, vmax = spectrogram_db.min(), spectrogram_db.max()
    return vmin, vmax

def plot_spectrograms(original_spectrogram, processed_spectrogram, sample_rate):
    fig, axs = plt.subplots(1, 2, figsize=(20, 4))
    hop_length = 512
    vmin, vmax = get_min_max_values(original_spectrogram)

    # Original Spectrogram
    # img1 = axs[0].imshow(10 * torch.log10(original_spectrogram + 1e-10).numpy(), origin='lower', aspect='auto', cmap='viridis', extent=[0, original_spectrogram.shape[-1], 0, sample_rate // 2])
    img1 = axs[0].imshow(10 * torch.log10(original_spectrogram + 1e-10).numpy(), origin='lower', aspect='auto', cmap='viridis', extent=[0, original_spectrogram.shape[-1], 0, sample_rate // 2], vmin=vmin, vmax=vmax)

    axs[0].set_title('Original Spectrogram')
    axs[0].set_xlabel('Frames')
    axs[0].set_ylabel('Frequency Bins')
    # fig.colorbar(img1, ax=axs[0], format="%+2.0f dB")

    # Processed Spectrogram
    # img2 = axs[1].imshow(10 * torch.log10(processed_spectrogram + 1e-10).numpy(), origin='lower', aspect='auto', cmap='viridis', extent=[0, processed_spectrogram.shape[-1], 0, sample_rate // 2])
    img2 = axs[1].imshow(10 * torch.log10(processed_spectrogram + 1e-10).numpy(), origin='lower', aspect='auto', cmap='viridis', extent=[0, processed_spectrogram.shape[-1], 0, sample_rate // 2], vmin=vmin, vmax=vmax)
    axs[1].set_title('Processed Spectrogram')
    axs[1].set_xlabel('Frames')
    axs[1].set_ylabel('Frequency Bins')
    
    fig.colorbar(img1, ax=axs[0], format="%+2.0f dB")
    fig.colorbar(img1, ax=axs[1], format="%+2.0f dB")
   
    plt.tight_layout()
    plt.show()

def load_original_phase(phase_path):
    return torch.load(phase_path)

def load_model(model_path):
    model = PhaseShiftCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    return model

def playback_model_output(model, audio_path):
    # Preprocess the input audio similarly to the training phase
    input_data, sample_rate, original_length, phase, magnitude = load_and_preprocess_audio(audio_path)
    
    original_phase = phase
    original_magnitude = magnitude 
    
    original_spectrogram = torch.abs(torch.stft(torchaudio.load(audio_path)[0], n_fft=2048, hop_length=512, return_complex=True))

    # Add batch dimension (model expects batch_size, channels, height, width)
    input_data = input_data.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        # processed_magnitude = model(input_data)
        processed_phase = model(original_phase)
    
    print(f"processed phase: {processed_phase[:10]}")
    print(f"original phase: {original_phase[:10]}")
    # output_magnitude = processed_magnitude[:, 0, :, :]

    # Maybe use STFT
    # complex_spec = torch.polar(output_magnitude, original_phase)
    complex_spec = torch.polar(original_magnitude, processed_phase + original_phase)
    
    waveform = torch.istft(complex_spec, n_fft = 2048, hop_length = 512, length = original_length)

    torchaudio.save("output_audio.wav", waveform.cpu(), sample_rate)
    # processed_spectrogram = torch.abs(torch.stft(torchaudio.load("output_audio.wav")[0], n_fft=2048, hop_length=512, return_complex=True))
    
    # plot_spectrograms(original_spectrogram.squeeze(), processed_spectrogram.squeeze(), sample_rate)
    
    
    # phase_angles = torch.angle(original_phase)
    # phase_angles_np = phase_angles.squeeze(0).numpy()
    
    # phase_angles_processed = torch.angle(processed_phase)
    # phase_angles_np_processed = phase_angles_processed.squeeze(0).numpy()
    
    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # axs[0].imshow(phase_angles_np, aspect='auto', origin='lower',
    #         extent=[0, phase_angles_np.shape[1], 0, phase_angles_np.shape[0]],
    #         cmap='twilight')
    # axs[0].set_xlabel('Frames')
    # axs[0].set_ylabel('Frequency Bins')
    # axs[0].set_title('Phase Plot of the Original STFT')
    
    # axs[0].imshow(phase_angles_np_processed, aspect='auto', origin='lower',
    #         extent=[0, phase_angles_np_processed.shape[1], 0, phase_angles_np_processed.shape[0]],
    #         cmap='twilight')
    # axs[0].set_xlabel('Frames')
    # axs[0].set_ylabel('Frequency Bins')
    # axs[0].set_title('Phase Plot of the Processed STFT')
    
    # fig.colorbar(axs[1].imshow(phase_angles_np, aspect='auto', origin='lower',
    #                        extent=[0, phase_angles_np.shape[1], 0, phase_angles_np.shape[0]],
    #                        cmap='twilight'), ax=axs[1], label='Phase (radians)')
    # plt.tight_layout()
    # plt.show()
    # plot_spectrogram(output_magnitude, "Processed Spectrogram by Model", sample_rate)
    # magnitude = output[:, 0, :, :]
    # phase = output[:, 1, :, :]

    # # Convert magnitude and phase into a complex tensor
    # complex_spec = torch.polar(magnitude, phase)
    # n_fft = 2048  # Number of FFT components, match this with your STFT settings
    # hop_length = 512  # Number of samples between successive frames, match this with your STFT settings
    # waveform = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, length=original_length)

    # # waveform = inverse_transform(output)  # Define inverse_transform according to your model's output processing
    
    # # Playback the waveform
    # torchaudio.play(waveform, sample_rate=16000)  # Adjust the sample rate as necessary

def apply_phase_shift_and_reconstruct(model, audio_path):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    original_length = waveform.shape[1]

    # Compute the STFT
    stft = torch.stft(waveform, n_fft=2048, hop_length=512, return_complex=True)
    original_magnitude = torch.abs(stft)
    original_phase = torch.angle(stft)

    # Assuming your model processes the phase and outputs a phase shift
    with torch.no_grad():
        phase_shift = model(original_phase.unsqueeze(0))  # Ensure input dimensions match model expectations

    # Calculate the new phase by adding the phase shift
    adjusted_phase = original_phase + phase_shift.squeeze(0)  # Adjust dimensions if necessary

    # Use torch.polar to reconstruct the complex spectrum from magnitude and adjusted phase
    complex_spec = torch.polar(original_magnitude, adjusted_phase)

    # Convert back to time domain
    reconstructed_waveform = torch.istft(complex_spec, n_fft=2048, hop_length=512, length=original_length)

    # Save the output audio
    torchaudio.save("output_audio.wav", reconstructed_waveform.cpu(), sample_rate)

    return reconstructed_waveform

if __name__ == "__main__":
    model_path = "../data/saved_models/phase_shift_cnn_model.pth"
    audio_path = "../data/datasets_fullband/noise_fullband/sine_wave.wav"
    # phase_path = "../data/datasets_fullband/noise_fullband/A2HaxSKLeqg.wav_phase.pt"

    # original_phase = load_original_phase(phase_path)
    model = load_model(model_path)
    apply_phase_shift_and_reconstruct(model, audio_path)
