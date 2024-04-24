import torch
import torchaudio
from torchaudio.transforms import GriffinLim
from your_model_file import YourModelClass
from preprocess import load_and_preprocess_audio

def generate_from_wav(input_path, model, device='cpu'):
    # Load and preprocess the input
    input_data = load_and_preprocess_audio(input_path)
    input_data = input_data.to(device)
    
    # Forward pass through the model
    model.eval()
    with torch.no_grad():
        output_data = model(input_data.unsqueeze(0)) # Todo: batch dimension
    
    # Convert model output back to time-domain waveform
    waveform = GriffinLim(n_fft=2048)(output_data)
    
    return waveform

def save_and_playback(waveform, output_path):
    # Save the waveform to a file
    torchaudio.save(output_path, waveform, sample_rate=16000)
    
    # Playback the sound
    print(f"Playing back generated sound from {output_path}")
    torchaudio.backend.sox_io_backend.play(output_path)

if __name__ == "__main__":
    model_path = 'path/to/your/model.pth'
    input_wav_path = 'path/to/your/input.wav'
    output_wav_path = 'path/to/save/output.wav'
    
    # Load your model
    model = YourModelClass()  # Instantiate your model class
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # Generate waveform from input
    generated_waveform = generate_from_wav(input_wav_path, model)

    # Save and playback
    save_and_playback(generated_waveform, output_wav_path)
