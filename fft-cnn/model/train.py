# model/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
# sys.path.insert(0, os.path.abspath('..'))

# from ..data.dataset import NoiseDataset

sys.path.append('../data') 

from network import PhaseShiftCNN
from dataset import NoiseDataset

def train(num_epochs=50):
    base_path = "../data/datasets_fullband/noise_fullband/"
    file_names = [
        "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav",
         "sine_wave.wav",
        "sine_wave.wav",
        "sine_wave.wav"
        # "0eE-ekO1HKE.wav",
        # "2wzUmr0idBo.wav",
        # # "35EOmSMTQ6I.wav",
        # "A2HaxSKLeqg.wav",
        # "e5m-4tzPSdA.wav",
        # "m7Ar5TKZZJM.wav",
        # "qZ55IXMaDOs.wav",
        # "u7E_VpjrbTw.wav",
        # "x8fp6MYE41s.wav",
        # "Zv8opdWx3kw.wav"
    ]
    file_paths = [base_path + file_name for file_name in file_names]

    dataset = NoiseDataset(file_paths, base_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = PhaseShiftCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (dirty, clean) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(dirty)
            loss = loss_function(outputs, clean)
            
            loss.backward() 
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')    
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    model_save_path = "../data/saved_models/phase_shift_cnn_model.pth"  # Adjust the path as needed
    train()