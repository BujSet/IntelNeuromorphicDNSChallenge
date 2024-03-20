# model/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

sys.path.append('../data') 

from nn import PhaseShiftCNN
from dataset import NoiseDataset

def train(num_epochs=50):
    base_path = "../data/datasets_fullband/noise_fullband/"
    file_names = [
        "0eE-ekO1HKE.wav",
        "2wzUmr0idBo.wav",
        "35EOmSMTQ6I.wav",
        "A2HaxSKLeqg.wav",
        "e5m-4tzPSdA.wav",
        "m7Ar5TKZZJM.wav",
        "qZ55IXMaDOs.wav",
        "u7E_VpjrbTw.wav",
        "x8fp6MYE41s.wav",
        "Zv8opdWx3kw.wav"
    ]
    file_paths = [base_path + file_name for file_name in file_names]

    dataset = NoiseDataset(file_paths, base_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = PhaseShiftCNN()
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_idx, data in enumerate(dataloader):
            inputs = data 
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            targets = torch.zeros_like(outputs)
            loss = loss_function(outputs, targets)
            
            loss.backward() 
            optimizer.step()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    model_save_path = "../data/saved_models/phase_shift_cnn_model.pth"  # Adjust the path as needed
    train()