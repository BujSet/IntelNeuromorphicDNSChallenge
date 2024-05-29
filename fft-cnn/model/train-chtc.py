# model/train.py
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import sys
import os 

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"script_dir: {script_dir}")
# Add the 'data' directory to the path
# Assuming 'data' and 'model' are sibling directories
data_dir = os.path.join(script_dir, '..', 'data')
print(f"data_dir: {data_dir}")
sys.path.insert(0, data_dir)

sys.path.append('./')

import glob
from torch.utils.tensorboard import SummaryWriter

from network import PhaseShiftCNN
from datasetCHTC import NoiseDataset

if __name__ == '__main__':
    print("Entered train-chtc.py from NOISE CANCEL")
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu',
                        type=int,
                        default=[0],
                        help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',
                        type=int,
                        default=32,
                        help='batch size for dataloader')
    parser.add_argument('-lr',
                        type=float,
                        default=0.01,
                        help='initial learning rate')
    parser.add_argument('-epoch',
                        type=int,
                        default=50,
                        help='number of epochs to run')
    parser.add_argument('-path',
                        type=str,
                        default='../../',
                        help='dataset path')

    args = parser.parse_args()
    identifier = 'test'
    trained_folder = 'Trained' + identifier
    logs_folder = 'Logs' + identifier
    print(trained_folder)
    
    writer = SummaryWriter('runs/' + identifier)
    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]) if torch.cuda.is_available() else 'cpu')
    
    if len(args.gpu) == 0:
        print("Building CPU network")
        net = PhaseShiftCNN().to(device)
        module = net
    else:
        print("Building GPU network")
        net = torch.nn.DataParallel(PhaseShiftCNN().to(device),
                                    device_ids=args.gpu)
        module = net.module
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_function = nn.MSELoss()
    train_set = NoiseDataset(root=args.path + 'training_set/')
    validation_set = NoiseDataset(root=args.path + 'validation_set/')
    
    train_loader = DataLoader(train_set, 
                              batch_size=args.b,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    
    validation_loader = DataLoader(validation_set, 
                                   batch_size=args.b,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True)
    
    net.train()
    best_loss = float('inf')
    model_save_path = "phase_shift_cnn_model.pth"
    # os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(args.epoch):
        total_loss = 0.0
        if (epoch % 10 == 0):
            print(f"Entering epoch: {epoch}")
        
        for batch_idx, (magnitude, phase, original_signal) in enumerate(train_loader):
            magnitude = magnitude.to(device)
            phase = phase.to(device)
            original_signal = original_signal.to(device)
            
            optimizer.zero_grad()
            
            adjusted_phase = net(phase).squeeze()
            
            # Combine with original magnitude to get adjusted FFT
            adjusted_fft = magnitude * torch.exp(1j * adjusted_phase)

            # Perform IFFT to convert the adjusted complex FFT back to the time domain
            adjusted_signal = torch.fft.ifft(adjusted_fft).real
            
            # Calculate the result of adding the original signal and the adjusted signal
            combined_signal = original_signal + adjusted_signal
            
            # The goal is to minimize the output of the combined_signal, aiming for silence
            loss = loss_function(combined_signal, torch.zeros_like(combined_signal))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{args.epoch}, Average Loss: {avg_loss:.6f}')    
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            
    params_dict = {}
    for key, val in args._get_kwargs():
        params_dict[key] = str(val)
        
    writer.flush()
    writer.close()
