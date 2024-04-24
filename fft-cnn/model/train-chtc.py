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
    print("Entered train-chtc.py")
    print("exiting")
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
                        default=0.001,
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
    device = torch.device('cuda:{}'.format(args.gpu[0]))
    
    if len(args.gpu) == 0:
        print("Building CPU network")
        net = PhaseShiftCNN().to(device)

        module = net
    else:
        print("Building GPU network")
        net = torch.nn.DataParallel(PhaseShiftCNN().to(device),
                                    device_ids=args.gpu)
        
        module = net.module
    
    optimizer = torch.optim.Adam(net.parameters(),
                                lr = args.lr)
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

    for epoch in range(args.epoch):
        if (epoch < 5 or epoch % 5 == 0):
            print(f"Entering epoch: {epoch}")
        
        for batch_idx, (dirty, clean) in enumerate(train_loader):
            net.train()
            
            inputs = data.to(device) 
            
            optimizer.zero_grad()
            outputs = net(inputs)
            
            # targets = torch.zeros_like(outputs).to(device) # maybe this works
            loss = loss_function(outputs, clean)
            
            loss.backward() 
            optimizer.step()
        
        # for batch_idx, data in enumerate(validation_loader):
        #     net.eval()
            
        #     inputs = data.to(device) 
        #     outputs = net(inputs)
        #     targets = torch.zeros_like(outputs).to(device) # maybe this works
        #     loss = loss_function(outputs, targets)
        
        # writer.add_scalar('Loss/train', stats.training.loss, epoch)
        # writer.add_scalar('Loss/valid', stats.validation.loss, epoch)
        
        # stats.update()
        # stats.plot(path=trained_folder + '/')
        
        # if (epoch>1):
        #     torch.save(module.state_dict(), trained_folder + '/fft-cnn.pt')

        # if stats.validation.best_accuracy is True:
        #     torch.save(module.state_dict(), trained_folder + '/fft-cnn.pt')

        # stats.save(trained_folder + '/')
    
    # module.load_state_dict(torch.load(trained_folder + '/fft-cnn.pt'))
    # module.export_hdf5(trained_folder + '/fft-cnn.net')
    params_dict = {}
    for key, val in args._get_kwargs():
        params_dict[key] = str(val)
    writer.flush()
    writer.close()