# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os, sys, math
sys.path.append('./')
from hrtfs.cipic_hrtf_dataloader import CipicHRTFs, Sample
from hrtfs.cipic_db import CipicDatabase 
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
import torchaudio
import random

class Network(torch.nn.Module):
    def __init__(self,
            input_dims=12, 
            hiddenLayerWidths=512,
            output_dims=200):
        super().__init__()
        self.hiddenLayerWidths = hiddenLayerWidths
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.blocks = torch.nn.ModuleList([
            torch.nn.Linear(in_features=input_dims, out_features=hiddenLayerWidths),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hiddenLayerWidths, out_features=hiddenLayerWidths),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hiddenLayerWidths, out_features=output_dims),
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def validate_gradients(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any()
                                       or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            self.zero_grad()


def run_training_loop(args, net, optimizer, scheduler, train_loader, startingEpoch=0):
    net.train()
    averageTrainingLoss = 0
    for epoch in range(args.epochs):
        trainingLosses = []
        for i, (x, y, idx) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            pred_y = net(x)

            loss = F.mse_loss(y, pred_y)

            if torch.isnan(loss).any():
                loss[torch.isnan(loss)] = 0
            assert torch.isnan(loss) == False

            optimizer.zero_grad()
            loss.backward()
            module.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            trainingLosses.append(torch.mean(loss).item())
            if args.printOutputWhileTraining:
                statString = "Train [" + str(epoch+startingEpoch+1) + " | " + str(i) + "] -> "
                statString += str(loss.item())
                print(statString)
        scheduler.step()
        averageTrainingLoss = sum(trainingLosses) / (1.0 * len(trainingLosses))
    return averageTrainingLoss

def run_validation_loop(args, net, validation_loader):
    validationLosses = []
    net.eval()
    for i, (x, y, idx) in enumerate(validation_loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():

            pred_y = net(x)
            

            loss = F.mse_loss(y, pred_y)
            if torch.isnan(loss).any():
                loss[torch.isnan(loss)] = 0

            validationLosses.append(torch.mean(loss).item())
            if args.printOutputWhileValidation:
                statString = "Valid [" + str(i) + "] -> "
                statString += str(loss.item())
                print(statString)
    averageValidationLoss = sum(validationLosses) / (1.0 * len(validationLosses))
    return averageValidationLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu',
                        type=int,
                        default=[0],
                        help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',
                        type=int,
                        default=25,
                        help='batch size for dataloader')
    parser.add_argument('-lr',
                        type=float,
                        default=0.001,
                        help='initial learning rate')
    parser.add_argument('-input_dims',
                        type=int,
                        default=12,
                        help='number of input dimensions')
    parser.add_argument('-output_dims',
                        type=int,
                        default=200,
                        help='number of output dimensions')
    parser.add_argument('-clip',
                        type=float,
                        default=10,
                        help='gradient clipping limit')
    parser.add_argument('-exp',
                        type=str,
                        default='',
                        help='experiment differentiater string')
    parser.add_argument('-seed',
                        type=int,
                        default=None,
                        help='random seed of the experiment')
    parser.add_argument('-epochs',
                        type=int,
                        default=50,
                        help='Number of training epochs to run')
    parser.add_argument('-path',
                        type=str,
                        default='../../',
                        help='dataset path')
    parser.add_argument('-training_samples',
                        type=int,
                        default=39375,
                        help='Number of samples training should use, (=0.9*35*1250))')
    parser.add_argument('-print_output_while_training',
                        dest='printOutputWhileTraining', 
                        action='store_true',
                        help='Switch flag to print score after every mini-batch during training')
    parser.add_argument('-validation_samples',
                        type=int,
                        default=4375,
                        help='Number of samples validation should use, (=0.1*35*1250))')
    parser.add_argument('-print_output_while_validation',
                        dest='printOutputWhileValidation', 
                        action='store_true',
                        help='Switch flag to print score after every mini-batch during validation')
    parser.add_argument('-useCheckpoint',
                        type=str,
                        default='',
                        help='Checkpoint to continue training from')
    parser.add_argument('-saveCheckpoint',
                        dest='saveCheckpoint', 
                        action='store_true',
                        help='Switch flag to enable saving a chekpoint after training')
    parser.add_argument('-hiddenLayerWidths',
                        type=int,
                        default=512,
                        help='# of nuerons in hidden layers')

    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}{}'.format(args.optim, args.seed)

    trained_folder = 'Trained' + identifier
    logs_folder = 'Logs' + identifier
    writer = SummaryWriter('runs/' + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    net = torch.nn.DataParallel(Network(
                args.input_dims,
                args.hiddenLayerWidths,
                args.output_dims).to(device),
                    device_ids=args.gpu)
    module = net.module

    # Define optimizer module.
    optimizer = torch.optim.RAdam(net.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    # Gather samples for training and validation split
    RightEarDataset = list()
    for subject in CipicDatabase.subjects.keys():
        Subject = CipicDatabase.subjects[subject]
        # If the data has all anthropometric measurements
        if (Subject._anthroDataIsComplete):

            # Get data for every orientation
            for i in range(0, 1250):
                RightEarDataset.append(Sample(Subject, i, "Right"))

    AvailableSampleIndices = set(range(0, len(RightEarDataset)))
    TrainingSetIndices = set()
    ValidationSetIndices = set()
    for _ in range(0, args.training_samples):
        choice = random.choice(tuple(AvailableSampleIndices))
        TrainingSetIndices.add(choice)
        AvailableSampleIndices.remove(choice)
    for _ in range(0, args.validation_samples):
        choice = random.choice(tuple(AvailableSampleIndices))
        ValidationSetIndices.add(choice)
        AvailableSampleIndices.remove(choice)
    # Ensure mutual exclusitivity
    assert(len(TrainingSetIndices.intersection(ValidationSetIndices)) == 0)

    train_set = CipicHRTFs(RightEarDataset, TrainingSetIndices)
    
    train_loader = DataLoader(train_set,
                          batch_size=args.b,
                          shuffle=True,
                          collate_fn=train_set.collate_fn,
                          num_workers=4,
                          pin_memory=True)

    startingEpoch = 0
    trackingInfo = dict()
    if args.useCheckpoint != "":
        if args.useCipic:
            run_warm_up_training_cipic(args, net, optimizer, scheduler, train_loader, orientList)
        else:
            run_warm_up_training(args, net, optimizer, scheduler, train_loader)
        checkpoint = torch.load(args.useCheckpoint, weights_only=True)
        module.load_state_dict(checkpoint['module_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        startingEpoch = checkpoint['epochs_completed']
        trackingInfo = checkpoint['tracking_info']
        print("Current Tracking info:")
        print("Epoch | Training Loss | Validation Loss")
        for i in range(0, startingEpoch+1):
            if i in trackingInfo.keys():
                tloss = trackingInfo[i]['training_loss']
                vLoss = trackingInfo[i]['validation_loss']
                checkpointStr  = str(i) + " | "
                checkpointStr += str(tloss) + " | "
                checkpointStr += str(vLoss) 
                print(checkpointStr)
        startingTrainingLoss = trackingInfo[startingEpoch]['training_loss']
        startingValidationLoss = trackingInfo[startingEpoch]['validation_loss']
        statusString  = "Resuming from checkpoint [epochs_completed:" 
        statusString += str(startingEpoch) + ", training loss=" 
        statusString += str(startingTrainingLoss) + ", validation loss="
        statusString += str(startingValidationLoss) + "]"
        print(statusString)

    lastTrainingLoss = run_training_loop(args, net, optimizer, scheduler, train_loader, startingEpoch)


    print("Completed training loop [epochs_completed:" + str(args.epochs) + ", training loss=" + str(lastTrainingLoss) + "]")

    validation_set = CipicHRTFs(RightEarDataset, ValidationSetIndices)
    
    validation_loader = DataLoader(validation_set,
                               batch_size=args.b,
                               shuffle=True,
                               collate_fn=validation_set.collate_fn,
                               num_workers=4,
                               pin_memory=True)
    finalValidationLoss = run_validation_loop(args, net, validation_loader)
    statusString  = "Completed training and validation [epochs_completed:" 
    statusString += str(startingEpoch+args.epochs) + ", training loss=" 
    statusString += str(lastTrainingLoss) + ", validation loss="
    statusString += str(finalValidationLoss) + "]"
    print(statusString)
    if (args.saveCheckpoint):
        trackingInfo[startingEpoch+args.epochs] = dict()
        currEpochStats = trackingInfo[startingEpoch+args.epochs]
        currEpochStats['training_loss'] = lastTrainingLoss
        currEpochStats['validation_loss'] = finalValidationLoss
        torch.save({
                'epochs_completed': startingEpoch + args.epochs,
                'module_state_dict': module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'tracking_info': trackingInfo,
                }, trained_folder + '/network.pt')
    print("Final validation loss: " + str(finalValidationLoss))
