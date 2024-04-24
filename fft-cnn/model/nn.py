import torch.nn as nn
import torch.nn.functional as F

class PhaseShiftCNN(nn.Module):
    def __init__(self):
        super(PhaseShiftCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.phase_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        phase_shift = self.phase_conv(x)
        return phase_shift

"""
Start with puretones, have it learn to phase shift that
then switch to a suite of frequencies, then switch to frequencies mixed together

For a pure tone, one would expect to see no change. There should be one frequency component that has a magnitude
the phase should be the original phase, and the output should be 180 degree phase offset
from the original symbol. Phase shift it by pi, then when we do the additions itll cancel each other out. 


For non-stationary noise: 
Have a recurrent nn or unroll a bunch of aperiodic noise and train a NN on that
Want to detect onset of noises and predict where its going to go, then generate a response 

Unroll the FFTs and present the NN with 10-20 samples, should be able to capture the lower frequencies then
Need magnitude and phase to go through the network still
"""