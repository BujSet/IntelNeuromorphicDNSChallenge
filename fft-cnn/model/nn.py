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