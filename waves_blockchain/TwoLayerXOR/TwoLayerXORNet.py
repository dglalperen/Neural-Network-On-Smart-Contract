import torch
import torch.nn as nn

class TwoLayerXORNet(nn.Module):
    def __init__(self):
        super(TwoLayerXORNet, self).__init__()
        # Define network layers
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x