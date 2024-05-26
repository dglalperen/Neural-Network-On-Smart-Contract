import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TwoLayerXORNetRelu(nn.Module):
    def __init__(self):
        super(TwoLayerXORNetRelu, self).__init__()
        # Define network layers
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)  # No activation on the output layer for now
        return x
