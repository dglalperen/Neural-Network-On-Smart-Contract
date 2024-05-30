import torch
import torch.nn as nn
import torch.optim as optim


class ThreeLayerXORNet(nn.Module):
    def __init__(self):
        super(ThreeLayerXORNet, self).__init__()
        # Define three layers
        self.layer1 = nn.Linear(2, 4)  # Input layer
        self.layer2 = nn.Linear(4, 2)  # Hidden layer
        self.layer3 = nn.Linear(2, 1)  # Output layer
        self.activations = {
            "layer1": "sigmoid",
            "layer2": "sigmoid",
            "layer3": "sigmoid",
        }

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x
