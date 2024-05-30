import torch
import torch.nn as nn


class TwoLayerXORNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayerXORNet, self).__init__()
        # Define network layers
        self.layer1 = torch.nn.Linear(2, 2)
        self.layer2 = torch.nn.Linear(2, 1)
        # Define activation functions for each layer
        self.activations = {"layer1": "sigmoid", "layer2": "sigmoid"}

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x
