import torch
from torch import nn


class TicTacNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(9, 36)
        self.layer2 = nn.Linear(36, 36)
        self.output_layer = nn.Linear(36, 9)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x
