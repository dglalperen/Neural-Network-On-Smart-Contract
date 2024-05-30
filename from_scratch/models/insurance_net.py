import torch.nn as nn
import torch.nn.functional as F


class InsuranceNet(nn.Module):
    def __init__(self):
        super(InsuranceNet, self).__init__()
        self.fc1 = nn.Linear(23, 15)
        self.fc2 = nn.Linear(15, 2)
        self.activations = {"fc1": "relu", "fc2": None}

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
