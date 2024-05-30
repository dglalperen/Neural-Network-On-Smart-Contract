from torch import nn
import torch


def get_network_architecture(model):
    """
    Gets the architecture of the neural network.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        list: A list containing the details of each layer.
    """
    architecture = []

    # Convert activations list to a dictionary for quick lookup
    activations_dict = dict(model.activations)

    for layer_name, layer in model.named_children():
        if isinstance(layer, torch.nn.Linear):
            layer_info = {
                "type": "Linear",
                "input_features": layer.in_features,
                "output_features": layer.out_features,
                "activation": activations_dict.get(layer_name, None),
            }
            architecture.append(layer_info)

    return architecture


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


class TicTacNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dl1 = nn.Linear(9, 36)
        self.dl2 = nn.Linear(36, 36)
        self.output_layer = nn.Linear(36, 9)
        self.activations = [
            ("dl1", "relu"),
            ("dl2", "relu"),
            ("output_layer", "sigmoid"),
        ]

    def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)

        x = self.dl2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    model = TwoLayerXORNet()
    architecture = get_network_architecture(model)
    print(architecture)

    model = TicTacNet()
    architecture = get_network_architecture(model)
    print(architecture)
