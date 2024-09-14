import torch
import pandas as pd
import os
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


def evaluate_model(model, test_inputs):
    test_tensor = torch.tensor(test_inputs, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(test_tensor)
    return outputs


def save_results(inputs, outputs, file_name="three_layer_xor_net_results.csv"):
    df = pd.DataFrame(inputs, columns=["Input 1", "Input 2"])
    df["Output"] = outputs.numpy().flatten()
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", file_name), index=False)


if __name__ == "__main__":
    # Load the model
    model = ThreeLayerXORNet()
    model.load_state_dict(torch.load("pth/three_layer_xor_net.pth"))
    model.eval()

    # Generate test inputs
    test_inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        # Add more test inputs as needed
    ]

    # Evaluate model
    outputs = evaluate_model(model, test_inputs)

    # Save results
    save_results(test_inputs, outputs)
