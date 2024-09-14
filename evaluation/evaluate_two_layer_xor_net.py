import torch
import pandas as pd
import os
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


def evaluate_model(model, test_inputs):
    test_tensor = torch.tensor(test_inputs, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(test_tensor)
    return outputs


def save_results(inputs, outputs, file_name="two_layer_xor_net_results.csv"):
    df = pd.DataFrame(inputs, columns=["Input 1", "Input 2"])
    df["Output"] = outputs.numpy().flatten()
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", file_name), index=False)


if __name__ == "__main__":
    # Load the model
    model = TwoLayerXORNet()
    model.load_state_dict(torch.load("pth/two_layer_xor_net.pth"))
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
