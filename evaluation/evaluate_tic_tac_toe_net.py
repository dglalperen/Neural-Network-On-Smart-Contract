import torch
import pandas as pd
import os
import torch
from torch import nn


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


def evaluate_model(model, test_inputs):
    test_tensor = torch.tensor(test_inputs, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(test_tensor)
    return outputs


def save_results(inputs, outputs, file_name="tic_tac_toe_net_results.csv"):
    df = pd.DataFrame(inputs, columns=[f"Input {i+1}" for i in range(9)])
    for i in range(9):
        df[f"Output {i+1}"] = outputs[:, i].numpy()
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", file_name), index=False)


if __name__ == "__main__":
    # Load the model
    model = TicTacNet()
    model.load_state_dict(torch.load("pth/tic_tac_toe_net.pth"))
    model.eval()

    # Generate test inputs based on provided board states
    test_inputs = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [-1, -1, 1, 0, -1, -1, 1, 0, 1],
        [-1, -1, 1, 0, -1, 0, 1, 0, 0],
        [-1, -1, 1, 0, 1, 0, 0, 0, 0],
        [-1, -1, 1, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 1, 0, 0, 0, 0, -1],
    ]

    # Evaluate model
    outputs = evaluate_model(model, test_inputs)

    # Save results
    save_results(test_inputs, outputs)
