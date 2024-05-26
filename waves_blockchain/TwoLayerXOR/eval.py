import torch
from TwoLayerXORNet import TwoLayerXORNet

# Initialize the model and load trained weights
model = TwoLayerXORNet()
model_path = "../drag_torch_model/two_layer_xor_net.pth"  # Corrected path if necessary
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

# Ensure the model is in evaluation mode
model.eval()

# Define the XOR input and output
XOR_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Run model predictions
with torch.no_grad():
    predictions = model(XOR_inputs)

# Compare predictions with expected outputs
print("XOR input combinations vs. Predictions:")
for input_val, prediction, expected in zip(XOR_inputs, predictions, XOR_outputs):
    print(
        f"{input_val.tolist()} => Predicted: {prediction.item():.4f}, Expected: {expected.item()}"
    )
