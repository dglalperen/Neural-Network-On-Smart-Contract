import torch
from ThreeLayerXORNet import ThreeLayerXORNet

# Initialize the model and load trained weights
model = ThreeLayerXORNet()
model_path = './three_layer_xor_net.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Ensure the model is in evaluation mode
model.eval()

# Define the XOR input and output
XOR_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
XOR_outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

# Run model predictions
with torch.no_grad():
    predictions = model(XOR_inputs)

# Compare predictions with expected outputs
print("XOR input combinations vs. Predictions:")
for i, (input_val, prediction) in enumerate(zip(XOR_inputs, predictions)):
    print(f"{input_val.tolist()} => {prediction.item():.4f} (Expected: {XOR_outputs[i].item()})")
