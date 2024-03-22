from TwoLayerXOR.train import XORNet
import torch
import hashlib
import numpy as np

# load xor_net_model.pth

def load_model(model_path):
    model = XORNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_xor(input_data):
    model = load_model('./model/xor_net_model.pth')
    with torch.no_grad():
        inputs = torch.tensor(input_data, dtype=torch.float)
        output = model(inputs)
        return output.numpy()
    
def hash_inputs(inputs):
    # Convert inputs to a string and encode to bytes
    input_str = str(inputs)
    encoded_input = input_str.encode('utf-8')

    # Create a hash of the encoded inputs
    input_hash = hashlib.sha256(encoded_input).hexdigest()
    return input_hash

def predict_and_hash(input_data):
    # Predict XOR output using the neural network
    prediction = predict_xor(input_data)

    # Compute hash of the inputs
    input_hash = hash_inputs(input_data)

    # Convert prediction to boolean (True for 1, False for 0) and round to nearest integer
    prediction_bool = bool(np.round(prediction))

    return input_hash, prediction_bool