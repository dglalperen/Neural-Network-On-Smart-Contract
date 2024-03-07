import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

layer1Weights = np.array([[4.051769, 4.062273], [-5.9485154, -6.0100846]])
layer1Biases = np.array([-6.3078423, 2.2298715])
layer2Weights = np.array([[-8.372358, -8.139317]])
layer2Biases = np.array([4.0836797])

# For input '1', we translate it to XOR input (0, 1)
input_data = np.array([0, 1])

hidden_layer_output = sigmoid(np.dot(input_data, layer1Weights.T) + layer1Biases)
output = sigmoid(np.dot(hidden_layer_output, layer2Weights.T) + layer2Biases)

print(output)
