import numpy as np


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Weights and biases
layer1_weights = np.array([[600497, 600733], [414197, 414253]])
layer1_biases = np.array([-259050, -635638])
layer2_weights = np.array([832966, -897142])
layer2_biases = np.array([-381179])

# Input sets
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Calculate outputs for each input set
for input_pair in inputs:
    input_vector = np.array(input_pair)

    # Layer 1 calculations
    layer1_z = np.dot(layer1_weights, input_vector) + layer1_biases
    layer1_output = sigmoid(layer1_z)

    # Layer 2 calculations
    layer2_z = np.dot(layer2_weights, layer1_output) + layer2_biases
    layer2_output = sigmoid(layer2_z)

    # Print results
    print(
        f"Input: {input_pair} -> Layer 1 Output: {layer1_output}, Layer 2 Output: {layer2_output}"
    )
