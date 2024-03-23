import torch

# Path to the saved model state
model_path = './three_layer_xor_net.pth'

# Load the saved model state
xor_net_state = torch.load(model_path, map_location=torch.device('cpu'))

print("State", xor_net_state)

# Extracting weights and biases
layer1_weights = xor_net_state['layer1.weight'].numpy()
layer1_biases = xor_net_state['layer1.bias'].numpy()
layer2_weights = xor_net_state['layer2.weight'].numpy()
layer2_biases = xor_net_state['layer2.bias'].numpy()
layer3_weights = xor_net_state['layer3.weight'].numpy()
layer3_biases = xor_net_state['layer3.bias'].numpy()

# print('Layer 1 weights:', layer1_weights)
# print('Layer 1 biases:', layer1_biases)
# print('Layer 2 weights:', layer2_weights)
# print('Layer 2 biases:', layer2_biases)
# print('Layer 3 weights:', layer3_weights)
# print('Layer 3 biases:', layer3_biases)
