import numpy as np
import torch
from TwoLayerXOR.TwoLayerXORNet import TwoLayerXORNet
from ThreeLayerXOR.ThreeLayerXORNet import ThreeLayerXORNet
import re

def extract_layer_index(name):
    # This function attempts to extract the layer index from the parameter name using a regular expression.
    match = re.search(r'layer(\d+)', name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract layer index from parameter name: {name}")

def format_parameters(param, scaling_factor):
    # Format the PyTorch parameter for RIDE, scaling and converting to integer
    param = param.detach().numpy()  # Assuming param is a tensor
    scaled_param = (param * scaling_factor).astype(int)
    formatted = ', '.join(str(x) for x in scaled_param.flatten())
    return formatted

def generate_forward_pass_function(layer_num, num_neurons, is_output_layer):
    if is_output_layer:
        input_weights = " + ".join([f"fraction(input[{i}], weights[0], 1000000)" for i in range(num_neurons)])  # Adjusted weights indexing for output layer
        return f"""
        func forwardPassLayer{layer_num}(input: List[Int], weights: List[Int], bias: Int) = {{
            let dotProduct = {input_weights}
            let sum = dotProduct + bias
            sigmoid(sum)
        }}
        """
    else:
        sums = "\n    ".join([f"let sum{i} = " + " + ".join([f"fraction(input[{j}], weights[{i}][{j}], 1000000)" for j in range(num_neurons)]) + f" + biases[{i}]" for i in range(num_neurons)])
        sigs = "\n    ".join([f"let sig{i} = sigmoid(sum{i})" for i in range(num_neurons)])
        outputs = "[" + ", ".join([f"sig{i}" for i in range(num_neurons)]) + "]"
        
        return f"""
        func forwardPassLayer{layer_num}(input: List[Int], weights: List[List[Int]], biases: List[Int]) = {{
            {sums}
            {sigs}
            {outputs}
        }}
        """
        
def generate_layer_functions(num_hidden_layers, neurons_per_hidden_layer):
    layer_functions = ""
    # Generate functions for hidden layers
    for i in range(1, num_hidden_layers + 1):
        layer_functions += generate_forward_pass_function(i, neurons_per_hidden_layer[i-1], False)
    # Generate function for the output layer, assuming output from the last hidden layer
    layer_functions += generate_forward_pass_function(num_hidden_layers + 1, neurons_per_hidden_layer[-1], True)
    return layer_functions

def generate_predict_function(layers_info):
    predict_function = "@Callable(i)\nfunc predict(input1: Int, input2: Int) = {\n"
    predict_function += "    let scaledInput1 = if(input1 == 1) then 1000000 else 0\n"
    predict_function += "    let scaledInput2 = if(input2 == 1) then 1000000 else 0\n"
    predict_function += "    let inputs = [scaledInput1, scaledInput2]\n"

    for i, layer in enumerate(layers_info[:-1]):  # Iterate through layers except the output layer
        if i == 0:  # Directly use inputs for the first layer
            predict_function += f"    let layer{i+1}Output = forwardPassLayer{i+1}(inputs, layer{i+1}Weights, layer{i+1}Biases)\n"
        else:  # Use output from the previous layer for subsequent layers
            predict_function += f"    let layer{i+1}Output = forwardPassLayer{i+1}(layer{i}Output, layer{i+1}Weights, layer{i+1}Biases)\n"

    # Handle the output layer
    last_layer_num = len(layers_info)
    predict_function += f"    let output = forwardPassLayer{last_layer_num}(layer{last_layer_num-1}Output, layer{last_layer_num}Weights, layer{last_layer_num}Biases)\n"

    # Construct the final output list
    predict_function += "    [\n"
    predict_function += "        IntegerEntry(\"result\", output)\n"
    predict_function += "    ]\n"
    predict_function += "}"

    return predict_function

def pytorch_to_waves_contract(model, scaling_factor=1000000):
    weight_bias_declarations = ""
    layers_info = []  # To store info about each layer

    for name, param in model.named_parameters():
        formatted_name = name.replace('.', '_')
        if 'weight' in name:
            layer_idx = extract_layer_index(name) - 1
            while len(layers_info) <= layer_idx:
                layers_info.append({'weights': None, 'biases': None, 'neurons': 0})
            layers_info[layer_idx]['weights'] = format_parameters(param, scaling_factor)
            layers_info[layer_idx]['neurons'] = param.size(0)  # Number of output neurons in the layer
        elif 'bias' in name:
            layer_idx = extract_layer_index(name) - 1
            layers_info[layer_idx]['biases'] = format_parameters(param, scaling_factor)

    # Generating weight and bias declarations
    for i, layer in enumerate(layers_info):
        weight_bias_declarations += f"let layer{i+1}Weights = [{layer['weights']}]\n    "
        weight_bias_declarations += f"let layer{i+1}Biases = [{layer['biases']}]\n    "

    num_hidden_layers = len(layers_info) - 1  # Assuming the last layer is output
    neurons_per_hidden_layer = [layer['neurons'] for layer in layers_info[:-1]]  # Excluding output layer

    forward_pass_functions = generate_layer_functions(num_hidden_layers, neurons_per_hidden_layer)

    predict_function = generate_predict_function(layers_info)

    # Construct the contract template
    contract_template = f"""
    {{-# STDLIB_VERSION 5 #-}}
    {{-# CONTENT_TYPE DAPP #-}}
    {{-# SCRIPT_TYPE ACCOUNT #-}}
    
    {weight_bias_declarations}
    
    func sigmoid(z: Int) = {{
        let e = 2718281
        let base = 1000000
        let positiveZ = if (z < 0) then -z else z
        let expPart = fraction(e, base, positiveZ)
        fraction(base, base, base + expPart)
    }}

    {forward_pass_functions}

    {predict_function}
    """
    return contract_template

if __name__ == "__main__":
    # Load and convert models as an example
    xor_net = TwoLayerXORNet()
    model_path = './TwoLayerXOR/two_layer_xor_net.pth'
    xor_net_state = torch.load(model_path, map_location=torch.device('cpu'))
    xor_net.load_state_dict(xor_net_state)

    # Convert the XORNet to a Waves smart contract
    contract = pytorch_to_waves_contract(xor_net)
    print("2 Layer XOR Net Smart Contract:")
    print(contract)
    
    # Three Layer XOR Net
    three_layer_xor_net = ThreeLayerXORNet()
    model_path = './ThreeLayerXOR/three_layer_xor_net.pth'
    three_layer_xor_net_state = torch.load(model_path, map_location=torch.device('cpu'))
    three_layer_xor_net.load_state_dict(three_layer_xor_net_state)
    contract = pytorch_to_waves_contract(three_layer_xor_net)
    print("3 Layer XOR Net Smart Contract:")
    print(contract)