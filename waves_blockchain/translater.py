import numpy as np
import torch
from TwoLayerXOR.TwoLayerXORNet import TwoLayerXORNet
from ThreeLayerXOR.ThreeLayerXORNet import ThreeLayerXORNet
import re

def save_contract_to_file(contract_content, file_name):
    with open(f"./generated_smart_contracts/{file_name}", 'w') as file:
        file.write(contract_content)
        
def extract_layer_index(name):
    # This function attempts to extract the layer index from the parameter name using a regular expression.
    match = re.search(r'layer(\d+)', name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract layer index from parameter name: {name}")

def flatten(list_of_lists):
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]

def format_parameters(param, scaling_factor, is_bias=False):
    param = param.detach().numpy()  # Assuming param is a tensor
    if is_bias:
        # Biases are still formatted as a single list
        scaled_param = (param * scaling_factor).astype(int)
        formatted = ', '.join(str(x) for x in scaled_param.flatten())
    else:
        # Weights need to be formatted as a list of lists
        scaled_param = (param * scaling_factor).astype(int)
        formatted = '[' + '], ['.join(', '.join(str(x) for x in row) for row in scaled_param) + ']'
    return formatted

def generate_sigmoid_function():
    return '''
    func sigmoid(z: Int, debugPrefix: String) = {
        let e = 2718281  # e scaled by 1,000,000
        let base = 1000000
        let positiveZ = if (z < 0) then -z else z
        let expPart = fraction(e, base, positiveZ)
        let sigValue = fraction(base, base, base + expPart)
        (
            [IntegerEntry(debugPrefix + "positiveZ", positiveZ), 
             IntegerEntry(debugPrefix + "expPart", expPart),
             IntegerEntry(debugPrefix + "sigValue", sigValue)],
            sigValue
        )
    }
    '''

def generate_forward_pass_function(layer_num, num_neurons, is_output_layer, include_debug=False):
    function_lines = []
    sums = []
    sigs = []
    debug_entries = []

    if is_output_layer:
        # Assuming a single output neuron for simplicity; adjust if your model differs
        input_weights = " + ".join([f"fraction(input[{i}], weights[0], 1000000)" for i in range(num_neurons)])
        sums.append(f"let sum = {input_weights} + biases")
        sigs.append("let (debug, sig) = sigmoid(sum, debugPrefix)")
        debug_entries.append("debug")
        output = "sig"
    else:
        for i in range(num_neurons):
            sums.append(f"let sum{i} = " + " + ".join([f"fraction(input[{j}], weights[{i}][{j}], 1000000)" for j in range(num_neurons)]) + f" + biases[{i}]")
            sigs.append(f'let (debug{i}, sig{i}) = sigmoid(sum{i}, debugPrefix + "L{layer_num}N{i}")')
            debug_entries.append(f"debug{i}")
        output = "[" + ", ".join([f"sig{i}" for i in range(num_neurons)]) + "]"

    # Combine all parts
    function_lines.extend(sums)
    function_lines.extend(sigs)
    function_body = "\n    ".join(function_lines)
    debug_info = " ++ ".join(debug_entries)

    # Generate the complete function definition
    function_definition = f'''
    func forwardPassLayer{layer_num}(input: List[Int], weights: {'List[Int]' if is_output_layer else 'List[List[Int]]'}, biases: {'Int' if is_output_layer else 'List[Int]'}, debugPrefix: String) = {{
        {function_body}
        ({output}, {debug_info})
    }}
    '''
    return function_definition
       
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
        debug_prefix = f'"Layer{i+1}"'
        if i == 0:  # Directly use inputs for the first layer
            predict_function += f"    let (layer{i+1}Output, debugLayer{i+1}) = forwardPassLayer{i+1}(inputs, layer{i+1}Weights, layer{i+1}Biases, {debug_prefix})\n"
        else:  # Use output from the previous layer for subsequent layers
            predict_function += f"    let (layer{i+1}Output, debugLayer{i+1}) = forwardPassLayer{i+1}(layer{i}Output, layer{i+1}Weights, layer{i+1}Biases, {debug_prefix})\n"

    # Adjusting for the output layer
    last_layer_num = len(layers_info)
    output_layer_weights = layers_info[-1]['weights']
    if isinstance(output_layer_weights[0], list):  # Check if the weights are in a list of lists
        output_layer_weights = flatten(output_layer_weights)  # Flatten the list of lists to a single list

    output_layer_biases = layers_info[-1]['biases']
    if isinstance(output_layer_biases, list) and len(output_layer_biases) == 1:
        output_layer_biases = output_layer_biases[0]  # Extract the single integer value

    output_layer_index = f'"Layer{last_layer_num}"'
    predict_function += f"    let (output, debugLayerLast) = forwardPassLayer{last_layer_num}(layer{last_layer_num-1}Output, {output_layer_weights}, {output_layer_biases}, {output_layer_index})\n"

    # Construct the final output list including debug entries, correctly formatted
    predict_function += "    [\n"
    predict_function += "        IntegerEntry(\"result\", output)\n"
    predict_function += "    ]\n"
    predict_function += "}"

    return predict_function

def pytorch_to_waves_contract(model, scaling_factor=1000000):
    weight_bias_declarations = ""
    layers_info = []  # To store info about each layer

    for name, param in model.items():
        formatted_name = name.replace('.', '_')
        if 'weight' in name:
            layer_idx = extract_layer_index(name) - 1
            while len(layers_info) <= layer_idx:
                layers_info.append({'weights': None, 'biases': None, 'neurons': 0})
            layers_info[layer_idx]['weights'] = format_parameters(param, scaling_factor, is_bias=False)
            layers_info[layer_idx]['neurons'] = param.size(0)  # Number of output neurons in the layer
        elif 'bias' in name:
            layer_idx = extract_layer_index(name) - 1
            layers_info[layer_idx]['biases'] = format_parameters(param, scaling_factor, is_bias=True)

    # Generating weight and bias declarations
    for i, layer in enumerate(layers_info):
        weight_bias_declarations += f"let layer{i+1}Weights = [{layer['weights']}]\n    "
        weight_bias_declarations += f"let layer{i+1}Biases = [{layer['biases']}]\n    "

    num_hidden_layers = len(layers_info) - 1  # Assuming the last layer is output
    neurons_per_hidden_layer = [layer['neurons'] for layer in layers_info[:-1]]  # Excluding output layer

    forward_pass_functions = generate_layer_functions(num_hidden_layers, neurons_per_hidden_layer)

    predict_function = generate_predict_function(layers_info)
    
    sigmoid_function = generate_sigmoid_function()

    # Construct the contract template
    contract_template = f"""
    {{-# STDLIB_VERSION 5 #-}}
    {{-# CONTENT_TYPE DAPP #-}}
    {{-# SCRIPT_TYPE ACCOUNT #-}}
    
    {weight_bias_declarations}
    
    {sigmoid_function}

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
    #print("2 Layer XOR Net Smart Contract:")
    #print(contract)
    save_contract_to_file(contract, 'TwoLayerXORNet.ride')
    
    # Three Layer XOR Net
    three_layer_xor_net = ThreeLayerXORNet()
    model_path = './ThreeLayerXOR/three_layer_xor_net.pth'
    three_layer_xor_net_state = torch.load(model_path, map_location=torch.device('cpu'))
    three_layer_xor_net.load_state_dict(three_layer_xor_net_state)
    contract = pytorch_to_waves_contract(three_layer_xor_net)
    # print("3 Layer XOR Net Smart Contract:")
    # print(contract)
    save_contract_to_file(contract, 'ThreeLayerXORNet.ride')