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

def generate_forward_pass_function(layer_num, num_neurons, is_output_layer, include_debug=True):
    # Initialize strings to hold function code
    sums = []
    sigs = []
    debug_infos = []

    # Generate sums and sigmoid applications with debugging for each neuron
    for i in range(num_neurons):
        input_refs = [f"input[{j}]" for j in range(num_neurons)] if not is_output_layer else ["input[0]", "input[1]"]
        weights_refs = [f"weights[{i}][{j}]" for j in range(num_neurons)] if not is_output_layer else [f"weights[0]", f"weights[1]"]
        sum_exp = " + ".join([f"fraction({inp}, {wgt}, 1000000)" for inp, wgt in zip(input_refs, weights_refs)]) + f" + biases[{i}]"
        
        sums.append(f"let sum{i} = {sum_exp}")
        sigs.append(f'let (debug{i}, sig{i}) = sigmoid(sum{i}, debugPrefix + "L{layer_num}N{i}")')
        debug_infos.append(f"debug{i}")

    # Combine all parts into a complete function definition
    function_body = "\n    ".join(sums + sigs)
    debug_concat = " ++ ".join(debug_infos)
    output = "[" + ", ".join([f"sig{i}" for i in range(num_neurons)]) + "]" if not is_output_layer else "sig0"

    # Final function definition
    function_definition = f'''
func forwardPassLayer{layer_num}(input: List[Int], weights: {'List[List[Int]]' if not is_output_layer else 'List[Int]'}, biases: {'List[Int]' if not is_output_layer else 'Int'}, debugPrefix: String) = {{
    {function_body}
    ({output}, {debug_concat})
}}
'''
    return function_definition.strip()
     
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
    
    debug_all = []

    # Iterate through each layer and generate forward pass calls
    for i, layer in enumerate(layers_info):
        layer_num = i + 1
        prev_output = "inputs" if i == 0 else f"layer{i}Output"
        debug_prefix = f'"Layer{layer_num}"'
        predict_function += f"    let (layer{layer_num}Output, debugLayer{layer_num}) = forwardPassLayer{layer_num}({prev_output}, layer{layer_num}Weights, layer{layer_num}Biases, {debug_prefix})\n"
        debug_all.append(f"debugLayer{layer_num}")

    # Reference the last layer's output directly
    last_layer_output = f"layer{len(layers_info)}Output"

    # Combining all debug information
    debug_concat = " ++ ".join(debug_all)
    predict_function += f"    [\n        IntegerEntry(\"result\", {last_layer_output})\n    ] ++ " + debug_concat + "\n}"
    
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