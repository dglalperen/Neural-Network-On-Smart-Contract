import numpy as np
import torch
import random
import re
from TwoLayerXOR.TwoLayerXORNet import TwoLayerXORNet
from ThreeLayerXOR.ThreeLayerXORNet import ThreeLayerXORNet


def save_contract_to_file(contract_content, file_name):
    with open(f"./generated_smart_contracts/{file_name}", "w") as file:
        file.write(contract_content)


def extract_layer_index(name):
    match = re.search(r"layer(\d+)", name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(
            "Could not extract layer index from parameter name: {}".format(name)
        )


def stochastic_rounding(number):
    floor_value = np.floor(number)
    return (
        int(floor_value + 1)
        if np.random.rand() < (number - floor_value)
        else int(floor_value)
    )


def quantize_parameters(param, scale_factor=100000, quantization_type="deterministic"):
    if quantization_type == "stochastic":
        vectorized_rounding = np.vectorize(stochastic_rounding)
        return vectorized_rounding(param * scale_factor).astype(int)
    return np.round(param * scale_factor).astype(int)


def format_parameters(
    param, scale_factor=100000, quantization_type="deterministic", is_bias=False
):
    param = param.detach().numpy()
    quantized_param = quantize_parameters(param, scale_factor, quantization_type)
    formatted = (
        ", ".join(str(x) for x in quantized_param.flatten())
        if is_bias
        else "[{}]".format(
            "], [".join(", ".join(str(x) for x in row) for row in quantized_param)
        )
    )
    return formatted


def generate_sigmoid_function():
    return """
func sigmoid(z: Int, debugPrefix: String) = {
    let e = 2718281  # e scaled by 1,000,000
    let base = 1000000
    let positiveZ = if (z < 0) then -z else z
    let expPart = fraction(e, base, positiveZ)
    let sigValue = fraction(base, base + expPart, base)
    ([IntegerEntry(debugPrefix + "positiveZ", positiveZ), IntegerEntry(debugPrefix + "expPart", expPart), IntegerEntry(debugPrefix + "sigValue", sigValue)], sigValue)
}
"""


def generate_forward_pass_function(
    layer_num, input_size, num_neurons, is_output_layer, include_debug=True
):
    function_lines = []
    sums = []
    sigs = []
    debug_infos = []

    bias_ref = "biases"
    weights_ref = "weights"

    for i in range(num_neurons):
        input_refs = [f"input[{j}]" for j in range(input_size)]
        weights_refs = [f"{weights_ref}[{i}][{j}]" for j in range(input_size)]

        sum_exp = " + ".join(
            f"fraction({inp}, {wgt}, 1000000)"
            for inp, wgt in zip(input_refs, weights_refs)
        )
        sum_exp += f" + {bias_ref}[{i}]"

        sums.append(f"let sum{i} = {sum_exp}")
        sigs.append(f'let (debug{i}, sig{i}) = sigmoid(sum{i}, "Layer{layer_num}N{i}")')
        debug_infos.append(f"debug{i}")

    function_body = "\n    ".join(sums + sigs)
    debug_concat = " ++ ".join(debug_infos)
    output = (
        f"sig0"
        if is_output_layer and num_neurons == 1
        else f"[{', '.join([f'sig{i}' for i in range(num_neurons)])}]"
    )

    function_definition = f"""
func forwardPassLayer{layer_num}(input: List[Int], {weights_ref}: List[List[Int]], {bias_ref}: List[Int], debugPrefix: String) = {{
    {function_body}
    ({output}, {debug_concat})
}}
"""
    return function_definition.strip()


def generate_layer_functions(layers_info):
    layer_functions = ""
    input_size = len(
        layers_info[0]["weights"].split("], [")[0].split(", ")
    )  # Determine input size from the first layer weights
    for i, layer in enumerate(layers_info):
        is_output_layer = i == len(layers_info) - 1
        num_neurons = layer["neurons"]
        layer_functions += generate_forward_pass_function(
            i + 1, input_size, num_neurons, is_output_layer
        )
        input_size = num_neurons  # Update input size to current layer's neuron count for the next layer
    return layer_functions


def generate_predict_function(layers_info):
    predict_function = "@Callable(i)\nfunc predict(input1: Int, input2: Int) = {\n"
    predict_function += "    let scaledInput1 = if(input1 == 1) then 1000000 else 0\n"
    predict_function += "    let scaledInput2 = if(input2 == 1) then 1000000 else 0\n"
    predict_function += "    let inputs = [scaledInput1, scaledInput2]\n"
    debug_all = []
    for i, layer in enumerate(layers_info):
        layer_num = i + 1
        prev_output = "inputs" if i == 0 else f"layer{i}Output"
        debug_prefix = f'"Layer{layer_num}"'
        predict_function += f"    let (layer{layer_num}Output, debugLayer{layer_num}) = forwardPassLayer{layer_num}({prev_output}, layer{layer_num}Weights, layer{layer_num}Biases, {debug_prefix})\n"
        debug_all.append(f"debugLayer{layer_num}")
    debug_concat = " ++ ".join(debug_all)
    predict_function += (
        f'    [\n        IntegerEntry("result", layer{len(layers_info)}Output)\n    ] ++ '
        + debug_concat
        + "\n}"
    )
    return predict_function


def pytorch_to_waves_contract(
    model, scaling_factor=100000, quantization_type="stochastic"
):
    weight_bias_declarations = ""
    layers_info = []

    for name, param in model.items():
        formatted_name = name.replace(".", "_")
        if "weight" in name:
            layer_idx = extract_layer_index(name) - 1
            while len(layers_info) <= layer_idx:
                layers_info.append({"weights": None, "biases": None, "neurons": 0})
            layers_info[layer_idx]["weights"] = format_parameters(
                param, scaling_factor, quantization_type, is_bias=False
            )
            layers_info[layer_idx]["neurons"] = param.size(0)
        elif "bias" in name:
            layer_idx = extract_layer_index(name) - 1
            layers_info[layer_idx]["biases"] = format_parameters(
                param, scaling_factor, quantization_type, is_bias=True
            )

    for i, layer in enumerate(layers_info):
        weight_bias_declarations += (
            f"let layer{i+1}Weights = [{layer['weights']}]\n    "
        )
        weight_bias_declarations += f"let layer{i+1}Biases = [{layer['biases']}]\n    "

    forward_pass_functions = generate_layer_functions(layers_info)
    predict_function = generate_predict_function(layers_info)
    sigmoid_function = generate_sigmoid_function()

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
    xor_net = TwoLayerXORNet()
    model_path = "./TwoLayerXOR/two_layer_xor_net.pth"
    xor_net_state = torch.load(model_path, map_location=torch.device("cpu"))
    xor_net.load_state_dict(xor_net_state)

    contract = pytorch_to_waves_contract(xor_net)
    save_contract_to_file(contract, "TwoLayerXORNet.ride")

    three_layer_xor_net = ThreeLayerXORNet()
    model_path = "./ThreeLayerXOR/three_layer_xor_net.pth"
    three_layer_xor_net_state = torch.load(model_path, map_location=torch.device("cpu"))
    three_layer_xor_net.load_state_dict(three_layer_xor_net_state)

    contract = pytorch_to_waves_contract(three_layer_xor_net)
    save_contract_to_file(contract, "ThreeLayerXORNet.ride")
