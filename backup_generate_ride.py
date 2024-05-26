import json

def generate_ride_script(json_file):
    with open(json_file, 'r') as f:
        model = json.load(f)
    
    architecture = model['architecture']
    parameters = model['parameters']
    
    weights = parameters['weights']
    biases = parameters['biases']
    
    layer_defs = []
    linear_funcs = []
    activation_funcs = []

    for i, layer in enumerate(architecture):
        input_features = layer['input_features']
        output_features = layer['output_features']
        weight_layer = weights[i]
        bias_layer = biases[i]
        
        # Define weights and biases
        weight_defs = f"let weights_layer_{i+1} = " + str([[int(w * 10000) for w in ws] for ws in weight_layer]).replace('[', '[[').replace(']', ']]').replace('], [[', '],\n[')
        bias_defs = f"let biases_layer_{i+1} = " + str([int(b * 10000) for b in bias_layer]).replace('[', '[').replace(']', ']')

        layer_defs.append(weight_defs)
        layer_defs.append(bias_defs)
        
        # Define linear forward function
        linear_sums_parts = []
        for j in range(output_features):
            parts = [f"input[{k}] * weights_layer_{i+1}[{j}][{k}]" for k in range(input_features)]
            linear_sums_parts.append(f"let weighted_sum{j+1} = (" + " + ".join(parts) + f") / 10000 + biases_layer_{i+1}[{j}]")
        linear_sums = "\n    ".join(linear_sums_parts)
        
        linear_func = (
            f"# Linear forward function for layer {i+1}\n"
            f"func linear_forward_layer_{i+1}(input: List[Int]) = {{\n"
            f"    {linear_sums}\n"
            f"    [{', '.join([f'weighted_sum{j+1}' for j in range(output_features)])}]\n"
            f"}}"
        )
        linear_funcs.append(linear_func)
        
        # Define activation function
        if i < len(architecture) - 1:
            activation_func = (
                f"# ReLU activation function for layer {i+1}\n"
                f"func relu_layer_{i+1}(input: List[Int]) = {{\n"
                f"    {'    '.join([f'let out{j+1} = if (input[{j}] > 0) then input[{j}] else 0\n' for j in range(output_features)])}\n"
                f"    [{', '.join([f'out{j+1}' for j in range(output_features)])}]\n"
                f"}}"
            )
        else:
            activation_func = (
                f"# Sigmoid activation function for layer {i+1}\n"
                f"func sigmoid_layer_{i+1}(input: Int) = {{\n"
                f"    if (input < -10000) then 0\n"
                f"    else if (input > 10000) then 10000\n"
                f"    else 5000 + input / 2\n"
                f"}}"
            )
        activation_funcs.append(activation_func)

    # Define the predict function
    input_scaling_parts = [f"let x{j+1}_scaled = x{j+1} * 10000" for j in range(architecture[0]['input_features'])]
    input_scaling = "\n    ".join(input_scaling_parts)
    inputs_list = ", ".join([f"x{j+1}_scaled" for j in range(architecture[0]['input_features'])])

    predict_func_parts = [
        "@Callable(i)",
        f"func predict({', '.join([f'x{j+1}: Int' for j in range(architecture[0]['input_features'])])}) = {{",
        "    # Scale inputs",
        f"    {input_scaling}",
        f"    let inputs = [{inputs_list}]"
    ]
    
    for i in range(len(architecture)):
        if i < len(architecture) - 1:
            predict_func_parts.extend([
                f"    let z{i+1} = linear_forward_layer_{i+1}(inputs)",
                f"    let a{i+1} = relu_layer_{i+1}(z{i+1})"
            ])
        else:
            predict_func_parts.extend([
                f"    let z{i+1} = linear_forward_layer_{i+1}(a{i})",
                f"    let a{i+1} = sigmoid_layer_{i+1}(z{i+1}[0])"
            ])
    
    predict_func_parts.extend([
        "    # Scaling back the output",
        f"    let result = a{len(architecture)} / 10000",
        "    # Debug outputs",
        "    let debug_outputs = ["
    ])
    
    debug_outputs_parts = []
    for i in range(len(architecture)):
        for j in range(architecture[i]['output_features']):
            debug_outputs_parts.append(f"        IntegerEntry('debug_z{i+1}_{j+1}', z{i+1}[{j}]), ")
            if i < len(architecture) - 1:
                debug_outputs_parts.append(f"        IntegerEntry('debug_a{i+1}_{j+1}', a{i+1}[{j}]), ")
            else:
                debug_outputs_parts.append(f"        IntegerEntry('debug_a{i+1}', a{i+1}), ")
    debug_outputs_parts.append("        IntegerEntry('debug_result', result)")
    
    predict_func_parts.extend(debug_outputs_parts)
    predict_func_parts.extend([
        "    ]",
        "    (",
        "        debug_outputs,",
        "        result",
        "    )",
        "}"
    ])
    
    predict_func = "\n".join(predict_func_parts)
    
    # Combine all parts
    ride_script = (
        "{-# STDLIB_VERSION 7 #-}\n"
        "{-# CONTENT_TYPE DAPP #-}\n"
        "{-# SCRIPT_TYPE ACCOUNT #-}\n\n"
        "# Weights and Biases\n"
        + "\n".join(layer_defs) + "\n\n"
        "# Linear Forward Functions\n"
        + "\n".join(linear_funcs) + "\n\n"
        "# Activation Functions\n"
        + "\n".join(activation_funcs) + "\n\n"
        "# Predict Function\n"
        + predict_func
    )
    
    return ride_script