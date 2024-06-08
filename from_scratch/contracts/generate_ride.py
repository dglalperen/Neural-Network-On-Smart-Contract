import json


def generate_weights_and_biases(architecture, parameters):
    weights = parameters["weights"]
    biases = parameters["biases"]
    layer_defs = []

    for i, (layer, weight_layer, bias_layer) in enumerate(
        zip(architecture, weights, biases)
    ):
        weight_defs = f"let weights_layer_{i+1} = " + str(
            [[int(w * 10000) for w in ws] for ws in weight_layer]
        ).replace("], [", "],\n    [")
        bias_defs = f"let biases_layer_{i+1} = " + str(
            [int(b * 10000) for b in bias_layer]
        ).replace("[", "[ ").replace("]", " ]")

        layer_defs.append(weight_defs)
        layer_defs.append(bias_defs)

    return "\n".join(layer_defs)


def generate_linear_forward_functions(architecture):
    linear_funcs = []

    def linear_func_def(i, input_size, output_size):
        weighted_sums = [
            f"let weighted_sum{j+1} = ("
            + " + ".join([f"input[{k}] * weights[{j}][{k}]" for k in range(input_size)])
            + f") / 10000 + biases[{j}]"
            for j in range(output_size)
        ]
        return (
            f"func linear_forward_{i}(input: List[Int], weights: List[List[Int]], biases: List[Int]) = {{\n"
            f"    " + "\n    ".join(weighted_sums) + "\n"
            f"    ["
            + ", ".join([f"weighted_sum{j+1}" for j in range(output_size)])
            + "]\n"
            f"}}"
        )

    for i, layer in enumerate(architecture):
        input_features = layer["input_features"]
        output_features = layer["output_features"]
        linear_funcs.append(linear_func_def(i + 1, input_features, output_features))

    return "\n\n".join(linear_funcs)


def generate_activation_functions(architecture):
    activation_funcs = {}

    for layer in architecture:
        activation = layer.get("activation")
        if activation and activation not in activation_funcs:
            if activation == "sigmoid":
                activation_funcs[activation] = (
                    f"# Approximate Sigmoid function using a piecewise linear function\n"
                    f"func sigmoid(input: Int) = {{\n"
                    f"    if (input < -80000) then 0\n"
                    f"    else if (input < -60000) then fraction(input + 80000, 125, 10000)\n"
                    f"    else if (input < -40000) then fraction(input + 60000, 100, 10000)\n"
                    f"    else if (input < -20000) then fraction(input + 40000, 75, 10000)\n"
                    f"    else if (input < 0) then fraction(input + 20000, 50, 10000)\n"
                    f"    else if (input < 20000) then fraction(input, 50, 10000) + 5000\n"
                    f"    else if (input < 40000) then fraction(input - 20000, 75, 10000) + 7500\n"
                    f"    else if (input < 60000) then fraction(input - 40000, 100, 10000) + 8750\n"
                    f"    else if (input < 80000) then fraction(input - 60000, 125, 10000) + 9375\n"
                    f"    else 10000\n"
                    f"}}\n"
                    f"# Sigmoid activation function for a list of values\n"
                    f"func sigmoid_activation(inputs: List[Int], num_outputs: Int) = {{\n"
                    f"    ["
                    + ", ".join(
                        [
                            f"sigmoid(inputs[{j}])"
                            for j in range(layer["output_features"])
                        ]
                    )
                    + "]\n"
                    f"}}"
                )
            elif activation == "relu":
                activation_funcs[activation] = (
                    f"# ReLU function\n"
                    f"func relu(input: Int) = {{\n"
                    f"    if (input < 0) then 0\n"
                    f"    else input\n"
                    f"}}\n"
                    f"# ReLU activation function for a list of values\n"
                    f"func relu_activation(inputs: List[Int], num_outputs: Int) = {{\n"
                    f"    ["
                    + ", ".join(
                        [f"relu(inputs[{j}])" for j in range(layer["output_features"])]
                    )
                    + "]\n"
                    f"}}"
                )
            # Add other activation functions as needed

    return "\n\n".join(activation_funcs.values())


def generate_predict_function(architecture, debug_mode=True):
    input_scaling_parts = [
        f"let x{j+1}_scaled = x{j+1} * 10000"
        for j in range(architecture[0]["input_features"])
    ]
    input_scaling = "\n    ".join(input_scaling_parts)
    inputs_list = ", ".join(
        [f"x{j+1}_scaled" for j in range(architecture[0]["input_features"])]
    )

    predict_func_parts = [
        "@Callable(i)",
        f"func predict({', '.join([f'x{j+1}: Int' for j in range(architecture[0]['input_features'])])}) = {{",
        "    # Scale inputs",
        f"    {input_scaling}",
        f"    let inputs = [{inputs_list}]",
    ]

    for i in range(len(architecture)):
        if i == 0:
            predict_func_parts.extend(
                [
                    f"    let z{i+1} = linear_forward_{i+1}(inputs, weights_layer_{i+1}, biases_layer_{i+1})"
                ]
            )
            if architecture[i]["activation"]:
                predict_func_parts.append(
                    f"    let a{i+1} = {architecture[i]['activation']}_activation(z{i+1}, {architecture[i]['output_features']})"
                )
        elif i < len(architecture) - 1:
            predict_func_parts.extend(
                [
                    f"    let z{i+1} = linear_forward_{i+1}(a{i}, weights_layer_{i+1}, biases_layer_{i+1})"
                ]
            )
            if architecture[i]["activation"]:
                predict_func_parts.append(
                    f"    let a{i+1} = {architecture[i]['activation']}_activation(z{i+1}, {architecture[i]['output_features']})"
                )
        else:
            predict_func_parts.extend(
                [
                    f"    let z{i+1} = linear_forward_{i+1}(a{i}, weights_layer_{i+1}, biases_layer_{i+1})"
                ]
            )
            if architecture[i]["activation"]:
                predict_func_parts.append(
                    f"    let a{i+1} = {architecture[i]['activation']}_activation(z{i+1}, {architecture[i]['output_features']})"
                )
            else:
                predict_func_parts.append(f"    let a{i+1} = z{i+1}")

    # Scaling back the output
    output_scaling = [
        f"let result{j} = a{len(architecture)}[{j}]"
        for j in range(architecture[-1]["output_features"])
    ]
    output_scaling_str = "\n    ".join(output_scaling)

    predict_func_parts.append("    # Scaling back the output")
    predict_func_parts.append(f"    {output_scaling_str}")

    if debug_mode:
        predict_func_parts.append("    # Debug outputs")
        predict_func_parts.append("    let debug_outputs = []")

        predict_func_parts.append("    [")
        for j in range(architecture[-1]["output_features"]):
            if j == architecture[-1]["output_features"] - 1:
                predict_func_parts.append(
                    f'        IntegerEntry("move_prediction_{j}", result{j})'
                )
            else:
                predict_func_parts.append(
                    f'        IntegerEntry("move_prediction_{j}", result{j}),'
                )
        predict_func_parts.append("    ] ++ debug_outputs")

    else:
        predict_func_parts.append("    let debug_outputs = []")
        predict_func_parts.append("    [")
        for j in range(architecture[-1]["output_features"]):
            if j == architecture[-1]["output_features"] - 1:
                predict_func_parts.append(
                    f'        IntegerEntry("move_prediction_{j}", result{j})'
                )
            else:
                predict_func_parts.append(
                    f'        IntegerEntry("move_prediction_{j}", result{j}),'
                )
        predict_func_parts.append("    ] ++ debug_outputs")

    predict_func_parts.append("}")

    return "\n".join(predict_func_parts)


def generate_ride_script(json_file, debug_mode=True):
    with open(json_file, "r") as f:
        model = json.load(f)

    architecture = model["architecture"]
    parameters = model["parameters"]

    weights_and_biases = generate_weights_and_biases(architecture, parameters)
    linear_forward_functions = generate_linear_forward_functions(architecture)
    activation_functions = generate_activation_functions(architecture)
    predict_function = generate_predict_function(architecture, debug_mode)

    # Combine all parts
    ride_script = (
        "{-# STDLIB_VERSION 7 #-}\n"
        "{-# CONTENT_TYPE DAPP #-}\n"
        "{-# SCRIPT_TYPE ACCOUNT #-}\n\n"
        "# Weights and Biases\n" + weights_and_biases + "\n\n"
        "# Linear Forward Functions\n" + linear_forward_functions + "\n\n"
        "# Activation Functions\n" + activation_functions + "\n\n"
        "# Predict Function\n" + predict_function
    )

    return ride_script
