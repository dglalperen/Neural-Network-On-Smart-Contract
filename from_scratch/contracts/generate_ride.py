import json


def generate_ride_script(json_file):
    with open(json_file, "r") as f:
        model = json.load(f)

    architecture = model["architecture"]
    parameters = model["parameters"]

    weights = parameters["weights"]
    biases = parameters["biases"]

    print("Architecture:", architecture)
    print("Weights:", weights)
    print("Biases:", biases)

    layer_defs = []
    linear_funcs = []
    activation_funcs = []

    for i, layer in enumerate(architecture):
        input_features = layer["input_features"]
        output_features = layer["output_features"]
        weight_layer = weights[i]
        bias_layer = biases[i]

        # Define weights and biases
        weight_defs = f"let weights_layer_{i+1} = " + str(
            [[int(w * 10000) for w in ws] for ws in weight_layer]
        ).replace("], [", "],\n    [")
        bias_defs = f"let biases_layer_{i+1} = " + str(
            [int(b * 10000) for b in bias_layer]
        ).replace("[", "[ ").replace("]", " ]")

        layer_defs.append(weight_defs)
        layer_defs.append(bias_defs)

    print("Layer definitions:", layer_defs)

    # General linear forward function
    linear_func = (
        f"# Linear forward function\n"
        f"func linear_forward(input: List[Int], weights: List[List[Int]], biases: List[Int]) = {{\n"
        f"    let weighted_sum1 = (input[0] * weights[0][0] + input[1] * weights[0][1]) / 10000 + biases[0]\n"
        f"    let weighted_sum2 = (input[0] * weights[1][0] + input[1] * weights[1][1]) / 10000 + biases[1]\n"
        f"    [weighted_sum1, weighted_sum2]\n"
        f"}}"
    )
    linear_funcs.append(linear_func)

    print("Linear functions:", linear_funcs)

    # Sigmoid activation function for a list of values
    # Sigmoid function definition
    activation_funcs.append(
        f"# Sigmoid function\n"
        f"func sigmoid(input: Int) = {{\n"
        f"    if (input < -10000) then 0\n"
        f"    else if (input > 10000) then 10000\n"
        f"    else 5000 + input / 2\n"
        f"}}"
    )

    activation_funcs.append(
        f"# Sigmoid activation function for a list of values\n"
        f"func sigmoid_activation(inputs: List[Int]) = {{\n"
        f"    [sigmoid(inputs[0]), sigmoid(inputs[1])]\n"
        f"}}"
    )

    print("Activation functions:", activation_funcs)

    # Define the predict function
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
                    f"    let z{i+1} = linear_forward(inputs, weights_layer_{i+1}, biases_layer_{i+1})",
                    f"    let a{i+1} = sigmoid_activation(z{i+1})",
                ]
            )
        else:
            if architecture[i]["output_features"] > 1:
                predict_func_parts.extend(
                    [
                        f"    let z{i+1} = linear_forward(a{i}, weights_layer_{i+1}, biases_layer_{i+1})",
                        f"    let a{i+1} = sigmoid_activation(z{i+1})",
                    ]
                )
            else:
                predict_func_parts.extend(
                    [
                        f"    let z{i+1} = (a{i}[0] * weights_layer_{i+1}[0][0] + a{i}[1] * weights_layer_{i+1}[0][1]) / 10000 + biases_layer_{i+1}[0]",
                        f"    let a{i+1} = sigmoid(z{i+1})",
                    ]
                )

    print("Predict function parts before debug outputs:", predict_func_parts)

    predict_func_parts.extend(
        [
            "    # Scaling back the output",
            f"    let result = a{len(architecture)} / 10000",
            "    # Debug outputs",
            "    let debug_outputs = [",
        ]
    )

    debug_outputs_parts = []
    for i in range(len(architecture)):
        if i < len(architecture) - 1:
            for j in range(architecture[i]["output_features"]):
                debug_outputs_parts.append(
                    f'        IntegerEntry("debug_z{i+1}_{j+1}", z{i+1}[{j}]), '
                )
                debug_outputs_parts.append(
                    f'        IntegerEntry("debug_a{i+1}_{j+1}", a{i+1}[{j}]), '
                )
        else:
            for j in range(architecture[i]["output_features"]):
                debug_outputs_parts.append(
                    f'        IntegerEntry("debug_a{i+1}", a{i+1}), '
                )
            debug_outputs_parts.append(
                f'        IntegerEntry("debug_z{i+1}", z{i+1}), '
            )

    debug_outputs_parts.append('        IntegerEntry("debug_result", result)')

    predict_func_parts.extend(debug_outputs_parts)
    predict_func_parts.extend(
        ["    ]", "    (", "        debug_outputs,", "        result", "    )", "}"]
    )

    predict_func = "\n".join(predict_func_parts)

    print("Final predict function parts:", predict_func_parts)

    # Combine all parts
    ride_script = (
        "{-# STDLIB_VERSION 7 #-}\n"
        "{-# CONTENT_TYPE DAPP #-}\n"
        "{-# SCRIPT_TYPE ACCOUNT #-}\n\n"
        "# Weights and Biases\n" + "\n".join(layer_defs) + "\n\n"
        "# Linear Forward Functions\n" + "\n".join(linear_funcs) + "\n\n"
        "# Activation Functions\n" + "\n".join(activation_funcs) + "\n\n"
        "# Predict Function\n" + predict_func
    )

    print("Generated RIDE script:\n", ride_script)

    return ride_script
