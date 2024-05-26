import json


def generate_ride_script(json_file, debug_mode=True):
    with open(json_file, "r") as f:
        model = json.load(f)

    architecture = model["architecture"]
    parameters = model["parameters"]

    weights = parameters["weights"]
    biases = parameters["biases"]

    layer_defs = []
    linear_funcs = []
    activation_funcs = []

    # Define weights and biases
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

    # General linear forward function
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

    # Sigmoid activation function for a list of values
    activation_funcs.append(
        f"# Sigmoid function\n"
        f"func sigmoid(input: Int) = {{\n"
        f"    if (input < -10000) then 0\n"
        f"    else if (input > 10000) then 10000\n"
        f"    else 5000 + input / 2\n"
        f"}}"
    )

    # Add sigmoid activations for layers
    activation_funcs.append(
        f"# Sigmoid activation function for a list of values\n"
        f"func sigmoid_activation(inputs: List[Int]) = {{\n"
        f"    ["
        + ", ".join(
            [
                f"sigmoid(inputs[{j}])"
                for j in range(architecture[-1]["output_features"])
            ]
        )
        + "]\n"
        f"}}"
    )

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
                    f"    let z{i+1} = linear_forward_{i+1}(inputs, weights_layer_{i+1}, biases_layer_{i+1})",
                    f"    let a{i+1} = sigmoid_activation(z{i+1})",
                ]
            )
        elif i < len(architecture) - 1:
            predict_func_parts.extend(
                [
                    f"    let z{i+1} = linear_forward_{i+1}(a{i}, weights_layer_{i+1}, biases_layer_{i+1})",
                    f"    let a{i+1} = sigmoid_activation(z{i+1})",
                ]
            )
        else:
            predict_func_parts.extend(
                [
                    f"    let z{i+1} = linear_forward_{i+1}(a{i}, weights_layer_{i+1}, biases_layer_{i+1})",
                    f"    let a{i+1} = sigmoid(z{i+1}[0])",
                ]
            )

    predict_func_parts.extend(
        [
            "    # Scaling back the output",
            f"    let result = a{len(architecture)} / 10000",
        ]
    )

    if debug_mode:
        predict_func_parts.extend(
            [
                "    # Debug outputs",
                "    let debug_outputs = [",
            ]
        )

        debug_outputs_parts = []
        for i in range(len(architecture)):
            for j in range(architecture[i]["output_features"]):
                debug_outputs_parts.append(
                    f'        IntegerEntry("debug_z{i+1}_{j+1}", z{i+1}[{j}]), '
                )
                if i < len(architecture) - 1:
                    debug_outputs_parts.append(
                        f'        IntegerEntry("debug_a{i+1}_{j+1}", a{i+1}[{j}]), '
                    )
                else:
                    debug_outputs_parts.append(
                        f'        IntegerEntry("debug_a{i+1}", a{i+1}), '
                    )
        debug_outputs_parts.append('        IntegerEntry("debug_result", result)')

        predict_func_parts.extend(debug_outputs_parts)
        predict_func_parts.extend(
            ["    ]", "    (", "        debug_outputs,", "        result", "    )", "}"]
        )
    else:
        predict_func_parts.extend(
            [
                "    let debug_outputs = []",  # Ensure an empty list for debug_outputs when not in debug mode
                "    (debug_outputs, result)",
                "}",
            ]
        )

    predict_func = "\n".join(predict_func_parts)

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

    return ride_script
