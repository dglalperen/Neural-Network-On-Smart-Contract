import json


def generate_ride_script(json_file):
    with open(json_file, "r") as f:
        model = json.load(f)

    architecture = model["architecture"]
    parameters = model["parameters"]

    weights = parameters["weights"]
    biases = parameters["biases"]

    layer_defs = []

    for i, layer in enumerate(architecture):
        input_features = layer["input_features"]
        output_features = layer["output_features"]
        weight_layer = weights[i]
        bias_layer = biases[i]

        # Define weights and biases
        weight_defs = f"let weights_layer_{i+1} = " + str(
            [[int(w * 10000) for w in ws] for ws in weight_layer]
        ).replace("[", "[[").replace("]", "]]").replace("], [[", "],\n[")
        bias_defs = f"let biases_layer_{i+1} = " + str(
            [int(b * 10000) for b in bias_layer]
        ).replace("[", "[").replace("]", "]")

        layer_defs.append(weight_defs)
        layer_defs.append(bias_defs)

    # Return the script for now
    ride_script = (
        "{-# STDLIB_VERSION 7 #-}\n"
        "{-# CONTENT_TYPE DAPP #-}\n"
        "{-# SCRIPT_TYPE ACCOUNT #-}\n\n"
        "# Weights and Biases\n" + "\n".join(layer_defs) + "\n\n"
    )

    return ride_script
