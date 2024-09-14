import json
from generate_ride import (
    generate_activation_functions,
    generate_linear_forward_functions,
    generate_predict_function,
    generate_weights_and_biases,
)


if __name__ == "__main__":
    json_file = "../model_info/two_layerxornet.json"
    with open(json_file, "r") as f:
        model = json.load(f)

    architecture = model["architecture"]
    parameters = model["parameters"]

    # Test weights and biases
    weights_and_biases = generate_weights_and_biases(architecture, parameters)
    # print("# Weights and Biases\n" + weights_and_biases + "\n")

    # Test linear forward functions
    linear_forward_functions = generate_linear_forward_functions(architecture)
    # print("# Linear Forward Functions\n" + linear_forward_functions + "\n")

    # Test activation functions
    activation_functions = generate_activation_functions(architecture)
    # print("# Activation Functions\n" + activation_functions + "\n")

    # Test predict function
    predict_function = generate_predict_function(architecture, debug_mode=False)
    print("# Predict Function\n" + predict_function + "\n")
