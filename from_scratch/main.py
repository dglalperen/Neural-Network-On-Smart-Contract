import json
import torch
import os
from src.extractors import extract_model_parameters
from src.architecture import get_network_architecture
from contracts.generate_ride import generate_ride_script


def extract_and_save_model_info(model_class, model_path, output_folder="model_info"):
    """
    Extracts the model parameters and architecture and saves them to a JSON file.

    Args:
        model_class (torch.nn.Module): The class of the model to be loaded.
        model_path (str): The path to the pretrained model.
        output_folder (str): The folder where the output file will be saved.
    """
    # Load the pretrained model
    model = model_class()
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    # Extract model parameters
    model_parameters = extract_model_parameters(model)

    # Get network architecture
    network_architecture = get_network_architecture(model)

    # Combine parameters and architecture into a single dictionary
    model_info = {"architecture": network_architecture, "parameters": model_parameters}

    # Dynamically create the output file name based on the model name
    output_file_name = f"{model.__class__.__name__.lower()}_info.json"

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the combined information to a JSON file
    with open(os.path.join(output_folder, output_file_name), "w") as f:
        json.dump(model_info, f, indent=4)


def serialize_parameters(parameters):
    serialized_params = {
        "layers": parameters["layers"],
        "weights": [
            [int(w * 10000) for w in layer.flatten()] for layer in parameters["weights"]
        ],
        "biases": [
            [int(b * 10000) for b in layer.flatten()] for layer in parameters["biases"]
        ],
    }
    return serialized_params


def save_generated_ride_script(script, model_name, output_folder="generated"):
    """
    Save the generated RIDE script to a file.

    Args:
        script (str): The generated RIDE script.
        model_name (str): The name of the model to include in the file name.
        output_folder (str): The folder where the output file will be saved.
    """

    # Dynamically create the output file name based on the model name
    output_file_name = f"{model_name.lower()}_generated_ride_script.ride"

    # Save the RIDE script to a file
    with open(os.path.join("contracts", output_folder, output_file_name), "w") as f:
        f.write(script)


if __name__ == "__main__":
    # Example usage
    from models.two_layer_xor_net import TwoLayerXORNet
    from models.three_layer_xor_net import ThreeLayerXORNet
    from models.tic_tac_toe_net import TicTacNet
    from models.insurance_net import InsuranceNet

    model_paths = {
        "twolayerxornet": os.path.join(
            os.path.dirname(__file__), "trained_torch_models", "two_layer_xor_net.pth"
        ),
        "threelayerxornet": os.path.join(
            os.path.dirname(__file__), "trained_torch_models", "three_layer_xor_net.pth"
        ),
        "tictacnet": os.path.join(
            os.path.dirname(__file__), "trained_torch_models", "tic_tac_toe_net.pth"
        ),
        # "insurancenet": os.path.join(
        #     os.path.dirname(__file__), "trained_torch_models", "insurance_net.pth"
        # ),
    }

    model_classes = {
        "twolayerxornet": TwoLayerXORNet,
        "threelayerxornet": ThreeLayerXORNet,
        "tictacnet": TicTacNet,
        # "insurancenet": InsuranceNet,
    }

    # Extract and save model info
    for model_name, model_path in model_paths.items():
        extract_and_save_model_info(model_classes[model_name], model_path)

    # Generate and save RIDE scripts
    for model_name in model_paths.keys():
        json_file_path = os.path.join("model_info", f"{model_name}_info.json")
        generated_script = generate_ride_script(json_file_path, debug_mode=False)
        save_generated_ride_script(generated_script, model_name)
        print(f"RIDE script for {model_name} generated successfully!")
