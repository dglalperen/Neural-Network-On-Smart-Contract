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


def save_generated_ride_script(script, output_folder="generated"):
    """
    Save the generated RIDE script to a file.

    Args:
        script (str): The generated RIDE script.
        output_folder (str): The folder where the output file will be saved.
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Dynamically create the output file name based on the model name
    output_file_name = "generated_ride_script.ride"

    # Save the RIDE script to a file
    with open(os.path.join("contracts", output_folder, output_file_name), "w") as f:
        f.write(script)


if __name__ == "__main__":
    # Example usage
    from models.two_layer_xor_net import TwoLayerXORNet
    from models.tic_tac_toe_net import TicTacNet

    model_path_tictac = os.path.join(
        os.path.dirname(__file__), "trained_torch_models", "tic_tac_toe_net.pth"
    )

    model_path_twoxor = os.path.join(
        os.path.dirname(__file__), "trained_torch_models", "two_layer_xor_net.pth"
    )

    extract_and_save_model_info(TwoLayerXORNet, model_path_twoxor)
    # tictac_model_info = extract_and_save_model_info(TicTacNet, model_path_tictac)

    # generate RIDE script
    generated_script = generate_ride_script(
        os.path.join("model_info", "twolayerxornet_info.json")
    )
    save_generated_ride_script(generated_script)
    print("Ride script generated successfully!")
    # generate_ride_script(os.path.join("model_info", "tictacnet_info.json"))
