import os
import torch
from translater import pytorch_to_waves_contract, save_contract_to_file

# Define the directory to search for PyTorch model files
directory_to_search = "./drag_torch_model"

def convert_model_to_contract(model_path):
    print(f"Converting {model_path} to Waves smart contract...")

    # Load the PyTorch model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print("Model loaded successfully.")
    print("Model architecture:", model)

    # Convert the PyTorch model to a Waves smart contract
    contract = pytorch_to_waves_contract(model)

    # Save the contract to a file
    model_name_no_ext = os.path.splitext(os.path.basename(model_path))[0]
    save_contract_to_file(contract, f"{model_name_no_ext}.ride")
    print(f"Smart contract generated: {model_name_no_ext}.ride")

def convert_models_to_contracts(directory):
    for file in os.listdir(directory):
        if file.endswith(".pth"):
            model_path = os.path.join(directory, file)
            convert_model_to_contract(model_path)

if __name__ == "__main__":
    convert_models_to_contracts(directory_to_search)
