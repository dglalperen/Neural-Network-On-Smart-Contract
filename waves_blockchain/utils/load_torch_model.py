import torch
import os

def load_torch_model(model_path):
    """
    Load a PyTorch model from a .pth file.

    Args:
        model_path (str): The path to the .pth file containing the PyTorch model.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    # Check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))

    return model
