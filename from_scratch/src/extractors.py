import torch


def extract_model_parameters(model):
    """
    Extracts weights and biases from a pretrained PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        dict: A dictionary containing the layers' weights, biases, and layer details.
    """
    model_parameters = {
        "layers": [],
        "weights": [],
        "biases": [],
    }

    for name, param in model.named_parameters():
        if "weight" in name:
            model_parameters["weights"].append(param.detach().numpy().tolist())
            layer_name = name.split(".")[0]
            model_parameters["layers"].append(
                {
                    "name": layer_name,
                    "num_neurons": param.size(0),
                    "num_inputs": param.size(1),
                }
            )
        elif "bias" in name:
            model_parameters["biases"].append(param.detach().numpy().tolist())

    return model_parameters
