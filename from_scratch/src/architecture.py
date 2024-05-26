import torch


def get_network_architecture(model):
    """
    Gets the architecture of the neural network.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        list: A list containing the details of each layer.
    """
    architecture = []
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            layer_info = {
                "type": "Linear",
                "input_features": layer.in_features,
                "output_features": layer.out_features,
            }
        elif isinstance(layer, torch.nn.ReLU):
            layer_info = {"type": "ReLU"}
        elif isinstance(layer, torch.nn.Sigmoid):
            layer_info = {"type": "Sigmoid"}
        # Add other activation functions and layers as needed

        architecture.append(layer_info)

    return architecture
