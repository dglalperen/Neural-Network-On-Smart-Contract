import numpy as np
import torch
from TwoLayerXOR.TwoLayerXORNet import TwoLayerXORNet
from ThreeLayerXOR.ThreeLayerXORNet import ThreeLayerXORNet

def pytorch_to_waves_contract(model, scaling_factor=1000000):
    # Initialize an empty string to store weight and bias declarations
    weight_bias_declarations = ""
    
    # Function to scale and format weights or biases
    def format_parameters(parameters):
        parameters_scaled = np.round(parameters.detach().numpy() * scaling_factor).astype(int)
        return np.array2string(parameters_scaled, separator=', ', threshold=np.inf).replace('[', '').replace(']', '').replace('\n', '')
    
    # Iterate over the state dictionary to process weights and biases for each layer
    for name, param in model.state_dict().items():
        # Replace '.' with '' to match Waves variable naming conventions
        formatted_name = name.replace('.', '')
        # Scale and format the parameter
        formatted_param = format_parameters(param)
        # Append the formatted parameter declaration to the overall declarations string
        weight_bias_declarations += f"let {formatted_name} = [{formatted_param}]\n    "
    
    # Smart contract template with dynamically inserted weights and biases
    contract_template = f"""
    {{-# STDLIB_VERSION 5 #-}}
    {{-# CONTENT_TYPE DAPP #-}}
    {{-# SCRIPT_TYPE ACCOUNT #-}}

    {weight_bias_declarations}

    func sigmoid(z: Int) = {{
        let e = 2718281
        let base = 1000000
        let positiveZ = if (z < 0) then -z else z
        let expPart = fraction(e, base, positiveZ)
        fraction(base, base, base + expPart)
    }}
    
    func accumulateProduct(acc: Int, idx: Int, a: List[Int], b: List[Int]) = {{
        let product = fraction(getElement(a, idx), getElement(b, idx), 1000000)
        acc + product
    }}

    func dotProduct(a: List[Int], b: List[Int]) = {{
        let size = min(size(a), size(b)) # Ensure you don't go out of bounds.
        let indices = range(0, size - 1) # Generating indices based on dynamic size.
        FOLD<indices>(0, indices, accumulateProduct, a, b)  # Passing a and b into accumulateProduct
    }}
    
    # Function to accumulate neuron outputs
    func accumulateNeuronOutput(accumOutputs: List[Int], idx: Int, input: List[Int], weights: List[List[Int]], biases: List[Int]) = {{
        let currentWeights = getElement(weights, idx)
        let currentBias = getElement(biases, idx)
        let z = dotProduct(input, currentWeights) + getElement(currentBias, 0) # Assume biases are [Int]
        let output = sigmoid(z)
        accumOutputs :+ output
    }}

    # Forward pass function applying accumulateNeuronOutput using FOLD
    func forwardPass(input: List[Int], weights: List[List[Int]], biases: List[Int], layerSize: Int) = {{
        let indices = range(0, layerSize - 1)  # Indices for the neurons in the layer
        FOLD<indices>([], indices, accumulateNeuronOutput, input, weights, biases)
    }}


    @Callable(i)
    func predict(input1: Int, input2: Int, numLayers: Int) = {{
        let input = [input1, input2]
        let hiddenOutputs1 = if(numLayers >= 1) then forwardPass(input, layer1Weights, layer1Biases, 2) else input
        let hiddenOutputs2 = if(numLayers >= 2) then forwardPass(hiddenOutputs1, layer2Weights, layer2Biases, 4) else hiddenOutputs1  # up to 4 neurons in layer 2
        let finalOutput = if(numLayers == 3) then forwardPass(hiddenOutputs2, layer3Weights, layer3Biases, 2) else hiddenOutputs2  # assuming up to 2 neurons in layer 3

        [IntegerEntry("result", getElement(finalOutput, 0))]
    }}
    """
    
    return contract_template

if __name__ == "__main__":
    
    # Two Layer XOR Net
    xor_net = TwoLayerXORNet()
    model_path = './TwoLayerXOR/two_layer_xor_net.pth'
    xor_net_state = torch.load(model_path, map_location=torch.device('cpu'))
    xor_net.load_state_dict(xor_net_state)

    # Convert the XORNet to a Waves smart contract
    contract = pytorch_to_waves_contract(xor_net)
    print("2 Layer XOR Net Smart Contract:")
    print(contract)
    
    # Three Layer XOR Net
    three_layer_xor_net = ThreeLayerXORNet()
    model_path = './ThreeLayerXOR/three_layer_xor_net.pth'
    three_layer_xor_net_state = torch.load(model_path, map_location=torch.device('cpu'))
    three_layer_xor_net.load_state_dict(three_layer_xor_net_state)
    contract = pytorch_to_waves_contract(three_layer_xor_net)
    print("3 Layer XOR Net Smart Contract:")
    print(contract)
