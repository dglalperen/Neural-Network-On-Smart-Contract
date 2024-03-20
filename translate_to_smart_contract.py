import numpy as np
import torch
from training_model import XORNet
from ThreeLayerXOR.ThreeLayerXORNet import ThreeLayerXORNet

def pytorch_to_waves_contract(model, scaling_factor=1000000):
    # Initialize an empty string to store weight and bias declarations
    weight_bias_declarations = ""
    
    # Function to scale and format weights or biases
    def format_parameters(parameters):
        parameters_scaled = np.round(parameters.detach().numpy() * scaling_factor).astype(int)
        return np.array2string(parameters_scaled, separator=', ', max_line_width=np.inf, threshold=np.inf).replace('\n', '')
    
    # Iterate over the state dictionary to process weights and biases for each layer
    for name, param in model.state_dict().items():
        # Replace '.' with '' to match Waves variable naming conventions
        formatted_name = name.replace('.', '')
        # Scale and format the parameter
        formatted_param = format_parameters(param)
        # Append the formatted parameter declaration to the overall declarations string
        weight_bias_declarations += f"let {formatted_name} = {formatted_param};\n    "
    
    # Smart contract template with dynamically inserted weights and biases
    contract_template = f"""
    {{-# STDLIB_VERSION 5 #}}
    {{-# CONTENT_TYPE DAPP #}}
    {{-# SCRIPT_TYPE ACCOUNT #}}

    {weight_bias_declarations}

    func sigmoid(z: Int) = {{
    let e = 2718281
    let base = 1000000
    let positiveZ = if (z < 0) then -z else z
    let expPart = fraction(e, base, positiveZ)
    fraction(base, base, base + expPart)
    }}

    func dotProduct(a: List[Int], b: List[Int]) = {{
        let sum = 0
        let length = size(a)
        let i = 0
        while(i < length) {{
            sum = sum + fraction(a[i], b[i], 1000000)
            i = i + 1
        }}
        sum
    }}

    func forwardPass(input: List[Int], weights: List[List[Int]], biases: List[Int]) = {{
        let outputs = []
        let length = size(weights)
        let i = 0
        while(i < length) {{
            let sum = dotProduct(input, weights[i]) + biases[i]
            let output = sigmoid(sum)
            outputs = outputs :+ output
            i = i + 1
        }}
        outputs
    }}

    @Callable(i)
    func predict(input1: Int, input2: Int, numLayers: Int) = {{
        let input = [input1, input2]
        let hiddenOutputs1 = if(numLayers >= 2) then forwardPass(input, layer1Weights, layer1Biases) else []
        let hiddenOutputs2 = if(numLayers == 3) then forwardPass(hiddenOutputs1, layer2Weights, layer2Biases) else []
        
        let output = if(numLayers == 1) then forwardPass(input, layer1Weights, layer1Biases)
                    else if(numLayers == 2) then forwardPass(hiddenOutputs1, layer2Weights, layer2Biases)
                    else if(numLayers == 3) then forwardPass(hiddenOutputs2, layer3Weights, layer3Biases)
                    else throw("Invalid number of layers")

        [IntegerEntry("result", output[0])]
    }}
    """
    
    return contract_template

if __name__ == "__main__":
    xor_net = XORNet()
    model_path = './xor_net.pth'
    xor_net_state = torch.load(model_path, map_location=torch.device('cpu'))
    xor_net.load_state_dict(xor_net_state)

    # Convert the XORNet to a Waves smart contract
    contract = pytorch_to_waves_contract(xor_net)
    print("2 Layer XOR Net Smart Contract:")
    print(contract)
    
    # Convert the ThreeLayerXORNet to a Waves smart contract
    three_layer_xor_net = ThreeLayerXORNet()
    model_path = './ThreeLayerXOR/three_layer_xor_net.pth'
    three_layer_xor_net_state = torch.load(model_path, map_location=torch.device('cpu'))
    three_layer_xor_net.load_state_dict(three_layer_xor_net_state)
    contract = pytorch_to_waves_contract(three_layer_xor_net)
    print("3 Layer XOR Net Smart Contract:")
    print(contract)
    