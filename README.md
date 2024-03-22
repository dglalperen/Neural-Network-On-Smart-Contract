# XOR Predictions Smart Contract and Neural Network

This project combines a Solidity smart contract with a neural network implemented in PyTorch to create a decentralized prediction system for XOR logic gates.

## Smart Contract (Solidity)

### Contract Overview

- **Contract Name**: XOR_Predictions
- **SPDX-License-Identifier**: UNKNOWN
- **Solidity Version**: ^0.8.0

### Functions

1. **storePrediction(bytes32 inputHash, bool prediction)**

   - Stores a prediction for a given input hash.
   - Input: `inputHash` - a unique hash of the input data, `prediction` - the predicted boolean value.
   - Access: Public
   - State Change: Modifies the `predictions` mapping.

2. **verifyPrediction(bytes32 inputHash, bool prediction)**
   - Verifies a prediction for a given input hash.
   - Input: `inputHash` - a unique hash of the input data, `prediction` - the expected boolean value.
   - Access: Public View
   - Returns: `true` if the stored prediction matches the expected prediction, `false` otherwise.

## Python Scripts

### 1. Connecting to Ethereum Network

Before interacting with the smart contract, make sure to set up your Infura API key in the `INFURA_API_KEY` variable and specify the contract address (`contract_address`) and ABI (`contract_abi`) for your deployed contract.

### 2. Neural Network for XOR Logic

The Python script includes the following components:

- **XORNet class**: A PyTorch neural network model for XOR logic.
- **Training**: Training the XORNet with XOR input and output data.
- **Loss Plotting**: Visualizing the loss during training.
- **Output Plotting**: Visualizing the network outputs after training.
- **Model Saving**: Saving the trained model to `xor_net_model.pth`.

### 3. Using the Neural Network for Predictions

This script allows you to load the trained XOR neural network model (`xor_net_model.pth`) and make predictions on XOR input data.

### 4. Hashing Inputs

The script provides a function to hash input data using SHA-256.

### 5. Deploying the Smart Contract

This script compiles and deploys the XOR_Predictions smart contract using Solidity. It sets the Solidity compiler version to 0.8.24, compiles the contract, and prints the compiled output.

## Prerequisites

Make sure you have the following installed:

- Python 3
- PyTorch
- Web3.py
- Matplotlib
- Solidity Compiler (solc)

## Getting Started

1. Clone this repository.
2. Set up your Infura API key in the Python script.
3. Deploy the smart contract using the provided Solidity code.
4. Train the XOR neural network using the Python script.
5. Make predictions on XOR input data using the neural network.
6. Hash input data as needed.

## License

This project is licensed under an UNKNOWN license.

Feel free to modify and adapt the code to your specific needs. Enjoy building and experimenting with decentralized predictions using XOR logic!

    func forwardPass(input: List[Int], weights: List[List[Int]], biases: List[Int], size: Int) = {{
        let initOutputs = []
        let indices = [0, 1, 2, 3]
        FOLD<indices>(initOutputs, indices, {{(outputs, i) =>
            if (i < size) then
                outputs :+ sigmoid(dotProduct(input, getElement(weights, i)) + getElement(biases, i))
            else
                outputs
        }})
    }}
