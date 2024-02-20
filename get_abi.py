import json
from solcx import compile_standard

def get_abi():
    compiled_sol = compile_standard({
    "language": "Solidity",
    "sources": {
        "XOR.sol": {
            "content": """
                pragma solidity ^0.8.0;

                contract XOR_Predictions {
                    mapping(bytes32 => bool) public predictions;

                    function storePrediction(bytes32 inputHash, bool prediction) public {
                        predictions[inputHash] = prediction;
                    }

                    function verifyPrediction(bytes32 inputHash, bool prediction) public view returns (bool) {
                        return predictions[inputHash] == prediction;
                    }
                }
            """
        }
    },
    "settings": {
        "outputSelection": {
            "*": {
                "*": ["abi", "metadata", "evm.bytecode", "evm.bytecode.sourceMap"]
            }
        }
    }
})

    # Extract bytecode and ABI
    abi = json.loads(compiled_sol['contracts']['XOR.sol']['XOR_Predictions']['metadata'])['output']['abi']
    
    return abi