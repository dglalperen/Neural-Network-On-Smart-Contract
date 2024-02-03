from solcx import compile_standard, install_solc, set_solc_version
import json

import solcx

# Set the Solidity compiler version to use
set_solc_version('0.8.24')

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

print(json.dumps(compiled_sol, indent=4))
