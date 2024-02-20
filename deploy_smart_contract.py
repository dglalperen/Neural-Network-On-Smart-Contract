from solcx import compile_standard, set_solc_version
import json
from web3 import Web3
import os
from dotenv import load_dotenv




if __name__ == "__main__":
    load_dotenv()

    INFURA_API_KEY = os.getenv("INFURA_API_KEY")
    PRIVATE_KEY = os.getenv("PRIVATE_KEY")
    
    # Connect to Web3
    w3 = Web3(Web3.HTTPProvider(f'https://sepolia.infura.io/v3/{INFURA_API_KEY}'))
    account = w3.eth.account.from_key(PRIVATE_KEY)
    
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
    bytecode = compiled_sol['contracts']['XOR.sol']['XOR_Predictions']['evm']['bytecode']['object']
    abi = json.loads(compiled_sol['contracts']['XOR.sol']['XOR_Predictions']['metadata'])['output']['abi']

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    constructor_tx = contract.constructor().build_transaction({
        'from': account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        'gas': 2000000,
        'gasPrice': w3.eth.gas_price
    })
    
    signed_txn = w3.eth.account.sign_transaction(constructor_tx, PRIVATE_KEY)
    
    # Sign the transaction
    signed_tx = w3.eth.account.sign_transaction(constructor_tx, PRIVATE_KEY)

    # Send the signed transaction
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

    # Wait for the transaction to be mined
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f'Contract deployed at address: {tx_receipt.contractAddress}')