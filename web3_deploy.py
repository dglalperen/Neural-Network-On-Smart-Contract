from web3 import Web3
from dotenv import load_dotenv
import os

def store_prediction(input_hash, prediction):
    # Transact a storePrediction function call
    tx_hash = contract.functions.storePrediction(input_hash, prediction).transact()
    
    # Wait for the transaction receipt
    receipt = w3.eth.waitForTransactionReceipt(tx_hash)
    
    return receipt

def verify_prediction(input_hash, expected_prediction):
    # Call the verifyPrediction function
    return contract.functions.verifyPrediction(input_hash, expected_prediction).call()

if __name__ == "__main__":
    load_dotenv()

    # Infura API Key (replace with your actual Infura API Key)
    INFURA_API_KEY = os.getenv('INFURA_API_KEY')

    # Connect to the Ethereum network using Infura
    w3 = Web3(Web3.HTTPProvider(f'https://mainnet.infura.io/v3/{INFURA_API_KEY}'))

    # Set the default Ethereum account (replace with your desired account)
    w3.eth.default_account = w3.eth.accounts[0]

    # Smart contract address (replace with your deployed contract address)
    contract_address = '0xYourContractAddress'

    # Smart contract ABI (replace with your actual ABI)
    contract_abi = [
                        {
                            "inputs": [
                                {
                                    "internalType": "bytes32",
                                    "name": "",
                                    "type": "bytes32"
                                }
                            ],
                            "name": "predictions",
                            "outputs": [
                                {
                                    "internalType": "bool",
                                    "name": "",
                                    "type": "bool"
                                }
                            ],
                            "stateMutability": "view",
                            "type": "function"
                        },
                        {
                            "inputs": [
                                {
                                    "internalType": "bytes32",
                                    "name": "inputHash",
                                    "type": "bytes32"
                                },
                                {
                                    "internalType": "bool",
                                    "name": "prediction",
                                    "type": "bool"
                                }
                            ],
                            "name": "storePrediction",
                            "outputs": [],
                            "stateMutability": "nonpayable",
                            "type": "function"
                        },
                        {
                            "inputs": [
                                {
                                    "internalType": "bytes32",
                                    "name": "inputHash",
                                    "type": "bytes32"
                                },
                                {
                                    "internalType": "bool",
                                    "name": "prediction",
                                    "type": "bool"
                                }
                            ],
                            "name": "verifyPrediction",
                            "outputs": [
                                {
                                    "internalType": "bool",
                                    "name": "",
                                    "type": "bool"
                                }
                            ],
                            "stateMutability": "view",
                            "type": "function"
                        }
                    ]

    # Create a contract object
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
