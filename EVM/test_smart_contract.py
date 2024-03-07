from web3 import Web3
import os
from dotenv import load_dotenv
from get_abi import get_abi

def generate_bytes32_hash(input_string):
    return Web3.sha3(text=input_string)[-32:]

def store_prediction(contract, account, hashed_input, prediction):
    tx = contract.functions.storePrediction(hashed_input, prediction).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 200000,
        'gasPrice': w3.eth.gas_price
    })
    signed_tx = w3.eth.account.sign_transaction(tx, account._private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt

def verify_prediction(contract, hashed_input, prediction):
    return contract.functions.verifyPrediction(hashed_input, prediction).call()

if __name__ == "__main__":
    load_dotenv()
    INFURA_API_KEY = os.getenv("INFURA_API_KEY")
    PRIVATE_KEY = os.getenv("PRIVATE_KEY")
    contract_address = os.getenv("CONTRACT_ADDRESS")

    w3 = Web3(Web3.HTTPProvider(f'https://sepolia.infura.io/v3/{INFURA_API_KEY}'))

    # Contract ABI and Address
    contract_abi = get_abi()

    # Initialize contract
    contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    account = w3.eth.account.from_key(PRIVATE_KEY)
    
    # Step 1: Generate a bytes32 hash
    input_a = "1"
    input_b = "1"
    combined_input = input_a + " " + input_b 
    
    hashed_input = w3.keccak(text=combined_input)

    # Step 2: Store the prediction
    prediction = True
    receipt = store_prediction(contract, account, hashed_input, prediction)
    print(f"Stored prediction. Transaction receipt: {receipt.transactionHash.hex()}")

    # Step 3: Verify the prediction
    result = verify_prediction(contract, hashed_input, prediction)
    print(f"Verification result: {result}")

