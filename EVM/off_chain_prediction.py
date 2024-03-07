from web3 import Web3
import torch
import torch.nn as nn
import os
from dotenv import load_dotenv
from get_abi import get_abi

class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

def load_model(model_path):
    model = XORNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def hash_input(input_data):
    return Web3.keccak(text=str(input_data))

def store_prediction(contract, account, w3, hashed_input, prediction, private_key):
    gas_price = w3.eth.gas_price * 2  # Example: double the current gas price for faster processing
    tx = contract.functions.storePrediction(hashed_input, prediction).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 200000,
        'gasPrice': gas_price
    })
    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    try:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=240)  # Increase timeout to 240 seconds
        return receipt
    except Web3.exceptions.TimeExhausted:
        print("Transaction timeout. Consider increasing gas price or trying again later.")
        return None


def verify_prediction(contract, hashed_input, prediction):
    return contract.functions.verifyPrediction(hashed_input, prediction).call()

def main():
    load_dotenv()
    INFURA_API_KEY = os.getenv("INFURA_API_KEY")
    PRIVATE_KEY = os.getenv("PRIVATE_KEY")
    CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
    MODEL_PATH = './model/xor_net.pth'

    w3 = Web3(Web3.HTTPProvider(f'https://sepolia.infura.io/v3/{INFURA_API_KEY}'))
    contract_abi = get_abi()
    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)
    account = w3.eth.account.from_key(PRIVATE_KEY)

    model = load_model(MODEL_PATH)

    # Example input for prediction
    input_data = [1, 1]  # Adjust this based on your model's expected input
    prediction_tensor = model(torch.tensor([input_data], dtype=torch.float))
    prediction = prediction_tensor.item() > 0.5

    hashed_input = hash_input(str(input_data))

    # Store the prediction in the smart contract
    receipt = store_prediction(contract, account,w3, hashed_input, prediction, PRIVATE_KEY)
    print(f"Stored prediction. Transaction receipt: {receipt.transactionHash.hex()}")

    # Verify the stored prediction
    result = verify_prediction(contract, hashed_input, prediction)
    print(f"Verification result: {result}")

if __name__ == "__main__":
    main()
