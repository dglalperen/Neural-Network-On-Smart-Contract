from web3_deploy import Web3
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()

    INFURA_API_KEY = os.getenv('INFURA_API_KEY')

    infura_url = f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"

    web3 = Web3(Web3.HTTPProvider(infura_url))

    print(web3.is_connected)
    print("Connected to Ethereum network")