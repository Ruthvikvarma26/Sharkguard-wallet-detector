import requests
import os

class EtherscanClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY")
        self.base_url = "https://api.etherscan.io/api"

    def get_wallet_balance(self, wallet_address):
        params = {
            "module": "account",
            "action": "balance",
            "address": wallet_address,
            "tag": "latest",
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "1":
            return float(data["result"]) / 10**18  # Convert from Wei to Ether
        else:
            raise ValueError(f"Error fetching balance: {data['message']}")

    def get_transaction_count(self, wallet_address):
        params = {
            "module": "proxy",
            "action": "eth_getTransactionCount",
            "address": wallet_address,
            "tag": "latest",
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "1":
            return int(data["result"], 16)  # Convert hex to int
        else:
            raise ValueError(f"Error fetching transaction count: {data['message']}")