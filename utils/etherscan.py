# utils/etherscan.py
"""
Simple wrapper to fetch Ethereum transactions from the Etherscan API.

Requires an Etherscan API key.
Usage:
    from utils.etherscan import fetch_transactions
    txs = fetch_transactions("0x1234...", "YOUR_API_KEY")
"""

import requests
import time
from typing import List, Dict, Optional

ETHERSCAN_BASE = "https://api.etherscan.io/api"


def fetch_transactions(address, api_key, startblock=0, endblock=99999999, sort="desc", page=1, offset=10000):
    """
    Return a list of transactions (dicts) for a given address with enhanced data.
    If API key is missing or API returns an error, returns [].
    """
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": startblock,
        "endblock": endblock,
        "page": page,
        "offset": offset,
        "sort": sort,
        "apikey": api_key,
    }
    try:
        r = requests.get(ETHERSCAN_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "1":
            print(f"⚠️  Etherscan API returned status: {data.get('status')}, message: {data.get('message')}")
            return []  # no txs or error
        
        transactions = data.get("result", [])
        
        # Enhance transactions with additional computed fields
        for tx in transactions:
            # Add transaction success status (1 = success, 0 = failed)
            tx['isError'] = tx.get('isError', '0')
            # Add computed fields
            tx['value_eth'] = float(tx.get('value', 0)) / 1e18
            tx['gasPrice_gwei'] = float(tx.get('gasPrice', 0)) / 1e9
            tx['gasUsed'] = tx.get('gasUsed', tx.get('gas', 0))
            # Determine if it's a contract interaction (simplified check)
            tx['is_contract_call'] = len(tx.get('input', '0x')) > 2
            
        return transactions
        
    except Exception as e:
        print("⚠️  Etherscan request failed:", e)
        return []

def fetch_account_balance(address, api_key):
    """
    Fetch the current ETH balance for an address.
    """
    params = {
        "module": "account",
        "action": "balance",
        "address": address,
        "tag": "latest",
        "apikey": api_key,
    }
    try:
        r = requests.get(ETHERSCAN_BASE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "1":
            balance_wei = int(data.get("result", 0))
            return balance_wei / 1e18
        return 0.0
    except Exception as e:
        print("⚠️  Balance fetch failed:", e)
        return 0.0

def is_contract_address(address, api_key):
    """
    Check if an address is a contract by getting its code.
    """
    params = {
        "module": "proxy",
        "action": "eth_getCode",
        "address": address,
        "tag": "latest",
        "apikey": api_key,
    }
    try:
        r = requests.get(ETHERSCAN_BASE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        code = data.get("result", "0x")
        return len(code) > 2  # Contract if has code beyond "0x"
    except Exception:
        return False
