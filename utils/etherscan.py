# utils/etherscan.py
"""
Simple wrapper to fetch Ethereum transactions from the Etherscan API.

This has been updated to use the **Etherscan V2 API**, since V1 has been
deprecated and now returns a NOTOK error.

Requires an Etherscan API key.
Usage:
    from utils.etherscan import fetch_transactions
    txs = fetch_transactions("0x1234...", "YOUR_API_KEY")
"""

import os
import requests
from typing import List, Dict, Any

# V2 base URL (see: https://docs.etherscan.io/v2-migration)
ETHERSCAN_BASE = os.getenv("ETHERSCAN_BASE", "https://api.etherscan.io/v2/api")

# Default to Ethereum mainnet chain id; can be overridden via env if needed
DEFAULT_CHAIN_ID = os.getenv("ETHERSCAN_CHAIN_ID", "1")


def fetch_transactions(address: str, api_key: str, sort: str = "desc", offset: int = 10000) -> List[Dict[str, Any]]:
    """
    Fetch normal transactions for address using Etherscan **V2** API.
    Returns list of tx dicts (may be empty). Raises on HTTP/parse error.
    """
    params = {
        "chainid": DEFAULT_CHAIN_ID,
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": offset,
        "sort": sort,
        "apikey": api_key,
    }
    r = requests.get(ETHERSCAN_BASE, params=params, timeout=15)
    r.raise_for_status()
    payload = r.json()

    status = str(payload.get("status", "0"))
    message = str(payload.get("message", "")).upper()

    # "No transactions found" is a valid, empty result
    if status != "1" and "NO TRANSACTIONS FOUND" not in message and message not in ("OK",):
        # Pass through useful error message to caller
        raise RuntimeError(f"Etherscan error: {payload}")

    # V2 still returns the actual data in "result"
    return payload.get("result", []) or []


def fetch_account_balance(address: str, api_key: str) -> float:
    """
    Fetch current balance (in ETH) for an address using Etherscan **V2** API.
    """
    params = {
        "chainid": DEFAULT_CHAIN_ID,
        "module": "account",
        "action": "balance",
        "address": address,
        "tag": "latest",
        "apikey": api_key,
    }
    r = requests.get(ETHERSCAN_BASE, params=params, timeout=10)
    r.raise_for_status()
    payload = r.json()
    status = str(payload.get("status", "0"))

    if status not in ("1", "0") and "result" not in payload:
        raise RuntimeError(f"Etherscan balance error: {payload}")
    try:
        wei = int(payload.get("result", 0))
    except Exception:
        wei = 0
    return float(wei) / 1e18
