# data/simulate.py
"""
Simulate synthetic wallet transactions and extract features for training SharkGuard.

Usage:
    python data/simulate.py

This will generate:
    data/simulated_features.csv
which you can then use to train a model with sharkguard.core.train_and_persist_model().
"""

import random
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Import feature extraction tools from core
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fakeacc.core import txs_to_dataframe, extract_wallet_features, features_to_df
from fakeacc.core import FEATURE_COLUMNS


def simulate_wallet(num_txs=50, start_days_ago=180, wallet_addr="0xwallet"):
    """
    Generate a list of fake transactions for a wallet.
    Each transaction has fields similar to Etherscan API output.
    """
    txs = []
    now = datetime.utcnow()
    for i in range(num_txs):
        ts = now - timedelta(seconds=random.randint(0, start_days_ago * 24 * 3600))
        other = "0x" + "".join(random.choices("0123456789abcdef", k=40))
        value = random.random() * (0.1 if random.random() > 0.95 else 1.0)
        tx = {
            "hash": "0x" + "".join(random.choices("0123456789abcdef", k=64)),
            "from": wallet_addr if random.random() < 0.5 else other,
            "to": other if random.random() < 0.5 else wallet_addr,
            "value": int(value * 1e18),  # in wei
            "gas": random.randint(21000, 200000),
            "gasPrice": random.randint(20, 200) * (10**9),  # in wei
            "timeStamp": int(ts.timestamp())
        }
        txs.append(tx)
    return txs


OUT = Path("data") / "simulated_features.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

N = 600
rng = np.random.default_rng(42)

def make_row():
    tx_count = rng.integers(0, 500)
    outgoing = rng.integers(0, tx_count+1)
    incoming = tx_count - outgoing
    avg_value = float(max(0.0, rng.normal(0.5, 2.0)))
    std_value = float(max(0.0, abs(rng.normal(0.2, 0.5))))
    unique_peers = int(min(200, rng.integers(0, tx_count+5)))
    days_active = int(rng.integers(0, 365*2))
    mean_t_between = float(abs(rng.normal(3600, 10000)))
    return {
        "tx_count": int(tx_count),
        "outgoing_tx_count": int(outgoing),
        "incoming_tx_count": int(incoming),
        "avg_value_eth": round(avg_value, 8),
        "std_value_eth": round(std_value, 8),
        "unique_peers": int(unique_peers),
        "days_active": int(days_active),
        "mean_time_between_tx": float(round(mean_t_between, 3)),
    }

rows = [make_row() for _ in range(N)]
df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
df.to_csv(OUT, index=False)
print("Wrote simulated features to", OUT)


def main():
    # Create synthetic dataset of multiple wallets
    wallets = []
    for i in range(200):
        w = "0x" + "".join(random.choices("0123456789abcdef", k=40))
        if i < 20:
            # make some wallets look more suspicious
            txs = simulate_wallet(num_txs=random.randint(100, 200), wallet_addr=w)
        else:
            txs = simulate_wallet(num_txs=random.randint(5, 80), wallet_addr=w)
        wallets.append({"wallet": w, "txs": txs})

    # Extract features
    feats = []
    for entry in wallets:
        df = txs_to_dataframe(entry["txs"])
        feat = extract_wallet_features(df, entry["wallet"])
        feats.append(feat)

    df_feats = features_to_df(feats)

    # Save to CSV
    out_path = os.path.join("data", "simulated_features.csv")
    df_feats.to_csv(out_path, index=False)
    print(f"✅ Saved {out_path}")


if __name__ == "__main__":
    main()
