# sharkguard/core.py
"""
Core engine for SharkGuard:
 - Convert transactions into DataFrames
 - Extract wallet-level features
 - Train and use IsolationForest model
"""

import json
from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest

FEATURE_COLUMNS = [
    "tx_count",
    "outgoing_tx_count",
    "incoming_tx_count",
    "avg_value_eth",
    "std_value_eth",
    "unique_peers",
    "days_active",
    "mean_time_between_tx"
]


class SharkGuardModel:
    def __init__(self):
        self.model = None
        self.cols = FEATURE_COLUMNS

    def load(self, path: str):
        data = joblib.load(path)
        # stored dict: {"model": model, "cols": cols}
        self.model = data["model"]
        self.cols = data.get("cols", FEATURE_COLUMNS)

    def predict_score(self, features: Dict[str, Any]):
        # accept dict or DataFrame-like
        x = [float(features.get(c, 0.0)) for c in self.cols]
        if self.model is None:
            raise RuntimeError("Model not loaded")
        # IsolationForest decision_function: higher -> more normal. we invert to anomaly score 0..1
        score_raw = float(self.model.decision_function([x])[0])
        # map roughly to 0..1 where higher is more suspicious
        # decision_function range typically around [-0.5, 0.5]; apply sigmoid-like transform
        score = 1.0 / (1.0 + np.exp(5.0 * score_raw))
        label = "SUSPICIOUS" if score > 0.6 else "NORMAL"
        return {"label": label, "score": float(score)}

def txs_to_dataframe(txs):
    """
    Convert list of transaction dicts (Etherscan-like) to pandas DataFrame with normalized fields.
    Safe for empty lists.
    """
    if not txs:
        return pd.DataFrame(columns=[
            "from", "to", "value", "timeStamp", "isError", "gasUsed", "gas"
        ])
    df = pd.DataFrame(txs)
    # Try to coerce value from wei to ETH
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0) / 1e18
    else:
        df["value"] = 0.0
    if "timeStamp" in df.columns:
        df["timeStamp"] = pd.to_datetime(pd.to_numeric(df["timeStamp"], errors="coerce"), unit="s", errors="coerce")
    else:
        df["timeStamp"] = pd.NaT
    # normalize from/to
    df["from"] = df.get("from", "").astype(str)
    df["to"] = df.get("to", "").astype(str)
    return df

def extract_wallet_features(df: pd.DataFrame, wallet: str) -> Dict[str, Any]:
    """Compute simple aggregate features for a wallet from normalized tx DataFrame."""
    wallet = (wallet or "").lower()
    if df is None or df.empty:
        return {c: 0.0 for c in FEATURE_COLUMNS}

    df = df.copy()
    df["from_lc"] = df["from"].str.lower()
    df["to_lc"] = df["to"].str.lower()
    outgoing = df[df["from_lc"] == wallet]
    incoming = df[df["to_lc"] == wallet]
    tx_count = len(outgoing) + len(incoming)
    outgoing_count = len(outgoing)
    incoming_count = len(incoming)
    avg_value = float(df["value"].mean()) if not df["value"].isna().all() else 0.0
    std_value = float(df["value"].std(ddof=0)) if tx_count > 1 else 0.0
    peers = pd.concat([outgoing["to_lc"], incoming["from_lc"]]).dropna().unique()
    unique_peers = int(len(peers))
    # days active
    times = df["timeStamp"].dropna().sort_values()
    if len(times) >= 2:
        days_active = (times.max() - times.min()).days or 0
        diffs = times.diff().dt.total_seconds().dropna()
        mean_t_between = float(diffs.mean()) if not diffs.empty else 0.0
    else:
        days_active = 0
        mean_t_between = 0.0

    return {
        "tx_count": int(tx_count),
        "outgoing_tx_count": int(outgoing_count),
        "incoming_tx_count": int(incoming_count),
        "avg_value_eth": float(round(avg_value, 8)),
        "std_value_eth": float(round(std_value, 8)),
        "unique_peers": int(unique_peers),
        "days_active": int(days_active),
        "mean_time_between_tx": float(round(mean_t_between, 3)),
    }

def train_and_persist_model(df: pd.DataFrame, path: str):
    """
    Train an IsolationForest on provided DataFrame (expects feature columns present or fallback)
    and persist with joblib as {"model": model, "cols": cols}.
    """
    # Ensure numeric features exist
    if df is None or df.empty:
        raise ValueError("DataFrame required for training")
    X = []
    for _, row in df.iterrows():
        X.append([float(row.get(c, 0.0)) for c in FEATURE_COLUMNS])
    X = np.array(X, dtype=float)
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X)
    joblib.dump({"model": model, "cols": FEATURE_COLUMNS}, path)
