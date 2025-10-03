# sharkguard/core.py
"""
Core engine for SharkGuard:
 - Convert transactions into DataFrames
 - Extract wallet-level features
 - Train and use IsolationForest model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

MODEL_PATH = "models/isolation_model.joblib"


def txs_to_dataframe(txs):
    """
    Convert a list of transactions into a DataFrame with enhanced processing.
    Expected tx fields: hash, from, to, value (wei), gas, gasPrice, timeStamp (unix).
    """
    if len(txs) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(txs)
    
    # Convert timestamp to datetime
    df["timeStamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s", errors="coerce")
    
    # Handle value conversion (might already be converted in enhanced fetch)
    if 'value_eth' not in df.columns:
        df["value_eth"] = df["value"].astype(float) / 1e18
    
    # Convert gas fields
    df["gas"] = df["gas"].astype(float)
    df["gasPrice"] = df["gasPrice"].astype(float)
    
    # Add gasUsed if available
    if 'gasUsed' in df.columns:
        df["gasUsed"] = df["gasUsed"].astype(float)
    else:
        df["gasUsed"] = df["gas"]  # Fallback to gas limit
    
    # Add transaction success status
    if 'isError' in df.columns:
        df["tx_success"] = (df["isError"].astype(str) == "0").astype(int)
    else:
        df["tx_success"] = 1  # Assume success if not available
    
    # Add contract interaction flag
    if 'is_contract_call' not in df.columns:
        df["is_contract_call"] = df.get("input", "0x").str.len() > 2
    
    return df


def extract_wallet_features(df, wallet_address):
    """
    Compute wallet-level features from a transaction DataFrame.
    Returns a dict of numeric features with enhanced analysis.
    """
    if df.empty:
        return {
            "tx_count": 0,
            "tx_freq_per_day": 0.0,
            "lifetime_days": 0.0,
            "avg_gas": 0.0,
            "avg_value_eth": 0.0,
            "unique_counterparties": 0,
            "repeated_ratio": 0.0,
            "hour_entropy": 0.0,
            "gas_efficiency": 0.0,
            "value_variance": 0.0,
            "weekend_activity": 0.0,
            "failed_tx_ratio": 0.0,
            "contract_interaction_ratio": 0.0,
        }

    df_sorted = df.sort_values("timeStamp")
    tx_count = len(df_sorted)
    lifetime_days = max(
        1.0,
        (df_sorted["timeStamp"].iloc[-1] - df_sorted["timeStamp"].iloc[0]).total_seconds() / 86400.0,
    )
    tx_freq_per_day = tx_count / lifetime_days
    avg_gas = df_sorted["gas"].mean()
    avg_value_eth = df_sorted["value_eth"].mean()

    # counterparties
    counterparties = set()
    wallet = wallet_address.lower()
    for _, r in df_sorted.iterrows():
        if str(r.get("from", "")).lower() != wallet:
            counterparties.add(str(r.get("from", "")).lower())
        if str(r.get("to", "")).lower() != wallet:
            counterparties.add(str(r.get("to", "")).lower())
    unique_counterparties = len(counterparties)

    # repeated behavior: fraction of txs to the same top counterparty
    cp = []
    for _, r in df_sorted.iterrows():
        if str(r["to"]).lower() != wallet:
            cp.append(str(r["to"]).lower())
        if str(r["from"]).lower() != wallet:
            cp.append(str(r["from"]).lower())
    repeated_ratio = 0.0
    if cp:
        top_count = max(pd.Series(cp).value_counts().iloc[0], 0)
        repeated_ratio = top_count / len(cp)

    # activity hour distribution entropy (bots often transact with regular hours)
    hours = df_sorted["timeStamp"].dt.hour.dropna()
    probs = hours.value_counts(normalize=True).values
    hour_entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    # Enhanced features for better detection
    # Gas efficiency (lower values might indicate automated transactions)
    gas_efficiency = avg_value_eth / (avg_gas / 1e9) if avg_gas > 0 else 0.0
    
    # Value variance (bots often use similar amounts)
    value_variance = df_sorted["value_eth"].var() if len(df_sorted) > 1 else 0.0
    
    # Weekend activity (bots are active 24/7)
    weekend_txs = df_sorted[df_sorted["timeStamp"].dt.weekday >= 5]
    weekend_activity = len(weekend_txs) / tx_count if tx_count > 0 else 0.0
    
    # Failed transaction ratio
    if 'tx_success' in df_sorted.columns:
        failed_tx_ratio = 1.0 - df_sorted['tx_success'].mean()
    else:
        failed_tx_ratio = 0.0
    
    # Contract interaction ratio
    if 'is_contract_call' in df_sorted.columns:
        contract_interaction_ratio = df_sorted['is_contract_call'].mean()
    else:
        contract_interaction_ratio = 0.0

    return {
        "tx_count": tx_count,
        "tx_freq_per_day": tx_freq_per_day,
        "lifetime_days": lifetime_days,
        "avg_gas": avg_gas,
        "avg_value_eth": avg_value_eth,
        "unique_counterparties": unique_counterparties,
        "repeated_ratio": repeated_ratio,
        "hour_entropy": hour_entropy,
        "gas_efficiency": gas_efficiency,
        "value_variance": value_variance,
        "weekend_activity": weekend_activity,
        "failed_tx_ratio": failed_tx_ratio,
        "contract_interaction_ratio": contract_interaction_ratio,
    }


def features_to_df(feat_dicts):
    """Turn a list of feature dicts into a DataFrame."""
    return pd.DataFrame(feat_dicts)


class SharkGuardModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        # The canonical full feature list supported by the current codebase
        self.features = [
            "tx_count",
            "tx_freq_per_day",
            "lifetime_days",
            "avg_gas",
            "avg_value_eth",
            "unique_counterparties",
            "repeated_ratio",
            "hour_entropy",
            "gas_efficiency",
            "value_variance",
            "weekend_activity",
            "failed_tx_ratio",
            "contract_interaction_ratio",
        ]
        # If a saved model provides the feature names it was trained with,
        # we store them here to align inputs at prediction time.
        self.trained_feature_names = None

    def train(self, X_df):
        """
        Train an IsolationForest model on a DataFrame of features.
        """
        X = X_df[self.features].fillna(0.0).values
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        # contamination small since fakes are rare in a mixed dataset
        self.model = IsolationForest(
            n_estimators=200, contamination=0.05, random_state=42
        )
        self.model.fit(Xs)

    def predict_score(self, feat_dict):
        """
        Predict a suspicion score for a single wallet's features.
        Returns dict with score (0..1), label, raw value.

        Handles backward-compatibility where an older saved model was trained on
        a different subset/ordering of features by aligning to the trained
        feature names when available.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded. Call load() first or train a model.")

        # Prefer the exact trained feature list if present
        if self.trained_feature_names is not None:
            feature_cols = list(self.trained_feature_names)
        else:
            # Fall back to scaler's expected feature count if available
            n_expected = getattr(self.scaler, "n_features_in_", None)
            if n_expected is None:
                feature_cols = list(self.features)
            else:
                # Best-effort alignment: use intersection in the canonical order
                available = [f for f in self.features if f in feat_dict]
                if len(available) >= n_expected:
                    feature_cols = available[:n_expected]
                else:
                    raise ValueError(
                        f"Prediction feature mismatch: only {len(available)} available, "
                        f"but scaler expects {n_expected}. Please retrain the model "
                        f"using the current feature set or provide all required features."
                    )

        try:
            df_row = pd.DataFrame([feat_dict])
            x = df_row[feature_cols].fillna(0.0).values
        except KeyError as e:
            missing = [c for c in feature_cols if c not in feat_dict]
            raise ValueError(
                "Missing required features for prediction: " + ", ".join(missing) +
                ". Consider retraining the model with the current code (data/simulate.py then train)."
            ) from e

        xs = self.scaler.transform(x)
        raw = self.model.decision_function(xs)[0]

        # convert raw anomaly score to 0..1
        score = 1.0 - ((raw - (-0.5)) / (0.5 - (-0.5)))
        score = max(0.0, min(1.0, score))
        label = "suspicious" if self.model.predict(xs)[0] == -1 else "normal"

        return {"score": float(score), "label": label, "raw": float(raw)}

    def save(self, path=MODEL_PATH):
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": list(self.features),
            },
            path,
        )

    def load(self, path=MODEL_PATH):
        obj = joblib.load(path)
        self.model = obj["model"]
        self.scaler = obj["scaler"]
        # Newer checkpoints include feature names; older ones won't.
        self.trained_feature_names = obj.get("feature_names", None)


def train_and_persist_model(feature_df, path=MODEL_PATH):
    """
    Helper to train a SharkGuard model and save it.
    """
    sg = SharkGuardModel()
    sg.train(feature_df)
    sg.save(path)
    return sg
