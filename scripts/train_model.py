"""
Pre-train the IsolationForest model and save to models/isolation_model.joblib

Usage (Windows PowerShell/CMD):
    python scripts/train_model.py
"""
from pathlib import Path
import subprocess
import sys
import pandas as pd

from fakeacc.core import train_and_persist_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
CSV = DATA / "simulated_features.csv"
MODEL_PATH = MODELS / "isolation_model.joblib"

if __name__ == "__main__":
    MODELS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    if not CSV.exists():
        print("[info] Simulated features not found; generating via data/simulate.py ...")
        subprocess.run([sys.executable, str(DATA / "simulate.py")], check=True)

    print("[info] Loading training data ...", CSV)
    df = pd.read_csv(CSV)
    print("[info] Training model and saving to", MODEL_PATH)
    train_and_persist_model(df, path=str(MODEL_PATH))
    print("[done] Model saved at:", MODEL_PATH)
