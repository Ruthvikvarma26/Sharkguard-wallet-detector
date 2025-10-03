# api/main.py
"""
FastAPI backend for SharkGuard
- Exposes endpoints to analyze wallets and predict using precomputed features
- Designed to pair with a static frontend (e.g., Netlify) calling this API

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Deploy:
- Render/Heroku/Railway/Fly.io: set start command to
    uvicorn api.main:app --host 0.0.0.0 --port $PORT
- Provide ETHERSCAN_API_KEY as an environment variable if desired (optional)
"""

from typing import List, Optional, Dict, Any
from functools import lru_cache
import os
import time
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from fakeacc.core import (
    SharkGuardModel,
    txs_to_dataframe,
    extract_wallet_features,
    train_and_persist_model,
)
from utils.etherscan import fetch_transactions, fetch_account_balance
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "isolation_model.joblib"
DATA_DIR = Path("data")
SIM_FEATURES = DATA_DIR / "simulated_features.csv"

app = FastAPI(title="SharkGuard API", version="1.0.0")

# Configurable CORS: set FRONTEND_ORIGIN (comma-separated) in env for production
_cors_env = os.getenv("FRONTEND_ORIGIN", "").strip()
if _cors_env:
    _origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    _origins = ["*"]  # permissive for local/dev

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Basic request logging
logger = logging.getLogger("sharkguard")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# In-memory rate limit state: key -> count in current minute window
_RATE_STATE: Dict[str, int] = {}


@app.middleware("http")
async def security_and_logging_middleware(request: Request, call_next):
    # HTTPS enforcement
    enforce_https = os.getenv("ENFORCE_HTTPS", "false").lower() == "true"
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    if enforce_https and proto != "https":
        return JSONResponse({"detail": "HTTPS required"}, status_code=426)

    # Simple per-IP rate limiting (requests/min)
    try:
        limit = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
    except Exception:
        limit = 60
    xfwd = request.headers.get("x-forwarded-for", "")
    client_ip = (xfwd.split(",")[0].strip() if xfwd else None) or (request.client.host if request.client else "?")
    now = time.time()
    window = int(now // 60)
    key = f"{client_ip}:{window}"
    count = _RATE_STATE.get(key, 0)
    if count >= limit:
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
    _RATE_STATE[key] = count + 1

    # Logging + timing
    start = time.perf_counter()
    response = await call_next(request)
    dur_ms = (time.perf_counter() - start) * 1000.0
    logger.info(f"{client_ip} {request.method} {request.url.path} -> {response.status_code} in {dur_ms:.1f}ms")
    return response


class AnalyzeRequest(BaseModel):
    wallet: str = Field(..., description="Ethereum wallet address (0x...)")
    etherscan_key: Optional[str] = Field(
        None, description="Optional Etherscan API key; falls back to ENV ETHERSCAN_API_KEY"
    )
    # If provided, bypass fetch and use these txs directly
    transactions: Optional[List[Dict[str, Any]]] = None


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class FeatureRequest(BaseModel):
    wallet: Optional[str] = None
    etherscan_key: Optional[str] = None
    transactions: Optional[List[Dict[str, Any]]] = None


@lru_cache(maxsize=1)
def get_model() -> Optional[SharkGuardModel]:
    """Ensure model exists; auto-train from simulated data if needed."""
    try:
        if not MODEL_PATH.exists():
            if not SIM_FEATURES.exists():
                # Generate synthetic data
                import subprocess, sys
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                subprocess.run([sys.executable, "data/simulate.py"], check=True)
            df = pd.read_csv(SIM_FEATURES)
            train_and_persist_model(df, path=str(MODEL_PATH))
        sg = SharkGuardModel()
        sg.load(str(MODEL_PATH))
        return sg
    except Exception as e:
        # Return None; endpoints can still compute heuristics without ML
        print("Model initialization failed:", e)
        return None


@app.get("/health")
def health():
    sg = get_model()
    return {
        "status": "ok",
        "model_loaded": bool(sg is not None),
        "model_path": str(MODEL_PATH),
        "version": "1.0.0",
    }


def _resolve_etherscan_key(provided: Optional[str]) -> Optional[str]:
    if provided:
        return provided
    return os.getenv("ETHERSCAN_API_KEY")


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    key = _resolve_etherscan_key(req.etherscan_key)

    # Transactions source: explicit list > fetch via etherscan > none
    txs: List[Dict[str, Any]] = []
    balance: float = 0.0

    if req.transactions:
        txs = req.transactions
    elif key:
        try:
            txs = fetch_transactions(req.wallet, key, sort="desc", offset=10000)
            balance = fetch_account_balance(req.wallet, key)
        except Exception as e:
            # Keep going with empty txs
            print("Etherscan fetch failed:", e)
            txs = []
    else:
        # No key => remain empty; caller can still use /predict with custom features
        pass

    # Feature extraction
    df = txs_to_dataframe(txs)
    feat = extract_wallet_features(df, req.wallet)

    result: Dict[str, Any] = {
        "wallet": req.wallet,
        "balance_eth": balance,
        "features": feat,
        "model": None,
    }

    sg = get_model()
    if sg is not None:
        try:
            pred = sg.predict_score(feat)
            result["model"] = pred
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return result


@app.post("/predict")
def predict(req: PredictRequest):
    sg = get_model()
    if sg is None:
        raise HTTPException(status_code=503, detail="Model not available")
    try:
        pred = sg.predict_score(req.features)
        return {"result": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid features: {e}")


@app.post("/features")
def features(req: FeatureRequest):
    """
    Compute features from either provided transactions or fetched transactions.
    Provide either transactions OR wallet(+optional key).
    """
    if req.transactions:
        df = txs_to_dataframe(req.transactions)
        wallet = req.wallet or ""
    elif req.wallet:
        key = _resolve_etherscan_key(req.etherscan_key)
        txs: List[Dict[str, Any]] = []
        if key:
            try:
                txs = fetch_transactions(req.wallet, key, sort="desc", offset=10000)
            except Exception as e:
                print("Etherscan fetch failed:", e)
                txs = []
        df = txs_to_dataframe(txs)
        wallet = req.wallet
    else:
        raise HTTPException(status_code=400, detail="Provide transactions or wallet")

    feat = extract_wallet_features(df, wallet)
    return {"features": feat}


# Local dev entry
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
