# 🦈 SharkGuard - Web3 Fake Account Detector

SharkGuard is a machine learning-powered tool for detecting suspicious wallet addresses in the Ethereum ecosystem. It analyzes transaction patterns and behavioral features to identify potentially fake or bot-controlled accounts.

## Features

- **Web Interface**: User-friendly Streamlit web application
- **Command Line Interface**: CLI tool for batch analysis
- **Machine Learning Model**: Isolation Forest-based anomaly detection
- **Etherscan Integration**: Real-time transaction fetching
- **Feature Extraction**: Comprehensive wallet behavior analysis
- **Simulated Data**: Built-in data generation for testing

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data (Optional)

```bash
python data/simulate.py
```

### 3. Train the Model

The model will be automatically trained when you first use the web interface, or you can train it manually:

```python
from fakeacc.core import train_and_persist_model
import pandas as pd

# Load simulated features
df = pd.read_csv("data/simulated_features.csv")
train_and_persist_model(df, path="models/isolation_model.joblib")
```

### 4. Run the Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 5. Use the Command Line Interface

```bash
# Analyze a wallet without Etherscan API
python cli.py --wallet 0x1234567890abcdef1234567890abcdef12345678

# Analyze with Etherscan API (requires API key)
python cli.py --wallet 0x1234567890abcdef1234567890abcdef12345678 --etherscan_key YOUR_API_KEY
```

## Project Structure

```
fake_acc/
├── app.py                          # Streamlit web application
├── cli.py                          # Command line interface
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── fakeacc/                        # Core package
│   ├── __init__.py
│   └── core.py                     # Main ML logic and feature extraction
├── utils/                          # Utility modules
│   ├── __init__.py
│   └── etherscan.py                # Etherscan API integration
├── data/                           # Data and simulation
│   ├── simulate.py                 # Generate synthetic training data
│   └── simulated_features.csv      # Generated training features
└── models/                         # Trained models
    └── isolation_model.joblib      # Pre-trained model
```

## How It Works

### Feature Extraction

SharkGuard analyzes wallets based on these behavioral features:

- **Transaction Count**: Total number of transactions
- **Transaction Frequency**: Transactions per day
- **Lifetime**: Account age in days
- **Gas Usage**: Average gas consumption patterns
- **Value Patterns**: Average transaction values
- **Counterparty Diversity**: Number of unique addresses interacted with
- **Repetition Ratio**: Frequency of interactions with same addresses
- **Timing Patterns**: Entropy of transaction hour distribution

### Machine Learning Model

- **Algorithm**: Isolation Forest (unsupervised anomaly detection)
- **Training**: Uses synthetic data with mixed normal/suspicious patterns
- **Output**: Suspicion score (0-1) and binary classification
- **Interpretation**: Higher scores indicate more suspicious behavior

### Detection Signals

The system flags accounts with:
- Very few transactions (new/dormant accounts)
- Extremely high transaction frequency (bot-like behavior)
- High repetition ratios (interacting with same counterparties)
- Low timing entropy (very regular transaction patterns)

## Web Interface Usage

1. **Load Model**: The app automatically loads the pre-trained model
2. **Enter Wallet**: Input a wallet address (0x...)
3. **API Key**: Optionally provide an Etherscan API key for real data
4. **Analyze**: Click "Analyze" to get results
5. **Results**: View suspicion score, features, and explanations

## API Integration

### Etherscan API

To fetch real transaction data, you need an Etherscan API key:

1. Visit [Etherscan.io](https://etherscan.io/apis)
2. Create a free account
3. Generate an API key
4. Use the key in the web interface or CLI

### Rate Limits

- Free Etherscan API: 5 calls/second, 100,000 calls/day
- The app handles rate limiting automatically

## Advanced Usage

### Custom Model Training

```python
from fakeacc.core import SharkGuardModel, train_and_persist_model
import pandas as pd

# Load your own training data
df = pd.read_csv("your_training_data.csv")

# Train and save model
train_and_persist_model(df, path="models/custom_model.joblib")
```

### Batch Analysis

```python
from fakeacc.core import SharkGuardModel, extract_wallet_features, txs_to_dataframe
from utils.etherscan import fetch_transactions

# Load model
sg = SharkGuardModel()
sg.load("models/isolation_model.joblib")

# Analyze multiple wallets
wallets = ["0x...", "0x...", "0x..."]
for wallet in wallets:
    txs = fetch_transactions(wallet, "YOUR_API_KEY")
    df = txs_to_dataframe(txs)
    features = extract_wallet_features(df, wallet)
    result = sg.predict_score(features)
    print(f"{wallet}: {result['score']:.3f} ({result['label']})")
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure all `__init__.py` files exist
2. **Model Loading Errors**: Train the model first using simulated data
3. **Etherscan API Errors**: Check your API key and rate limits
4. **Dependency Issues**: Use `pip install -r requirements.txt`

### Performance Tips

- Use simulated data for testing (faster)
- Cache model loading in production
- Batch API calls to respect rate limits
- Monitor memory usage with large datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Open an issue on GitHub

---

**Note**: This tool is for educational and research purposes. Always verify results with additional analysis before making important decisions based on the output.

## Deploying on Netlify (Static UI) + FastAPI (Backend)

SharkGuard includes a FastAPI backend (`api/main.py`) and a minimal static frontend (`web/index.html`) so you can host the UI on Netlify and the API on a Python-friendly platform (Render/Heroku/Railway/Fly.io).

### Backend (FastAPI)

Files:
- `api/main.py` — FastAPI app exposing `/health`, `/analyze`, `/predict`, `/features`
- `api/requirements.txt` — Backend dependencies
- `Procfile` — For platforms like Heroku/Render:
  - `web: uvicorn api.main:app --host 0.0.0.0 --port $PORT`

Run locally:

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Render (example):
- Create a Web Service from your repo
- Build command: `pip install -r api/requirements.txt`
- Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Set env var (optional): `ETHERSCAN_API_KEY`

Heroku (example):
- `heroku create`
- `heroku buildpacks:add heroku/python`
- `heroku config:set PORT=8000` (Heroku sets this automatically; uvicorn uses `$PORT`)
- `git push heroku main`

Health check:

```bash
curl https://your-backend.example.com/health
```

### Frontend (Netlify)

Files:
- `web/index.html` — Static UI that calls the API

Deploy options:
- Drag-and-drop the `web/` folder in Netlify UI; or
- Connect Git repo and set `Base directory` to `web` (no build command required)

Usage:
1. Open the Netlify site URL
2. Enter `Backend API URL` (e.g., `https://your-backend.example.com`)
3. (Optional) Paste your Etherscan API key
4. Enter a wallet address and click `Analyze`

Security & CORS:
- The backend enables permissive CORS (`*`) by default. For production, restrict `allow_origins` in `api/main.py` to your Netlify domain.
- Prefer storing secrets server-side (e.g., `ETHERSCAN_API_KEY` env var on the backend). Leave the UI key empty to rely on the backend secret.

### API Reference (summary)

- `GET /health` → `{ status, model_loaded, model_path, version }`
- `POST /analyze` → body `{ wallet, etherscan_key? }` → `{ wallet, balance_eth, features, model? }`
- `POST /features` → body `{ wallet, etherscan_key? }` or `{ transactions: [...] }` → `{ features }`
- `POST /predict` → body `{ features: {...} }` → `{ result }`

### Notes

- The backend will auto-generate training data and a model on first run if `models/isolation_model.joblib` is missing.
- To speed up cold start, you can pre-train locally and commit `models/isolation_model.joblib`.
- The original Streamlit app (`app.py`) remains available for local use.

#  SharkGuard — Web3 Wallet Risk Analyzer (Comprehensive README)

SharkGuard detects suspicious or fake Ethereum wallets using machine learning (IsolationForest) and on‑chain heuristics. It offers:
- A static web UI (Netlify-ready)
- A FastAPI backend (Render/Heroku/Railway/Docker)
- CLI and Streamlit app for local use

## Features

- ML anomaly detection (IsolationForest) for wallet “suspicion score” (0–1)
- Behavioral features: tx frequency, lifetime, timing entropy, counterparty diversity, gas/value patterns, contract interaction ratio
- Heuristic overlay with human‑readable flags and recommendations
- Etherscan integration (real data) or simulated/demo mode
- Netlify + proxy to a fixed backend (no user URL input)
- Pre‑trained model to avoid cold start delays

## Project Structure

```
fake_acc/
├─ api/                     # FastAPI backend
│  ├─ main.py
│  └─ requirements.txt
├─ data/
│  ├─ simulate.py
│  └─ simulated_features.csv
├─ fakeacc/                 # Core logic
│  ├─ __init__.py
│  ├─ core.py               # Features + IsolationForest train/predict
│  └─ heuristics.py         # Heuristic risk indicators
├─ models/
│  └─ isolation_model.joblib
├─ scripts/
│  └─ train_model.py        # Pre-train model helper
├─ utils/
│  └─ etherscan.py
├─ web/
│  └─ index.html            # Netlify-ready static UI (uses /api proxy)
├─ netlify.toml             # Publishes web/ + /api proxy to backend
├─ render.yaml              # One-click Render backend
├─ Procfile                 # Heroku/Render start command
├─ Dockerfile               # Container for backend
├─ app.py                   # Streamlit app (local)
├─ cli.py                   # CLI analyzer (local)
└─ README.md
```

## Quick Start (Local)

- **Backend**
  - `pip install -r api/requirements.txt`
  - `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
  - Health: `http://127.0.0.1:8000/health`

- **Frontend**
  - Open `web/index.html`
  - Backend API URL: `http://127.0.0.1:8000`
  - Click “Check Health”, then enter a wallet and “Analyze”

- **Pre-train model** (optional, speeds up startup)
  - `python scripts/train_model.py`
  - Writes `models/isolation_model.joblib`

## Deploy (Netlify + Render)

- **Backend (Render)**
  - Connect repo in Render; it detects `render.yaml`
  - Build: `pip install -r api/requirements.txt`
  - Start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
  - Copy backend URL (e.g., `https://your-backend.onrender.com`)
  - Environment variables (Render → Environment):
    - `FRONTEND_ORIGIN=https://YOUR-SITE.netlify.app`
    - `ENFORCE_HTTPS=true`
    - `RATE_LIMIT_PER_MIN=60`
    - `ETHERSCAN_API_KEY=your_key_here` (optional)
  - Verify: `https://your-backend.onrender.com/health`

- **Frontend (Netlify)**
  - `netlify.toml` should proxy `/api/*` to backend:
    ```toml
    [[redirects]]
      from = "/api/*"
      to = "https://your-backend.onrender.com/:splat"
      status = 200
      force = true
    ```
  - In `web/index.html` set:
    ```js
    const BACKEND_URL = "/api";
    ```
  - Deploy on Netlify → open your site → “Check Health”, then Analyze.

## Environment Variables (Backend)

- `FRONTEND_ORIGIN` → Required for production CORS (e.g., `https://YOUR-SITE.netlify.app`)
- `ENFORCE_HTTPS=true` → Forces HTTPS
- `RATE_LIMIT_PER_MIN=60` → Per‑IP request limit
- `ETHERSCAN_API_KEY=your_key` → Optional; use server-side secret

## API Reference

- `GET /health` → `{ status, model_loaded, model_path, version }`
- `POST /analyze` → Body `{ wallet, etherscan_key? }` → `{ wallet, balance_eth, features, model? }`
- `POST /features` → Body `{ wallet, etherscan_key? }` or `{ transactions: [...] }` → `{ features }`
- `POST /predict` → Body `{ features: {...} }` → `{ result }`

## CLI Usage

```
python cli.py --wallet 0x1234567890abcdef1234567890abcdef1234567890
python cli.py --wallet 0x... --etherscan_key YOUR_API_KEY
```

## Streamlit App (Local)

```
streamlit run app.py
```

## Security & Ops

- CORS via `FRONTEND_ORIGIN` in `api/main.py`
- HTTPS via `ENFORCE_HTTPS=true`
- Rate-limiting via `RATE_LIMIT_PER_MIN`
- Prefer backend env secrets for `ETHERSCAN_API_KEY`

## Customization

- Branding and hero copy in `web/index.html`
- Background animation (constellation) can be swapped on request
- Heuristics in `fakeacc/heuristics.py`
- Re-train model with `scripts/train_model.py`

## Troubleshooting

- Health fails → backend not deployed or CORS/HTTPS misconfigured
- Rate limited → adjust `RATE_LIMIT_PER_MIN`
- Etherscan API errors → check key and rate limits

## License

Open source; see LICENSE.

---

Note: This tool provides heuristic and anomaly‑based signals; always pair with additional due diligence.
