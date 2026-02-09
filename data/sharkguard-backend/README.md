# SharkGuard Backend

SharkGuard is a Web3 Fake Account Detector that analyzes wallet addresses based on on-chain behavior and anomaly detection.

## Project Structure

```
sharkguard-backend
├── app
│   ├── main.py               # Entry point of the application
│   ├── api                   # API routes
│   │   ├── __init__.py       # Marks the api directory as a package
│   │   ├── routes.py         # Main API routes
│   │   ├── health.py         # Health check endpoint
│   │   └── analyze.py        # Wallet analysis endpoint
│   ├── core                  # Core application settings
│   │   ├── config.py         # Configuration settings
│   │   └── logging.py        # Logging setup
│   ├── services              # Business logic and external API interactions
│   │   ├── analyzer.py       # Wallet analysis logic
│   │   └── etherscan_client.py # Etherscan API interactions
│   ├── models                # Machine learning models
│   │   └── ml_model.py       # ML model definition
│   ├── schemas               # Request and response validation
│   │   └── analyze.py        # Pydantic schemas for analysis
│   └── utils                 # Utility functions
│       └── helpers.py        # Helper functions
├── tests                     # Unit tests
│   ├── test_health.py        # Tests for health check endpoint
│   └── test_analyze.py       # Tests for analyze endpoint
├── Dockerfile                # Docker image instructions
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project dependencies and configurations
├── .env.example              # Example environment variables
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/sharkguard-backend.git
   cd sharkguard-backend
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Copy `.env.example` to `.env` and fill in the required values.

5. **Run the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

## Usage

- **Health Check Endpoint:**  
  Access the health check at `http://localhost:8000/health` to verify that the backend is running.

- **Analyze Wallet Endpoint:**  
  Use the analyze endpoint at `http://localhost:8000/analyze` to analyze wallet addresses. Send a POST request with the required data.

## Testing

Run the tests using:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.