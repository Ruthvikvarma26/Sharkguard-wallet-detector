import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_analyze_valid_wallet():
    response = client.post("/analyze", json={"wallet": "0xValidWalletAddress"})
    assert response.status_code == 200
    data = response.json()
    assert "wallet" in data
    assert "balance_eth" in data
    assert "model" in data

def test_analyze_invalid_wallet():
    response = client.post("/analyze", json={"wallet": "0xInvalidWalletAddress"})
    assert response.status_code == 400
    assert "detail" in response.json()

def test_analyze_missing_wallet():
    response = client.post("/analyze", json={})
    assert response.status_code == 422
    assert "detail" in response.json()