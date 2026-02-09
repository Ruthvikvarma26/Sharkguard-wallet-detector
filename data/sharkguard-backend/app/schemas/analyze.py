from pydantic import BaseModel
from typing import Optional, List

class AnalyzeRequest(BaseModel):
    wallet: str
    etherscan_key: Optional[str] = None

class AnalyzeResponse(BaseModel):
    wallet: str
    balance_eth: float
    label: str
    score: float
    features: dict
    model: dict

class HealthResponse(BaseModel):
    status: str
    message: str