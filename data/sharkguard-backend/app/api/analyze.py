from fastapi import APIRouter, HTTPException
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.analyzer import analyze_wallet

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        result = await analyze_wallet(request.wallet, request.etherscan_key)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))