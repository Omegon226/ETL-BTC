from fastapi import APIRouter, HTTPException

from logic.make_analysis import make_sentiment_analysis

router = APIRouter(prefix="/make_analysis")


@router.post("/full/")
def forecast_with_arch(title: str, text: str):
    analysis_result = make_sentiment_analysis(title, text)
    return analysis_result.dict()
