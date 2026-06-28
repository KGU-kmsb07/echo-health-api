from fastapi import APIRouter
from services.predict_service import predict

router = APIRouter()

@router.post("/analyze")
def analyze(data: dict):
    try:
        return predict(data)
    except Exception as e:
        return {"error": str(e)}

@router.post("/simulate")
def simulate(data: dict):
    try:
        return predict(data)
    except Exception as e:
        return {"error": str(e)}