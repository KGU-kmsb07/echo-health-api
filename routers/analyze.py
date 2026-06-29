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
    # This endpoint is read-only and does not modify the user's permanent profile or DB.
    # It takes the virtual simulated input and runs inference through predict_service.py.
    try:
        return predict(data)
    except Exception as e:
        return {"error": str(e)}