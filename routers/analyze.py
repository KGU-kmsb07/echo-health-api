from fastapi import APIRouter

router = APIRouter()

@router.post("/analyze")
def analyze():
    return {"status": "준비 중입니다"}