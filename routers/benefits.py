from fastapi import APIRouter

router = APIRouter()

@router.get("/benefits")
def benefits():
    return {"status": "준비 중입니다"}