from fastapi import APIRouter

router = APIRouter()

@router.get("/plan")
def plan():
    return {"status": "준비 중입니다"}