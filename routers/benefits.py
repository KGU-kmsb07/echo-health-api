from fastapi import APIRouter

router = APIRouter()

@router.get("/benefits")
def benefits(region: str = "", age: int = 0, risks: str = ""):
    return {
        "status": "준비 중",
        "benefits": [],
        "message": "혜택 데이터는 준비 중입니다."
    }