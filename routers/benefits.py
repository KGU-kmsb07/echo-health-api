from fastapi import APIRouter
import os
import json

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@router.get("/benefits")
def benefits(region: str = "", age: int = 0, risks: str = ""):
    try:
        benefits_path = os.path.join(BASE_DIR, "config/benefits.json")
        with open(benefits_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if region:
            filtered = [b for b in data if region in b.get("region", "") or b.get("region", "") in region]
            return {
                "status": "성공",
                "benefits": filtered,
                "message": f"{region} 지역의 혜택 데이터입니다."
            }
            
        return {
            "status": "성공",
            "benefits": data,
            "message": "전체 혜택 데이터입니다."
        }
    except Exception as e:
        return {
            "status": "에러",
            "benefits": [],
            "message": f"혜택 데이터를 불러오는 중 오류가 발생했습니다: {str(e)}"
        }