from fastapi import APIRouter, Query
from services.kdca_service import fetch_kdca_content

router = APIRouter()

@router.get("/kdca/content")
async def get_kdca_content(category: str = Query(..., description="KDCA category name (e.g. hypertension, diabetes, obesity, smoking, activity)")):
    """
    지정된 카테고리에 해당하는 질병관리청 국가건강정보포털 가이드를 조회합니다.
    """
    content_dict = await fetch_kdca_content([category])
    return {
        "category": category,
        "content": content_dict.get(category, "관련 정보를 찾을 수 없습니다.")
    }
