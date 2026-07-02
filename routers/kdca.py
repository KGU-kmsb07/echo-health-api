from fastapi import APIRouter, Query
from services.kdca_service import fetch_kdca_content

router = APIRouter()

@router.get("/kdca/content")
async def get_kdca_content(
    category: str = Query("general", description="KDCA category name or general"),
    query: str = Query("", description="Search keyword such as anemia"),
):
    """
    지정된 카테고리에 해당하는 질병관리청 국가건강정보포털 가이드를 조회합니다.
    """
    search_query = query or category
    content_dict = await fetch_kdca_content([category], query=search_query)
    content = content_dict.get(category, "")
    return {
        "status": "success" if content else "error",
        "category": category,
        "query": search_query,
        "source": "질병관리청 국가건강정보포털 API",
        "content": content,
        "message": "" if content else "질병관리청 국가건강정보포털 API 응답을 받지 못했습니다."
    }
