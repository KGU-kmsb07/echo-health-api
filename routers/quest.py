from fastapi import APIRouter
from pydantic import BaseModel
from services.quest_service import generate_quests

router = APIRouter()

class QuestRequest(BaseModel):
    user_data: dict
    predict_result: dict

@router.post("/quest/generate")
async def quest_generate(data: QuestRequest):
    """
    사용자의 건강 지표 및 위험도 예측 결과를 전달받아 개인 맞춤형 건강 실천 퀘스트(5종)를 생성합니다.
    """
    result = await generate_quests(data.user_data, data.predict_result)
    return result
