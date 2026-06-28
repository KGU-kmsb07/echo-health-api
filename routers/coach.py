from fastapi import APIRouter
from services.gemini_service import generate_coach_reply

router = APIRouter()

@router.post("/coach")
def coach(data: dict):
    messages = data.get("messages", [])
    user_context = data.get("userContext", "")
    if not messages:
        return {
            "reply": "안녕하세요! 건강에 대해 궁금한 점을 물어보세요.",
            "source": "출처: 질병관리청 국민건강영양조사"
        }
    return generate_coach_reply(messages, user_context)