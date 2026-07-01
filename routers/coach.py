from fastapi import APIRouter
from services.gemini_service import generate_coach_reply

router = APIRouter()

@router.post("/coach")
def coach(data: dict):
    messages = data.get("messages", [])
    user_context = data.get("userContext", "")
    if not messages:
        return {
            "reply": "안녕하세요. 건강정보나 건강 관련 복지정보에 대해 물어보세요.",
            "source": "응답 범위: 건강정보 및 건강 복지정보"
        }
    return generate_coach_reply(messages, user_context)
