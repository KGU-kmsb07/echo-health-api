"""
quest_service.py
- RAG(Retrieval-Augmented Generation) 기반 퀘스트 생성 서비스.
- 흐름: 위험요인 분석 → KDCA 콘텐츠 수신 → Gemini 생성 (grounding 없음)
- Gemini tools 파라미터 없음 → 웹 검색 완전 제거.
"""
import os
import json
from google import genai
from services.kdca_service import fetch_kdca_content, select_categories

client = genai.Client()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_quest_prompt(user_data: dict, predict_result: dict, kdca_content: dict) -> str:
    """config/quest_prompt.txt 로드 후 변수 치환."""
    path = os.path.join(BASE_DIR, "config/quest_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()

    kdca_text = "\n".join(
        f"[{k}] {v}" for k, v in kdca_content.items() if v
    )
    sex_label = "남성" if user_data.get("sex", 1) == 1 else "여성"

    return template\
        .replace("{age}", str(user_data.get("age", "")))\
        .replace("{sex}", sex_label)\
        .replace("{bmi}", str(predict_result.get("bmi", "")))\
        .replace("{obesity_status}", str(predict_result.get("obesity_status", "")))\
        .replace("{hypertension_prob}", str(predict_result.get("hypertension_prob", "")))\
        .replace("{diabetes_prob}", str(predict_result.get("diabetes_prob", "")))\
        .replace("{current_smoking}", str(user_data.get("current_smoking", 0)))\
        .replace("{aerobic_activity}", str(user_data.get("aerobic_activity", 1)))\
        .replace("{kdca_content}", kdca_text)

def _parse_quest_response(text: str) -> dict:
    """Gemini 응답에서 JSON만 추출."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

async def generate_quests(user_data: dict, predict_result: dict) -> dict:
    """
    1. predict_result에서 위험요인 분석
    2. KDCA 콘텐츠 수신 (실시간 시도 → fallback)
    3. Gemini에 KDCA 콘텐츠 + 사용자 데이터를 주입하여 퀘스트 생성
    4. 생성된 퀘스트를 반환
    """
    try:
        categories = select_categories(predict_result, user_data)
        kdca_content = await fetch_kdca_content(categories)

        prompt = _load_quest_prompt(user_data, predict_result, kdca_content)
        print(f"[Quest] Generating quests for categories: {categories}")

        # tools 파라미터 없음 → 웹 검색(grounding) 완전 제거
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return _parse_quest_response(response.text)
    except Exception as e:
        print(f"[Quest] Gemini 퀘스트 생성 실패, fallback 반환: {e}")
        return _default_quests()

def _default_quests() -> dict:
    """Gemini 실패 시 기본 퀘스트 fallback."""
    return {
        "sos_message": "오늘 하루도 건강한 한 걸음을 내딛어 보세요!",
        "quests": [
            {
                "id": 1,
                "title": "신체활동 늘리기",
                "todos": ["30분 걷기", "계단 이용하기", "식후 10분 산책"]
            },
            {
                "id": 2,
                "title": "식습관 개선",
                "todos": ["물 8잔 마시기", "채소 매끼 포함", "나트륨 줄이기"]
            },
            {
                "id": 3,
                "title": "수면 & 스트레스",
                "todos": ["취침 전 스트레칭", "7시간 수면 목표", "명상 5분"]
            },
            {
                "id": 4,
                "title": "금연 & 절주",
                "todos": ["흡연 충동 일지 작성", "니코틴 대체제 활용", "금연 앱 기록"]
            },
            {
                "id": 5,
                "title": "정기 체크",
                "todos": ["체중 측정 기록", "혈압 체크", "건강 일기 작성"]
            }
        ],
        "caution": "이 내용은 건강 행동 개선을 위한 참고 정보이며, 진단이나 치료를 대신하지 않습니다."
    }
