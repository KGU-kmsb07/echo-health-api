from google import genai
from google.genai import types
import os
import json

# Initialize the Gemini Client with explicit API key.
# os.environ에서 직접 읽어 배포 환경(Cloud Run 등)에서도 안전하게 동작.
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일 또는 배포 환경변수를 확인하세요.")
client = genai.Client(api_key=_api_key)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_plan_prompt(age, diabetes, hypertension, metabolic, obesity, matched_kdca) -> str:
    path = os.path.join(BASE_DIR, "config/plan_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content.replace("{age}", str(age))\
                  .replace("{diabetes}", str(diabetes))\
                  .replace("{hypertension}", str(hypertension))\
                  .replace("{metabolic}", str(metabolic))\
                  .replace("{obesity}", str(obesity))\
                  .replace("{matched_kdca}", matched_kdca)

def _load_coach_prompt(user_context) -> str:
    path = os.path.join(BASE_DIR, "config/coach_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content.replace("{user_context}", str(user_context) if not isinstance(user_context, str) else user_context)

def _load_kdca_contents() -> dict:
    path = os.path.join(BASE_DIR, "config/kdca_contents.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_plan(diabetes: float, hypertension: float,
                  metabolic: float, obesity: float, age: int,
                  current_smoking: int = 0, aerobic_activity: int = 1) -> dict:
    try:
        kdca_content_map = _load_kdca_contents()

        h_pct = hypertension * 100 if hypertension <= 1.0 else hypertension
        d_pct = diabetes * 100 if diabetes <= 1.0 else diabetes
        o_pct = obesity * 100 if obesity <= 1.0 else obesity

        selected_contents = []
        if h_pct >= 35.0:
            selected_contents.append(kdca_content_map["hypertension"])
        if d_pct >= 35.0:
            selected_contents.append(kdca_content_map["diabetes"])
        if o_pct >= 35.0:
            selected_contents.append(kdca_content_map["obesity"])
        if current_smoking == 1:
            selected_contents.append(kdca_content_map["smoking"])
        if aerobic_activity == 0:
            selected_contents.append(kdca_content_map["activity"])

        if not selected_contents:
            selected_contents.append(kdca_content_map["activity"])

        matched_kdca = "\n".join(selected_contents)
        print(f"[KDCA Grounding LOG] Matched health contents:\n{matched_kdca}")

        prompt = _load_plan_prompt(
            age=age,
            diabetes=diabetes,
            hypertension=hypertension,
            metabolic=metabolic,
            obesity=obesity,
            matched_kdca=matched_kdca
        )

        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
        )
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        print(f"Gemini plan 실패: {e}")
        return {
            "plan": [
                {"week": 1, "title": "건강 기초 다지기", "color": "#2563EB",
                 "items": ["매일 30분 걷기", "물 하루 8잔", "취침 전 스트레칭", "식사 규칙적으로"]},
                {"week": 2, "title": "생활습관 강화", "color": "#7C3AED",
                 "items": ["유산소 운동 주 3회", "나트륨 줄이기", "수면 7시간 확보", "계단 이용"]},
                {"week": 3, "title": "집중 관리", "color": "#059669",
                 "items": ["근력 운동 추가", "채소 매 끼니 포함", "음주 줄이기", "스트레스 관리"]},
                {"week": 4, "title": "습관 정착", "color": "#D97706",
                 "items": ["운동 루틴 점검", "한 달 변화 기록", "재분석으로 확인", "다음 달 목표 설정"]}
            ],
            "weeklyGoals": {"steps": 1000, "exerciseMinutes": 30}
        }

def generate_coach_reply(messages: list, user_context: str) -> dict:
    try:
        system_prompt = _load_coach_prompt(user_context=user_context)

        history = []
        for msg in messages[:-1]:
            role = "user" if msg.get("role") == "user" else "model"
            history.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.get("text", ""))]
                )
            )

        chat = client.chats.create(
            model="gemini-3.1-flash-lite",
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        last_message = messages[-1].get("text", "") if messages else ""
        response = chat.send_message(last_message)

        return {
            "reply": response.text,
            "source": "출처: 질병관리청 국민건강영양조사"
        }
    except Exception as e:
        print(f"Gemini coach 실패: {e}")
        return {
            "reply": "현재 AI 코치 서비스에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.",
            "source": "서비스 점검 중"
        }
