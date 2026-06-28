from google import genai
from google.genai import types
import os
import json

# Initialize the Gemini Client. It will load GEMINI_API_KEY from environment variables automatically.
client = genai.Client()

def generate_plan(diabetes: float, hypertension: float,
                  metabolic: float, obesity: float, age: int) -> dict:
    try:
        prompt = f"""
다음 건강 분석 결과를 가진 {age}세 사용자의 4주 실천 플랜과 주간 목표를 JSON으로만 반환해줘.
설명 텍스트, 마크다운 없이 JSON만.

분석 결과:
- 당뇨 위험도: {diabetes}%
- 고혈압 위험도: {hypertension}%
- 대사증후군 위험도: {metabolic}%
- 비만 위험도: {obesity}%

형식:
{{
  "plan": [
    {{
      "week": 1,
      "title": "주차 제목",
      "color": "#2563EB",
      "items": ["실천 항목1", "실천 항목2", "실천 항목3", "실천 항목4"]
    }}
  ],
  "weeklyGoals": {{
    "steps": 목표 걸음수 숫자,
    "exerciseMinutes": 하루 목표 운동시간 숫자
  }}
}}
4주차까지 작성. JSON만 반환.
"""
        response = client.models.generate_content(
            model='gemini-2.5-flash',
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
            "weeklyGoals": {"steps": 8000, "exerciseMinutes": 30}
        }

def generate_coach_reply(messages: list, user_context: str) -> dict:
    try:
        system_prompt = f"""너는 공공 건강 데이터 기반 AI 건강 코치야.
KNHANES 2022-2024, 질병관리청 데이터를 근거로 답변해.
의료 진단은 절대 하지 마.
답변은 친절하고 간결하게 3-4문장으로.
답변 마지막에 반드시 '출처: 질병관리청 국민건강영양조사' 형식으로 표기해.

사용자 건강 정보: {user_context}"""

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
            model="gemini-2.5-flash",
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
