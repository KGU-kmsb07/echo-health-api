import json
from services.gemini_service import client
from services.kdca_service import select_categories, fetch_kdca_content

async def generate_quests(user_data: dict, predict_result: dict) -> dict:
    """
    user_data 및 predict_result를 바탕으로 질병관리청(KDCA) 가이드를 RAG로 연동하여 
    Gemini를 통해 맞춤형 건강 퀘스트(5개)를 생성합니다. (웹 검색 도구는 완전히 배제함)
    """
    categories = select_categories(predict_result, user_data)
    kdca_content_dict = await fetch_kdca_content(categories)
    
    # 질병청 정보를 줄바꿈으로 구성
    kdca_content_str = "\n".join([f"- {cat}: {val}" for cat, val in kdca_content_dict.items()])
    
    sex_str = "남성" if user_data.get('sex', 1) == 1 else "여성"
    
    prompt = f"""
너는 예방 중심 헬스케어 코치다.
건강 조언은 반드시 아래 질병관리청 국가건강정보포털 내용을 기반으로 작성한다.
질병을 진단하거나 치료를 지시하는 표현은 사용하지 않는다.
사용자가 오늘 실천할 수 있는 작고 구체적인 행동으로 제안한다.

[사용자 정보]
나이: {user_data.get('age', 30)}세, 성별: {sex_str}
BMI: {predict_result.get('bmi', 22.0)}, 비만여부: {predict_result.get('obesity_status', 0)}
고혈압 위험: {predict_result.get('hypertension_prob', 0.2)}
당뇨 위험: {predict_result.get('diabetes_prob', 0.1)}
흡연: {user_data.get('current_smoking', 0)}, 운동: {user_data.get('aerobic_activity', 1)}

[질병관리청 건강정보]
{kdca_content_str}

위 정보를 바탕으로 아래 JSON 형식으로만 응답하라. 설명 텍스트나 마크다운 블록 없이 JSON만 반환하라.
{{
  "sos_message": "1문장 동기부여 메시지",
  "quests": [
    {{"id": 1, "title": "퀘스트 제목", "todos": ["todo1", "todo2", "todo3"]}},
    {{"id": 2, "title": "퀘스트 제목", "todos": ["todo1", "todo2", "todo3"]}},
    {{"id": 3, "title": "퀘스트 제목", "todos": ["todo1", "todo2", "todo3"]}},
    {{"id": 4, "title": "퀘스트 제목", "todos": ["todo1", "todo2", "todo3"]}},
    {{"id": 5, "title": "퀘스트 제목", "todos": ["todo1", "todo2", "todo3"]}}
  ],
  "caution": "이 내용은 건강 행동 개선을 위한 참고 정보이며, 진단이나 치료를 대신하지 않습니다."
}}
"""

    try:
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
        print(f"Gemini quest 생성 실패: {e}")
        # API 통신 장애 등 예외 상황 시 견고한 기본 퀘스트 제공
        return {
            "sos_message": "매일 작은 습관이 당신의 건강 수명을 늘립니다. 함께 시작해봐요!",
            "quests": [
                {"id": 1, "title": "하루 걷기 습관", "todos": ["가벼운 운동화 착용하기", "점심 식사 후 10분 산책", "엘리베이터 대신 계단 이용"]},
                {"id": 2, "title": "규칙적인 수분 섭취", "todos": ["기상 후 미지근한 물 한 잔", "일할 때 텀블러 옆에 두기", "커피 대신 물로 수분 보충"]},
                {"id": 3, "title": "나트륨 줄인 식사", "todos": ["국물 요리 시 건더기 위주로 먹기", "가공식품 섭취 줄이기", "음식에 소금 대신 천연 조미료 사용"]},
                {"id": 4, "title": "금연 및 절주 실천", "todos": ["흡연 욕구 발생 시 껌이나 물 이용", "주변에 금연 선언하기", "술자리 횟수 절반으로 줄이기"]},
                {"id": 5, "title": "스트레칭으로 몸 풀기", "todos": ["기상 직후 가벼운 기지개", "의자에 오래 앉아있을 때 목/어깨 스트레칭", "취침 10분 전 전신 스트레칭"]}
            ],
            "caution": "이 내용은 건강 행동 개선을 위한 참고 정보이며, 진단이나 치료를 대신하지 않습니다."
        }
