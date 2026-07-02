from google import genai
from google.genai import types
import os
import json
import re
import html
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from routers.benefits import _filter_benefits

# Initialize the Gemini Client with explicit API key.
# os.environ에서 직접 읽어 배포 환경(Cloud Run 등)에서도 안전하게 동작.
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일 또는 배포 환경변수를 확인하세요.")
client = genai.Client(api_key=_api_key)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEALTH_WORDS = [
    "암", "간암", "위암", "폐암", "대장암", "유방암", "빈혈", "감기", "천식", "간염", "검진", "예방",
    "건강", "질병", "증상", "운동", "식단", "영양", "비만", "체중", "혈압", "고혈압",
    "혈당", "당뇨", "흡연", "금연", "음주", "수면", "스트레스", "검진", "예방",
    "콜레스테롤", "허리둘레", "bmi", "BMI", "심혈관", "정신건강", "우울", "불안",
    "질환", "질병", "병", "예방법", "예방정보", "치료", "원인"
]
WELFARE_WORDS = [
    "복지", "혜택", "지원", "지원금", "서비스", "보조", "신청", "대상", "자격",
    "건강검진", "보건소", "금연클리닉", "상담", "의료비", "국가건강검진"
]
BLOCKED_REPLY = (
    "죄송하지만 저는 건강정보와 건강 관련 복지정보만 답변할 수 있습니다. "
    "건강 습관, 질병 예방, 검진, 금연, 운동, 식단, 또는 받을 수 있는 건강 복지 혜택을 질문해 주세요."
)


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    return str(messages[-1].get("text", "") or "").strip()


def _is_health_question(text: str) -> bool:
    stripped = text.strip()
    if any(word in stripped for word in HEALTH_WORDS):
        return True
    return bool(re.search(r"[가-힣A-Za-z0-9]{2,}(이|가)?\s*(뭐야|무엇|원인|증상|예방|관리)", stripped))


def _is_welfare_question(text: str) -> bool:
    return any(word in text for word in WELFARE_WORDS)


def _extract_context_value(user_context: Any, names: list[str], default: Any = "") -> Any:
    if not isinstance(user_context, dict):
        return default
    for name in names:
        value = user_context.get(name)
        if value not in (None, ""):
            return value
    return default


def _extract_age(user_context: Any) -> int | None:
    value = _extract_context_value(user_context, ["나이", "age", "Age"], "")
    match = re.search(r"\d{1,3}", str(value))
    return int(match.group(0)) if match else None


def _risk_list_from_context(user_context: Any) -> list[str]:
    risks: list[str] = []
    if not isinstance(user_context, dict):
        return risks
    risk_map = [
        ("diabetes", ["당뇨위험도", "당뇨 위험도", "diabetes"]),
        ("hypertension", ["고혈압위험도", "고혈압 위험도", "hypertension"]),
        ("obesity", ["BMI", "bmi", "비만"]),
    ]
    for risk, keys in risk_map:
        raw = _extract_context_value(user_context, keys, "")
        numbers = re.findall(r"\d+(?:\.\d+)?", str(raw))
        score = float(numbers[0]) if numbers else None
        if risk == "obesity":
            if score is not None and score >= 25:
                risks.append(risk)
        elif score is not None and score >= 35:
            risks.append(risk)
    smoking = str(_extract_context_value(user_context, ["흡연", "smoking"], ""))
    if "흡연" in smoking or "smok" in smoking.lower():
        risks.append("smoking")
    return risks


def _keyword_from_question(text: str) -> str:
    priority = ["금연", "건강검진", "검진", "운동", "비만", "고혈압", "당뇨", "정신건강", "보건소", "의료비"]
    return " ".join([word for word in priority if word in text])


def _load_health_fallback() -> dict:
    path = os.path.join(BASE_DIR, "config", "kdca_contents.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_health_categories(question: str, user_context: Any) -> list[str]:
    categories: list[str] = []
    keyword_map = {
        "hypertension": ["고혈압", "혈압"],
        "diabetes": ["당뇨", "혈당"],
        "obesity": ["비만", "체중", "BMI", "bmi"],
        "smoking": ["흡연", "금연"],
        "activity": ["운동", "신체활동", "걷기"],
        "cancer": ["암", "간암", "위암", "폐암", "대장암", "유방암", "검진"],
        "anemia": ["빈혈"],
    }
    for category, words in keyword_map.items():
        if any(word in question for word in words):
            categories.append(category)
    risks = _risk_list_from_context(user_context)
    if "hypertension" in risks:
        categories.append("hypertension")
    if "diabetes" in risks:
        categories.append("diabetes")
    if "obesity" in risks:
        categories.append("obesity")
    if "smoking" in risks:
        categories.append("smoking")
    return list(dict.fromkeys(categories or ["general"]))


def _health_intent(question: str) -> str:
    text = question or ""
    if any(word in text for word in ["예방", "예방법", "막으", "낮추", "줄이", "관리", "생활습관"]):
        return "prevention"
    if any(word in text for word in ["증상", "아플", "통증", "징후"]):
        return "symptom"
    if any(word in text for word in ["치료", "수술", "약", "검사", "진단"]):
        return "care"
    if any(word in text for word in ["뭐야", "무엇", "정의", "개요", "뜻"]):
        return "definition"
    return "definition"


def _extract_health_keyword(question: str, category: str) -> str:
    text = re.sub(r"\s+", " ", question or "").strip()
    known_terms = [
        "간암", "위암", "폐암", "대장암", "유방암", "빈혈", "고혈압", "당뇨", "비만",
        "직장탈출증", "천식", "감기", "간염", "심혈관질환", "우울증", "불안"
    ]
    for term in known_terms:
        if term in text:
            return term
    text = re.sub(r"(에\s*대해|에\s*관해|관련|예방정보|예방 정보|예방법|예방|정보|정의|개요|뜻)", " ", text)
    text = re.sub(r"(이|가|은|는|을|를)?\s*(뭐야|무엇인가요|무엇|알려줘|찾아줘|설명해줘|낮추려면|줄이려면|관리하려면)", " ", text)
    text = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
    words = [word for word in text.split() if len(word) >= 2]
    if words:
        return words[0]
    if category == "anemia":
        return "빈혈"
    return "" if category == "general" else category


def _health_search_query(question: str, category: str) -> str:
    return _extract_health_keyword(question, category) or category


def _api_category(category: str) -> str:
    allowed = {"hypertension", "diabetes", "obesity", "smoking", "activity", "cancer"}
    return category if category in allowed else ""


def _xml_text(element: ET.Element, path: str) -> str:
    node = element.find(path)
    return html.unescape("".join(node.itertext()).strip()) if node is not None else ""


def _extract_kdca_sections(payload: str, intent: str) -> str:
    text = (payload or "").strip()
    if not text:
        return ""

    if text.startswith("{") or text.startswith("["):
        try:
            return json.dumps(json.loads(text), ensure_ascii=False)[:1600]
        except Exception:
            return text[:1600]

    try:
        root = ET.fromstring(f"<root>{text}</root>")
    except Exception:
        return re.sub(r"\s+", " ", text)[:1600]

    title = _xml_text(root, ".//CNTNTSSJ")
    sections = []
    for item in root.findall(".//cntntsCl"):
        name = _xml_text(item, "CNTNTS_CL_NM")
        content = _xml_text(item, "CNTNTS_CL_CN")
        if not content or content.startswith("http"):
            continue
        sections.append({"name": name, "content": re.sub(r"\s+", " ", content)})

    if not sections:
        return title

    preferences = {
        "definition": ["개요-정의", "정의", "개요"],
        "prevention": ["예방", "위험요인", "생활", "관리", "자가관리"],
        "symptom": ["증상", "진단"],
        "care": ["치료", "검사", "진단", "수술"],
    }.get(intent, ["개요-정의", "정의", "개요"])

    selected = []
    for preferred in preferences:
        for section in sections:
            if preferred in section["name"] and section not in selected:
                selected.append(section)
        if selected:
            break

    if not selected:
        selected = sections[:2]

    lines = [f"질환명: {title}"] if title else []
    lines.extend([f"{section['name']}: {section['content']}" for section in selected[:3]])
    return "\n".join(lines)[:1800]


def _fetch_kdca_health_context(question: str, user_context: Any) -> tuple[str, str]:
    categories = _select_health_categories(question, user_context)
    intent = _health_intent(question)
    kdca_value = os.environ.get("KDCA", "").strip()
    kdca_base_url = os.environ.get("KDCA_API_BASE", "").strip()
    api_key = os.environ.get("KDCA_API_KEY", "").strip()

    if kdca_value.startswith("http"):
        kdca_base_url = kdca_value
    elif kdca_value:
        api_key = kdca_value

    results: list[str] = []
    source = "질병관리청 국가건강정보포털"
    for category in categories:
        text = ""
        if kdca_base_url:
            try:
                query = _health_search_query(question, category)
                params = {
                    "query": query,
                    "q": query,
                    "keyword": query,
                    "searchWrd": query,
                    "category": _api_category(category),
                    "serviceKey": api_key,
                }
                url = f"{kdca_base_url}?{urlencode({k: v for k, v in params.items() if v})}"
                request = Request(url, headers={"Accept": "application/json"})
                with urlopen(request, timeout=4) as response:
                    payload = response.read().decode("utf-8")
                try:
                    data = json.loads(payload)
                    text = json.dumps(data, ensure_ascii=False)[:1600]
                except Exception:
                    text = _extract_kdca_sections(payload, intent)
            except Exception as error:
                print(f"[KDCA RAG Warning] {category} API request failed: {error}")
        if text:
            results.append(f"- 검색어: {query} / 의도: {intent}\n{text}")
        else:
            results.append(f"- {category}: 국가건강정보포털 API 응답을 받지 못했습니다.")
    return "\n".join(results), source


def _fetch_welfare_context(question: str, user_context: Any) -> tuple[str, str]:
    age = _extract_age(user_context)
    gender = str(_extract_context_value(user_context, ["성별", "gender"], ""))
    smoking = str(_extract_context_value(user_context, ["흡연", "smoking"], ""))
    region = str(_extract_context_value(user_context, ["지역", "region"], ""))
    sub_region = str(_extract_context_value(user_context, ["시군구", "district", "subRegion"], ""))
    keyword = _keyword_from_question(question)
    response = _filter_benefits(
        age=age,
        gender=gender,
        smoking=smoking,
        risks=_risk_list_from_context(user_context),
        region=region,
        sub_region=sub_region,
        keyword=keyword,
        sort="relevance",
    )
    items = response.get("items", [])[:5]
    if not items and keyword:
        response = _filter_benefits(
            age=age,
            gender=gender,
            smoking=smoking,
            risks=_risk_list_from_context(user_context),
            region=region,
            sub_region=sub_region,
            keyword="",
            sort="relevance",
        )
        items = response.get("items", [])[:5]
    if not items:
        return "현재 조건에 맞는 건강 복지정보를 찾지 못했습니다.", "정부24 API"
    lines = []
    for index, item in enumerate(items, start=1):
        lines.append(
            f"{index}. {item.get('title', '공공서비스')} | 기관: {item.get('provider', '')} | "
            f"요약: {item.get('summary', '')} | 대상: {item.get('target', '') or '기관 문의'} | "
            f"신청: {item.get('applicationMethod', '') or '기관 문의'}"
        )
    return "\n".join(lines), "정부24 API"


def _build_coach_system_prompt(user_context: Any, health_context: str, welfare_context: str) -> str:
    return f"""
너는 Echo Health의 RAG 기반 AI 건강 코치다.
답변 가능 범위는 건강정보와 건강 관련 복지정보뿐이다.
범위 밖 질문에는 답변하지 말고, 건강정보/건강 복지정보 질문만 가능하다고 짧게 안내한다.
건강정보는 반드시 [KDCA 건강정보] 안의 내용과 사용자 건강정보를 근거로 답한다.
질문이 "무엇/뭐야/정의"이면 개요-정의 내용을 우선 설명하고, "예방/낮추려면/관리"이면 예방, 위험요인, 생활습관 내용을 우선 설명한다.
복지정보는 반드시 [건강 복지정보] 안의 항목만 근거로 소개한다.
자료에 없는 세부 자격, 금액, 신청기한은 추측하지 말고 기관 확인이 필요하다고 말한다.
진단, 처방, 치료 지시는 하지 말고 예방과 생활습관 중심으로 3-5문장 안에 답한다.
응급 증상이나 심한 증상은 즉시 의료기관/119 상담을 권한다.

[사용자 건강정보]
{json.dumps(user_context, ensure_ascii=False) if isinstance(user_context, dict) else user_context}

[KDCA 건강정보]
{health_context}

[건강 복지정보]
{welfare_context}
""".strip()

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
        print(f"Gemini plan failed: {e}")
        return {"status": "error", "message": str(e), "plan": [], "weeklyGoals": None}

def generate_coach_reply(messages: list, user_context: Any) -> dict:
    try:
        last_message = _latest_user_text(messages)
        wants_health = _is_health_question(last_message)
        wants_welfare = _is_welfare_question(last_message)

        if not wants_health and not wants_welfare:
            return {
                "reply": BLOCKED_REPLY,
                "source": "응답 범위: 건강정보 및 건강 복지정보"
            }

        health_context, health_source = _fetch_kdca_health_context(last_message, user_context) if wants_health else ("", "")
        welfare_context, welfare_source = _fetch_welfare_context(last_message, user_context) if wants_welfare else ("", "")
        system_prompt = _build_coach_system_prompt(user_context, health_context, welfare_context)

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
        response = chat.send_message(last_message)
        sources = [source for source in [health_source, welfare_source] if source]

        return {
            "reply": response.text,
            "source": f"출처: {', '.join(sources)}" if sources else "출처: Echo Health"
        }
    except Exception as e:
        print(f"Gemini coach 실패: {e}")
        return {
            "reply": "현재 AI 코치 서비스에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.",
            "source": "서비스 점검 중"
        }
