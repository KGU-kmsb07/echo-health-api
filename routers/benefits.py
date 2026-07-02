import json
import os
import re
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import APIRouter
from pydantic import BaseModel
from services.benefit_text_mining_service import get_benefit_text_mining_model

router = APIRouter(tags=["benefits"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOV24_BASE_URL = os.environ.get("KOR_GOV24_BASE_URL", "https://api.odcloud.kr/api")
HEALTH_KEYWORDS = ["건강", "금연", "건강검진", "운동", "비만", "고혈압", "당뇨", "보건소", "정신건강"]
HEALTH_CONTEXT_WORDS = ["검진", "진료", "의료", "보건", "질병", "예방", "상담", "운동", "정신", "금연", "비만", "고혈압", "당뇨", "혈압", "혈당"]
NATIONAL_MARKERS = ["전국", "중앙부처", "보건복지부", "국민건강보험공단", "질병관리청", "한국건강관리협회"]
CACHE_TTL_SECONDS = 60 * 60 * 6
_CACHE: dict[str, Any] = {"expires_at": 0.0, "items": []}


class BenefitMatchRequest(BaseModel):
    age: int | None = None
    gender: str | None = None
    smoking: str | None = None
    region: str | None = None
    subRegion: str | None = None
    district: str | None = None
    keyword: str | None = None
    query: str | None = None
    sort: str | None = "latest"
    risks: list[str] | None = None


class BenefitProfile(BaseModel):
    age: int | None = None
    gender: str = ""
    smoking: str = ""
    risks: list[str] | None = None
    region: str = ""
    sub_region: str = ""
    keyword: str = ""


def _clean(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _first(row: dict[str, Any], names: list[str], default: str = "") -> str:
    for name in names:
        value = _clean(row.get(name))
        if value:
            return value
    return default


def _split_tags(*values: Any) -> list[str]:
    tags: list[str] = []
    for value in values:
        text = _clean(value)
        if not text:
            continue
        for part in re.split(r"[,/|>\s]+", text):
            tag = part.strip()
            if tag and tag not in tags:
                tags.append(tag)
    return tags[:8]


def _make_tags(item: dict[str, Any]) -> list[str]:
    haystack = _benefit_text(item)
    tags = _split_tags(item.get("supportType"), item.get("serviceField"), item.get("userType"))
    for keyword in HEALTH_KEYWORDS:
        if keyword in haystack and keyword not in tags:
            tags.append(keyword)
    return tags[:8]


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    provider = _first(row, ["소관기관명", "접수기관명", "기관명", "부서명"], "정부24")
    region = _first(row, ["지역", "시도", "소관기관명", "접수기관명"], provider)
    source_url = _first(row, ["상세조회URL", "온라인신청사이트URL", "URL", "서비스URL"])
    detail_id = _first(row, ["서비스ID", "서비스아이디", "serviceId", "id"])
    item = {
        "id": detail_id or _first(row, ["서비스명", "title"], provider),
        "title": _first(row, ["서비스명", "서비스명칭", "title"], "공공서비스"),
        "provider": provider,
        "summary": _first(row, ["서비스목적요약", "서비스목적", "지원내용", "서비스내용", "summary"]),
        "target": _first(row, ["지원대상", "선정기준", "target"]),
        "applicationMethod": _first(row, ["신청방법", "신청절차", "applicationMethod"]),
        "deadline": _first(row, ["신청기한", "접수기간", "deadline"], "상시 또는 기관 문의"),
        "sourceUrl": source_url,
        "region": region,
        "supportType": _first(row, ["지원유형", "지원형태"]),
        "serviceField": _first(row, ["서비스분야", "분야"]),
        "userType": _first(row, ["사용자구분", "생애주기"]),
        "updatedAt": _first(row, ["수정일시", "수정일", "최종수정일", "등록일시", "등록일"]),
        "raw": row,
    }
    item["tags"] = _make_tags(item)
    return item


def _gov24_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    service_key = os.environ.get("DATA_GO_KR_SERVICE_KEY") or os.environ.get("KOR_GOV24")
    if not service_key:
        raise RuntimeError("DATA_GO_KR_SERVICE_KEY 환경변수가 설정되어 있지 않습니다.")

    query = {
        "page": params.get("page", 1),
        "perPage": params.get("perPage", 100),
        "serviceKey": service_key,
    }
    query.update({k: v for k, v in params.items() if v not in (None, "")})
    url = f"{GOV24_BASE_URL}{path}?{urlencode(query, safe='%')}"
    request = Request(url, headers={"Accept": "application/json"})

    with urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _benefit_text(benefit: dict[str, Any]) -> str:
    return " ".join(
        _clean(benefit.get(key))
        for key in [
            "title",
            "provider",
            "summary",
            "target",
            "applicationMethod",
            "deadline",
            "region",
            "supportType",
            "serviceField",
            "userType",
        ]
    )


def _is_health_related(benefit: dict[str, Any]) -> bool:
    return get_benefit_text_mining_model().is_health_related(benefit)


def _is_national(benefit: dict[str, Any]) -> bool:
    return get_benefit_text_mining_model().is_national(benefit)


def _matches_region(benefit: dict[str, Any], region: str = "", sub_region: str = "") -> bool:
    if not region and not sub_region:
        return True
    if _is_national(benefit):
        return True
    text = _benefit_text(benefit)
    region_ok = not region or region in text
    sub_region_ok = not sub_region or sub_region in text
    return region_ok and sub_region_ok


def _age_group_terms(age: int) -> list[str]:
    if age < 13:
        return ["아동", "어린이", "영유아"]
    if age < 19:
        return ["청소년", "학생"]
    if age < 35:
        return ["청년", "대학생", "사회초년생"]
    if age < 50:
        return ["중장년", "성인"]
    if age < 65:
        return ["중장년", "장년", "성인"]
    return ["노인", "어르신", "고령", "시니어", "65세"]


def _matches_age(benefit: dict[str, Any], age: int | None) -> bool:
    if not age:
        return True

    text = f"{benefit.get('target', '')} {benefit.get('summary', '')} {benefit.get('userType', '')}"
    title = _clean(benefit.get("title"))
    age_text = f"{title} {text}"
    has_age_rule = False
    has_range_rule = False

    if age >= 19 and any(word in age_text for word in ["초등학교", "중학교", "고등학교", "청소년", "학교 밖"]):
        return False
    if age < 19 and any(word in age_text for word in ["청년", "대학생", "중장년", "장년", "노인", "어르신", "고령", "시니어"]):
        return False

    for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*세\s*이상(?:에서)?\s*(?:만\s*)?(\d{1,3})\s*세\s*(이하|미만|까지)", age_text):
        has_age_rule = True
        has_range_rule = True
        start, end = int(match.group(1)), int(match.group(2))
        end_inclusive = match.group(3) != "미만"
        if start <= age and (age <= end if end_inclusive else age < end):
            return True

    for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*(?:세)?\s*[~\-–]\s*(?:만\s*)?(\d{1,3})\s*세", age_text):
        has_age_rule = True
        has_range_rule = True
        start, end = int(match.group(1)), int(match.group(2))
        if start <= age <= end:
            return True

    if has_range_rule:
        return False

    for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*세\s*이상", age_text):
        has_age_rule = True
        if age >= int(match.group(1)):
            return True

    for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*세\s*(이하|미만)", age_text):
        has_age_rule = True
        limit = int(match.group(1))
        if (match.group(2) == "미만" and age < limit) or (match.group(2) != "미만" and age <= limit):
            return True

    for match in re.finditer(r"(?<!\d)(\d{1,3})\s*세(?!\s*(?:이상|이하|미만|초과|부터|까지|~|\-|–))", age_text):
        exact_age = int(match.group(1))
        if 0 < exact_age <= 120:
            has_age_rule = True
            if age == exact_age:
                return True

    for match in re.finditer(r"(?<!\d)(\d{1,2})0대", age_text):
        has_age_rule = True
        decade = int(match.group(1)) * 10
        if decade <= age <= decade + 9:
            return True

    terms = _age_group_terms(age)
    all_age_terms = ["아동", "어린이", "영유아", "청소년", "청년", "대학생", "중장년", "장년", "노인", "어르신", "고령", "시니어"]
    mentioned_terms = [term for term in all_age_terms if term in text]
    if mentioned_terms:
        return any(term in text for term in terms)

    if has_age_rule:
        return False

    if any(word in text for word in ["전 국민", "전국민", "누구나", "전체", "모든 국민", "일반 국민"]):
        return True

    return True


def _matches_gender_and_life_stage(benefit: dict[str, Any], age: int | None = None, gender: str = "") -> bool:
    text = _benefit_text(benefit)
    gender_text = _clean(gender)

    maternity_words = ["임산부", "산모", "임신", "출산", "분만", "태아", "난임", "모자보건"]
    infant_words = ["신생아", "영유아", "영아", "유아", "소아", "어린이"]
    female_only_words = ["여성", "여자"]
    male_only_words = ["남성", "남자", "전립선"]

    if gender_text.startswith("남"):
        if any(word in text for word in maternity_words + female_only_words):
            return False
    if gender_text.startswith("여"):
        if any(word in text for word in male_only_words):
            return False

    if age and age >= 19 and any(word in text for word in infant_words):
        return False
    return True


def _matches_general_profile_scope(benefit: dict[str, Any], keyword: str = "") -> bool:
    text = _benefit_text(benefit)
    keyword_text = _clean(keyword)
    restricted_groups = [
        ("장애", ["장애인", "근로장애인", "장애아"]),
        ("보훈", ["국가유공자", "보훈", "참전", "무공수훈", "애국지사"]),
        ("암", ["암환자", "항암", "중증질환", "희귀질환", "난치병"]),
        ("정신", ["정신질환자"]),
        ("응급", ["응급환자", "응급의료비"]),
        ("건설", ["건설노동자", "건설근로자"]),
        ("군인", ["군인"]),
        ("위안부", ["위안부"]),
        ("한부모", ["한부모"]),
        ("당뇨", ["당뇨병 환자", "당뇨환자", "당뇨병 투약자"]),
        ("고혈압", ["고혈압 환자"]),
        ("치매", ["치매"]),
        ("석면", ["석면"]),
        ("산정특례", ["산정특례"]),
        ("수급", ["의료급여수급권자", "수급자", "차상위", "저소득"]),
        ("가습기", ["가습기살균제"]),
        ("한센", ["한센"]),
    ]

    for unlock_word, blocked_words in restricted_groups:
        if any(word in text for word in blocked_words) and unlock_word not in keyword_text:
            return False
    return True


def _matches_keyword(benefit: dict[str, Any], keyword: str = "") -> bool:
    words = [word for word in re.split(r"\s+", keyword.strip()) if word]
    if not words:
        return True
    text = _benefit_text(benefit)
    return all(word in text for word in words)


def _extract_age_constraints(text: str) -> dict[str, Any]:
    ranges: list[tuple[int, int]] = []
    exact: list[int] = []
    min_age: int | None = None
    max_age: int | None = None

    for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*세\s*이상(?:에서)?\s*(?:만\s*)?(\d{1,3})\s*세\s*(이하|미만|까지)", text):
        start, end = int(match.group(1)), int(match.group(2))
        if match.group(3) == "미만":
            end -= 1
        ranges.append((start, end))

    for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*(?:세)?\s*[~\-–]\s*(?:만\s*)?(\d{1,3})\s*세", text):
        ranges.append((int(match.group(1)), int(match.group(2))))

    if not ranges:
        for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*세\s*이상", text):
            value = int(match.group(1))
            min_age = value if min_age is None else max(min_age, value)

        for match in re.finditer(r"(?:만\s*)?(\d{1,3})\s*세\s*(이하|미만)", text):
            value = int(match.group(1)) - (1 if match.group(2) == "미만" else 0)
            max_age = value if max_age is None else min(max_age, value)

    for match in re.finditer(r"(?<!\d)(\d{1,3})\s*세(?!\s*(?:이상|이하|미만|초과|부터|까지|~|\-|–))", text):
        value = int(match.group(1))
        if 0 < value <= 120:
            exact.append(value)

    for match in re.finditer(r"(?<!\d)(\d{1,2})0대", text):
        decade = int(match.group(1)) * 10
        ranges.append((decade, decade + 9))

    return {"ranges": ranges, "exact": exact, "min": min_age, "max": max_age}


def _age_constraint_match(constraints: dict[str, Any], age: int | None) -> tuple[bool, int, list[str]]:
    if not age:
        return True, 0, []

    ranges = constraints["ranges"]
    exact = constraints["exact"]
    min_age = constraints["min"]
    max_age = constraints["max"]
    reasons: list[str] = []

    if ranges:
        if any(start <= age <= end for start, end in ranges):
            reasons.append(f"{age}세 연령 조건 일치")
            return True, 35, reasons
        return False, -100, ["연령 조건 불일치"]

    if exact:
        if age in exact:
            reasons.append(f"{age}세 전용 조건 일치")
            return True, 35, reasons
        return False, -100, ["연령 조건 불일치"]

    if min_age is not None and age < min_age:
        return False, -100, ["최소 연령 조건 불일치"]
    if max_age is not None and age > max_age:
        return False, -100, ["최대 연령 조건 불일치"]
    if min_age is not None or max_age is not None:
        reasons.append(f"{age}세 연령 조건 일치")
        return True, 30, reasons

    return True, 8, ["연령 제한 없음"]


def _extract_profile_traits(text: str) -> dict[str, Any]:
    topic_words = {
        "금연": ["금연", "흡연"],
        "검진": ["건강검진", "검진"],
        "운동": ["운동", "신체활동", "체력"],
        "비만": ["비만", "체중"],
        "고혈압": ["고혈압", "혈압"],
        "당뇨": ["당뇨", "혈당"],
        "정신건강": ["정신건강", "마음건강", "정신질환"],
        "치아": ["치아", "구강"],
    }
    conditions = {
        "흡연자": ["흡연자", "금연치료", "금연클리닉", "금연상담"],
        "당뇨": ["당뇨병 환자", "당뇨환자", "당뇨병 투약자"],
        "고혈압": ["고혈압 환자", "고혈압"],
        "저소득": ["저소득", "수급자", "차상위", "의료급여"],
        "근로자": ["근로자", "직장인", "사업장", "자영업자"],
        "군인": ["군인"],
        "장애": ["장애인"],
        "보훈": ["국가유공자", "보훈", "참전"],
        "질환자": ["환자", "질환", "치매", "암", "석면"],
    }
    topics = [name for name, words in topic_words.items() if any(word in text for word in words)]
    required = [name for name, words in conditions.items() if any(word in text for word in words)]

    life_stage = ""
    if any(word in text for word in ["임산부", "산모", "임신", "출산", "신생아", "영유아"]):
        life_stage = "maternity_or_infant"
    elif any(word in text for word in ["초등학교", "중학교", "고등학교", "청소년", "학교 밖"]):
        life_stage = "youth"
    elif any(word in text for word in ["청년", "대학생", "사회초년생"]):
        life_stage = "young_adult"
    elif any(word in text for word in ["노인", "어르신", "고령", "시니어"]):
        life_stage = "senior"
    elif any(word in text for word in ["전 국민", "전국민", "누구나", "일반 국민"]):
        life_stage = "all"
    else:
        life_stage = "general"

    gender = "all"
    if any(word in text for word in ["임산부", "산모", "임신", "출산", "여성", "여자"]):
        gender = "female"
    elif any(word in text for word in ["남성", "남자", "전립선"]):
        gender = "male"

    return {"topics": topics, "required": required, "lifeStage": life_stage, "gender": gender}


def _score_region(benefit: dict[str, Any], region: str = "", sub_region: str = "") -> tuple[int, list[str]]:
    if not region and not sub_region:
        return 0, ["전국 기준"]
    if _is_national(benefit):
        return 30, ["전국 혜택"]
    text = _benefit_text(benefit)
    score = 0
    reasons: list[str] = []
    if region and region in text:
        score += 35
        reasons.append(f"{region} 지역 일치")
    if sub_region and sub_region in text:
        score += 45
        reasons.append(f"{sub_region} 세부지역 일치")
    if score == 0:
        score -= 30
        reasons.append("지역 불일치")
    return score, reasons


def _score_personalization(benefit: dict[str, Any], profile: BenefitProfile) -> dict[str, Any]:
    return get_benefit_text_mining_model().score_benefit(
        benefit,
        {
            "age": profile.age,
            "gender": profile.gender,
            "smoking": profile.smoking,
            "risks": profile.risks or [],
            "region": profile.region,
            "sub_region": profile.sub_region,
            "keyword": profile.keyword,
        },
    )


def _relevance_score(benefit: dict[str, Any], keyword: str = "") -> int:
    return get_benefit_text_mining_model().relevance_score(benefit, keyword)


def _date_rank(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    return digits[:14].ljust(14, "0")


def _sort_items(items: list[dict[str, Any]], sort: str = "latest", keyword: str = "") -> list[dict[str, Any]]:
    if sort == "relevance":
        return sorted(items, key=lambda item: (_relevance_score(item, keyword), _date_rank(item.get("updatedAt", ""))), reverse=True)
    return sorted(items, key=lambda item: (_date_rank(item.get("updatedAt", "")), str(item.get("id", ""))), reverse=True)


def _fetch_health_benefits() -> list[dict[str, Any]]:
    now = time.time()
    if _CACHE["expires_at"] > now:
        return list(_CACHE["items"])

    per_page = 1000
    first = _gov24_get("/gov24/v3/serviceList", {"page": 1, "perPage": per_page})
    total_count = int(first.get("totalCount") or len(first.get("data", [])) or 0)
    max_pages = min(12, max(1, (total_count + per_page - 1) // per_page))

    rows = list(first.get("data", []))
    for page in range(2, max_pages + 1):
        result = _gov24_get("/gov24/v3/serviceList", {"page": page, "perPage": per_page})
        rows.extend(result.get("data", []))

    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = _normalize(row)
        key = item["id"] or item["title"]
        if key in seen or not _is_health_related(item):
            continue
        seen.add(key)
        items.append(item)

    _CACHE["items"] = items
    _CACHE["expires_at"] = now + CACHE_TTL_SECONDS
    return list(items)


def _local_benefits() -> list[dict[str, Any]]:
    return []

def _shape_response(items: list[dict[str, Any]], status: str = "success", source: str = "gov24", message: str = ""):
    return {
        "status": status,
        "source": source,
        "message": message,
        "items": [
            {
                "id": item.get("id", ""),
                "title": item.get("title", ""),
                "provider": item.get("provider", ""),
                "summary": item.get("summary", ""),
                "target": item.get("target", ""),
                "applicationMethod": item.get("applicationMethod", ""),
                "deadline": item.get("deadline", ""),
                "sourceUrl": item.get("sourceUrl", ""),
                "tags": item.get("tags", []),
                "region": item.get("region", ""),
                "updatedAt": item.get("updatedAt", ""),
                "matchReasons": item.get("matchReasons", []),
                "conditionWarnings": item.get("conditionWarnings", []),
                "scores": item.get("scores", {}),
            }
            for item in items
        ],
        "benefits": items,
        "totalCount": len(items),
        "matchCount": len(items),
    }


def _filter_benefits(
    age: int | None = None,
    gender: str = "",
    smoking: str = "",
    risks: list[str] | None = None,
    region: str = "",
    sub_region: str = "",
    keyword: str = "",
    sort: str = "latest",
) -> dict[str, Any]:
    try:
        items = _fetch_health_benefits()
        status, source, message = "success", "gov24", ""
    except Exception as error:
        return _shape_response([], "error", "gov24", str(error))

    profile = BenefitProfile(
        age=age,
        gender=gender,
        smoking=smoking,
        risks=risks or [],
        region=region,
        sub_region=sub_region,
        keyword=keyword,
    )
    scored: list[dict[str, Any]] = []
    for item in items:
        if not _matches_keyword(item, keyword):
            continue
        analysis = _score_personalization(item, profile)
        if not analysis["eligible"]:
            continue
        if region == "국가기관" and analysis["regionScore"] < 0:
            continue
        ranked = {
            **item,
            "matchReasons": analysis["matchReasons"],
            "conditionWarnings": analysis["conditionWarnings"],
            "scores": {
                "personal": analysis["personalScore"],
                "region": analysis["regionScore"],
                "keyword": analysis["keywordScore"],
                "total": analysis["totalScore"],
            },
        }
        scored.append(ranked)

    if sort == "relevance":
        sorted_items = sorted(
            scored,
            key=lambda item: (
                item.get("scores", {}).get("keyword", 0),
                item.get("scores", {}).get("personal", 0),
                item.get("scores", {}).get("region", 0),
                _date_rank(item.get("updatedAt", "")),
            ),
            reverse=True,
        )
    else:
        sorted_items = sorted(
            scored,
            key=lambda item: (
                item.get("scores", {}).get("personal", 0),
                item.get("scores", {}).get("region", 0),
                item.get("scores", {}).get("keyword", 0),
                _date_rank(item.get("updatedAt", "")),
            ),
            reverse=True,
        )
    return _shape_response(sorted_items[:100], status, source, message)


@router.get("/api/benefits/search")
def search_benefits(
    query: str = "",
    keyword: str = "",
    region: str = "",
    subRegion: str = "",
    district: str = "",
    age: int | None = None,
    gender: str = "",
    smoking: str = "",
    risks: str = "",
    sort: str = "latest",
    page: int = 1,
    perPage: int = 100,
):
    search_word = keyword or query
    risk_list = [risk for risk in risks.split(",") if risk]
    response = _filter_benefits(
        age=age,
        gender=gender,
        smoking=smoking,
        risks=risk_list,
        region=region,
        sub_region=subRegion or district,
        keyword=search_word,
        sort=sort,
    )
    start = max(0, (page - 1) * perPage)
    end = start + perPage
    response["items"] = response["items"][start:end]
    response["benefits"] = response["benefits"][start:end]
    return response


@router.post("/api/benefits/match")
def match_benefits(payload: BenefitMatchRequest):
    return _filter_benefits(
        age=payload.age,
        gender=payload.gender or "",
        smoking=payload.smoking or "",
        risks=payload.risks or [],
        region=payload.region or "",
        sub_region=payload.subRegion or payload.district or "",
        keyword=payload.keyword or payload.query or "",
        sort=payload.sort or "latest",
    )


@router.get("/api/benefits/detail/{service_id}")
def benefit_detail(service_id: str):
    try:
        result = _gov24_get("/gov24/v3/serviceDetail", {"서비스ID": service_id, "perPage": 10})
        rows = result.get("data", [])
        return {"status": "success", "source": "gov24", "benefit": _normalize(rows[0]) if rows else None}
    except Exception as error:
        return {"status": "error", "message": str(error), "benefit": None}


@router.get("/benefits")
def legacy_benefits(region: str = "", age: int = 0, risks: str = ""):
    return search_benefits(query=risks, region=region, age=age or None, page=1, perPage=100)
