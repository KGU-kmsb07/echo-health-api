import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["benefits"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOV24_BASE_URL = os.environ.get("KOR_GOV24_BASE_URL", "https://api.odcloud.kr/api")


class BenefitMatchRequest(BaseModel):
    age: int | None = None
    region: str | None = None
    district: str | None = None
    risks: list[str] | None = None
    query: str | None = None


def _clean(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _split_tags(*values: Any) -> list[str]:
    tags: list[str] = []
    for value in values:
        text = _clean(value)
        if not text:
            continue
        for part in text.replace(",", "|").replace("/", "|").split("|"):
            tag = part.strip()
            if tag and tag not in tags:
                tags.append(tag)
    return tags[:6]


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    service_id = _clean(row.get("서비스ID"))
    provider = _clean(row.get("소관기관명")) or _clean(row.get("접수기관명")) or _clean(row.get("접수기관"))
    summary = _clean(row.get("서비스목적요약")) or _clean(row.get("서비스목적")) or _clean(row.get("지원내용"))
    source_url = _clean(row.get("상세조회URL")) or _clean(row.get("온라인신청사이트URL"))
    tags = _split_tags(row.get("지원유형"), row.get("서비스분야"), row.get("사용자구분"))

    return {
        "id": service_id,
        "title": _clean(row.get("서비스명"), "공공서비스"),
        "provider": provider or "정부24",
        "summary": summary,
        "target": _clean(row.get("지원대상")),
        "selectionCriteria": _clean(row.get("선정기준")),
        "supportContent": _clean(row.get("지원내용")),
        "applicationMethod": _clean(row.get("신청방법")),
        "applicationDeadline": _clean(row.get("신청기한"), "상시 또는 기관별 상이"),
        "sourceUrl": source_url,
        "contact": _clean(row.get("전화문의")) or _clean(row.get("문의처")),
        "region": provider,
        "tags": tags,
        "raw": row,
    }


def _gov24_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    service_key = os.environ.get("KOR_GOV24") or os.environ.get("DATA_GO_KR_SERVICE_KEY")
    if not service_key:
        raise RuntimeError("KOR_GOV24 환경변수가 설정되어 있지 않습니다.")

    query = {
        "page": params.get("page", 1),
        "perPage": params.get("perPage", 100),
        "serviceKey": service_key,
    }
    query.update({k: v for k, v in params.items() if v not in (None, "")})
    url = f"{GOV24_BASE_URL}{path}?{urlencode(query)}"
    request = Request(url, headers={"Accept": "application/json"})

    with urlopen(request, timeout=12) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def _matches(benefit: dict[str, Any], keywords: list[str], region: str = "") -> bool:
    haystack = " ".join(
        _clean(benefit.get(key))
        for key in ["title", "provider", "summary", "target", "supportContent", "applicationMethod", "region"]
    )
    region_ok = not region or region in haystack
    keyword_ok = not keywords or any(keyword in haystack for keyword in keywords)
    return region_ok and keyword_ok


def _local_benefits() -> list[dict[str, Any]]:
    path = os.path.join(BASE_DIR, "config", "benefits.json")
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return [
            {
                "id": str(item.get("id", "")),
                "title": _clean(item.get("title"), "공공서비스"),
                "provider": _clean(item.get("region"), "정부24"),
                "summary": _clean(item.get("desc")),
                "target": "",
                "selectionCriteria": "",
                "supportContent": _clean(item.get("desc")),
                "applicationMethod": "",
                "applicationDeadline": "상시 또는 기관별 상이",
                "sourceUrl": "",
                "contact": "",
                "region": _clean(item.get("region")),
                "tags": item.get("tags", []),
                "raw": item,
            }
            for item in data
        ]
    except Exception:
        return []


@router.get("/api/benefits/search")
def search_benefits(query: str = "건강", region: str = "", page: int = 1, perPage: int = 1000):
    keywords = [word for word in query.split() if word]
    try:
        scan_size = max(100, min(perPage, 1000))
        first_result = _gov24_get("/gov24/v3/serviceList", {"page": page, "perPage": scan_size})
        results = [first_result]
        total_count = int(first_result.get("totalCount") or 0)
        max_pages = min(12, max(1, (total_count + scan_size - 1) // scan_size))

        benefits = [_normalize(row) for row in first_result.get("data", [])]
        filtered = [benefit for benefit in benefits if _matches(benefit, keywords, region)]

        next_page = page + 1
        while len(filtered) < 30 and next_page <= max_pages:
            result = _gov24_get("/gov24/v3/serviceList", {"page": next_page, "perPage": scan_size})
            results.append(result)
            page_benefits = [_normalize(row) for row in result.get("data", [])]
            filtered.extend([benefit for benefit in page_benefits if _matches(benefit, keywords, region)])
            next_page += 1

        return {
            "status": "success",
            "source": "gov24",
            "page": page,
            "perPage": scan_size,
            "scannedPages": len(results),
            "totalCount": total_count,
            "matchCount": len(filtered),
            "benefits": filtered[:50],
        }
    except Exception as error:
        local = [benefit for benefit in _local_benefits() if _matches(benefit, keywords, region)]
        return {
            "status": "fallback",
            "source": "local",
            "message": str(error),
            "matchCount": len(local),
            "benefits": local,
        }


@router.get("/api/benefits/detail/{service_id}")
def benefit_detail(service_id: str):
    try:
        result = _gov24_get("/gov24/v3/serviceDetail", {"서비스ID": service_id, "perPage": 10})
        rows = result.get("data", [])
        return {
            "status": "success",
            "source": "gov24",
            "benefit": _normalize(rows[0]) if rows else None,
        }
    except Exception as error:
        return {"status": "error", "message": str(error), "benefit": None}


@router.post("/api/benefits/match")
def match_benefits(payload: BenefitMatchRequest):
    risk_words = {
        "diabetes": "당뇨 혈당 건강검진",
        "hypertension": "고혈압 혈압 건강관리",
        "metabolic": "대사증후군 비만 운동 영양",
        "obesity": "비만 체중 운동 영양",
        "smoking": "금연 흡연",
    }
    query_parts = [payload.query or "건강"]
    for risk in payload.risks or []:
        query_parts.append(risk_words.get(risk, risk))
    if payload.age and payload.age >= 65:
        query_parts.append("노인 어르신")

    return search_benefits(
        query=" ".join(query_parts),
        region=payload.district or payload.region or "",
        page=1,
        perPage=100,
    )


@router.get("/benefits")
def legacy_benefits(region: str = "", age: int = 0, risks: str = ""):
    query = "건강"
    if risks:
        query = f"{query} {risks}"
    return search_benefits(query=query, region=region, page=1, perPage=100)
