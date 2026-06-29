"""
kdca_service.py
- 질병관리청 국가건강정보포털 API를 실시간으로 호출하여 콘텐츠를 가져온다.
- API 호출 실패 시 config/kdca_contents.json의 정적 데이터로 fallback 처리.
- 사용자의 위험요인에 맞는 카테고리를 선택하여 관련 콘텐츠를 반환한다.
"""
import os
import json
import httpx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# KDCA 국가건강정보포털 공개 API
# 실제 엔드포인트 및 인증 키는 추후 확인 후 .env에 추가 필요
# 현재는 실시간 호출 시도 → 실패 시 정적 fallback
KDCA_API_BASE = os.environ.get(
    "KDCA_API_BASE",
    "https://health.kdca.go.kr/healthinfo/biz/pblcnth/getPblcnthList.do"
)

# 위험요인 → KDCA 카테고리 매핑 테이블
# (플랜 v3 표 기준)
RISK_CATEGORY_MAP = {
    "hypertension": "고혈압 예방",
    "diabetes": "당뇨 예방",
    "obesity": "비만 관리",
    "smoking": "금연",
    "activity": "신체활동",
}

def _load_static_fallback() -> dict:
    """정적 fallback: config/kdca_contents.json"""
    path = os.path.join(BASE_DIR, "config/kdca_contents.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

async def fetch_kdca_content(categories: list) -> dict:
    """
    categories: ["hypertension", "diabetes", "smoking", ...]
    각 카테고리별 KDCA API 호출 후 요약 콘텐츠 반환.
    API 호출 실패 시 정적 fallback 사용.
    """
    static = _load_static_fallback()
    results = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        for category in categories:
            try:
                resp = await client.get(
                    KDCA_API_BASE,
                    params={"category": RISK_CATEGORY_MAP.get(category, category)}
                )
                resp.raise_for_status()
                data = resp.json()
                # KDCA API 응답 파싱 (포맷 확인 후 조정 필요)
                results[category] = data.get("content") or data.get("items", [{}])[0].get("cn", static.get(category, ""))
            except Exception as e:
                print(f"[KDCA API Fallback] {category}: {e}")
                results[category] = static.get(category, "")

    return results

def select_categories(predict_result: dict, user_data: dict = None) -> list:
    """
    predict_result의 위험요인 분석 결과를 기반으로 해당하는 카테고리 목록을 반환.
    플랜 v3 표 기준.
    """
    categories = []
    if predict_result.get("hypertension_prob", 0) > 0.5:
        categories.append("hypertension")
    if predict_result.get("diabetes_prob", 0) > 0.5:
        categories.append("diabetes")
    if predict_result.get("obesity_status", 0) == 1:
        categories.append("obesity")
    if (user_data or {}).get("current_smoking", predict_result.get("current_smoking", 0)) == 1:
        categories.append("smoking")
    if (user_data or {}).get("aerobic_activity", predict_result.get("aerobic_activity", 1)) == 0:
        categories.append("activity")

    # 매핑 결과가 없는 경우 기본값: activity
    return categories if categories else ["activity"]
