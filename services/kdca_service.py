import os

import httpx


_kdca_value = os.environ.get("KDCA", "").strip()
KDCA_API_BASE = os.environ.get("KDCA_API_BASE", "").strip()
KDCA_API_KEY = os.environ.get("KDCA_API_KEY", "").strip()

if _kdca_value.startswith("http"):
    KDCA_API_BASE = _kdca_value
elif _kdca_value and not KDCA_API_KEY:
    KDCA_API_KEY = _kdca_value


def _api_category(category: str) -> str:
    allowed = {"hypertension", "diabetes", "obesity", "smoking", "activity", "cancer"}
    return category if category in allowed else ""


async def fetch_kdca_content(categories: list[str], query: str | None = None) -> dict:
    """Fetch health guidance from the National Health Information Portal API.

    Local fallback text is intentionally not used. If the API is not configured
    or does not return data, the category is returned with an empty value so the
    caller can handle the missing source explicitly.
    """
    results: dict[str, str | dict] = {}

    if not KDCA_API_BASE:
        return {category: "" for category in categories}

    async with httpx.AsyncClient(timeout=4.0) as client:
        for category in categories:
            try:
                keyword = (query or category or "").strip()
                params = {
                    "category": _api_category(category),
                    "query": keyword,
                    "q": keyword,
                    "keyword": keyword,
                    "searchWrd": keyword,
                }
                if KDCA_API_KEY and not KDCA_API_KEY.startswith("http"):
                    params["serviceKey"] = KDCA_API_KEY
                response = await client.get(KDCA_API_BASE, params=params)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                results[category] = response.json() if "json" in content_type else response.text
            except Exception as error:
                print(f"[KDCA Service Warning] API request failed for {category}: {error}")
                results[category] = ""

    return results


def select_categories(predict_result: dict, user_data: dict | None = None) -> list[str]:
    """Select KDCA categories from prediction and user profile data."""
    categories: list[str] = []

    if predict_result.get("hypertension_prob", 0.0) > 0.5:
        categories.append("hypertension")
    if predict_result.get("diabetes_prob", 0.0) > 0.5:
        categories.append("diabetes")
    if predict_result.get("obesity_status", 0) == 1:
        categories.append("obesity")

    current_smoking = predict_result.get("current_smoking")
    if current_smoking is None and user_data:
        current_smoking = user_data.get("current_smoking")

    aerobic_activity = predict_result.get("aerobic_activity")
    if aerobic_activity is None and user_data:
        aerobic_activity = user_data.get("aerobic_activity")

    if current_smoking == 1:
        categories.append("smoking")
    if aerobic_activity == 0:
        categories.append("activity")

    return categories or ["activity"]
