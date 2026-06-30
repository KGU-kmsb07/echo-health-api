import os
import json
import httpx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KDCA_API_BASE = "https://health.kdca.go.kr/healthinfo/biz/pblcnth/..."  # 실제 엔드포인트 미확정이므로 대기

def _load_fallback_kdca():
    path = os.path.join(BASE_DIR, "config/kdca_contents.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

async def fetch_kdca_content(categories: list[str]) -> dict:
    """
    categories: ["hypertension", "diabetes", "obesity", "smoking", "activity"]
    각 카테고리별 KDCA API 호출을 시도하며, 엔드포인트 미작동 시 static kdca_contents.json에서 복원합니다.
    """
    fallback_data = _load_fallback_kdca()
    results = {}
    
    for category in categories:
        try:
            # 실제 KDCA API 엔드포인트가 제대로 설정되었을 때 작동하도록 가드
            if KDCA_API_BASE and not KDCA_API_BASE.endswith("..."):
                async with httpx.AsyncClient() as client:
                    response = await client.get(KDCA_API_BASE, params={"category": category}, timeout=2.0)
                    if response.status_code == 200:
                        results[category] = response.json()
                        continue
            
            # API 호출 불가능하거나 실패 시 로컬 정적 데이터 제공
            results[category] = fallback_data.get(category, "건강을 위해 규칙적인 운동과 식단 관리를 권장합니다. (출처: 질병관리청)")
        except Exception as e:
            print(f"[KDCA Service Warning] API 호출 실패로 정적 데이터 활용: {e}")
            results[category] = fallback_data.get(category, "건강을 위해 규칙적인 운동과 식단 관리를 권장합니다. (출처: 질병관리청)")
            
    return results

def select_categories(predict_result: dict, user_data: dict = None) -> list[str]:
    """
    사용자의 예측 결과 및 프로필 요인을 분석하여 관련 KDCA 카테고리를 선택합니다.
    """
    categories = []
    
    # 1. 고혈압
    if predict_result.get("hypertension_prob", 0.0) > 0.5:
        categories.append("hypertension")
        
    # 2. 당뇨
    if predict_result.get("diabetes_prob", 0.0) > 0.5:
        categories.append("diabetes")
        
    # 3. 비만
    if predict_result.get("obesity_status", 0) == 1:
        categories.append("obesity")
        
    # 4. 흡연 및 운동 여부 (predict_result 혹은 user_data에서 확인)
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
        
    return categories if categories else ["activity"]
