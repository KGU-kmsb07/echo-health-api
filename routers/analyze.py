import os
import re
import tempfile

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from services.disease_predict_service import predict

router = APIRouter()
_latest_privacy_consent = {}


@router.post("/analyze")
def analyze(data: dict):
    try:
        return predict(data)
    except Exception as e:
        return {"error": str(e)}


@router.post("/simulate")
def simulate(data: dict):
    # This endpoint is read-only and does not modify the user's permanent profile or DB.
    # It takes the virtual simulated input and runs inference through predict_service.py.
    try:
        return predict(data)
    except Exception as e:
        return {"error": str(e)}



CHECKUP_FIELD_ALIASES = {
    "height_cm": ["키", "신장", "height"],
    "weight_kg": ["체중", "몸무게", "weight"],
    "waist_cm": ["허리둘레", "허리", "waist"],
    "systolic_bp": ["수축기혈압", "최고혈압", "systolic"],
    "diastolic_bp": ["이완기혈압", "최저혈압", "diastolic"],
    "fasting_glucose": ["공복혈당", "혈당", "glucose"],
    "hba1c": ["당화혈색소", "HbA1c", "A1c"],
    "total_cholesterol": ["총콜레스테롤", "총 콜레스테롤", "total cholesterol"],
    "hdl_cholesterol": ["HDL", "hdl"],
    "triglyceride": ["중성지방", "triglyceride", "TG"],
    "ldl_direct": ["LDL", "ldl"],
}


def _extract_checkup_values(text: str):
    values = {}
    normalized = re.sub(r"\s+", " ", text or "")
    for key, aliases in CHECKUP_FIELD_ALIASES.items():
        for alias in aliases:
            pattern = rf"{re.escape(alias)}[^0-9]{{0,20}}([0-9]+(?:\.[0-9]+)?)"
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                values[key] = float(match.group(1))
                break
    return values


@router.post("/checkup/extract")
async def extract_checkup(request: Request, source: str = "ocr"):
    body = await request.body()
    text = ""

    try:
        if source == "ocr":
            import easyocr

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(body)
                tmp_path = tmp.name
            try:
                reader = easyocr.Reader(["ko", "en"], gpu=False)
                text = " ".join(reader.readtext(tmp_path, detail=0))
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            text = body.decode("utf-8", errors="ignore")

        health_checkup = _extract_checkup_values(text)
        required = set(CHECKUP_FIELD_ALIASES.keys())
        if required.issubset(health_checkup.keys()):
            return {"status": "ok", "healthCheckup": health_checkup, "source": source}
        return {
            "status": "error",
            "message": "정보가 나오지 않습니다. 다른 방법으로 다시 시도해주세요.",
            "healthCheckup": health_checkup,
            "source": source,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "정보가 나오지 않습니다. 다른 방법으로 다시 시도해주세요.",
            "detail": str(e),
            "source": source,
        }

@router.post("/privacy/consent")
def save_privacy_consent(data: dict):
    global _latest_privacy_consent
    _latest_privacy_consent = {
        "requiredConsent": bool(data.get("requiredConsent")),
        "optionalConsent": bool(data.get("optionalConsent")),
        "consentVersion": data.get("consentVersion", "2024"),
        "consentedAt": data.get("consentedAt"),
    }
    return {"status": "ok", "consent": _latest_privacy_consent}


@router.get("/privacy/consent")
def get_privacy_consent():
    return {"status": "ok", "consent": _latest_privacy_consent}


@router.get("/privacy/consent/document", response_class=HTMLResponse)
def privacy_consent_document():
    return """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Echo Health 개인정보 수집 및 이용 동의</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: #f8fafc; }
    main { max-width: 720px; margin: 0 auto; padding: 32px 20px 48px; background: #fff; min-height: 100vh; box-sizing: border-box; }
    h1 { font-size: 24px; margin: 0 0 16px; }
    h2 { font-size: 17px; margin: 28px 0 10px; }
    p, li { font-size: 14px; line-height: 1.7; color: #374151; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
    th, td { border: 1px solid #e5e7eb; padding: 10px; vertical-align: top; text-align: left; }
    th { background: #f3f4f6; color: #374151; }
    .notice { background: #eff6ff; color: #1d4ed8; padding: 12px 14px; border-radius: 10px; }
  </style>
</head>
<body>
  <main>
    <h1>Echo Health 개인정보 수집 및 이용 동의</h1>
    <p class="notice">이 문서는 프론트 화면에 삽입하거나 별도 링크로 연결할 수 있는 개인정보 동의 기본 HTML 틀입니다.</p>
    <h2>1. 수집 및 이용 목적</h2>
    <p>건강 위험도 예측, 건강검진 정보 기반 분석, 맞춤 실천 플랜 생성, 마일리지 기록 관리, 정부 건강 혜택 추천에 사용합니다.</p>
    <h2>2. 수집 항목</h2>
    <table>
      <thead><tr><th>구분</th><th>항목</th><th>필수 여부</th></tr></thead>
      <tbody>
        <tr><td>기본 정보</td><td>이름, 나이, 성별, 지역</td><td>필수</td></tr>
        <tr><td>건강 정보</td><td>키, 몸무게, 허리둘레, 혈압, 흡연, 음주, 운동 빈도</td><td>필수</td></tr>
        <tr><td>건강검진 정보</td><td>혈압, 혈당, 당화혈색소, 콜레스테롤, 중성지방 등 사용자가 입력하거나 업로드한 정보</td><td>선택</td></tr>
        <tr><td>알림 정보</td><td>건강 혜택 안내 및 실천 리마인더 수신 여부</td><td>선택</td></tr>
      </tbody>
    </table>
    <h2>3. 보유 및 이용 기간</h2>
    <p>서비스 이용 기간 동안 보관하며, 사용자가 앱 내 데이터 삭제를 요청하면 지체 없이 삭제합니다.</p>
    <h2>4. 동의 거부 권리</h2>
    <p>사용자는 개인정보 수집 및 이용에 동의하지 않을 수 있습니다. 필수 항목에 동의하지 않는 경우 건강 분석 서비스 이용이 제한될 수 있습니다.</p>
    <h2>5. 의료 고지</h2>
    <p>Echo Health의 분석 결과는 공공데이터 기반 통계적 참고 정보이며 의료 진단이 아닙니다. 증상이 있거나 결과가 걱정되는 경우 의료진 상담을 우선해 주세요.</p>
  </main>
</body>
</html>
    """

