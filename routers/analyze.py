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


@router.get("/privacy", response_class=HTMLResponse)
@router.get("/privacy/", response_class=HTMLResponse)
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
    .muted { color: #6b7280; font-size: 12px; }
  </style>
</head>
<body>
  <main>
    <h1>Echo Health 개인정보 수집 및 이용 동의</h1>
    <p class="notice">본 문서는 Echo Health 서비스 이용을 위한 개인정보 수집 및 이용 동의, 선택 알림 동의, 서비스 이용상 유의사항을 안내합니다.</p>
    <p class="muted">시행일: 2026년 7월 1일</p>

    <h2>1. 서비스 개요</h2>
    <p>Echo Health는 사용자가 입력한 건강 관련 정보와 공공 통계 데이터를 바탕으로 건강 위험도 참고 정보, 맞춤 실천 플랜, 마일리지 기록, 건강 혜택 안내를 제공하는 건강관리 보조 서비스입니다. 본 서비스의 결과는 의료기관의 진단, 처방 또는 치료를 대체하지 않습니다.</p>

    <h2>2. 개인정보 수집 및 이용 목적</h2>
    <p>회사는 다음 목적을 위해 개인정보 및 건강 관련 정보를 수집·이용합니다.</p>
    <ul>
      <li>사용자별 건강 위험도 예측 및 결과 화면 제공</li>
      <li>건강검진 정보 기반 분석 및 생활습관 개선 제안</li>
      <li>4주 실천 플랜, 마일리지, 리마인더 등 개인화 기능 제공</li>
      <li>지역 기반 정부·공공 건강 혜택 추천</li>
      <li>서비스 오류 확인, 품질 개선, 이용 기록 관리</li>
    </ul>

    <h2>3. 수집 항목</h2>
    <table>
      <thead><tr><th>구분</th><th>항목</th><th>필수 여부</th></tr></thead>
      <tbody>
        <tr><td>기본 정보</td><td>이름, 나이, 성별, 지역</td><td>필수</td></tr>
        <tr><td>건강 정보</td><td>키, 몸무게, 허리둘레, 혈압, 흡연, 음주, 운동 빈도</td><td>필수</td></tr>
        <tr><td>건강검진 정보</td><td>혈압, 혈당, 당화혈색소, 콜레스테롤, 중성지방 등 사용자가 입력하거나 업로드한 정보</td><td>선택</td></tr>
        <tr><td>알림 정보</td><td>건강 혜택 안내 및 실천 리마인더 수신 여부</td><td>선택</td></tr>
      </tbody>
    </table>

    <h2>4. 보유 및 이용 기간</h2>
    <p>개인정보는 서비스 이용 기간 동안 보관하며, 사용자가 앱 내 데이터 삭제 또는 동의 철회를 요청하는 경우 지체 없이 삭제합니다. 다만 관계 법령에 따라 보관이 필요한 정보가 있는 경우 해당 법령에서 정한 기간 동안 보관할 수 있습니다.</p>

    <h2>5. 제3자 제공 및 처리 위탁</h2>
    <p>회사는 원칙적으로 사용자의 개인정보를 외부에 제공하지 않습니다. 향후 서비스 운영을 위해 개인정보 처리 위탁 또는 제3자 제공이 필요한 경우, 제공받는 자, 제공 목적, 제공 항목, 보유 기간을 사전에 고지하고 필요한 동의를 받습니다.</p>

    <h2>6. 선택 동의: 건강 혜택 알림</h2>
    <p>사용자가 선택 동의한 경우 Echo Health는 지역 건강 혜택 안내, 실천 리마인더, 건강관리 콘텐츠 알림을 제공할 수 있습니다. 선택 동의는 서비스 이용에 필수적이지 않으며, 사용자는 언제든지 마이페이지에서 동의를 철회할 수 있습니다.</p>

    <h2>7. 동의 거부 및 철회 권리</h2>
    <p>사용자는 개인정보 수집 및 이용에 동의하지 않을 권리가 있습니다. 다만 필수 항목에 동의하지 않는 경우 건강 분석, 맞춤 플랜 생성 등 주요 서비스 이용이 제한될 수 있습니다. 동의 철회 또는 데이터 삭제는 앱 내 개인정보 관리 메뉴에서 진행할 수 있습니다.</p>

    <h2>8. 사용자 입력 정보의 정확성</h2>
    <p>Echo Health의 분석 결과는 사용자가 입력한 정보에 따라 달라질 수 있습니다. 부정확하거나 최신 상태가 아닌 정보가 입력된 경우 결과의 신뢰도가 낮아질 수 있으므로, 가능한 한 정확한 정보를 입력해 주세요.</p>

    <h2>9. 의료 고지</h2>
    <p>Echo Health의 분석 결과와 추천 내용은 공공데이터 기반 통계적 참고 정보이며 의료적 진단, 치료, 처방, 예방을 목적으로 하지 않습니다. 통증, 이상 증상, 질환 의심, 복약 또는 치료 관련 의사결정이 필요한 경우 반드시 의사 등 의료 전문가와 상담해야 합니다.</p>

    <h2>10. 아동 및 민감정보 안내</h2>
    <p>본 서비스는 일반적인 건강관리 참고 정보를 제공하기 위한 서비스입니다. 법정대리인의 동의가 필요한 연령의 사용자는 보호자와 함께 서비스를 이용해야 하며, 서비스 목적과 무관한 민감한 의료 기록 또는 타인의 개인정보를 입력하지 않아야 합니다.</p>

    <h2>11. 안전성 확보 조치</h2>
    <p>회사는 개인정보가 분실, 도난, 유출, 변조 또는 훼손되지 않도록 합리적인 기술적·관리적 보호 조치를 취합니다. 사용자는 공동 기기 사용 시 앱 이용 후 브라우저 또는 기기 저장 정보를 직접 관리해야 합니다.</p>

    <h2>12. 약관의 변경</h2>
    <p>본 약관은 서비스 개선, 법령 변경, 운영 정책 변경에 따라 수정될 수 있습니다. 중요한 변경이 있는 경우 앱 화면 또는 별도 공지 수단을 통해 안내합니다.</p>
  </main>
</body>
</html>
    """

