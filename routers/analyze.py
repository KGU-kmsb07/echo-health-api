from fastapi import APIRouter
import joblib
import pandas as pd
import json
import os

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

hypertension_model = None
diabetes_model = None
feature_columns = None


def load_models():
    global hypertension_model, diabetes_model, feature_columns

    if hypertension_model is None:
        hypertension_model = joblib.load(
            os.path.join(BASE_DIR, "models/hypertension_model.pkl")
        )

    if diabetes_model is None:
        diabetes_model = joblib.load(
            os.path.join(BASE_DIR, "models/diabetes_model.pkl")
        )

    if feature_columns is None:
        with open(
            os.path.join(BASE_DIR, "config/feature_columns.json"),
            "r",
            encoding="utf-8"
        ) as f:
            feature_columns = json.load(f)


def calc_bmi(weight_kg, height_cm):
    h = height_cm / 100
    return round(weight_kg / (h ** 2), 1)


def calc_obesity(bmi):
    if bmi >= 30:
        return 75
    if bmi >= 25:
        return 50
    if bmi >= 23:
        return 25
    return 10


def calc_metabolic(data):
    score = 0

    if data.get("waist_cm", 0) >= 90:
        score += 1
    if data.get("systolic_bp", 0) >= 130:
        score += 1
    if data.get("fasting_glucose", 0) >= 100:
        score += 1
    if data.get("triglyceride", 0) >= 150:
        score += 1
    if data.get("hdl_cholesterol", 0) < 40:
        score += 1

    if score >= 3:
        return 70
    if score == 2:
        return 45
    if score == 1:
        return 20
    return 10


@router.post("/analyze")
def analyze(data: dict):
    load_models()

    try:
        # 1. BMI 자동 계산
        bmi = calc_bmi(
            data.get("weight_kg", 70),
            data.get("height_cm", 170)
        )

        # 2. 입력 데이터 복사 후 bmi 추가
        request_data = data.copy()
        request_data["bmi"] = bmi

        # 3. feature_columns.json 구조에 맞게 컬럼 불러오기
        feature_map = feature_columns["feature_columns_by_disease"]

        hyper_cols = feature_map["고혈압"]
        diab_cols = feature_map["당뇨"]

        # 4. 고혈압 모델 입력 생성
        hyper_input = {
            col: request_data.get(col, 0)
            for col in hyper_cols
        }
        hyper_df = pd.DataFrame([hyper_input], columns=hyper_cols)
        hyper_prob = float(hypertension_model.predict_proba(hyper_df)[0][1]) * 100

        # 5. 당뇨 모델 입력 생성
        diab_input = {
            col: request_data.get(col, 0)
            for col in diab_cols
        }
        diab_df = pd.DataFrame([diab_input], columns=diab_cols)
        diab_prob = float(diabetes_model.predict_proba(diab_df)[0][1]) * 100

        # 6. 룰 기반 비만/대사증후군 계산
        obesity_prob = calc_obesity(bmi)
        metabolic_prob = calc_metabolic(request_data)

        # 7. 종합 건강 점수 계산
        health_score = max(
            0,
            round(
                100 - (
                    hyper_prob * 0.3
                    + diab_prob * 0.3
                    + obesity_prob * 0.2
                    + metabolic_prob * 0.2
                ) / 2
            )
        )

        # 8. 건강 나이 계산
        base_age = request_data.get("age", 30)
        health_age = base_age + round((hyper_prob + diab_prob) / 20 - 3)
        health_age = max(base_age - 5, health_age)

        return {
            "diabetes": round(diab_prob, 1),
            "hypertension": round(hyper_prob, 1),
            "metabolic": metabolic_prob,
            "obesity": obesity_prob,
            "bmi": bmi,
            "healthScore": health_score,
            "healthAge": health_age
        }

    except Exception as e:
        return {"error": str(e)}


"""
FastAPI Swagger /analyze 테스트용 입력 예시

주의:
- smoking 말고 current_smoking 사용
- exercise 말고 aerobic_activity 사용
- bmi는 서버에서 자동 계산하므로 입력하지 않아도 됨
- hba1c, total_cholesterol, ldl_direct도 모델 컬럼에 있으므로 가능하면 넣는 게 좋음

{
  "age": 30,
  "sex": 1,
  "height_cm": 175,
  "weight_kg": 72,
  "waist_cm": 82,
  "systolic_bp": 118,
  "diastolic_bp": 76,
  "fasting_glucose": 92,
  "hba1c": 5.4,
  "total_cholesterol": 180,
  "hdl_cholesterol": 55,
  "triglyceride": 110,
  "ldl_direct": 105,
  "current_smoking": 0,
  "aerobic_activity": 1
}

curl 테스트 예시:

curl -X POST "https://echo-health-api-716121321457.asia-northeast3.run.app/analyze" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "age": 30,
  "sex": 1,
  "height_cm": 175,
  "weight_kg": 72,
  "waist_cm": 82,
  "systolic_bp": 118,
  "diastolic_bp": 76,
  "fasting_glucose": 92,
  "hba1c": 5.4,
  "total_cholesterol": 180,
  "hdl_cholesterol": 55,
  "triglyceride": 110,
  "ldl_direct": 105,
  "current_smoking": 0,
  "aerobic_activity": 1
}'
"""