import joblib
import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_hypertension_model = None
_diabetes_model = None
_feature_columns = None

def _load_models():
    global _hypertension_model, _diabetes_model, _feature_columns
    try:
        if _hypertension_model is None:
            _hypertension_model = joblib.load(
                os.path.join(BASE_DIR, "models/hypertension_model.pkl"))
        if _diabetes_model is None:
            _diabetes_model = joblib.load(
                os.path.join(BASE_DIR, "models/diabetes_model.pkl"))
    except Exception as e:
        print(f"모델 로드 실패, rule-based 사용: {e}")
    if _feature_columns is None:
        with open(os.path.join(BASE_DIR, "config/feature_columns.json"), "r", encoding="utf-8") as f:
            _feature_columns = json.load(f)

def _calc_bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / (height_cm / 100) ** 2, 1)

def _calc_obesity(bmi: float) -> int:
    if bmi >= 30: return 75
    if bmi >= 25: return 50
    if bmi >= 23: return 25
    return 10

def _calc_metabolic(data: dict) -> int:
    score = 0
    if data.get("waist_cm", 0) >= 90: score += 1
    if data.get("systolic_bp", 0) >= 130: score += 1
    if data.get("fasting_glucose", 0) >= 100: score += 1
    if data.get("triglyceride", 0) >= 150: score += 1
    if data.get("hdl_cholesterol", 0) < 40: score += 1
    if score >= 3: return 70
    if score == 2: return 45
    if score == 1: return 20
    return 10

def predict(data: dict) -> dict:
    _load_models()

    # BMI 자동 계산
    bmi = data.get("bmi") or _calc_bmi(data["weight_kg"], data["height_cm"])

    disease_feature_map = _feature_columns.get("feature_columns_by_disease", {})

    # 고혈압
    if _hypertension_model is not None:
        hyper_cols = disease_feature_map.get("고혈압", [])
        hyper_input = {col: data.get(col, 0) for col in hyper_cols}
        hyper_input["bmi"] = bmi
        hyper_df = pd.DataFrame([hyper_input], columns=hyper_cols)
        hyper_prob = float(
            _hypertension_model.predict_proba(hyper_df)[0][1]
        ) * 100
    else:
        hyper_prob = 20.0
        if data.get("systolic_bp", 0) >= 140: hyper_prob = 70.0
        elif data.get("systolic_bp", 0) >= 130: hyper_prob = 45.0

    # 당뇨
    if _diabetes_model is not None:
        diab_cols = disease_feature_map.get("당뇨", [])
        diab_input = {col: data.get(col, 0) for col in diab_cols}
        diab_input["bmi"] = bmi
        diab_df = pd.DataFrame([diab_input], columns=diab_cols)
        diab_prob = float(
            _diabetes_model.predict_proba(diab_df)[0][1]
        ) * 100
    else:
        diab_prob = 15.0
        if data.get("fasting_glucose", 0) >= 126: diab_prob = 75.0
        elif data.get("fasting_glucose", 0) >= 100: diab_prob = 40.0

    obesity_prob = _calc_obesity(bmi)
    metabolic_prob = _calc_metabolic(data)

    health_score = max(0, round(100 - (
        hyper_prob * 0.3 + diab_prob * 0.3 +
        obesity_prob * 0.2 + metabolic_prob * 0.2
    ) / 2))

    base_age = data.get("age", 30)
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
