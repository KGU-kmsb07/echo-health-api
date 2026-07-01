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
                os.path.join(BASE_DIR, "models/simple/hypertension_logistic_regression.pkl"))
        if _diabetes_model is None:
            _diabetes_model = joblib.load(
                os.path.join(BASE_DIR, "models/simple/diabetes_logistic_regression.pkl"))
    except Exception as e:
        print(f"모델 로드 실패, rule-based 사용: {e}")
    if _feature_columns is None:
        with open(os.path.join(BASE_DIR, "config/feature_columns.json"), "r", encoding="utf-8") as f:
            _feature_columns = json.load(f)

def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

def calculate_vitality_score(hypertension_prob: float, diabetes_prob: float, bmi: float,
                              current_smoking: int, aerobic_activity: int) -> int:
    # 소수(0.0~1.0) → 퍼센트(0.0~100.0) 변환
    hp = hypertension_prob * 100 if hypertension_prob <= 1.0 else hypertension_prob
    dp = diabetes_prob * 100 if diabetes_prob <= 1.0 else diabetes_prob

    score = 100.0
    score -= hp * 0.2
    score -= dp * 0.2
    if bmi >= 25:
        score -= 15
    if current_smoking == 1:
        score -= 15
    if aerobic_activity == 0:
        score -= 15

    return max(0, min(100, round(score)))

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

def calculate_health_age(age: int, hypertension_prob: float, diabetes_prob: float, 
                         bmi: float, current_smoking: int, aerobic_activity: int) -> int:
    health_age = float(age)
    
    # 1. 고혈압 위험도 영향성 (혈관 건강 나이 가중치)
    if hypertension_prob < 0.2:
        health_age -= 1.0
    elif hypertension_prob >= 0.6:
        health_age += 4.0
    elif hypertension_prob >= 0.35:
        health_age += 2.0
        
    # 2. 당뇨 위험도 영향성 (세포/대사 기능 나이 가중치)
    if diabetes_prob < 0.1:
        health_age -= 1.0
    elif diabetes_prob >= 0.6:
        health_age += 4.0
    elif diabetes_prob >= 0.35:
        health_age += 2.0
        
    # 3. 비만도 (BMI) 영향성
    if bmi < 18.5:
        health_age += 1.0
    elif 18.5 <= bmi < 23.0:
        health_age -= 1.5
    elif 23.0 <= bmi < 25.0:
        health_age += 1.0
    else:  # bmi >= 25.0
        health_age += 3.0
        
    # 4. 흡연 습관 (산화 스트레스 및 기대수명 감점)
    if current_smoking == 1:
        health_age += 4.0
    else:
        health_age -= 1.0
        
    # 5. 유산소 운동 여부 (세포 노화 지연 및 심폐 기능 가점)
    if aerobic_activity == 1:
        health_age -= 1.5
    else:
        health_age += 2.0
        
    # 실제 나이보다 최대 5살까지만 젊어지게 보정 (하한 장벽)
    min_age = age - 5
    if health_age < min_age:
        health_age = min_age
        
    return int(round(health_age))

def predict(data: dict) -> dict:
    _load_models()

    # BMI 자동 계산
    bmi = calculate_bmi(data["height_cm"], data["weight_kg"])
    obesity_status = 1 if bmi >= 25 else 0

    disease_feature_map = _feature_columns.get("feature_columns_by_disease", {})

    # 고혈압 (소수점 형태 확률 계산)
    if _hypertension_model is not None:
        hyper_cols = disease_feature_map.get("고혈압", [])
        hyper_input = {col: data.get(col, 0) for col in hyper_cols}
        hyper_input["bmi"] = bmi
        hyper_df = pd.DataFrame([hyper_input], columns=hyper_cols)
        hypertension_prob = float(
            _hypertension_model.predict_proba(hyper_df)[0][1]
        )
    else:
        hypertension_prob = 0.20
        if data.get("systolic_bp", 0) >= 140: hypertension_prob = 0.70
        elif data.get("systolic_bp", 0) >= 130: hypertension_prob = 0.45

    # 당뇨 (소수점 형태 확률 계산)
    if _diabetes_model is not None:
        diab_cols = disease_feature_map.get("당뇨", [])
        diab_input = {col: data.get(col, 0) for col in diab_cols}
        diab_input["bmi"] = bmi
        diab_df = pd.DataFrame([diab_input], columns=diab_cols)
        diabetes_prob = float(
            _diabetes_model.predict_proba(diab_df)[0][1]
        )
    else:
        diabetes_prob = 0.15
        if data.get("fasting_glucose", 0) >= 126: diabetes_prob = 0.75
        elif data.get("fasting_glucose", 0) >= 100: diabetes_prob = 0.40

    current_smoking = data.get("current_smoking", 0)
    aerobic_activity = data.get("aerobic_activity", 1)

    vitality_score = calculate_vitality_score(
        hypertension_prob=hypertension_prob,
        diabetes_prob=diabetes_prob,
        bmi=bmi,
        current_smoking=current_smoking,
        aerobic_activity=aerobic_activity
    )
    metabolic = _calc_metabolic(data)

    health_age = calculate_health_age(
        age=data.get("age", 30),
        hypertension_prob=hypertension_prob,
        diabetes_prob=diabetes_prob,
        bmi=bmi,
        current_smoking=current_smoking,
        aerobic_activity=aerobic_activity
    )

    return {
        "bmi": bmi,
        "obesity_status": obesity_status,
        "hypertension_prob": round(hypertension_prob, 4),
        "diabetes_prob": round(diabetes_prob, 4),
        "metabolic": metabolic,
        "vitality_score": vitality_score,
        "health_age": health_age,
        "healthAge": health_age
    }
