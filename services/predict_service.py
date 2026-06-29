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

def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

def calculate_vitality_score(hypertension_prob: float, diabetes_prob: float, bmi: float,
                              current_smoking: int, aerobic_activity: int) -> int:
    # 소수 → 퍼센트 변환
    if hypertension_prob <= 1.0:
        hypertension_prob *= 100
    if diabetes_prob <= 1.0:
        diabetes_prob *= 100

    score = 100.0
    score -= hypertension_prob * 0.2
    score -= diabetes_prob * 0.2
    if bmi >= 25:
        score -= 15
    if current_smoking == 1:
        score -= 15
    if aerobic_activity == 0:
        score -= 15

    return max(0, min(100, round(score)))

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

    current_smoking = data.get("current_smoking", 0)
    aerobic_activity = data.get("aerobic_activity", 1)

    vitality_score = calculate_vitality_score(
        hypertension_prob=hyper_prob,
        diabetes_prob=diab_prob,
        bmi=bmi,
        current_smoking=current_smoking,
        aerobic_activity=aerobic_activity
    )

    health_age = calculate_health_age(
        age=data.get("age", 30),
        hypertension_prob=hyper_prob / 100.0,
        diabetes_prob=diab_prob / 100.0,
        bmi=bmi,
        current_smoking=current_smoking,
        aerobic_activity=aerobic_activity
    )

    return {
        "bmi": bmi,
        "obesity_status": obesity_status,
        "hypertension_prob": round(hyper_prob / 100.0, 4),
        "diabetes_prob": round(diab_prob / 100.0, 4),
        "vitality_score": vitality_score,
        "health_age": health_age,
        "healthAge": health_age
    }
