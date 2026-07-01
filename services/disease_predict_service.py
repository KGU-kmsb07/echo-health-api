import os
import warnings

import joblib
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

SIMPLE_FEATURES = [
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "waist_cm",
    "bmi",
    "current_smoking",
    "aerobic_activity",
]

CHECKUP_FEATURES = SIMPLE_FEATURES + [
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "hba1c",
    "total_cholesterol",
    "hdl_cholesterol",
    "triglyceride",
    "ldl_direct",
]

DEFAULTS = {
    "waist_cm": 80,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "fasting_glucose": 90,
    "hba1c": 5.2,
    "total_cholesterol": 180,
    "hdl_cholesterol": 50,
    "triglyceride": 120,
    "ldl_direct": 110,
    "current_smoking": 0,
    "aerobic_activity": 1,
}

MODEL_REGISTRY = {
    "simple": {
        "hypertension": {
            "label": "고혈압",
            "file": "simple/hypertension_logistic_regression.pkl",
            "model": "Logistic Regression",
            "threshold": 0.36,
            "metrics": {"accuracy": 0.7032, "precision": 0.6476, "recall": 0.8779, "f1": 0.7454, "roc_auc": 0.7864, "pr_auc": 0.7511},
        },
        "diabetes": {
            "label": "당뇨",
            "file": "simple/diabetes_logistic_regression.pkl",
            "model": "Logistic Regression",
            "threshold": 0.56,
            "metrics": {"accuracy": 0.7123, "precision": 0.3078, "recall": 0.6536, "f1": 0.4185, "roc_auc": 0.7688, "pr_auc": 0.3384},
        },
        "stroke": {
            "label": "뇌졸중",
            "file": None,
            "model": "Simple Rule",
            "threshold": 0.02,
            "metrics": {"accuracy": 0.9163, "precision": 0.0843, "recall": 0.3043, "f1": 0.1321, "roc_auc": 0.7405, "pr_auc": 0.0797},
        },
        "heart_disease": {
            "label": "심장질환",
            "file": "simple/heart_disease_logistic_regression.pkl",
            "model": "Logistic Regression",
            "threshold": 0.68,
            "metrics": {"accuracy": 0.8417, "precision": 0.1297, "recall": 0.6486, "f1": 0.2162, "roc_auc": 0.8513, "pr_auc": 0.1216},
            "notice": "이 결과는 의료 진단이 아닙니다. 심장질환 위험 가능성을 미리 살피기 위한 참고 신호이며, 흉통·호흡곤란·갑작스러운 불편감이 있거나 결과가 걱정된다면 의료진 상담을 우선해 주세요.",
        },
        "cancer": {
            "label": "암",
            "file": "simple/cancer_logistic_regression.pkl",
            "model": "Logistic Regression",
            "threshold": 0.05,
            "metrics": {"accuracy": 0.5469, "precision": 0.0911, "recall": 0.6184, "f1": 0.1588, "roc_auc": 0.6106, "pr_auc": 0.0908},
        },
    },
}

_model_cache = {}


def _to_float(value, default=0):
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default=0):
    return int(round(_to_float(value, default)))


def _load_model(path):
    if path not in _model_cache:
        full_path = os.path.join(MODEL_DIR, path.replace("/", os.sep))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(full_path)
        for _, step in getattr(model, "steps", []):
            if step.__class__.__name__ == "SimpleImputer" and not hasattr(step, "_fill_dtype"):
                step._fill_dtype = getattr(step, "_fit_dtype", None)
        _model_cache[path] = model
    return _model_cache[path]


def _prepare_input(raw):
    data = dict(raw)
    for key, value in DEFAULTS.items():
        if data.get(key) in (None, ""):
            data[key] = value
    data["age"] = _to_float(data.get("age"), 30)
    data["sex"] = _to_float(data.get("sex"), 1)
    data["height_cm"] = _to_float(data.get("height_cm"), 170)
    data["weight_kg"] = _to_float(data.get("weight_kg"), 70)
    data["waist_cm"] = _to_float(data.get("waist_cm"), 80)
    data["bmi"] = round(data["weight_kg"] / ((data["height_cm"] / 100) ** 2), 2)
    for key in CHECKUP_FEATURES:
        data[key] = _to_float(data.get(key), DEFAULTS.get(key, 0))
    return data


def _detect_input_mode(raw):
    if raw.get("input_mode") == "checkup":
        return "checkup"
    checkup_keys = set(CHECKUP_FEATURES) - set(SIMPLE_FEATURES)
    return "checkup" if any(raw.get(key) not in (None, "") for key in checkup_keys) else "simple"


def _predict_one(model, data, threshold):
    if model is None:
        age = data.get("age", 30)
        bmi = data.get("bmi", 22)
        smoking = data.get("current_smoking", 0)
        probability = min(0.25, max(0.005, 0.004 + (age - 40) * 0.0015 + (bmi - 23) * 0.003 + smoking * 0.015))
        return {
            "probability": round(probability, 4),
            "percent": round(probability * 100, 1),
            "prediction": int(probability >= threshold),
        }
    columns = list(getattr(model, "feature_names_in_", CHECKUP_FEATURES))
    frame = pd.DataFrame([{column: data.get(column, 0) for column in columns}], columns=columns)
    probability = float(model.predict_proba(frame)[0][1])
    return {
        "probability": round(probability, 4),
        "percent": round(probability * 100, 1),
        "prediction": int(probability >= threshold),
    }


def _apply_checkup_adjustments(key, result, data, input_mode):
    if input_mode != "checkup":
        return result

    probability = result["probability"]
    if key == "hypertension":
        systolic = data.get("systolic_bp", 120)
        diastolic = data.get("diastolic_bp", 80)
        if systolic >= 140 or diastolic >= 90:
            probability = max(probability, 0.72)
        elif systolic >= 130 or diastolic >= 85:
            probability = max(probability, 0.45)
    elif key == "diabetes":
        glucose = data.get("fasting_glucose", 90)
        hba1c = data.get("hba1c", 5.2)
        if glucose >= 126 or hba1c >= 6.5:
            probability = max(probability, 0.75)
        elif glucose >= 100 or hba1c >= 5.7:
            probability = max(probability, 0.40)
    elif key == "heart_disease":
        total = data.get("total_cholesterol", 180)
        hdl = data.get("hdl_cholesterol", 50)
        triglyceride = data.get("triglyceride", 120)
        systolic = data.get("systolic_bp", 120)
        if total >= 240 or hdl < 40 or triglyceride >= 200 or systolic >= 140:
            probability = max(probability, min(0.85, probability + 0.08))

    result.update({
        "probability": round(probability, 4),
        "percent": round(probability * 100, 1),
        "prediction": int(probability >= result.get("threshold", 0.5)),
    })
    return result


def _metabolic_percent(data, input_mode, weight):
    if input_mode != "checkup":
        return 70 if weight["prediction"] else 10

    score = 0
    if data.get("waist_cm", 0) >= 90:
        score += 1
    if data.get("systolic_bp", 0) >= 130 or data.get("diastolic_bp", 0) >= 85:
        score += 1
    if data.get("fasting_glucose", 0) >= 100 or data.get("hba1c", 0) >= 5.7:
        score += 1
    if data.get("triglyceride", 0) >= 150:
        score += 1
    if data.get("hdl_cholesterol", 999) < 40:
        score += 1

    if score >= 3:
        return 70
    if score == 2:
        return 45
    if score == 1:
        return 20
    return 10


def _weight_management(data):
    bmi = data["bmi"]
    if bmi >= 25:
        level = "비만"
        score = 0.75
    elif bmi >= 23:
        level = "과체중"
        score = 0.55
    elif bmi < 18.5:
        level = "저체중"
        score = 0.45
    else:
        level = "정상"
        score = 0.10
    return {
        "probability": score,
        "percent": round(score * 100, 1),
        "prediction": int(bmi >= 23),
        "level": level,
        "rule": "BMI 23 이상은 과체중 이상 체중관리군으로 분류합니다.",
    }


def _health_age(age, risks, weight_management):
    delta = 0
    if risks["hypertension"]["probability"] >= 0.6:
        delta += 4
    elif risks["hypertension"]["probability"] >= 0.35:
        delta += 2
    if risks["diabetes"]["probability"] >= 0.6:
        delta += 4
    elif risks["diabetes"]["probability"] >= 0.35:
        delta += 2
    if risks["heart_disease"]["probability"] >= 0.6:
        delta += 3
    elif risks["heart_disease"]["probability"] >= 0.35:
        delta += 1.5
    if weight_management["prediction"]:
        delta += 2
    return int(round(max(age - 5, age + delta)))


def predict(data):
    input_mode = _detect_input_mode(data)
    prepared = _prepare_input(data)
    registry = MODEL_REGISTRY["simple"]
    risks = {}

    for key, config in registry.items():
        model = _load_model(config["file"]) if config.get("file") else None
        result = _predict_one(model, prepared, config["threshold"])
        result.update({
            "label": config["label"],
            "model": config["model"],
            "threshold": config["threshold"],
            "metrics": config["metrics"],
        })
        result = _apply_checkup_adjustments(key, result, prepared, input_mode)
        if config.get("notice"):
            result["notice"] = config["notice"]
        risks[key] = result

    weight = _weight_management(prepared)
    metabolic_percent = _metabolic_percent(prepared, input_mode, weight)
    vitality_score = round(max(0, min(100, 100 - risks["hypertension"]["percent"] * 0.18 - risks["diabetes"]["percent"] * 0.18 - risks["heart_disease"]["percent"] * 0.12 - (15 if weight["prediction"] else 0))))
    health_age = _health_age(_to_int(prepared["age"], 30), risks, weight)

    return {
        "input_mode": input_mode,
        "feature_set_used": "건강검진연동형" if input_mode == "checkup" else "간편입력형",
        "bmi": prepared["bmi"],
        "obesity_status": 1 if prepared["bmi"] >= 25 else 0,
        "weight_management": weight,
        "hypertension_prob": risks["hypertension"]["probability"],
        "diabetes_prob": risks["diabetes"]["probability"],
        "stroke_prob": risks["stroke"]["probability"],
        "heart_disease_prob": risks["heart_disease"]["probability"],
        "cancer_prob": risks["cancer"]["probability"],
        "cardiovascular_signal": risks["heart_disease"]["percent"],
        "cardiovascular_signal_prob": risks["heart_disease"]["probability"],
        "cardiovascular_notice": risks["heart_disease"].get("notice"),
        "metabolic": metabolic_percent,
        "vitality_score": vitality_score,
        "health_age": health_age,
        "healthAge": health_age,
        "risks": risks,
        "checkup_used": input_mode == "checkup",
        "model_notice": "질병 위험도는 국민건강영양조사 2024 기반 통계 모델 결과이며 의료 진단이 아닙니다.",
    }
