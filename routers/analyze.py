from fastapi import APIRouter
import pickle
import numpy as np
import pandas as pd
import json
import os

router = APIRouter()

# 모델 로드
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "models/hypertension_model.pkl"), "rb") as f:
    hypertension_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "models/diabetes_model.pkl"), "rb") as f:
    diabetes_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "config/feature_columns.json"), "r") as f:
    feature_columns = json.load(f)

def calc_bmi(weight_kg, height_cm):
    h = height_cm / 100
    return round(weight_kg / (h ** 2), 1)

def calc_obesity(bmi):
    if bmi >= 30: return 75
    if bmi >= 25: return 50
    if bmi >= 23: return 25
    return 10

def calc_metabolic(data):
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

@router.post("/analyze")
def analyze(data: dict):
    try:
        bmi = calc_bmi(data.get("weight_kg", 70), data.get("height_cm", 170))

        # 고혈압 피처 (혈압 제외)
        hyper_cols = feature_columns.get("hypertension", [])
        hyper_input = {col: data.get(col, 0) for col in hyper_cols}
        hyper_df = pd.DataFrame([hyper_input])
        hyper_prob = float(hypertension_model.predict_proba(hyper_df)[0][1]) * 100

        # 당뇨 피처 (혈당 제외)
        diab_cols = feature_columns.get("diabetes", [])
        diab_input = {col: data.get(col, 0) for col in diab_cols}
        diab_df = pd.DataFrame([diab_input])
        diab_prob = float(diabetes_model.predict_proba(diab_df)[0][1]) * 100

        # 비만 rule-based
        obesity_prob = calc_obesity(bmi)

        # 대사증후군 rule-based
        metabolic_prob = calc_metabolic(data)

        # 건강점수 계산
        health_score = max(0, round(100 - (
            hyper_prob * 0.3 +
            diab_prob * 0.3 +
            obesity_prob * 0.2 +
            metabolic_prob * 0.2
        ) / 2))

        # 건강나이
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

    except Exception as e:
        return {"error": str(e)}