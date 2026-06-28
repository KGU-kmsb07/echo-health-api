from fastapi import APIRouter
from services.gemini_service import generate_plan

router = APIRouter()

@router.get("/plan")
def get_plan(diabetes: float = 0, hypertension: float = 0,
             metabolic: float = 0, obesity: float = 0, age: int = 30):
    return generate_plan(diabetes, hypertension, metabolic, obesity, age)