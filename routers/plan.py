from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze, plan, benefits

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(plan.router)
app.include_router(benefits.router)

@app.get("/health")
def health():
    return {"status": "ok"}

5. 각 router 파일에:
from fastapi import APIRouter, HTTPException
router = APIRouter()

@app.post("/analyze")
def analyze():
    raise HTTPException(status_code=503, detail="준비 중입니다")