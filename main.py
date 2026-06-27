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