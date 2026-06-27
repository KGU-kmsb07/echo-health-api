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


import os
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)