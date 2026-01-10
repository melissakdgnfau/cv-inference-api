from fastapi import FastAPI

app = FastAPI(
    title="CV Inference API",
    description="Object detection inference service",
    version="0.1.0",
)

@app.get("/health")
def health_check():
    return {"status": "ok"}
