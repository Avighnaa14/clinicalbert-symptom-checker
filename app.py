from fastapi import FastAPI
from pydantic import BaseModel
from model.service import diagnose

app = FastAPI(
    title="AI Symptom Checker API",
    description="ClinicalBERT-based symptom-to-disease prediction",
    version="1.0"
)

class SymptomRequest(BaseModel):
    symptoms: str


@app.get("/")
def root():
    return {"status": "running", "message": "Symptom Checker API is live"}


@app.post("/diagnose")
def diagnose_symptoms(request: SymptomRequest):
    """
    Input : symptoms (string)
    Output: prediction + recommendation JSON
    """
    return diagnose(request.symptoms)
