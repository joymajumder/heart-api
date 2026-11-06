from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import HeartInput, PredictionOut

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "heart_model.joblib"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURES = artifact["features"]

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model_type": type(model.named_steps["clf"]).__name__,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "target": "heart_disease (True = presence, False = absence)",
    }


@app.post("/predict", response_model=PredictionOut)
def predict(payload: HeartInput):
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    return {"heart_disease": bool(model.predict(row)[0] == 1)}
