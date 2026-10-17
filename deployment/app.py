
from pathlib import Path
import logging

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

# Use the most recently saved model folder.
model_paths = sorted(
    MODEL_DIR.glob("heart_disease_model_*"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if not model_paths:
    raise RuntimeError(
        "No saved model found. Run `python -m src.inference` first."
    )

MODEL_PATH = model_paths[0]
model = mlflow.sklearn.load_model(str(MODEL_PATH))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational ML inference API using the UCI Cleveland dataset.",
    version="1.0.0",
)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


class PatientInput(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: float = Field(ge=0, le=1)
    cp: float = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=700)
    fbs: float = Field(ge=0, le=1)
    restecg: float = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=250)
    exang: float = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: float = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=3, le=7)


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running",
        "model": MODEL_PATH.name,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientInput):
    try:
        input_data = pd.DataFrame(
            [patient.model_dump()],
            columns=FEATURES,
        )

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        positive_probability = float(probabilities[1])

        result = {
            "prediction": prediction,
            "prediction_label": (
                "disease_present" if prediction == 1 else "no_disease_detected"
            ),
            "probability_disease_present": round(positive_probability, 4),
            "model": MODEL_PATH.name,
        }

        logger.info("Prediction request processed successfully")
        return result

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the application logs.",
        )
