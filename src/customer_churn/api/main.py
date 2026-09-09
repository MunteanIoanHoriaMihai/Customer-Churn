from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd
from fastapi import Depends, FastAPI

from customer_churn.api.schemas import ChurnPrediction, CustomerFeatures
from customer_churn.features.encoding import transform_features
from customer_churn.features.imputation import impute_total_charges

ENCODER_PATH = Path("models/encoder.joblib")
MODEL_URI = "mlruns/1/models/m-59833d26fca14832aaca40f73f29e38b/artifacts"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(title="Customer Churn Prediction API")

_encoder = None
_model = None


def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = joblib.load(ENCODER_PATH)
    return _encoder


def get_model():
    global _model
    if _model is None:
        _model = mlflow.sklearn.load_model(MODEL_URI)
    return _model


@app.post("/predict", response_model=ChurnPrediction)
def predict(
    customer: CustomerFeatures,
    model=Depends(get_model),
    encoder=Depends(get_encoder),
) -> ChurnPrediction:
    df = pd.DataFrame([customer.model_dump()])
    df = impute_total_charges(df)
    features = transform_features(df, encoder)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return ChurnPrediction(
        churn_prediction="Yes" if prediction == 1 else "No",
        churn_probability=float(probability),
    )
