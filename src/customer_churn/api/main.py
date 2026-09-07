from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI

from customer_churn.api.schemas import ChurnPrediction, CustomerFeatures
from customer_churn.features.encoding import transform_features
from customer_churn.features.imputation import impute_total_charges

ENCODER_PATH = Path("models/encoder.joblib")
MODEL_URI = "models:/churn_xgboost/2"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(title="Customer Churn Prediction API")

encoder = joblib.load(ENCODER_PATH)
model = mlflow.sklearn.load_model(MODEL_URI)


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures) -> ChurnPrediction:
    df = pd.DataFrame([customer.model_dump()])
    df = impute_total_charges(df)
    features = transform_features(df, encoder)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return ChurnPrediction(
        churn_prediction="Yes" if prediction == 1 else "No",
        churn_probability=float(probability),
    )
