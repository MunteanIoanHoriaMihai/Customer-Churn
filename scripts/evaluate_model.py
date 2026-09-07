from pathlib import Path

import mlflow
import mlflow.sklearn

from customer_churn.data.loader import load_data
from customer_churn.models.evaluation import evaluate_model

PROCESSED_TEST_PATH = Path("data/processed/test.csv")
MODEL_URI = "models:/churn_xgboost/2"

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def main() -> None:
    df = load_data(PROCESSED_TEST_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    model = mlflow.sklearn.load_model(MODEL_URI)
    metrics = evaluate_model(model, X, y)

    print(f"Model: {MODEL_URI}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
