from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from customer_churn.data.loader import load_data
from customer_churn.models.training import train_and_register

PROCESSED_TRAIN_PATH = Path("data/processed/train.csv")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customer_churn")

RF_BEST_PARAMS = {
    "n_estimators": 250,
    "max_depth": 30,
    "min_samples_split": 8,
    "min_samples_leaf": 5,
    "max_features": "log2",
    "class_weight": "balanced",
    "random_state": 42,
}

XGB_BEST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 8,
    "learning_rate": 0.022634013027296136,
    "subsample": 0.7241699532621577,
    "colsample_bytree": 0.7815370212512751,
    "scale_pos_weight": 2.77,
    "random_state": 42,
    "eval_metric": "logloss",
}


def main() -> None:
    df = load_data(PROCESSED_TRAIN_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    train_and_register(
        RandomForestClassifier,
        RF_BEST_PARAMS,
        "random_forest_registered",
        "churn_random_forest",
        mlflow.sklearn.log_model,
        X,
        y,
    )
    train_and_register(
        XGBClassifier,
        XGB_BEST_PARAMS,
        "xgboost_registered",
        "churn_xgboost",
        mlflow.xgboost.log_model,
        X,
        y,
    )

    print("Both models trained on full train set and registered in MLflow.")


if __name__ == "__main__":
    main()
