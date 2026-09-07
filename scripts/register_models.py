from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from customer_churn.data.loader import load_data
from customer_churn.models.training import make_oversampled_xgb, train_and_register

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

XGB_OVERSAMPLED_BEST_PARAMS = {
    "n_estimators": 350,
    "max_depth": 5,
    "learning_rate": 0.015707794120719844,
    "subsample": 0.8419266701068213,
    "colsample_bytree": 0.6761745692348862,
    "scale_pos_weight": 1,
    "random_state": 42,
    "eval_metric": "logloss",
}


def register_random_forest(X: pd.DataFrame, y: pd.Series) -> None:
    train_and_register(
        RandomForestClassifier,
        RF_BEST_PARAMS,
        "random_forest_registered",
        "churn_random_forest",
        mlflow.sklearn.log_model,
        X,
        y,
    )


def register_xgboost(X: pd.DataFrame, y: pd.Series) -> None:
    train_and_register(
        XGBClassifier,
        XGB_BEST_PARAMS,
        "xgboost_registered",
        "churn_xgboost",
        mlflow.xgboost.log_model,
        X,
        y,
    )


def register_xgboost_oversampled(X: pd.DataFrame, y: pd.Series) -> None:
    train_and_register(
        make_oversampled_xgb,
        XGB_OVERSAMPLED_BEST_PARAMS,
        "xgboost_oversampled_registered",
        "churn_xgboost",
        mlflow.sklearn.log_model,
        X,
        y,
        log_model_kwargs={
            "skops_trusted_types": [
                "imblearn.over_sampling._random_over_sampler.RandomOverSampler",
                "imblearn.pipeline.Pipeline",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ]
        },
    )


def main() -> None:
    df = load_data(PROCESSED_TRAIN_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # register_random_forest(X, y)  # already registered as churn_random_forest v1
    # register_xgboost(X, y)  # already registered as churn_xgboost v1
    register_xgboost_oversampled(X, y)  # registers as churn_xgboost v2

    print("XGBoost (oversampled) trained on full train set and registered as a new version.")


if __name__ == "__main__":
    main()
