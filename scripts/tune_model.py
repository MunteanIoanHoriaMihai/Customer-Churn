import argparse
from pathlib import Path

import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from customer_churn.data.loader import load_data
from customer_churn.models.training import (
    make_oversampled_rf,
    make_oversampled_xgb,
    rf_param_space,
    tune_with_optuna,
    xgb_param_space,
)

PROCESSED_TRAIN_PATH = Path("data/processed/train.csv")
N_TRIALS = 25

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customer_churn")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oversample", action="store_true")
    args = parser.parse_args()

    rf_model = make_oversampled_rf if args.oversample else RandomForestClassifier
    xgb_model = make_oversampled_xgb if args.oversample else XGBClassifier
    suffix = "_oversampled" if args.oversample else "_tuned"

    df = load_data(PROCESSED_TRAIN_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_best = tune_with_optuna(
        rf_model,
        rf_param_space,
        {"random_state": 42},
        f"random_forest{suffix}",
        X,
        y,
        cv,
        n_trials=N_TRIALS,
    )
    print("Random Forest best params:", rf_best)

    xgb_best = tune_with_optuna(
        xgb_model,
        xgb_param_space,
        {"random_state": 42, "eval_metric": "logloss"},
        f"xgboost{suffix}",
        X,
        y,
        cv,
        n_trials=N_TRIALS,
    )
    print("XGBoost best params:", xgb_best)


if __name__ == "__main__":
    main()
