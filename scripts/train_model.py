from pathlib import Path

import mlflow
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from customer_churn.data.loader import load_data
from customer_churn.models.training import train_and_log

PROCESSED_TRAIN_PATH = Path("data/processed/train.csv")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customer_churn")


def main() -> None:
    df = load_data(PROCESSED_TRAIN_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(random_state=42),
        "xgboost": XGBClassifier(random_state=42, eval_metric="logloss"),
    }

    for name, model in candidates.items():
        metrics = train_and_log(model, name, X, y, cv)
        print(name, metrics)


if __name__ == "__main__":
    main()
