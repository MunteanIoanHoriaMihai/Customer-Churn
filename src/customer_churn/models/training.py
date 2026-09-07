from typing import Callable

import mlflow
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate

SCORING = ["f1", "roc_auc", "precision", "recall"]


def train_and_log(
    model: BaseEstimator,
    run_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
) -> dict:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(model.get_params())
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
        metrics = {metric: scores[f"test_{metric}"].mean() for metric in SCORING}
        mlflow.log_metrics(metrics)
        return metrics


def tune_with_optuna(
    model_class: type[BaseEstimator],
    param_space_fn: Callable[[optuna.Trial], dict],
    fixed_params: dict,
    run_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    n_trials: int = 25,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = param_space_fn(trial)
        model = model_class(**fixed_params, **params)
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
            metrics = {metric: scores[f"test_{metric}"].mean() for metric in SCORING}
            mlflow.log_metrics(metrics)
        return metrics["f1"]

    def log_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        completed = trial.number + 1
        if completed % 10 == 0 or completed == n_trials:
            print(
                f"[{run_name}] {completed}/{n_trials} trials done "
                f"— best f1 so far: {study.best_value:.4f}"
            )

    with mlflow.start_run(run_name=run_name):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[log_progress])
        mlflow.log_params(study.best_params)
        mlflow.log_metric("f1", study.best_value)
        return study.best_params


def train_and_register(
    model_class: type[BaseEstimator],
    params: dict,
    run_name: str,
    registered_model_name: str,
    log_model_fn: Callable[..., None],
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    model = model_class(**params)
    model.fit(X, y)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        log_model_fn(model, artifact_path="model", registered_model_name=registered_model_name)


def rf_param_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def xgb_param_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1, 2.77]),
    }
