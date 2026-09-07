# Customer Churn Prediction

Machine learning project for predicting customer churn, using the Telco Customer Churn dataset (IBM/Kaggle, ~7043 rows, target `Churn`).

## Setup

```
poetry install
poetry run pre-commit install
```

## Project structure

```
data/
├── raw/          # raw data, untouched
├── interim/      # cleaned data + train/test split
└── processed/    # final, encoded data, ready for the model
notebooks/
└── eda.ipynb     # quick raw-data audit + in-depth EDA on train
scripts/          # pipeline orchestration (entry points, no business logic)
src/customer_churn/
├── data/         # loading, cleaning
├── features/     # imputation, encoding
└── models/       # training + MLflow logging
tests/            # unit tests for the project's own logic
```

## Pipeline (step by step)

1. **Technical cleaning** (`scripts/prepare_data.py`) — dtype conversion (`TotalCharges` string → numeric), exact duplicate removal, filtering out impossible values (negative tenure, etc.). Saves `data/interim/cleaned.csv`.
2. **Train/test split** (`scripts/split_data.py`) — 80/20, stratified on `Churn`, done **before** the in-depth EDA.
3. **EDA** (`notebooks/eda.ipynb`) — quick audit on raw data (dtypes, duplicates, missing values), then in-depth analysis on train only (distributions, correlations, churn rate by category, the missing-`TotalCharges` pattern).
4. **Feature engineering** (`scripts/build_features.py`) — `TotalCharges` imputation (→ 0, justified by the pattern found in EDA: only occurs for brand-new customers, `tenure=0`), one-hot encoding (fit on train only, applied to test). Saves to `data/processed/`.
5. **Initial training + tracking** (`scripts/train_model.py`) — baseline (`DummyClassifier`), Random Forest, XGBoost, compared via cross-validation, logged to MLflow. Both models clearly beat baseline; XGBoost slightly ahead with default parameters.
6. **Hyperparameter tuning** (`scripts/tune_model.py`) — Random Forest and XGBoost, both tuned via Optuna (Bayesian/TPE search, 25 trials each, optimizing F1), search spaces including `class_weight`/`scale_pos_weight` to address class imbalance. Every trial logged to MLflow as a nested run. Result: Random Forest ends up ahead after tuning (F1 = 0.6356 vs. XGBoost's 0.6281) — the opposite of the untuned comparison, which is why both were tuned instead of just the initial winner.
7. **Model registration** (`scripts/register_models.py`) — both tuned models retrained on the full train set and registered in the MLflow Model Registry (`churn_random_forest` v1, `churn_xgboost` v1), giving a solid, versioned checkpoint before trying further improvements.
8. **SMOTE variant** — planned next: retry tuning with SMOTE-based oversampling (via `imbalanced-learn`, applied only within CV training folds to avoid leakage) to see if it beats the currently registered models; if so, register as v2.

## Key architectural decisions

A few decisions we deliberately insisted on along the way:

- **Split before EDA** — in-depth EDA runs only on train, not the full dataset, so decisions (outlier handling, imputation) aren't informed by data that should stay "unseen" until final evaluation.
- **Cleaning vs. imputation, kept separate** — `cleaning.py` only does "blind" operations (dtypes, duplicates, impossible values) that don't depend on any pattern discovered in the data. Imputation lives separately, in `features/`, explicitly justified by a finding from EDA rather than assumed upfront.
- **Same CV folds across all models** — a single `StratifiedKFold` reused for every comparison, so score differences between models come from the model itself, not from lucky splits.
- **MLflow with SQLite backend** (not file-based) — chosen specifically to enable the Model Registry (formal model versioning) at the tuning/final-model stage.

## Experiment tracking

```
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Opens a local dashboard at `http://localhost:5000`. Use the **"Model training"** tab (not "GenAI") to see runs, params, and metrics. Tracking metadata lives in `mlflow.db`; the actual model artifacts live in `mlruns/` — both are local-only in this setup and must be kept together (deleting one breaks the other).

## Testing

```
poetry run pytest
```

Unit tests only for the project's own logic (cleaning, imputation, encoding)

## Linting & formatting

```
poetry run ruff check .
poetry run ruff format .
```

Runs automatically on every commit (pre-commit hook, with `--fix`).
