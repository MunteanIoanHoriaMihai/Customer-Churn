import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
