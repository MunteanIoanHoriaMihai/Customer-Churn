import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from customer_churn.models.evaluation import evaluate_model


def test_evaluate_model_returns_expected_keys():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    model = DecisionTreeClassifier(random_state=42).fit(X, y)

    metrics = evaluate_model(model, X, y)

    assert set(metrics.keys()) == {"f1", "roc_auc", "precision", "recall", "confusion_matrix"}


def test_evaluate_model_perfect_predictions():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    model = DecisionTreeClassifier(random_state=42).fit(X, y)

    metrics = evaluate_model(model, X, y)

    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"] == [[3, 0], [0, 3]]
