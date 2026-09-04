import pandas as pd

from customer_churn.features.encoding import encode_features


def _make_df(contract_values):
    n = len(contract_values)
    return pd.DataFrame(
        {
            "gender": ["Female"] * n,
            "Partner": ["Yes"] * n,
            "Dependents": ["No"] * n,
            "PhoneService": ["Yes"] * n,
            "MultipleLines": ["No"] * n,
            "InternetService": ["DSL"] * n,
            "OnlineSecurity": ["No"] * n,
            "OnlineBackup": ["No"] * n,
            "DeviceProtection": ["No"] * n,
            "TechSupport": ["No"] * n,
            "StreamingTV": ["No"] * n,
            "StreamingMovies": ["No"] * n,
            "Contract": contract_values,
            "PaperlessBilling": ["Yes"] * n,
            "PaymentMethod": ["Mailed check"] * n,
            "SeniorCitizen": [0] * n,
            "tenure": [10] * n,
            "MonthlyCharges": [50.0] * n,
            "TotalCharges": [500.0] * n,
        }
    )


def test_encode_features_produces_matching_columns():
    train_df = _make_df(["Month-to-month", "One year"])
    test_df = _make_df(["Two year"])

    train_out, test_out = encode_features(train_df, test_df)

    assert list(train_out.columns) == list(test_out.columns)


def test_encode_features_handles_category_unseen_in_train():
    train_df = _make_df(["Month-to-month", "One year"])
    test_df = _make_df(["Two year"])

    train_out, test_out = encode_features(train_df, test_df)

    contract_cols = [c for c in test_out.columns if c.startswith("Contract_")]
    assert test_out.loc[0, contract_cols].sum() == 0
