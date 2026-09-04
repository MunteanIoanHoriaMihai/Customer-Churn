import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def encode_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[CATEGORICAL_COLS])

    train_encoded = pd.DataFrame(
        encoder.transform(train_df[CATEGORICAL_COLS]),
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=train_df.index,
    )
    test_encoded = pd.DataFrame(
        encoder.transform(test_df[CATEGORICAL_COLS]),
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=test_df.index,
    )

    train_out = pd.concat([train_df[NUMERIC_COLS], train_encoded], axis=1)
    test_out = pd.concat([test_df[NUMERIC_COLS], test_encoded], axis=1)

    return train_out, test_out
