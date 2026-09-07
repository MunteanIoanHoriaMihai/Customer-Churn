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


def transform_features(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    encoded = pd.DataFrame(
        encoder.transform(df[CATEGORICAL_COLS]),
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=df.index,
    )
    return pd.concat([df[NUMERIC_COLS], encoded], axis=1)


def encode_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[CATEGORICAL_COLS])

    train_out = transform_features(train_df, encoder)
    test_out = transform_features(test_df, encoder)

    return train_out, test_out, encoder
