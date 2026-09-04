import pandas as pd


def impute_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    # EDA finding: TotalCharges is NaN only for tenure=0 (brand-new customers, not yet billed)
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df
