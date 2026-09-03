import pandas as pd


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def remove_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    # TotalCharges may be NaN (missing, to be handled after EDA)
    # Only reject values that are present and negative
    mask = (
        (df["tenure"] >= 0)
        & (df["MonthlyCharges"] > 0)
        & ((df["TotalCharges"] >= 0) | df["TotalCharges"].isna())
    )
    return df[mask].reset_index(drop=True)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = fix_dtypes(df)
    df = remove_duplicate_rows(df)
    df = remove_impossible_values(df)
    return df
