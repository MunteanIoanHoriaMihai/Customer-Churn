import numpy as np
import pandas as pd

from customer_churn.data.cleaning import (
    fix_dtypes,
    remove_duplicate_rows,
    remove_impossible_values,
)


def test_fix_dtypes_converts_totalcharges_to_numeric():
    df = pd.DataFrame({"TotalCharges": ["29.85", " ", "108.15"]})

    result = fix_dtypes(df)

    assert result["TotalCharges"].tolist()[0] == 29.85
    assert pd.isna(result["TotalCharges"].iloc[1])
    assert result["TotalCharges"].tolist()[2] == 108.15


def test_remove_duplicate_rows_drops_exact_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    result = remove_duplicate_rows(df)

    assert len(result) == 2


def test_remove_impossible_values_filters_negative_tenure():
    df = pd.DataFrame(
        {
            "tenure": [5, -1, 10],
            "MonthlyCharges": [50.0, 60.0, 70.0],
            "TotalCharges": [100.0, 200.0, 300.0],
        }
    )

    result = remove_impossible_values(df)

    assert result["tenure"].tolist() == [5, 10]


def test_remove_impossible_values_filters_non_positive_monthly_charges():
    df = pd.DataFrame(
        {
            "tenure": [5, 10, 15],
            "MonthlyCharges": [50.0, 0.0, -5.0],
            "TotalCharges": [100.0, 200.0, 300.0],
        }
    )

    result = remove_impossible_values(df)

    assert result["MonthlyCharges"].tolist() == [50.0]


def test_remove_impossible_values_keeps_nan_totalcharges():
    df = pd.DataFrame(
        {
            "tenure": [0],
            "MonthlyCharges": [50.0],
            "TotalCharges": [np.nan],
        }
    )

    result = remove_impossible_values(df)

    assert len(result) == 1
