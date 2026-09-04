import numpy as np
import pandas as pd

from customer_churn.features.imputation import impute_total_charges


def test_impute_total_charges_fills_nan_with_zero():
    df = pd.DataFrame({"TotalCharges": [29.85, np.nan, 108.15]})

    result = impute_total_charges(df)

    assert result["TotalCharges"].tolist() == [29.85, 0.0, 108.15]


def test_impute_total_charges_does_not_mutate_input():
    df = pd.DataFrame({"TotalCharges": [np.nan]})

    impute_total_charges(df)

    assert pd.isna(df["TotalCharges"].iloc[0])
