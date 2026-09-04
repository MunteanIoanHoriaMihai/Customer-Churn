from pathlib import Path

from customer_churn.data.loader import load_data
from customer_churn.features.encoding import encode_features
from customer_churn.features.imputation import impute_total_charges

TRAIN_PATH = Path("data/interim/train.csv")
TEST_PATH = Path("data/interim/test.csv")
PROCESSED_TRAIN_PATH = Path("data/processed/train.csv")
PROCESSED_TEST_PATH = Path("data/processed/test.csv")


def main() -> None:
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    train_df = impute_total_charges(train_df)
    test_df = impute_total_charges(test_df)

    train_df["Churn"] = train_df["Churn"].map({"Yes": 1, "No": 0})
    test_df["Churn"] = test_df["Churn"].map({"Yes": 1, "No": 0})

    train_features, test_features = encode_features(train_df, test_df)
    train_features["Churn"] = train_df["Churn"].to_numpy()
    test_features["Churn"] = test_df["Churn"].to_numpy()

    train_features.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_features.to_csv(PROCESSED_TEST_PATH, index=False)
    print(f"Saved {len(train_features)} rows to {PROCESSED_TRAIN_PATH}")
    print(f"Saved {len(test_features)} rows to {PROCESSED_TEST_PATH}")


if __name__ == "__main__":
    main()
