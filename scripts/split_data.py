from pathlib import Path

from sklearn.model_selection import train_test_split

from customer_churn.data.loader import load_data

INTERIM_PATH = Path("data/interim/cleaned.csv")
TRAIN_PATH = Path("data/interim/train.csv")
TEST_PATH = Path("data/interim/test.csv")


def main() -> None:
    df = load_data(INTERIM_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Churn"])
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Saved {len(train_df)} rows to {TRAIN_PATH}")
    print(f"Saved {len(test_df)} rows to {TEST_PATH}")


if __name__ == "__main__":
    main()
