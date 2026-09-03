from pathlib import Path

from customer_churn.data.cleaning import clean_raw_data
from customer_churn.data.loader import load_data

RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
INTERIM_PATH = Path("data/interim/cleaned.csv")


def main() -> None:
    df = load_data(RAW_PATH)
    df = clean_raw_data(df)
    df.to_csv(INTERIM_PATH, index=False)
    print(f"Saved {len(df)} rows to {INTERIM_PATH}")


if __name__ == "__main__":
    main()
