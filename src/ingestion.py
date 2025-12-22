import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}")

    df = pd.read_csv(path)

    return df


def basic_data_checks(df: pd.DataFrame):
    print("=== SHAPE ===")
    print(df.shape)

    print("\n=== FRAUD RATIO ===")
    print(df["Class"].value_counts(normalize=True))

    print("\n=== AMOUNT STATS ===")
    print(df["Amount"].describe())

    print("\n=== TIME RANGE ===")
    print(df["Time"].min(), "→", df["Time"].max())