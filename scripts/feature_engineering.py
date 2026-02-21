"""Engineer Z-score and velocity features for one bank partition."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import PARTITIONED_DIR, PARTITION_KEY, VELOCITY_WINDOWS


def add_zscore_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card1 rolling Z-score of TransactionAmt."""
    grp = df.groupby(PARTITION_KEY)["TransactionAmt"]
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(1.0).replace(0, 1.0)
    df["amt_zscore"] = (df["TransactionAmt"] - mean) / std
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling count and sum per card1 within time windows (no data leakage)."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    for label, window_sec in VELOCITY_WINDOWS.items():
        short = label.replace("min", "m").replace("hr", "h")
        count_col = f"tx_count_{short}"
        sum_col = f"tx_sum_{short}"

        counts, sums = [], []
        for _, group in df.groupby(PARTITION_KEY, sort=False):
            times = group["TransactionDT"].values
            amts = group["TransactionAmt"].values
            n = len(times)
            grp_counts = np.zeros(n, dtype=np.int32)
            grp_sums = np.zeros(n, dtype=np.float64)
            left = 0
            for right in range(n):
                while times[right] - times[left] > window_sec:
                    left += 1
                # Intentionally excludes the current transaction (index `right`).
                # Count and sum cover [left, right-1]: transactions that occurred
                # strictly before the current one within the window. This is the
                # correct leakage-free design — the current transaction's amount
                # must not contribute to its own velocity feature.
                grp_counts[right] = right - left
                grp_sums[right] = amts[left:right].sum()
            counts.append(pd.Series(grp_counts, index=group.index))
            sums.append(pd.Series(grp_sums, index=group.index))

        df[count_col] = pd.concat(counts).reindex(df.index).values
        df[sum_col] = pd.concat(sums).reindex(df.index).values

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns in place."""
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def engineer(bank: str) -> None:
    src = PARTITIONED_DIR / f"bank_{bank}.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — run partition_data.py first")

    print(f"Loading {src}...")
    df = pd.read_csv(src)
    print(f"  Rows: {len(df):,}")

    df = add_zscore_feature(df)
    df = add_velocity_features(df)
    df = encode_categoricals(df)

    out = PARTITIONED_DIR / f"bank_{bank}_engineered.csv"
    df.to_csv(out, index=False)
    print(f"Saved engineered data → {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True, choices=["a", "b", "c"])
    args = parser.parse_args()
    engineer(args.bank)


if __name__ == "__main__":
    main()
