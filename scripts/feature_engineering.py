"""Engineer Z-score, velocity, V-group, and client-aggregate features for one bank partition."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import PARTITIONED_DIR, PARTITION_KEY, VELOCITY_WINDOWS, V_GROUPS, AGG_COLS


def add_zscore_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card1 Z-score of TransactionAmt."""
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
                grp_counts[right] = right - left
                grp_sums[right] = amts[left:right].sum()
            counts.append(pd.Series(grp_counts, index=group.index))
            sums.append(pd.Series(grp_sums, index=group.index))

        df[count_col] = pd.concat(counts).reindex(df.index).values
        df[sum_col] = pd.concat(sums).reindex(df.index).values

    return df


def add_v_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each V-column NAN group into mean and std."""
    for group_name, cols in V_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            df[f"{group_name}_mean"] = 0.0
            df[f"{group_name}_std"] = 0.0
            continue
        sub = df[present]
        df[f"{group_name}_mean"] = sub.mean(axis=1).fillna(0.0)
        df[f"{group_name}_std"] = sub.std(axis=1).fillna(0.0)
    return df


def build_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a client-group identifier from card1 + addr1. Used for aggregation only."""
    df["_uid"] = (
        df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(int).astype(str)
    )
    return df


def add_uid_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-client-group aggregated features, then drop the UID column."""
    df = build_uid(df)

    df["uid_tx_count"] = df.groupby("_uid")["_uid"].transform("count")

    for col in AGG_COLS:
        if col in df.columns:
            df[f"uid_{col}_mean"] = df.groupby("_uid")[col].transform("mean")
        else:
            df[f"uid_{col}_mean"] = 0.0

    df.drop(columns=["_uid"], inplace=True)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply global category maps for consistent encoding across banks."""
    maps_path = PARTITIONED_DIR / "category_maps.json"
    if not maps_path.exists():
        raise FileNotFoundError(
            f"{maps_path} not found — run build_category_maps.py first"
        )

    with open(maps_path) as f:
        category_maps = json.load(f)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col in category_maps:
            mapping = category_maps[col]
            df[col] = df[col].fillna("_NAN_").astype(str).map(mapping).fillna(-1).astype(int)
        else:
            df[col] = df[col].astype("category").cat.codes
    return df


def engineer(bank: str) -> None:
    src = PARTITIONED_DIR / f"bank_{bank}.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — run partition_data.py first")

    print(f"Loading {src}...")
    df = pd.read_csv(src)
    print(f"  Rows: {len(df):,}")

    df = add_zscore_feature(df)
    print("  Added z-score feature")
    df = add_velocity_features(df)
    print("  Added velocity features")
    df = add_v_group_features(df)
    print("  Added V-group summary features")
    df = add_uid_aggregated_features(df)
    print("  Added UID-aggregated features")
    df = encode_categoricals(df)
    print("  Encoded categoricals")

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
