"""Split IEEE-CIS dataset into 3 non-IID bank partitions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import DATA_DIR, PARTITIONED_DIR, PARTITION_KEY, TARGET_COL, BANKS


def load_raw_data() -> pd.DataFrame:
    tx = pd.read_csv(DATA_DIR / "train_transaction.csv")
    identity = pd.read_csv(DATA_DIR / "train_identity.csv")
    df = tx.merge(identity, on="TransactionID", how="left")
    return df


def partition(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Assign card1 groups to banks non-IID (fraud rates differ per bank)."""
    rng = np.random.default_rng(seed)

    # Compute per-card1 fraud rate
    card_stats = (
        df.groupby(PARTITION_KEY)[TARGET_COL]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COL: "fraud_rate"})
    )
    # Sort by fraud rate and split: bank_a gets low-fraud cards, bank_c gets high
    card_stats = card_stats.sort_values("fraud_rate").reset_index(drop=True)
    n = len(card_stats)
    thirds = [n // 3, 2 * (n // 3), n]
    card_sets = {
        "a": set(card_stats.iloc[: thirds[0]][PARTITION_KEY]),
        "b": set(card_stats.iloc[thirds[0] : thirds[1]][PARTITION_KEY]),
        "c": set(card_stats.iloc[thirds[1] :][PARTITION_KEY]),
    }
    return {bank: df[df[PARTITION_KEY].isin(cards)].copy() for bank, cards in card_sets.items()}


def verify_non_iid(partitions: dict[str, pd.DataFrame]) -> None:
    rates = {bank: df[TARGET_COL].mean() for bank, df in partitions.items()}
    print("Fraud rates per bank:", {k: f"{v:.4f}" for k, v in rates.items()})
    rate_vals = list(rates.values())
    for i in range(len(rate_vals)):
        for j in range(i + 1, len(rate_vals)):
            diff = abs(rate_vals[i] - rate_vals[j])
            assert diff > 0.01, f"Banks too similar (diff={diff:.4f}); partition may be IID"


def main() -> None:
    PARTITIONED_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading raw data...")
    df = load_raw_data()
    print(f"Total rows: {len(df):,}  |  Fraud rate: {df[TARGET_COL].mean():.4f}")

    partitions = partition(df)
    verify_non_iid(partitions)

    for bank, shard in partitions.items():
        out = PARTITIONED_DIR / f"bank_{bank}.csv"
        shard.to_csv(out, index=False)
        print(f"Saved bank_{bank}: {len(shard):,} rows → {out}")


if __name__ == "__main__":
    main()
