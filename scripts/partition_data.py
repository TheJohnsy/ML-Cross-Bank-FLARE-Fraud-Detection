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
    """Assign card1 groups to banks non-IID (fraud rates differ per bank).

    Strategy: split cards into fraud-only, mixed, and clean buckets, then
    distribute them unevenly so every bank has fraud but at different rates.
    """
    rng = np.random.default_rng(seed)

    card_stats = (
        df.groupby(PARTITION_KEY)[TARGET_COL]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "fraud_rate", "count": "tx_count"})
    )

    # Separate cards that have any fraud from clean cards
    fraud_cards = card_stats[card_stats["fraud_rate"] > 0].copy()
    clean_cards = card_stats[card_stats["fraud_rate"] == 0].copy()

    # Shuffle both sets
    fraud_cards = fraud_cards.sample(frac=1, random_state=seed).reset_index(drop=True)
    clean_cards = clean_cards.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Distribute fraud cards unevenly: bank_c gets 60%, bank_b 28%, bank_a 12%
    nf = len(fraud_cards)
    fraud_split = {
        "a": set(fraud_cards.iloc[: int(nf * 0.12)][PARTITION_KEY]),
        "b": set(fraud_cards.iloc[int(nf * 0.12) : int(nf * 0.40)][PARTITION_KEY]),
        "c": set(fraud_cards.iloc[int(nf * 0.40) :][PARTITION_KEY]),
    }

    # Give bank_a more clean cards (dilutes its fraud rate further),
    # bank_c fewer clean cards (raises its fraud rate).
    nc = len(clean_cards)
    clean_split = {
        "a": set(clean_cards.iloc[: int(nc * 0.45)][PARTITION_KEY]),
        "b": set(clean_cards.iloc[int(nc * 0.45) : int(nc * 0.75)][PARTITION_KEY]),
        "c": set(clean_cards.iloc[int(nc * 0.75) :][PARTITION_KEY]),
    }

    card_sets = {
        bank: fraud_split[bank] | clean_split[bank] for bank in BANKS
    }
    return {bank: df[df[PARTITION_KEY].isin(cards)].copy() for bank, cards in card_sets.items()}


def verify_non_iid(partitions: dict[str, pd.DataFrame]) -> None:
    rates = {bank: df[TARGET_COL].mean() for bank, df in partitions.items()}
    print("Fraud rates per bank:", {k: f"{v:.4f}" for k, v in rates.items()})
    rate_vals = list(rates.values())
    for i in range(len(rate_vals)):
        for j in range(i + 1, len(rate_vals)):
            diff = abs(rate_vals[i] - rate_vals[j])
            assert diff > 0.005, f"Banks too similar (diff={diff:.4f}); partition may be IID"


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
