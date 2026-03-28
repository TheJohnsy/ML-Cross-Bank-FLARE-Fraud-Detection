from pathlib import Path
from typing import Final

BASE_DIR: Final[Path] = Path(__file__).parent
DATA_DIR: Final[Path] = BASE_DIR / "data" / "raw"
PARTITIONED_DIR: Final[Path] = BASE_DIR / "data" / "partitioned"
MODEL_DIR: Final[Path] = BASE_DIR / "data" / "models"

TARGET_COL: Final[str] = "isFraud"
PARTITION_KEY: Final[str] = "card1"

# V-column groups by shared NAN pattern (discovered via EDA).
# Each group is summarised into mean+std during feature engineering.
# Group 12 (V322-V339) excluded — temporally inconsistent.
V_GROUPS: Final[dict[str, list[str]]] = {
    "vg1":  [f"V{i}" for i in range(1, 12)],
    "vg2":  [f"V{i}" for i in range(12, 35)],
    "vg3":  [f"V{i}" for i in range(35, 53)],
    "vg4":  [f"V{i}" for i in range(53, 75)],
    "vg5":  [f"V{i}" for i in range(75, 95)],
    "vg6":  [f"V{i}" for i in range(95, 138)],
    "vg7":  [f"V{i}" for i in range(138, 167)],
    "vg8":  [f"V{i}" for i in range(167, 217)],
    "vg9":  [f"V{i}" for i in list(range(217, 220)) + list(range(223, 227)) + list(range(228, 234)) + list(range(235, 279))],
    "vg10": [f"V{i}" for i in [220, 221, 222, 227, 234] + list(range(279, 284))],
    "vg11": [f"V{i}" for i in list(range(284, 322))],
    # vg12 (V322-V339) intentionally excluded
}

# Columns used for client-group aggregation
AGG_COLS: Final[list[str]] = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
]

FEATURE_COLS: Final[list[str]] = [
    "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4",
    "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    # V-group summaries (added during feature engineering)
    *[f"{g}_mean" for g in V_GROUPS],
    *[f"{g}_std" for g in V_GROUPS],
    # engineered
    "amt_zscore", "tx_count_30m", "tx_sum_30m", "tx_count_2h", "tx_sum_2h",
    # client-group aggregated features
    "uid_tx_count",
    *[f"uid_{c}_mean" for c in AGG_COLS],
]

XGBOOST_PARAMS: Final[dict] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

FL_ROUNDS: Final[int] = 15
FL_ESTIMATORS_PER_ROUND: Final[int] = 50
MIN_CLIENTS: Final[int] = 3

VELOCITY_WINDOWS: Final[dict[str, int]] = {
    "30min": 1800,
    "2hr": 7200,
}

BANKS: Final[list[str]] = ["a", "b", "c"]
