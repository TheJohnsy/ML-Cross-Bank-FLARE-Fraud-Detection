from pathlib import Path
from typing import Final

BASE_DIR: Final[Path] = Path(__file__).parent
DATA_DIR: Final[Path] = BASE_DIR / "data" / "raw"
PARTITIONED_DIR: Final[Path] = BASE_DIR / "data" / "partitioned"
MODEL_DIR: Final[Path] = BASE_DIR / "data" / "models"

TARGET_COL: Final[str] = "isFraud"
PARTITION_KEY: Final[str] = "card1"

FEATURE_COLS: Final[list[str]] = [
    "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "V1", "V2", "V3", "V4", "V5", "V6",
    # engineered
    "amt_zscore", "tx_count_30m", "tx_sum_30m", "tx_count_2h", "tx_sum_2h",
]

XGBOOST_PARAMS: Final[dict] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

FL_ROUNDS: Final[int] = 10
MIN_CLIENTS: Final[int] = 3

VELOCITY_WINDOWS: Final[dict[str, int]] = {
    "30min": 1800,
    "2hr": 7200,
}

BANKS: Final[list[str]] = ["a", "b", "c"]
