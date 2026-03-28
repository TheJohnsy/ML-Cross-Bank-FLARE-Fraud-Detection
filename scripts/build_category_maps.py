"""Build a global category mapping from all bank partitions for consistent label encoding."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config import PARTITIONED_DIR, BANKS


def main() -> None:
    frames = []
    for bank in BANKS:
        src = PARTITIONED_DIR / f"bank_{bank}.csv"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found — run partition_data.py first")
        frames.append(pd.read_csv(src))

    combined = pd.concat(frames, ignore_index=True)

    category_maps: dict[str, dict[str, int]] = {}
    for col in combined.select_dtypes(include=["object", "string"]).columns:
        unique_vals = sorted(combined[col].fillna("_NAN_").astype(str).unique())
        category_maps[col] = {v: i for i, v in enumerate(unique_vals)}

    out = PARTITIONED_DIR / "category_maps.json"
    with open(out, "w") as f:
        json.dump(category_maps, f, indent=2)
    print(f"Saved category maps ({len(category_maps)} columns) → {out}")


if __name__ == "__main__":
    main()
