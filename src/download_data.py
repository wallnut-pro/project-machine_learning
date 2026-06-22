from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "uci"
OUTPUT_PATH = OUTPUT_DIR / "mushroom.csv"
DATASET_ID = 73


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if targets.shape[1] != 1:
        raise ValueError(
            f"Expected a single target column, but found {targets.shape[1]} columns."
        )

    target_column = targets.columns[0]
    combined_df = pd.concat([features, targets], axis=1).rename(
        columns={target_column: "class"}
    )
    combined_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Downloaded UCI dataset ID {DATASET_ID}.")
    print(f"Saved file: {OUTPUT_PATH}")
    print(f"Shape: {combined_df.shape}")


if __name__ == "__main__":
    main()
