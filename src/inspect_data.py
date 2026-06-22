from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_CANDIDATES = {
    "UCI Mushroom Dataset": [
        PROJECT_ROOT / "data" / "raw" / "uci" / "mushroom.csv",
        PROJECT_ROOT / "data" / "raw" / "uci" / "agaricus-lepiota.csv",
        PROJECT_ROOT / "data" / "raw" / "uci" / "uci_mushroom.csv",
        PROJECT_ROOT / "data" / "raw" / "mushroom.csv",
    ],
    "Secondary Mushroom Dataset": [
        PROJECT_ROOT / "data" / "raw" / "secondary" / "secondary_mushroom.csv",
        PROJECT_ROOT / "data" / "raw" / "secondary" / "secondary_data.csv",
        PROJECT_ROOT / "data" / "raw" / "secondary" / "mushroom_secondary.csv",
        PROJECT_ROOT / "data" / "raw" / "secondary.csv",
    ],
}

TARGET_COLUMN_CANDIDATES = {
    "class",
    "target",
    "label",
    "labels",
    "poisonous",
    "edible",
    "classes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect raw mushroom datasets without preprocessing."
    )
    parser.add_argument(
        "--uci-path",
        type=Path,
        default=None,
        help="Relative or absolute path to the UCI Mushroom CSV.",
    )
    parser.add_argument(
        "--secondary-path",
        type=Path,
        default=None,
        help="Relative or absolute path to the Secondary Mushroom CSV.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to display from the top of each dataset.",
    )
    return parser.parse_args()


def resolve_path(path_value: Path | None, dataset_name: str) -> Path | None:
    if path_value is not None:
        candidate = normalize_path(path_value)
        return candidate if candidate.exists() else None

    for candidate in DEFAULT_DATASET_CANDIDATES[dataset_name]:
        if candidate.exists():
            return candidate

    return None


def normalize_path(path_value: Path) -> Path:
    return path_value if path_value.is_absolute() else PROJECT_ROOT / path_value


def detect_target_column(df: pd.DataFrame) -> tuple[str | None, str]:
    normalized_columns = {column.lower(): column for column in df.columns}

    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in normalized_columns:
            original_name = normalized_columns[candidate]
            return original_name, f"Detected target column: '{original_name}'."

    if not df.columns.empty:
        fallback_column = df.columns[-1]
        return (
            fallback_column,
            f"No standard target column found. Falling back to last column: '{fallback_column}'.",
        )

    return None, "Dataset has no columns, so target distribution cannot be shown."


def read_csv_with_auto_separator(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=None, engine="python")


def format_missing_values(df: pd.DataFrame) -> str:
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if missing_counts.empty:
        return "No missing values detected."

    return missing_counts.to_string()


def format_target_distribution(df: pd.DataFrame, target_column: str | None) -> str:
    if target_column is None:
        return "Target distribution unavailable."

    distribution = df[target_column].value_counts(dropna=False)
    percentages = df[target_column].value_counts(dropna=False, normalize=True).mul(100)
    summary = pd.DataFrame(
        {
            "count": distribution,
            "percentage": percentages.round(2),
        }
    )
    return summary.to_string()


def format_info(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue().strip()


def print_expected_locations(dataset_name: str, candidates: Iterable[Path]) -> None:
    print(f"Expected locations for {dataset_name}:")
    for candidate in candidates:
        print(f"- {candidate.relative_to(PROJECT_ROOT)}")


def inspect_dataset(dataset_name: str, csv_path: Path | None, head_rows: int) -> None:
    print("=" * 80)
    print(dataset_name)
    print("=" * 80)

    if csv_path is None:
        print("CSV file not found.")
        print_expected_locations(dataset_name, DEFAULT_DATASET_CANDIDATES[dataset_name])
        print()
        return

    try:
        df = read_csv_with_auto_separator(csv_path)
    except EmptyDataError:
        print(f"File is empty: {csv_path.relative_to(PROJECT_ROOT)}")
        print()
        return
    except Exception as error:
        print(f"Failed to read CSV: {csv_path.relative_to(PROJECT_ROOT)}")
        print(f"Error: {error}")
        print()
        return
    target_column, target_message = detect_target_column(df)

    print(f"File: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Shape: {df.shape}")
    print()
    print(f"Head ({head_rows} rows):")
    print(df.head(head_rows).to_string(index=False))
    print()
    print("Data types and dataframe info:")
    print(format_info(df))
    print()
    print("Missing values:")
    print(format_missing_values(df))
    print()
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print()
    print(target_message)
    print("Target class distribution:")
    print(format_target_distribution(df, target_column))
    print()


def main() -> None:
    args = parse_args()

    dataset_paths = {
        "UCI Mushroom Dataset": resolve_path(args.uci_path, "UCI Mushroom Dataset"),
        "Secondary Mushroom Dataset": resolve_path(
            args.secondary_path, "Secondary Mushroom Dataset"
        ),
    }

    for dataset_name, csv_path in dataset_paths.items():
        inspect_dataset(dataset_name, csv_path, args.head)


if __name__ == "__main__":
    main()
