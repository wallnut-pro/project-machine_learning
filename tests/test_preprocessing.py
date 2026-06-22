from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.preprocessing import DATASET_CONFIGS, preprocess_all_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OUTPUT_FILES = [
    "X_train.npz",
    "X_test.npz",
    "y_train.npy",
    "y_test.npy",
    "preprocessor.joblib",
    "feature_names.csv",
    "preprocessing_summary.json",
]


@pytest.fixture(scope="session")
def preprocessing_outputs() -> dict[str, dict]:
    return preprocess_all_datasets()


def load_summary(dataset_key: str) -> dict:
    summary_path = DATASET_CONFIGS[dataset_key].output_dir / "preprocessing_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_sparse_matrix(path: Path) -> sparse.csr_matrix:
    return sparse.load_npz(path).tocsr()


@pytest.mark.parametrize("dataset_key", sorted(DATASET_CONFIGS))
def test_output_files_exist(preprocessing_outputs: dict[str, dict], dataset_key: str) -> None:
    output_dir = DATASET_CONFIGS[dataset_key].output_dir
    for filename in EXPECTED_OUTPUT_FILES:
        assert (output_dir / filename).exists(), f"Missing {filename} for {dataset_key}"


@pytest.mark.parametrize("dataset_key", sorted(DATASET_CONFIGS))
def test_transformed_outputs_are_consistent(
    preprocessing_outputs: dict[str, dict], dataset_key: str
) -> None:
    output_dir = DATASET_CONFIGS[dataset_key].output_dir
    X_train = load_sparse_matrix(output_dir / "X_train.npz")
    X_test = load_sparse_matrix(output_dir / "X_test.npz")
    y_train = np.load(output_dir / "y_train.npy")
    y_test = np.load(output_dir / "y_test.npy")
    feature_names = pd.read_csv(output_dir / "feature_names.csv")
    preprocessor = joblib.load(output_dir / "preprocessor.joblib")

    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1] == len(feature_names)
    assert X_train.shape[1] == len(preprocessor.named_steps["column_transformer"].get_feature_names_out())
    assert not np.isnan(X_train.data).any()
    assert not np.isnan(X_test.data).any()


def test_uci_summary_matches_expected_rules(preprocessing_outputs: dict[str, dict]) -> None:
    summary = load_summary("uci")

    assert summary["duplicate_rows_removed"] == 0
    assert summary["dropped_columns"] == []
    assert summary["shape_before_cleaning"] == summary["shape_after_cleaning"]
    assert summary["missing_values_after_preprocessing"]["X_train_total"] == 0
    assert summary["missing_values_after_preprocessing"]["X_test_total"] == 0


def test_secondary_summary_matches_expected_rules(preprocessing_outputs: dict[str, dict]) -> None:
    summary = load_summary("secondary")

    assert summary["duplicate_rows_removed"] == 146
    assert sorted(summary["dropped_columns"]) == [
        "spore-print-color",
        "stem-root",
        "veil-color",
        "veil-type",
    ]
    assert summary["shape_before_cleaning"]["rows"] == 61069
    assert summary["shape_after_cleaning"]["rows"] == 60923
    assert summary["shape_after_cleaning"]["columns"] == 17
    assert summary["missing_values_after_preprocessing"]["X_train_total"] == 0
    assert summary["missing_values_after_preprocessing"]["X_test_total"] == 0
