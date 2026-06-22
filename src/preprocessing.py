from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMN = "class"
TARGET_MAPPING = {"e": 0, "p": 1}
TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    name: str
    raw_path: Path
    output_dir: Path
    auto_detect_separator: bool = False
    remove_duplicates: bool = False
    drop_missing_threshold: float | None = None


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "uci": DatasetConfig(
        key="uci",
        name="UCI Mushroom Dataset",
        raw_path=PROJECT_ROOT / "data" / "raw" / "uci" / "mushroom.csv",
        output_dir=PROJECT_ROOT / "data" / "processed" / "uci",
    ),
    "secondary": DatasetConfig(
        key="secondary",
        name="Secondary Mushroom Dataset",
        raw_path=PROJECT_ROOT / "data" / "raw" / "secondary" / "secondary_mushroom.csv",
        output_dir=PROJECT_ROOT / "data" / "processed" / "secondary",
        auto_detect_separator=True,
        remove_duplicates=True,
        drop_missing_threshold=0.8,
    ),
}


def read_dataset(config: DatasetConfig) -> pd.DataFrame:
    if config.auto_detect_separator:
        return pd.read_csv(config.raw_path, sep=None, engine="python")
    return pd.read_csv(config.raw_path)


def encode_target(target: pd.Series) -> pd.Series:
    encoded_target = target.map(TARGET_MAPPING)
    if encoded_target.isna().any():
        invalid_labels = sorted(target[encoded_target.isna()].dropna().unique().tolist())
        raise ValueError(f"Unexpected target labels found: {invalid_labels}")
    return encoded_target.astype(np.int64)


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

    return Pipeline(steps=[("column_transformer", column_transformer)])


def get_feature_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]
    return numeric_features, categorical_features


def filter_missing_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(column): int(count)
        for column, count in series.items()
        if int(count) > 0
    }


def count_missing_in_matrix(matrix: sparse.spmatrix | np.ndarray) -> int:
    if sparse.issparse(matrix):
        return int(np.isnan(matrix.data).sum())
    return int(np.isnan(matrix).sum())


def to_sparse_csr(matrix: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(matrix)


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_artifacts(
    output_dir: Path,
    X_train: sparse.csr_matrix,
    X_test: sparse.csr_matrix,
    y_train: np.ndarray,
    y_test: np.ndarray,
    preprocessor: Pipeline,
    feature_names: list[str],
    summary: dict[str, Any],
) -> None:
    ensure_output_dir(output_dir)

    save_npz(output_dir / "X_train.npz", X_train)
    save_npz(output_dir / "X_test.npz", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        output_dir / "feature_names.csv", index=False
    )
    with (output_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def build_summary(
    *,
    config: DatasetConfig,
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    dropped_columns: list[str],
    duplicate_rows_removed: int,
    y_full: pd.Series,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_train_before_encoding: pd.DataFrame,
    X_train_transformed: sparse.csr_matrix,
    X_test_transformed: sparse.csr_matrix,
) -> dict[str, Any]:
    return {
        "dataset_key": config.key,
        "dataset_name": config.name,
        "raw_path": str(config.raw_path.relative_to(PROJECT_ROOT)),
        "output_dir": str(config.output_dir.relative_to(PROJECT_ROOT)),
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "shape_before_cleaning": {
            "rows": int(raw_df.shape[0]),
            "columns": int(raw_df.shape[1]),
        },
        "shape_after_cleaning": {
            "rows": int(cleaned_df.shape[0]),
            "columns": int(cleaned_df.shape[1]),
        },
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "dropped_columns": dropped_columns,
        "missing_values_before_cleaning": filter_missing_counts(raw_df.isna().sum()),
        "missing_values_after_preprocessing": {
            "X_train_total": count_missing_in_matrix(X_train_transformed),
            "X_test_total": count_missing_in_matrix(X_test_transformed),
        },
        "class_distribution": {
            "full": {str(label): int(count) for label, count in y_full.value_counts().sort_index().items()},
            "train": {
                str(label): int(count)
                for label, count in pd.Series(y_train).value_counts().sort_index().items()
            },
            "test": {
                str(label): int(count)
                for label, count in pd.Series(y_test).value_counts().sort_index().items()
            },
        },
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "feature_counts": {
            "before_encoding": int(X_train_before_encoding.shape[1]),
            "after_encoding": int(X_train_transformed.shape[1]),
        },
    }


def preprocess_dataset(dataset_key: str) -> dict[str, Any]:
    if dataset_key not in DATASET_CONFIGS:
        valid_keys = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset key '{dataset_key}'. Valid values: {valid_keys}")

    config = DATASET_CONFIGS[dataset_key]
    raw_df = read_dataset(config)
    duplicate_rows_removed = 0
    working_df = raw_df.copy()

    if config.remove_duplicates:
        duplicate_rows_removed = int(working_df.duplicated().sum())
        working_df = working_df.drop_duplicates().reset_index(drop=True)

    if TARGET_COLUMN not in working_df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in {config.raw_path}")

    X = working_df.drop(columns=[TARGET_COLUMN])
    y = encode_target(working_df[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    dropped_columns: list[str] = []
    if config.drop_missing_threshold is not None:
        missing_ratio = X_train.isna().mean()
        dropped_columns = sorted(
            missing_ratio[missing_ratio > config.drop_missing_threshold].index.tolist()
        )
        X_train = X_train.drop(columns=dropped_columns)
        X_test = X_test.drop(columns=dropped_columns)

    cleaned_df = working_df.drop(columns=dropped_columns, errors="ignore")
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_features, categorical_features = get_feature_groups(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_train)

    X_train_transformed = to_sparse_csr(preprocessor.transform(X_train))
    X_test_transformed = to_sparse_csr(preprocessor.transform(X_test))
    feature_names = preprocessor.named_steps["column_transformer"].get_feature_names_out().tolist()

    summary = build_summary(
        config=config,
        raw_df=raw_df,
        cleaned_df=cleaned_df,
        dropped_columns=dropped_columns,
        duplicate_rows_removed=duplicate_rows_removed,
        y_full=y,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        X_train_before_encoding=X_train,
        X_train_transformed=X_train_transformed,
        X_test_transformed=X_test_transformed,
    )

    save_artifacts(
        output_dir=config.output_dir,
        X_train=X_train_transformed,
        X_test=X_test_transformed,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        preprocessor=preprocessor,
        feature_names=feature_names,
        summary=summary,
    )

    return summary


def preprocess_all_datasets() -> dict[str, dict[str, Any]]:
    return {dataset_key: preprocess_dataset(dataset_key) for dataset_key in DATASET_CONFIGS}
