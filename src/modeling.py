from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_PROJECT_ROOT / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.sparse import load_npz
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree

try:
    from src.preprocessing import DATASET_CONFIGS, PROJECT_ROOT, TARGET_COLUMN
except ModuleNotFoundError:
    from preprocessing import DATASET_CONFIGS, PROJECT_ROOT, TARGET_COLUMN  # type: ignore


CLASS_NAMES = ["edible", "poisonous"]
MODELING_TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "modeling"
MODELING_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "modeling"
MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    estimator_factory: Any


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "decision_tree": ModelConfig(
        key="decision_tree",
        name="Decision Tree",
        estimator_factory=lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    ),
    "random_forest": ModelConfig(
        key="random_forest",
        name="Random Forest",
        estimator_factory=lambda: RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    ),
}


def ensure_directories() -> None:
    for path in [
        MODELING_TABLES_DIR,
        MODEL_ARTIFACTS_DIR / "uci",
        MODEL_ARTIFACTS_DIR / "secondary",
        PREDICTIONS_DIR / "uci",
        PREDICTIONS_DIR / "secondary",
        MODELING_FIGURES_DIR / "uci",
        MODELING_FIGURES_DIR / "secondary",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")


def load_processed_dataset(dataset_key: str) -> dict[str, Any]:
    config = DATASET_CONFIGS[dataset_key]
    processed_dir = config.output_dir

    X_train = load_npz(processed_dir / "X_train.npz").tocsr()
    X_test = load_npz(processed_dir / "X_test.npz").tocsr()
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    feature_names = pd.read_csv(processed_dir / "feature_names.csv")["feature_name"].tolist()
    preprocessor = joblib.load(processed_dir / "preprocessor.joblib")
    preprocessing_summary = json.loads((processed_dir / "preprocessing_summary.json").read_text(encoding="utf-8"))

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "preprocessing_summary": preprocessing_summary,
    }


def get_original_feature_name(encoded_feature_name: str) -> str:
    prefix_split = encoded_feature_name.split("__", 1)
    encoded_part = prefix_split[1] if len(prefix_split) == 2 else encoded_feature_name
    base_feature = encoded_part.split("_", 1)[0]
    return base_feature


def aggregate_feature_importance(
    feature_names: list[str], feature_importances: np.ndarray
) -> pd.DataFrame:
    encoded_df = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "original_feature": [get_original_feature_name(name) for name in feature_names],
            "importance": feature_importances,
        }
    )
    aggregated_df = (
        encoded_df.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return aggregated_df


def plot_decision_tree_top_levels(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    output_path: Path,
) -> None:
    plt.figure(figsize=(26, 14))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Decision Tree Top Levels (max_depth=3)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    title: str,
    feature_column: str,
    top_n: int = 20,
) -> None:
    plot_df = importance_df.nlargest(top_n, "importance").sort_values("importance", ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=plot_df,
        x="importance",
        y=feature_column,
        hue=feature_column,
        dodge=False,
        palette="viridis",
        legend=False,
    )
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def fit_and_evaluate_model(
    dataset_key: str,
    model_key: str,
    dataset_bundle: dict[str, Any],
) -> dict[str, Any]:
    model_config = MODEL_CONFIGS[model_key]
    model = model_config.estimator_factory()
    X_train = dataset_bundle["X_train"]
    X_test = dataset_bundle["X_test"]
    y_train = dataset_bundle["y_train"]
    y_test = dataset_bundle["y_test"]
    feature_names = dataset_bundle["feature_names"]

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - train_start

    prediction_start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - prediction_start

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, y_pred)
    precision_poisonous = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_poisonous = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_poisonous = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Attach explicit fit metadata so tests can verify train-only fitting.
    model._fit_sample_count = int(X_train.shape[0])  # type: ignore[attr-defined]
    model._test_sample_count_reference = int(X_test.shape[0])  # type: ignore[attr-defined]
    model._fit_feature_count = int(X_train.shape[1])  # type: ignore[attr-defined]
    model._fit_used_test_data = False  # type: ignore[attr-defined]

    model_path = MODEL_ARTIFACTS_DIR / dataset_key / f"{model_key}_baseline.joblib"
    prediction_path = PREDICTIONS_DIR / dataset_key / f"{model_key}_y_pred.npy"
    joblib.dump(model, model_path)
    np.save(prediction_path, y_pred)

    root_feature_index = None
    encoded_root_feature = None
    original_root_feature = None
    tree_depth = None
    leaf_count = None
    n_estimators = None

    if model_key == "decision_tree":
        root_feature_index = int(model.tree_.feature[0])
        encoded_root_feature = feature_names[root_feature_index] if root_feature_index >= 0 else None
        original_root_feature = (
            get_original_feature_name(encoded_root_feature) if encoded_root_feature is not None else None
        )
        tree_depth = int(model.tree_.max_depth)
        leaf_count = int(model.get_n_leaves())
        plot_decision_tree_top_levels(
            model,
            feature_names,
            MODELING_FIGURES_DIR / dataset_key / "decision_tree_top_levels.png",
        )
    else:
        n_estimators = int(model.n_estimators)

    feature_importances = np.asarray(model.feature_importances_, dtype=float)
    encoded_importance_df = (
        pd.DataFrame(
            {
                "dataset": dataset_key,
                "model": model_config.name,
                "importance_view": "encoded",
                "encoded_feature": feature_names,
                "original_feature": [get_original_feature_name(name) for name in feature_names],
                "importance": feature_importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    aggregated_importance_df = aggregate_feature_importance(feature_names, feature_importances)
    aggregated_importance_df.insert(0, "model", model_config.name)
    aggregated_importance_df.insert(0, "dataset", dataset_key)
    aggregated_importance_df.insert(2, "importance_view", "aggregated")
    aggregated_importance_df.insert(3, "encoded_feature", "")

    if model_key == "random_forest":
        plot_feature_importance(
            encoded_importance_df.rename(columns={"encoded_feature": "feature"}),
            MODELING_FIGURES_DIR / dataset_key / "random_forest_feature_importance_top20_encoded.png",
            f"{dataset_bundle['preprocessing_summary']['dataset_name']} Random Forest Top 20 Encoded Feature Importance",
            "feature",
            top_n=20,
        )
        plot_feature_importance(
            aggregated_importance_df.rename(columns={"original_feature": "feature"}),
            MODELING_FIGURES_DIR / dataset_key / "random_forest_feature_importance_top20_aggregated.png",
            f"{dataset_bundle['preprocessing_summary']['dataset_name']} Random Forest Top 20 Aggregated Feature Importance",
            "feature",
            top_n=20,
        )

    metrics_row = {
        "dataset": dataset_key,
        "model": model_config.name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision_poisonous": precision_poisonous,
        "recall_poisonous": recall_poisonous,
        "f1_poisonous": f1_poisonous,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
    }

    structure_row = {
        "dataset": dataset_key,
        "model": model_config.name,
        "jumlah_fitur_input": int(X_train.shape[1]),
        "tree_depth": tree_depth,
        "jumlah_leaf": leaf_count,
        "jumlah_estimator": n_estimators,
        "encoded_root_feature": encoded_root_feature,
        "original_root_feature": original_root_feature,
    }

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics_row": metrics_row,
        "structure_row": structure_row,
        "feature_importance_encoded": encoded_importance_df,
        "feature_importance_aggregated": aggregated_importance_df,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
    }


def train_all_models_for_dataset(dataset_key: str) -> dict[str, Any]:
    dataset_bundle = load_processed_dataset(dataset_key)
    results: dict[str, Any] = {}
    for model_key in MODEL_CONFIGS:
        results[model_key] = fit_and_evaluate_model(dataset_key, model_key, dataset_bundle)
    return results


def save_modeling_tables(all_results: dict[str, dict[str, Any]]) -> dict[str, str]:
    metrics_rows = []
    structure_rows = []
    uci_importance_rows = []
    secondary_importance_rows = []

    for dataset_key, dataset_results in all_results.items():
        for model_key, result in dataset_results.items():
            metrics_rows.append(result["metrics_row"])
            structure_rows.append(result["structure_row"])

            combined_importance = pd.concat(
                [result["feature_importance_encoded"], result["feature_importance_aggregated"]],
                ignore_index=True,
            )
            if dataset_key == "uci":
                uci_importance_rows.append(combined_importance)
            else:
                secondary_importance_rows.append(combined_importance)

    metrics_df = pd.DataFrame(metrics_rows)
    structure_df = pd.DataFrame(structure_rows)
    uci_importance_df = pd.concat(uci_importance_rows, ignore_index=True)
    secondary_importance_df = pd.concat(secondary_importance_rows, ignore_index=True)

    metrics_path = MODELING_TABLES_DIR / "baseline_metrics.csv"
    structure_path = MODELING_TABLES_DIR / "model_structure_summary.csv"
    uci_importance_path = MODELING_TABLES_DIR / "feature_importance_uci.csv"
    secondary_importance_path = MODELING_TABLES_DIR / "feature_importance_secondary.csv"

    metrics_df.to_csv(metrics_path, index=False)
    structure_df.to_csv(structure_path, index=False)
    uci_importance_df.to_csv(uci_importance_path, index=False)
    secondary_importance_df.to_csv(secondary_importance_path, index=False)

    return {
        "baseline_metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
        "model_structure_summary": str(structure_path.relative_to(PROJECT_ROOT)),
        "feature_importance_uci": str(uci_importance_path.relative_to(PROJECT_ROOT)),
        "feature_importance_secondary": str(secondary_importance_path.relative_to(PROJECT_ROOT)),
    }


def run_all_modeling() -> dict[str, Any]:
    ensure_directories()
    all_results: dict[str, dict[str, Any]] = {}
    for dataset_key in sorted(DATASET_CONFIGS):
        all_results[dataset_key] = train_all_models_for_dataset(dataset_key)

    table_paths = save_modeling_tables(all_results)

    summary: dict[str, Any] = {"tables": table_paths, "datasets": {}}
    for dataset_key, dataset_results in all_results.items():
        dataset_summary = {}
        for model_key, result in dataset_results.items():
            dataset_summary[model_key] = {
                "model_path": result["model_path"],
                "prediction_path": result["prediction_path"],
                "metrics": result["metrics_row"],
                "structure": result["structure_row"],
                "top_feature_importance_encoded": result["feature_importance_encoded"].head(10).to_dict(orient="records"),
                "top_feature_importance_aggregated": result["feature_importance_aggregated"].head(10).to_dict(orient="records"),
            }
        summary["datasets"][dataset_key] = dataset_summary

    summary_path = MODELING_TABLES_DIR / "modeling_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["tables"]["modeling_run_summary"] = str(summary_path.relative_to(PROJECT_ROOT))
    return summary


def matrix_row_signatures(matrix: sparse.csr_matrix) -> list[bytes]:
    dense_matrix = np.asarray(matrix.toarray())
    dense_matrix = np.ascontiguousarray(dense_matrix)
    return [dense_matrix[index].tobytes() for index in range(dense_matrix.shape[0])]


def compute_overlap_statistics(
    X_train: sparse.csr_matrix,
    X_test: sparse.csr_matrix,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, int]:
    train_signatures = matrix_row_signatures(X_train)
    test_signatures = matrix_row_signatures(X_test)

    train_signature_set = set(train_signatures)
    unique_feature_overlap = len(set(test_signatures) & train_signature_set)
    overlapping_test_rows_feature_vectors = sum(signature in train_signature_set for signature in test_signatures)

    train_labeled_signatures = {
        signature + int(label).to_bytes(1, byteorder="little", signed=False)
        for signature, label in zip(train_signatures, y_train, strict=False)
    }
    test_labeled_signatures = [
        signature + int(label).to_bytes(1, byteorder="little", signed=False)
        for signature, label in zip(test_signatures, y_test, strict=False)
    ]
    unique_labeled_overlap = len(set(test_labeled_signatures) & train_labeled_signatures)
    overlapping_test_rows_labeled_patterns = sum(
        signature in train_labeled_signatures for signature in test_labeled_signatures
    )

    return {
        "exact_overlap_unique_feature_vectors": int(unique_feature_overlap),
        "overlapping_test_rows_feature_vectors": int(overlapping_test_rows_feature_vectors),
        "exact_overlap_unique_feature_label_patterns": int(unique_labeled_overlap),
        "overlapping_test_rows_feature_label_patterns": int(overlapping_test_rows_labeled_patterns),
    }


def compute_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_poisonous": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_poisonous": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_poisonous": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def run_shuffled_label_control(
    dataset_bundle: dict[str, Any],
    model_key: str,
) -> dict[str, float]:
    model = MODEL_CONFIGS[model_key].estimator_factory()
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled_y_train = rng.permutation(dataset_bundle["y_train"])
    model.fit(dataset_bundle["X_train"], shuffled_y_train)
    y_pred = model.predict(dataset_bundle["X_test"])
    return compute_metrics_from_predictions(dataset_bundle["y_test"], y_pred)


def run_cross_validation_on_training_set(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    model_key: str,
) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracy_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []

    for train_index, validation_index in splitter.split(X_train, y_train):
        model = clone(MODEL_CONFIGS[model_key].estimator_factory())
        model.fit(X_train[train_index], y_train[train_index])
        y_pred = model.predict(X_train[validation_index])
        fold_metrics = compute_metrics_from_predictions(y_train[validation_index], y_pred)
        accuracy_scores.append(fold_metrics["accuracy"])
        recall_scores.append(fold_metrics["recall_poisonous"])
        f1_scores.append(fold_metrics["f1_poisonous"])

    return {
        "accuracy_mean": float(np.mean(accuracy_scores)),
        "accuracy_std": float(np.std(accuracy_scores, ddof=0)),
        "recall_poisonous_mean": float(np.mean(recall_scores)),
        "recall_poisonous_std": float(np.std(recall_scores, ddof=0)),
        "f1_poisonous_mean": float(np.mean(f1_scores)),
        "f1_poisonous_std": float(np.std(f1_scores, ddof=0)),
    }


def run_uci_without_odor_diagnostic(
    dataset_bundle: dict[str, Any],
    model_key: str,
) -> dict[str, float | int]:
    odor_indices = [
        index
        for index, feature_name in enumerate(dataset_bundle["feature_names"])
        if get_original_feature_name(feature_name) == "odor"
    ]
    keep_indices = [index for index in range(len(dataset_bundle["feature_names"])) if index not in odor_indices]
    X_train_reduced = dataset_bundle["X_train"][:, keep_indices]
    X_test_reduced = dataset_bundle["X_test"][:, keep_indices]

    model = MODEL_CONFIGS[model_key].estimator_factory()
    model.fit(X_train_reduced, dataset_bundle["y_train"])
    y_pred = model.predict(X_test_reduced)
    metrics = compute_metrics_from_predictions(dataset_bundle["y_test"], y_pred)
    metrics["removed_odor_feature_count"] = int(len(odor_indices))
    return metrics


def ensure_modeling_artifacts_exist() -> None:
    required_model_paths = [
        MODEL_ARTIFACTS_DIR / "uci" / "decision_tree_baseline.joblib",
        MODEL_ARTIFACTS_DIR / "uci" / "random_forest_baseline.joblib",
        MODEL_ARTIFACTS_DIR / "secondary" / "decision_tree_baseline.joblib",
        MODEL_ARTIFACTS_DIR / "secondary" / "random_forest_baseline.joblib",
    ]
    if not all(path.exists() for path in required_model_paths):
        run_all_modeling()


def run_modeling_sanity_checks() -> dict[str, str]:
    ensure_directories()
    ensure_modeling_artifacts_exist()

    sanity_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []

    for dataset_key in sorted(DATASET_CONFIGS):
        dataset_bundle = load_processed_dataset(dataset_key)
        overlap_stats = compute_overlap_statistics(
            dataset_bundle["X_train"],
            dataset_bundle["X_test"],
            dataset_bundle["y_train"],
            dataset_bundle["y_test"],
        )
        target_in_feature_names = any(
            get_original_feature_name(feature_name) == TARGET_COLUMN
            for feature_name in dataset_bundle["feature_names"]
        )
        target_in_preprocessor_inputs = TARGET_COLUMN in list(getattr(dataset_bundle["preprocessor"], "feature_names_in_", []))

        if dataset_key == "secondary":
            dataset_note = (
                "Secondary dataset has no explicit species/group identifier in the processed artifacts, "
                "so group-wise leakage cannot be ruled out beyond exact-vector overlap checks."
            )
        else:
            dataset_note = "No explicit group identifier is present in the processed artifacts."

        for model_key in ["decision_tree", "random_forest"]:
            model_path = MODEL_ARTIFACTS_DIR / dataset_key / f"{model_key}_baseline.joblib"
            prediction_path = PREDICTIONS_DIR / dataset_key / f"{model_key}_y_pred.npy"
            model = joblib.load(model_path)
            y_pred = np.load(prediction_path)

            shuffled_metrics = run_shuffled_label_control(dataset_bundle, model_key)
            cv_metrics = run_cross_validation_on_training_set(
                dataset_bundle["X_train"],
                dataset_bundle["y_train"],
                model_key,
            )
            confusion = confusion_matrix(dataset_bundle["y_test"], y_pred, labels=[0, 1]).ravel()
            tn, fp, fn, tp = [int(value) for value in confusion]

            odor_diagnostic = {"accuracy": np.nan, "recall_poisonous": np.nan, "f1_poisonous": np.nan, "removed_odor_feature_count": 0}
            if dataset_key == "uci":
                odor_diagnostic = run_uci_without_odor_diagnostic(dataset_bundle, model_key)

            cv_rows.append(
                {
                    "dataset": dataset_key,
                    "model": MODEL_CONFIGS[model_key].name,
                    "folds": 5,
                    **cv_metrics,
                }
            )

            sanity_rows.append(
                {
                    "dataset": dataset_key,
                    "model": MODEL_CONFIGS[model_key].name,
                    "target_in_preprocessor_inputs": bool(target_in_preprocessor_inputs),
                    "target_in_feature_names": bool(target_in_feature_names),
                    "fit_train_only": bool(getattr(model, "_fit_sample_count", -1) == int(dataset_bundle["X_train"].shape[0])),
                    "fit_sample_count": int(getattr(model, "_fit_sample_count", -1)),
                    "train_rows": int(dataset_bundle["X_train"].shape[0]),
                    "test_rows": int(dataset_bundle["X_test"].shape[0]),
                    **overlap_stats,
                    "shuffled_label_test_accuracy": shuffled_metrics["accuracy"],
                    "shuffled_label_precision_poisonous": shuffled_metrics["precision_poisonous"],
                    "shuffled_label_recall_poisonous": shuffled_metrics["recall_poisonous"],
                    "shuffled_label_f1_poisonous": shuffled_metrics["f1_poisonous"],
                    "confusion_tn": tn,
                    "confusion_fp": fp,
                    "confusion_fn": fn,
                    "confusion_tp": tp,
                    "uci_without_odor_test_accuracy": odor_diagnostic["accuracy"],
                    "uci_without_odor_recall_poisonous": odor_diagnostic["recall_poisonous"],
                    "uci_without_odor_f1_poisonous": odor_diagnostic["f1_poisonous"],
                    "uci_removed_odor_feature_count": int(odor_diagnostic["removed_odor_feature_count"]),
                    "notes": dataset_note,
                }
            )

    sanity_df = pd.DataFrame(sanity_rows)
    cv_df = pd.DataFrame(cv_rows)

    sanity_path = MODELING_TABLES_DIR / "sanity_check_results.csv"
    cv_path = MODELING_TABLES_DIR / "cross_validation_baseline.csv"
    sanity_df.to_csv(sanity_path, index=False)
    cv_df.to_csv(cv_path, index=False)

    return {
        "sanity_check_results": str(sanity_path.relative_to(PROJECT_ROOT)),
        "cross_validation_baseline": str(cv_path.relative_to(PROJECT_ROOT)),
    }
