from __future__ import annotations

import json
import os
import time
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

try:
    from src.modeling import (
        MODEL_ARTIFACTS_DIR,
        MODEL_CONFIGS,
        MODELING_TABLES_DIR,
        PREDICTIONS_DIR,
        get_original_feature_name,
        load_processed_dataset,
    )
    from src.optimization import EVALUATION_FIGURES_DIR, EVALUATION_TABLES_DIR, OPTIMIZATION_MODEL_CONFIGS
    from src.preprocessing import DATASET_CONFIGS, PROJECT_ROOT
except ModuleNotFoundError:
    from modeling import (  # type: ignore
        MODEL_ARTIFACTS_DIR,
        MODEL_CONFIGS,
        MODELING_TABLES_DIR,
        PREDICTIONS_DIR,
        get_original_feature_name,
        load_processed_dataset,
    )
    from optimization import EVALUATION_FIGURES_DIR, EVALUATION_TABLES_DIR, OPTIMIZATION_MODEL_CONFIGS  # type: ignore
    from preprocessing import DATASET_CONFIGS, PROJECT_ROOT  # type: ignore


sns.set_theme(style="whitegrid", context="talk")


def load_baseline_metrics_table() -> pd.DataFrame:
    return pd.read_csv(MODELING_TABLES_DIR / "baseline_metrics.csv")


def get_model_artifact_path(dataset_key: str, model_key: str, version: str) -> Path:
    suffix = "baseline" if version == "baseline" else "tuned"
    return MODEL_ARTIFACTS_DIR / dataset_key / f"{model_key}_{suffix}.joblib"


def get_prediction_artifact_path(dataset_key: str, model_key: str, version: str) -> Path:
    suffix = "" if version == "baseline" else "_tuned"
    return PREDICTIONS_DIR / dataset_key / f"{model_key}{suffix}_y_pred.npy"


def compute_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_poisonous": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_poisonous": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_poisonous": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def describe_model_complexity(model: Any, model_key: str, feature_names: list[str]) -> dict[str, Any]:
    if model_key == "decision_tree":
        root_feature_index = int(model.tree_.feature[0])
        encoded_root_feature = feature_names[root_feature_index] if root_feature_index >= 0 else None
        return {
            "tree_depth": int(model.tree_.max_depth),
            "leaf_count": int(model.get_n_leaves()),
            "n_estimators": np.nan,
            "encoded_root_feature": encoded_root_feature,
            "original_root_feature": (
                get_original_feature_name(encoded_root_feature) if encoded_root_feature is not None else None
            ),
        }
    return {
        "tree_depth": np.nan,
        "leaf_count": np.nan,
        "n_estimators": int(model.n_estimators),
        "encoded_root_feature": None,
        "original_root_feature": None,
    }


def plot_confusion_matrix(
    matrix: np.ndarray,
    dataset_key: str,
    model_name: str,
    version: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred edible", "Pred poisonous"],
        yticklabels=["True edible", "True poisonous"],
    )
    plt.title(f"{DATASET_CONFIGS[dataset_key].name}\n{model_name} ({version.title()})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_metric_comparison(metric_df: pd.DataFrame, dataset_key: str, metric_column: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    plot_df = metric_df.copy()
    plot_df["model_version"] = plot_df["model"] + " (" + plot_df["version"] + ")"
    sns.barplot(
        data=plot_df,
        x=metric_column,
        y="model_version",
        hue="model",
        dodge=False,
        palette="Set2",
    )
    plt.title(f"{DATASET_CONFIGS[dataset_key].name} - {title}")
    plt.xlim(0, max(1.0, float(plot_df[metric_column].max()) * 1.05))
    plt.xlabel(metric_column)
    plt.ylabel("Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_count_comparison(metric_df: pd.DataFrame, dataset_key: str, count_column: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    plot_df = metric_df.copy()
    plot_df["model_version"] = plot_df["model"] + " (" + plot_df["version"] + ")"
    sns.barplot(
        data=plot_df,
        x=count_column,
        y="model_version",
        hue="model",
        dodge=False,
        palette="crest",
    )
    plt.title(f"{DATASET_CONFIGS[dataset_key].name} - {title}")
    plt.xlabel(count_column)
    plt.ylabel("Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def evaluate_model_version(
    dataset_key: str,
    model_key: str,
    version: str,
    dataset_bundle: dict[str, Any],
    baseline_metrics_df: pd.DataFrame,
    tuning_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    model = joblib.load(get_model_artifact_path(dataset_key, model_key, version))

    prediction_start = time.perf_counter()
    y_pred = model.predict(dataset_bundle["X_test"])
    prediction_time_seconds = time.perf_counter() - prediction_start

    prediction_path = get_prediction_artifact_path(dataset_key, model_key, version)
    np.save(prediction_path, y_pred)

    metrics = compute_evaluation_metrics(dataset_bundle["y_test"], y_pred)

    if version == "baseline":
        baseline_row = baseline_metrics_df[
            (baseline_metrics_df["dataset"] == dataset_key)
            & (baseline_metrics_df["model"] == MODEL_CONFIGS[model_key].name)
        ].iloc[0]
        training_or_tuning_time_seconds = float(baseline_row["training_time_seconds"])
    else:
        training_or_tuning_time_seconds = float(tuning_lookup[(dataset_key, model_key)]["tuning_time_seconds"])

    complexity = describe_model_complexity(model, model_key, dataset_bundle["feature_names"])

    evaluation_row = {
        "dataset": dataset_key,
        "model": MODEL_CONFIGS[model_key].name,
        "version": version,
        **metrics,
        "training_or_tuning_time_seconds": float(training_or_tuning_time_seconds),
        "prediction_time_seconds": float(prediction_time_seconds),
    }
    evaluation_row.update(complexity)

    matrix = confusion_matrix(dataset_bundle["y_test"], y_pred, labels=[0, 1])
    plot_confusion_matrix(
        matrix,
        dataset_key,
        MODEL_CONFIGS[model_key].name,
        version,
        EVALUATION_FIGURES_DIR / dataset_key / f"{model_key}_{version}_confusion_matrix.png",
    )

    return {
        "row": evaluation_row,
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
    }


def save_evaluation_tables(
    evaluation_rows: list[dict[str, Any]],
    best_params_rows: list[dict[str, Any]],
    summary_payload: dict[str, Any],
) -> dict[str, str]:
    evaluation_df = pd.DataFrame(evaluation_rows)
    final_model_comparison_df = evaluation_df[
        [
            "dataset",
            "model",
            "version",
            "accuracy",
            "precision_poisonous",
            "recall_poisonous",
            "f1_poisonous",
            "TN",
            "FP",
            "FN",
            "TP",
            "training_or_tuning_time_seconds",
            "prediction_time_seconds",
        ]
    ].copy()
    confusion_df = evaluation_df[["dataset", "model", "version", "TN", "FP", "FN", "TP"]].copy()
    best_hyperparameters_df = pd.DataFrame(best_params_rows)

    final_path = EVALUATION_TABLES_DIR / "final_model_comparison.csv"
    best_params_path = EVALUATION_TABLES_DIR / "best_hyperparameters.csv"
    confusion_path = EVALUATION_TABLES_DIR / "confusion_matrix_values.csv"
    summary_path = EVALUATION_TABLES_DIR / "optimization_summary.json"

    final_model_comparison_df.to_csv(final_path, index=False)
    best_hyperparameters_df.to_csv(best_params_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "final_model_comparison": str(final_path.relative_to(PROJECT_ROOT)),
        "best_hyperparameters": str(best_params_path.relative_to(PROJECT_ROOT)),
        "confusion_matrix_values": str(confusion_path.relative_to(PROJECT_ROOT)),
        "optimization_summary": str(summary_path.relative_to(PROJECT_ROOT)),
    }


def generate_comparison_figures(evaluation_rows: list[dict[str, Any]]) -> None:
    evaluation_df = pd.DataFrame(evaluation_rows)
    for dataset_key in sorted(DATASET_CONFIGS):
        dataset_df = evaluation_df[evaluation_df["dataset"] == dataset_key].copy()
        plot_metric_comparison(
            dataset_df,
            dataset_key,
            "accuracy",
            "Accuracy Comparison",
            EVALUATION_FIGURES_DIR / dataset_key / "accuracy_comparison.png",
        )
        plot_metric_comparison(
            dataset_df,
            dataset_key,
            "precision_poisonous",
            "Precision Poisonous Comparison",
            EVALUATION_FIGURES_DIR / dataset_key / "precision_poisonous_comparison.png",
        )
        plot_metric_comparison(
            dataset_df,
            dataset_key,
            "recall_poisonous",
            "Recall Poisonous Comparison",
            EVALUATION_FIGURES_DIR / dataset_key / "recall_poisonous_comparison.png",
        )
        plot_metric_comparison(
            dataset_df,
            dataset_key,
            "f1_poisonous",
            "F1 Poisonous Comparison",
            EVALUATION_FIGURES_DIR / dataset_key / "f1_poisonous_comparison.png",
        )
        plot_count_comparison(
            dataset_df,
            dataset_key,
            "FN",
            "False Negative Comparison",
            EVALUATION_FIGURES_DIR / dataset_key / "false_negative_comparison.png",
        )
        plot_count_comparison(
            dataset_df,
            dataset_key,
            "training_or_tuning_time_seconds",
            "Training or Tuning Time Comparison",
            EVALUATION_FIGURES_DIR / dataset_key / "training_tuning_time_comparison.png",
        )


def build_best_params_rows(optimization_results: dict[str, dict[str, Any]], evaluation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluation_df = pd.DataFrame(evaluation_rows)
    rows: list[dict[str, Any]] = []
    for dataset_key, dataset_results in optimization_results.items():
        for model_key, result in dataset_results.items():
            matching_eval_row = evaluation_df[
                (evaluation_df["dataset"] == dataset_key)
                & (evaluation_df["model"] == OPTIMIZATION_MODEL_CONFIGS[model_key]["name"])
                & (evaluation_df["version"] == "tuned")
            ].iloc[0]
            rows.append(
                {
                    "dataset": dataset_key,
                    "model": OPTIMIZATION_MODEL_CONFIGS[model_key]["name"],
                    "best_params_json": json.dumps(result["best_params"], sort_keys=True),
                    "best_recall_poisonous_cv": float(result["best_recall_poisonous_cv"]),
                    "tuning_time_seconds": float(result["tuning_time_seconds"]),
                    "tree_depth": matching_eval_row["tree_depth"],
                    "leaf_count": matching_eval_row["leaf_count"],
                    "n_estimators": matching_eval_row["n_estimators"],
                    "encoded_root_feature": matching_eval_row["encoded_root_feature"],
                    "original_root_feature": matching_eval_row["original_root_feature"],
                    "test_recall_poisonous": float(matching_eval_row["recall_poisonous"]),
                    "test_f1_poisonous": float(matching_eval_row["f1_poisonous"]),
                }
            )
    return rows


def build_summary_payload(
    optimization_results: dict[str, dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluation_df = pd.DataFrame(evaluation_rows)
    summary: dict[str, Any] = {"datasets": {}}

    for dataset_key in sorted(DATASET_CONFIGS):
        dataset_df = evaluation_df[evaluation_df["dataset"] == dataset_key].copy()
        tuned_df = dataset_df[dataset_df["version"] == "tuned"].copy()
        best_row = tuned_df.sort_values(
            ["recall_poisonous", "f1_poisonous", "accuracy", "FN", "FP"],
            ascending=[False, False, False, True, True],
        ).iloc[0]

        model_summaries: dict[str, Any] = {}
        for model_key, result in optimization_results[dataset_key].items():
            model_name = OPTIMIZATION_MODEL_CONFIGS[model_key]["name"]
            baseline_row = dataset_df[
                (dataset_df["model"] == model_name) & (dataset_df["version"] == "baseline")
            ].iloc[0]
            tuned_row = dataset_df[
                (dataset_df["model"] == model_name) & (dataset_df["version"] == "tuned")
            ].iloc[0]
            model_summaries[model_key] = {
                "best_params": result["best_params"],
                "best_recall_poisonous_cv": float(result["best_recall_poisonous_cv"]),
                "baseline_metrics": {
                    "accuracy": float(baseline_row["accuracy"]),
                    "precision_poisonous": float(baseline_row["precision_poisonous"]),
                    "recall_poisonous": float(baseline_row["recall_poisonous"]),
                    "f1_poisonous": float(baseline_row["f1_poisonous"]),
                    "FP": int(baseline_row["FP"]),
                    "FN": int(baseline_row["FN"]),
                },
                "tuned_metrics": {
                    "accuracy": float(tuned_row["accuracy"]),
                    "precision_poisonous": float(tuned_row["precision_poisonous"]),
                    "recall_poisonous": float(tuned_row["recall_poisonous"]),
                    "f1_poisonous": float(tuned_row["f1_poisonous"]),
                    "FP": int(tuned_row["FP"]),
                    "FN": int(tuned_row["FN"]),
                },
                "complexity_change": {
                    "baseline_tree_depth": None if pd.isna(baseline_row["tree_depth"]) else int(baseline_row["tree_depth"]),
                    "tuned_tree_depth": None if pd.isna(tuned_row["tree_depth"]) else int(tuned_row["tree_depth"]),
                    "baseline_leaf_count": None if pd.isna(baseline_row["leaf_count"]) else int(baseline_row["leaf_count"]),
                    "tuned_leaf_count": None if pd.isna(tuned_row["leaf_count"]) else int(tuned_row["leaf_count"]),
                    "baseline_n_estimators": None if pd.isna(baseline_row["n_estimators"]) else int(baseline_row["n_estimators"]),
                    "tuned_n_estimators": None if pd.isna(tuned_row["n_estimators"]) else int(tuned_row["n_estimators"]),
                },
            }

        summary["datasets"][dataset_key] = {
            "best_tuned_model": {
                "model": best_row["model"],
                "accuracy": float(best_row["accuracy"]),
                "precision_poisonous": float(best_row["precision_poisonous"]),
                "recall_poisonous": float(best_row["recall_poisonous"]),
                "f1_poisonous": float(best_row["f1_poisonous"]),
                "FP": int(best_row["FP"]),
                "FN": int(best_row["FN"]),
            },
            "models": model_summaries,
        }
    return summary


def evaluate_all_models(optimization_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_metrics_df = load_baseline_metrics_table()
    tuning_lookup = {
        (dataset_key, model_key): result
        for dataset_key, dataset_results in optimization_results.items()
        for model_key, result in dataset_results.items()
    }

    evaluation_rows: list[dict[str, Any]] = []
    prediction_paths: dict[str, dict[str, str]] = {}

    for dataset_key in sorted(DATASET_CONFIGS):
        dataset_bundle = load_processed_dataset(dataset_key)
        prediction_paths[dataset_key] = {}
        for version in ["baseline", "tuned"]:
            for model_key in ["decision_tree", "random_forest"]:
                result = evaluate_model_version(
                    dataset_key,
                    model_key,
                    version,
                    dataset_bundle,
                    baseline_metrics_df,
                    tuning_lookup,
                )
                evaluation_rows.append(result["row"])
                prediction_paths[dataset_key][f"{model_key}_{version}"] = result["prediction_path"]

    generate_comparison_figures(evaluation_rows)
    best_params_rows = build_best_params_rows(optimization_results, evaluation_rows)
    summary_payload = build_summary_payload(optimization_results, evaluation_rows)
    table_paths = save_evaluation_tables(evaluation_rows, best_params_rows, summary_payload)

    return {
        "tables": table_paths,
        "predictions": prediction_paths,
        "summary": summary_payload,
    }
