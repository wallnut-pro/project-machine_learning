from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_PROJECT_ROOT / ".matplotlib"))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

try:
    from src.modeling import (
        MODEL_ARTIFACTS_DIR,
        PREDICTIONS_DIR,
        RANDOM_STATE,
        ensure_directories as ensure_modeling_directories,
        load_processed_dataset,
    )
    from src.preprocessing import DATASET_CONFIGS, PROJECT_ROOT
except ModuleNotFoundError:
    from modeling import (  # type: ignore
        MODEL_ARTIFACTS_DIR,
        PREDICTIONS_DIR,
        RANDOM_STATE,
        ensure_directories as ensure_modeling_directories,
        load_processed_dataset,
    )
    from preprocessing import DATASET_CONFIGS, PROJECT_ROOT  # type: ignore


GRID_SEARCH_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "grid_search"
EVALUATION_TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "evaluation"
EVALUATION_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "evaluation"
CV_SPLITTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
RECALL_POISONOUS_SCORER = make_scorer(recall_score, pos_label=1, zero_division=0)

OPTIMIZATION_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "decision_tree": {
        "name": "Decision Tree",
        "estimator_factory": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced"],
        },
    },
    "random_forest": {
        "name": "Random Forest",
        "estimator_factory": lambda: RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [None, 20],
            "max_features": ["sqrt", "log2"],
            "min_samples_split": [2, 5],
            "class_weight": [None, "balanced"],
        },
    },
}


def ensure_optimization_directories() -> None:
    ensure_modeling_directories()
    for dataset_key in DATASET_CONFIGS:
        (GRID_SEARCH_ARTIFACTS_DIR / dataset_key).mkdir(parents=True, exist_ok=True)
        (EVALUATION_FIGURES_DIR / dataset_key).mkdir(parents=True, exist_ok=True)
    EVALUATION_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def get_tuned_model_path(dataset_key: str, model_key: str) -> Path:
    return MODEL_ARTIFACTS_DIR / dataset_key / f"{model_key}_tuned.joblib"


def get_tuned_prediction_path(dataset_key: str, model_key: str) -> Path:
    return PREDICTIONS_DIR / dataset_key / f"{model_key}_tuned_y_pred.npy"


def run_grid_search_for_model(dataset_key: str, model_key: str) -> dict[str, Any]:
    if dataset_key not in DATASET_CONFIGS:
        valid_keys = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset key '{dataset_key}'. Valid values: {valid_keys}")
    if model_key not in OPTIMIZATION_MODEL_CONFIGS:
        valid_keys = ", ".join(sorted(OPTIMIZATION_MODEL_CONFIGS))
        raise ValueError(f"Unknown model key '{model_key}'. Valid values: {valid_keys}")

    dataset_bundle = load_processed_dataset(dataset_key)
    model_config = OPTIMIZATION_MODEL_CONFIGS[model_key]
    estimator = model_config["estimator_factory"]()

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=model_config["param_grid"],
        scoring=RECALL_POISONOUS_SCORER,
        cv=CV_SPLITTER,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    tuning_start = time.perf_counter()
    grid_search.fit(dataset_bundle["X_train"], dataset_bundle["y_train"])
    tuning_time = time.perf_counter() - tuning_start

    best_model = grid_search.best_estimator_
    best_model._fit_sample_count = int(dataset_bundle["X_train"].shape[0])  # type: ignore[attr-defined]
    best_model._fit_feature_count = int(dataset_bundle["X_train"].shape[1])  # type: ignore[attr-defined]
    best_model._fit_used_test_data = False  # type: ignore[attr-defined]
    best_model._grid_search_used_test_data = False  # type: ignore[attr-defined]
    best_model._grid_search_cv_folds = CV_SPLITTER.get_n_splits()  # type: ignore[attr-defined]

    prediction_start = time.perf_counter()
    y_pred = best_model.predict(dataset_bundle["X_test"])
    prediction_time = time.perf_counter() - prediction_start

    model_path = get_tuned_model_path(dataset_key, model_key)
    prediction_path = get_tuned_prediction_path(dataset_key, model_key)
    joblib.dump(best_model, model_path)
    np.save(prediction_path, y_pred)

    best_params_payload = {
        "dataset": dataset_key,
        "dataset_name": DATASET_CONFIGS[dataset_key].name,
        "model_key": model_key,
        "model_name": model_config["name"],
        "best_params": grid_search.best_params_,
        "best_recall_poisonous_cv": float(grid_search.best_score_),
        "tuning_time_seconds": float(tuning_time),
        "train_rows": int(dataset_bundle["X_train"].shape[0]),
        "test_rows": int(dataset_bundle["X_test"].shape[0]),
        "test_set_used_in_grid_search": False,
    }

    return {
        "dataset_bundle": dataset_bundle,
        "model_key": model_key,
        "model_name": model_config["name"],
        "grid_search": grid_search,
        "best_model": best_model,
        "best_params_payload": best_params_payload,
        "cv_results_df": pd.DataFrame(grid_search.cv_results_),
        "tuning_time_seconds": float(tuning_time),
        "prediction_time_seconds": float(prediction_time),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
    }


def optimize_models_for_dataset(dataset_key: str) -> dict[str, Any]:
    dataset_results: dict[str, Any] = {}
    for model_key in ["decision_tree", "random_forest"]:
        dataset_results[model_key] = run_grid_search_for_model(dataset_key, model_key)
    return dataset_results


def save_grid_search_artifacts(all_results: dict[str, dict[str, Any]]) -> dict[str, str]:
    combined_best_params: dict[str, dict[str, Any]] = {}
    combined_cv_paths: dict[str, str] = {}

    for dataset_key, dataset_results in all_results.items():
        dataset_dir = GRID_SEARCH_ARTIFACTS_DIR / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        combined_cv_results: list[pd.DataFrame] = []
        combined_best_params[dataset_key] = {}

        for model_key, result in dataset_results.items():
            model_cv_results = result["cv_results_df"].copy()
            model_cv_results.insert(0, "model", result["model_name"])
            model_cv_results.insert(0, "dataset", dataset_key)
            combined_cv_results.append(model_cv_results)

            per_model_best_params_path = dataset_dir / f"{model_key}_best_params.json"
            per_model_best_params_path.write_text(
                json.dumps(result["best_params_payload"], indent=2),
                encoding="utf-8",
            )
            combined_best_params[dataset_key][model_key] = result["best_params_payload"]

        best_params_path = dataset_dir / "best_params.json"
        best_params_path.write_text(json.dumps(combined_best_params[dataset_key], indent=2), encoding="utf-8")

        combined_cv_results_df = pd.concat(combined_cv_results, ignore_index=True)
        combined_cv_results_path = dataset_dir / "cv_results.csv"
        combined_cv_results_df.to_csv(combined_cv_results_path, index=False)
        combined_cv_paths[dataset_key] = str(combined_cv_results_path.relative_to(PROJECT_ROOT))

    return {
        "uci_best_params": str((GRID_SEARCH_ARTIFACTS_DIR / "uci" / "best_params.json").relative_to(PROJECT_ROOT)),
        "secondary_best_params": str(
            (GRID_SEARCH_ARTIFACTS_DIR / "secondary" / "best_params.json").relative_to(PROJECT_ROOT)
        ),
        "uci_cv_results": combined_cv_paths["uci"],
        "secondary_cv_results": combined_cv_paths["secondary"],
    }


def run_all_optimizations() -> dict[str, Any]:
    ensure_optimization_directories()
    all_results: dict[str, dict[str, Any]] = {}
    for dataset_key in sorted(DATASET_CONFIGS):
        all_results[dataset_key] = optimize_models_for_dataset(dataset_key)

    artifact_paths = save_grid_search_artifacts(all_results)

    summary: dict[str, Any] = {"grid_search_artifacts": artifact_paths, "datasets": {}}
    for dataset_key, dataset_results in all_results.items():
        summary["datasets"][dataset_key] = {}
        for model_key, result in dataset_results.items():
            summary["datasets"][dataset_key][model_key] = {
                "best_params": result["best_params_payload"]["best_params"],
                "best_recall_poisonous_cv": result["best_params_payload"]["best_recall_poisonous_cv"],
                "tuning_time_seconds": result["tuning_time_seconds"],
                "prediction_time_seconds": result["prediction_time_seconds"],
                "model_path": result["model_path"],
                "prediction_path": result["prediction_path"],
            }

    return summary
