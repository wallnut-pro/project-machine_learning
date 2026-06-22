from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from src.evaluation import EVALUATION_FIGURES_DIR, EVALUATION_TABLES_DIR
from src.modeling import MODEL_ARTIFACTS_DIR, PREDICTIONS_DIR
from src.optimization import GRID_SEARCH_ARTIFACTS_DIR, run_all_optimizations
from src.preprocessing import DATASET_CONFIGS, PROJECT_ROOT


@pytest.fixture(scope="session")
def optimization_evaluation_outputs() -> dict:
    summary_path = EVALUATION_TABLES_DIR / "optimization_summary.json"
    final_table_path = EVALUATION_TABLES_DIR / "final_model_comparison.csv"
    if summary_path.exists() and final_table_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    optimization_summary = run_all_optimizations()
    from src.evaluation import evaluate_all_models

    evaluation_summary = evaluate_all_models(optimization_summary["datasets"])
    return evaluation_summary["summary"]


def test_four_tuned_models_are_saved(optimization_evaluation_outputs: dict) -> None:
    expected_paths = [
        MODEL_ARTIFACTS_DIR / "uci" / "decision_tree_tuned.joblib",
        MODEL_ARTIFACTS_DIR / "uci" / "random_forest_tuned.joblib",
        MODEL_ARTIFACTS_DIR / "secondary" / "decision_tree_tuned.joblib",
        MODEL_ARTIFACTS_DIR / "secondary" / "random_forest_tuned.joblib",
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing tuned model at {path}"


def test_grid_search_artifacts_and_best_params_exist(optimization_evaluation_outputs: dict) -> None:
    expected_paths = [
        GRID_SEARCH_ARTIFACTS_DIR / "uci" / "best_params.json",
        GRID_SEARCH_ARTIFACTS_DIR / "uci" / "cv_results.csv",
        GRID_SEARCH_ARTIFACTS_DIR / "secondary" / "best_params.json",
        GRID_SEARCH_ARTIFACTS_DIR / "secondary" / "cv_results.csv",
        GRID_SEARCH_ARTIFACTS_DIR / "uci" / "decision_tree_best_params.json",
        GRID_SEARCH_ARTIFACTS_DIR / "uci" / "random_forest_best_params.json",
        GRID_SEARCH_ARTIFACTS_DIR / "secondary" / "decision_tree_best_params.json",
        GRID_SEARCH_ARTIFACTS_DIR / "secondary" / "random_forest_best_params.json",
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing grid search artifact at {path}"

    for dataset_key in sorted(DATASET_CONFIGS):
        payload = json.loads((GRID_SEARCH_ARTIFACTS_DIR / dataset_key / "best_params.json").read_text(encoding="utf-8"))
        assert {"decision_tree", "random_forest"} == set(payload)


@pytest.mark.parametrize("dataset_key", sorted(DATASET_CONFIGS))
@pytest.mark.parametrize("model_key", ["decision_tree", "random_forest"])
def test_tuned_predictions_and_grid_search_metadata_are_valid(
    optimization_evaluation_outputs: dict,
    dataset_key: str,
    model_key: str,
) -> None:
    model_path = MODEL_ARTIFACTS_DIR / dataset_key / f"{model_key}_tuned.joblib"
    prediction_path = PREDICTIONS_DIR / dataset_key / f"{model_key}_tuned_y_pred.npy"
    y_test = np.load(DATASET_CONFIGS[dataset_key].output_dir / "y_test.npy")
    y_train = np.load(DATASET_CONFIGS[dataset_key].output_dir / "y_train.npy")

    assert model_path.exists()
    assert prediction_path.exists()

    model = joblib.load(model_path)
    y_pred = np.load(prediction_path)

    assert len(y_pred) == len(y_test)
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert getattr(model, "_fit_sample_count") == int(len(y_train))
    assert getattr(model, "_fit_used_test_data") is False
    assert getattr(model, "_grid_search_used_test_data") is False

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    assert matrix.shape == (2, 2)


def test_final_tables_have_eight_results_and_valid_metrics(optimization_evaluation_outputs: dict) -> None:
    final_df = pd.read_csv(EVALUATION_TABLES_DIR / "final_model_comparison.csv")
    confusion_df = pd.read_csv(EVALUATION_TABLES_DIR / "confusion_matrix_values.csv")
    best_params_df = pd.read_csv(EVALUATION_TABLES_DIR / "best_hyperparameters.csv")

    assert len(final_df) == 8
    assert len(confusion_df) == 8
    assert len(best_params_df) == 4
    assert set(final_df["version"]) == {"baseline", "tuned"}

    for column in ["accuracy", "precision_poisonous", "recall_poisonous", "f1_poisonous"]:
        assert final_df[column].between(0, 1).all()

    assert best_params_df["best_params_json"].str.len().gt(2).all()

    expected_test_sizes = {
        dataset_key: int(np.load(config.output_dir / "y_test.npy").shape[0])
        for dataset_key, config in DATASET_CONFIGS.items()
    }
    for _, row in confusion_df.iterrows():
        total = int(row["TN"] + row["FP"] + row["FN"] + row["TP"])
        assert total == expected_test_sizes[row["dataset"]]


def test_main_evaluation_figures_exist(optimization_evaluation_outputs: dict) -> None:
    expected_figure_names = [
        "decision_tree_baseline_confusion_matrix.png",
        "decision_tree_tuned_confusion_matrix.png",
        "random_forest_baseline_confusion_matrix.png",
        "random_forest_tuned_confusion_matrix.png",
        "accuracy_comparison.png",
        "precision_poisonous_comparison.png",
        "recall_poisonous_comparison.png",
        "f1_poisonous_comparison.png",
        "false_negative_comparison.png",
        "training_tuning_time_comparison.png",
    ]
    for dataset_key in sorted(DATASET_CONFIGS):
        for figure_name in expected_figure_names:
            figure_path = EVALUATION_FIGURES_DIR / dataset_key / figure_name
            assert figure_path.exists(), f"Missing evaluation figure at {figure_path}"
