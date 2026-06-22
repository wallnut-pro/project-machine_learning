from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    MODEL_ARTIFACTS_DIR,
    MODELING_FIGURES_DIR,
    MODELING_TABLES_DIR,
    PREDICTIONS_DIR,
    run_all_modeling,
    run_modeling_sanity_checks,
)
from src.preprocessing import DATASET_CONFIGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def modeling_outputs() -> dict:
    summary_path = MODELING_TABLES_DIR / "modeling_run_summary.json"
    if not summary_path.exists():
        run_all_modeling()
    return json.loads(summary_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def modeling_sanity_outputs() -> dict:
    sanity_path = MODELING_TABLES_DIR / "sanity_check_results.csv"
    cv_path = MODELING_TABLES_DIR / "cross_validation_baseline.csv"
    if not sanity_path.exists() or not cv_path.exists():
        return run_modeling_sanity_checks()
    return {
        "sanity_check_results": str(sanity_path.relative_to(PROJECT_ROOT)),
        "cross_validation_baseline": str(cv_path.relative_to(PROJECT_ROOT)),
    }


def test_four_models_are_trained_and_saved(modeling_outputs: dict) -> None:
    expected_paths = [
        MODEL_ARTIFACTS_DIR / "uci" / "decision_tree_baseline.joblib",
        MODEL_ARTIFACTS_DIR / "uci" / "random_forest_baseline.joblib",
        MODEL_ARTIFACTS_DIR / "secondary" / "decision_tree_baseline.joblib",
        MODEL_ARTIFACTS_DIR / "secondary" / "random_forest_baseline.joblib",
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing trained model at {path}"


@pytest.mark.parametrize("dataset_key", sorted(DATASET_CONFIGS))
@pytest.mark.parametrize("model_key", ["decision_tree", "random_forest"])
def test_predictions_and_model_metadata_are_valid(modeling_outputs: dict, dataset_key: str, model_key: str) -> None:
    prediction_path = PREDICTIONS_DIR / dataset_key / f"{model_key}_y_pred.npy"
    model_path = MODEL_ARTIFACTS_DIR / dataset_key / f"{model_key}_baseline.joblib"
    y_test = np.load(DATASET_CONFIGS[dataset_key].output_dir / "y_test.npy")
    feature_names = pd.read_csv(DATASET_CONFIGS[dataset_key].output_dir / "feature_names.csv")

    assert prediction_path.exists()
    assert model_path.exists()

    y_pred = np.load(prediction_path)
    model = joblib.load(model_path)

    assert len(y_pred) == len(y_test)
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert hasattr(model, "feature_importances_")
    assert len(model.feature_importances_) == len(feature_names)
    assert np.isclose(float(np.sum(model.feature_importances_)), 1.0, atol=1e-6)
    assert getattr(model, "_fit_sample_count") == int(np.load(DATASET_CONFIGS[dataset_key].output_dir / "y_train.npy").shape[0])
    assert getattr(model, "_fit_used_test_data") is False


def test_main_tables_and_figures_exist(modeling_outputs: dict) -> None:
    expected_tables = [
        MODELING_TABLES_DIR / "baseline_metrics.csv",
        MODELING_TABLES_DIR / "model_structure_summary.csv",
        MODELING_TABLES_DIR / "feature_importance_uci.csv",
        MODELING_TABLES_DIR / "feature_importance_secondary.csv",
        MODELING_TABLES_DIR / "modeling_run_summary.json",
    ]
    expected_figures = [
        MODELING_FIGURES_DIR / "uci" / "decision_tree_top_levels.png",
        MODELING_FIGURES_DIR / "secondary" / "decision_tree_top_levels.png",
        MODELING_FIGURES_DIR / "uci" / "random_forest_feature_importance_top20_encoded.png",
        MODELING_FIGURES_DIR / "uci" / "random_forest_feature_importance_top20_aggregated.png",
        MODELING_FIGURES_DIR / "secondary" / "random_forest_feature_importance_top20_encoded.png",
        MODELING_FIGURES_DIR / "secondary" / "random_forest_feature_importance_top20_aggregated.png",
    ]
    for path in expected_tables + expected_figures:
        assert path.exists(), f"Missing artifact {path}"


def test_root_nodes_and_metrics_are_recorded(modeling_outputs: dict) -> None:
    structure_df = pd.read_csv(MODELING_TABLES_DIR / "model_structure_summary.csv")
    metrics_df = pd.read_csv(MODELING_TABLES_DIR / "baseline_metrics.csv")

    decision_tree_rows = structure_df[structure_df["model"] == "Decision Tree"]
    assert decision_tree_rows["encoded_root_feature"].notna().all()
    assert decision_tree_rows["original_root_feature"].notna().all()

    assert len(metrics_df) == 4
    for column in [
        "train_accuracy",
        "test_accuracy",
        "precision_poisonous",
        "recall_poisonous",
        "f1_poisonous",
    ]:
        assert metrics_df[column].between(0, 1).all()


def test_sanity_outputs_and_leakage_checks(modeling_sanity_outputs: dict) -> None:
    sanity_df = pd.read_csv(MODELING_TABLES_DIR / "sanity_check_results.csv")
    cv_df = pd.read_csv(MODELING_TABLES_DIR / "cross_validation_baseline.csv")

    assert len(sanity_df) == 4
    assert len(cv_df) == 4
    assert (~sanity_df["target_in_preprocessor_inputs"]).all()
    assert (~sanity_df["target_in_feature_names"]).all()
    assert sanity_df["fit_train_only"].all()
    assert (sanity_df["shuffled_label_test_accuracy"] < 0.7).all()
    assert cv_df["accuracy_mean"].between(0, 1).all()
    assert cv_df["accuracy_std"].ge(0).all()
