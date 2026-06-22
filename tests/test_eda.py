from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.eda import run_all_eda


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "eda"
UCI_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "eda" / "uci"
SECONDARY_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "eda" / "secondary"


@pytest.fixture(scope="session")
def eda_outputs() -> dict[str, dict]:
    return run_all_eda()


def test_uci_outputs_exist_and_use_actual_dataset(eda_outputs: dict[str, dict]) -> None:
    summary = eda_outputs["uci"]
    assert summary["raw_path"] == "data\\raw\\uci\\mushroom.csv"
    assert summary["analysis_split"]["train_rows"] == 6499
    assert summary["analysis_split"]["test_rows"] == 1625

    expected_tables = [
        "uci_class_distribution.csv",
        "uci_missing_values_before_cleaning.csv",
        "uci_odor_vs_class.csv",
        "uci_gill_color_vs_class.csv",
        "uci_feature_association_ranking.csv",
        "uci_selected_cramers_v_matrix.csv",
        "uci_eda_summary.json",
    ]
    expected_figures = [
        "class_distribution.png",
        "missing_values_before_cleaning.png",
        "odor_vs_class.png",
        "gill_color_vs_class.png",
        "selected_cramers_v_heatmap.png",
        "feature_association_ranking.png",
    ]

    for filename in expected_tables:
        assert (TABLES_DIR / filename).exists()
    for filename in expected_figures:
        assert (UCI_FIGURES_DIR / filename).exists()

    ranking = pd.read_csv(TABLES_DIR / "uci_feature_association_ranking.csv")
    assert "odor" in ranking["feature"].values
    assert ranking["cramers_v"].between(0, 1).all()


def test_secondary_outputs_exist_and_use_actual_dataset(eda_outputs: dict[str, dict]) -> None:
    summary = eda_outputs["secondary"]
    assert summary["raw_path"] == "data\\raw\\secondary\\secondary_mushroom.csv"
    assert summary["duplicate_rows_removed"] == 146
    assert sorted(summary["dropped_columns"]) == [
        "spore-print-color",
        "stem-root",
        "veil-color",
        "veil-type",
    ]

    expected_tables = [
        "secondary_class_distribution.csv",
        "secondary_missing_values_before_cleaning.csv",
        "secondary_numeric_descriptive_stats.csv",
        "secondary_numeric_descriptive_by_class.csv",
        "secondary_gill_color_vs_class.csv",
        "secondary_habitat_vs_class.csv",
        "secondary_season_vs_class.csv",
        "secondary_numeric_correlation.csv",
        "secondary_mutual_information_ranking.csv",
        "secondary_eda_summary.json",
    ]
    expected_figures = [
        "class_distribution.png",
        "missing_values_before_cleaning.png",
        "cap_diameter_by_class.png",
        "stem_height_by_class.png",
        "stem_width_by_class.png",
        "gill_color_vs_class.png",
        "habitat_vs_class.png",
        "season_vs_class.png",
        "mutual_information_ranking.png",
    ]

    for filename in expected_tables:
        assert (TABLES_DIR / filename).exists()
    for filename in expected_figures:
        assert (SECONDARY_FIGURES_DIR / filename).exists()

    ranking = pd.read_csv(TABLES_DIR / "secondary_mutual_information_ranking.csv")
    secondary_features = {
        "cap-diameter",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "does-bruise-or-bleed",
        "gill-attachment",
        "gill-spacing",
        "gill-color",
        "stem-height",
        "stem-width",
        "stem-surface",
        "stem-color",
        "has-ring",
        "ring-type",
        "habitat",
        "season",
    }
    assert set(ranking["feature"]).issubset(secondary_features)
    assert ranking["mutual_information"].ge(0).all()


def test_audit_corrections_and_docs_exist(eda_outputs: dict[str, dict]) -> None:
    doc_path = PROJECT_ROOT / "docs" / "milestone_02_exploratory_data_analysis.md"
    draft_path = PROJECT_ROOT / "reports" / "eda_section_4_2_draft.md"
    notebook_path = PROJECT_ROOT / "notebooks" / "03_exploratory_data_analysis.ipynb"

    assert doc_path.exists()
    assert draft_path.exists()
    assert notebook_path.exists()

    draft_text = draft_path.read_text(encoding="utf-8")
    assert "Secondary Dataset bukan salinan UCI" in draft_text
    assert "Dennis Wagner" in draft_text
    assert "OneHotEncoder" in draft_text
    assert "LabelEncoder" in draft_text
    assert "stalk-root" in draft_text
    assert "root node" in draft_text
