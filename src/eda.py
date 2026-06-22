from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

try:
    from src.preprocessing import (
        DATASET_CONFIGS,
        PROJECT_ROOT,
        RANDOM_STATE,
        TARGET_COLUMN,
        TEST_SIZE,
        encode_target,
        get_feature_groups,
        read_dataset,
    )
except ModuleNotFoundError:
    from preprocessing import (  # type: ignore
        DATASET_CONFIGS,
        PROJECT_ROOT,
        RANDOM_STATE,
        TARGET_COLUMN,
        TEST_SIZE,
        encode_target,
        get_feature_groups,
        read_dataset,
    )


REPORTS_ROOT = PROJECT_ROOT / "reports"
EDA_TABLES_DIR = REPORTS_ROOT / "tables" / "eda"
EDA_FIGURES_DIR = REPORTS_ROOT / "figures" / "eda"
UCI_FIGURES_DIR = EDA_FIGURES_DIR / "uci"
SECONDARY_FIGURES_DIR = EDA_FIGURES_DIR / "secondary"
CLASS_LABELS = {"e": "edible", "p": "poisonous", 0: "edible", 1: "poisonous"}


@dataclass
class EDADataBundle:
    dataset_key: str
    raw_df: pd.DataFrame
    working_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    dropped_columns: list[str]
    duplicate_rows_removed: int
    numeric_features: list[str]
    categorical_features: list[str]


def ensure_report_directories() -> None:
    EDA_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    UCI_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SECONDARY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def label_target_series(series: pd.Series) -> pd.Series:
    return series.map(CLASS_LABELS)


def prepare_eda_dataset(dataset_key: str) -> EDADataBundle:
    config = DATASET_CONFIGS[dataset_key]
    raw_df = read_dataset(config)
    working_df = raw_df.copy()
    duplicate_rows_removed = 0

    if config.remove_duplicates:
        duplicate_rows_removed = int(working_df.duplicated().sum())
        working_df = working_df.drop_duplicates().reset_index(drop=True)

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

    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()

    numeric_features, categorical_features = get_feature_groups(X_train_clean)

    if numeric_features:
        median_values = X_train_clean[numeric_features].median()
        X_train_clean[numeric_features] = X_train_clean[numeric_features].fillna(median_values)
        X_test_clean[numeric_features] = X_test_clean[numeric_features].fillna(median_values)

    if categorical_features:
        X_train_clean[categorical_features] = X_train_clean[categorical_features].fillna("unknown")
        X_test_clean[categorical_features] = X_test_clean[categorical_features].fillna("unknown")

    train_df = X_train_clean.copy()
    train_df[TARGET_COLUMN] = y_train.to_numpy()
    train_df["class_label"] = label_target_series(y_train)

    test_df = X_test_clean.copy()
    test_df[TARGET_COLUMN] = y_test.to_numpy()
    test_df["class_label"] = label_target_series(y_test)

    return EDADataBundle(
        dataset_key=dataset_key,
        raw_df=raw_df,
        working_df=working_df.drop(columns=dropped_columns, errors="ignore"),
        train_df=train_df,
        test_df=test_df,
        dropped_columns=dropped_columns,
        duplicate_rows_removed=duplicate_rows_removed,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def save_table(dataframe: pd.DataFrame, filename: str) -> Path:
    path = EDA_TABLES_DIR / filename
    dataframe.to_csv(path, index=False)
    return path


def save_json(data: dict[str, Any], filename: str) -> Path:
    path = EDA_TABLES_DIR / filename
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_class_distribution(raw_df: pd.DataFrame, output_dir: Path, filename: str, title: str) -> Path:
    distribution = raw_df[TARGET_COLUMN].map(CLASS_LABELS).value_counts().rename_axis("class_label").reset_index(name="count")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=distribution, x="class_label", y="count", hue="class_label", dodge=False, palette="Set2", legend=False)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    figure_path = output_dir / filename
    save_figure(figure_path)
    return figure_path


def plot_missing_values(raw_df: pd.DataFrame, output_dir: Path, filename: str, title: str) -> Path:
    missing_df = (
        raw_df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "feature"})
        .query("missing_count > 0")
        .sort_values("missing_count", ascending=False)
    )

    plt.figure(figsize=(10, 5))
    if missing_df.empty:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=16)
        plt.axis("off")
    else:
        sns.barplot(data=missing_df, x="feature", y="missing_count", hue="feature", dodge=False, palette="crest", legend=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Missing Count")
        plt.xlabel("Feature")
    plt.title(title)
    figure_path = output_dir / filename
    save_figure(figure_path)
    return figure_path


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    if confusion.empty:
        return 0.0

    chi2 = chi2_contingency(confusion, correction=False)[0]
    n = confusion.values.sum()
    if n <= 1:
        return 0.0

    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denominator = min(kcorr - 1, rcorr - 1)
    if denominator <= 0:
        return 0.0
    return float(math.sqrt(phi2corr / denominator))


def build_categorical_association_ranking(train_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_series = train_df[TARGET_COLUMN]

    for feature in feature_columns:
        contingency = pd.crosstab(train_df[feature], target_series)
        chi2, p_value, _, _ = chi2_contingency(contingency, correction=False)
        rows.append(
            {
                "feature": feature,
                "cramers_v": round(cramers_v(train_df[feature], target_series), 6),
                "chi_square": round(float(chi2), 6),
                "p_value": float(p_value),
            }
        )

    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def build_cramers_v_matrix(train_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(index=features, columns=features, dtype=float)
    for row_feature in features:
        for col_feature in features:
            matrix.loc[row_feature, col_feature] = cramers_v(train_df[row_feature], train_df[col_feature])
    return matrix


def plot_stacked_categorical_distribution(
    train_df: pd.DataFrame,
    feature: str,
    output_dir: Path,
    filename: str,
    title: str,
    top_n: int | None = None,
) -> Path:
    plot_df = train_df.copy()
    if top_n is not None:
        top_values = plot_df[feature].value_counts().head(top_n).index
        plot_df[feature] = np.where(plot_df[feature].isin(top_values), plot_df[feature], "other")

    contingency = pd.crosstab(plot_df[feature], plot_df["class_label"])
    normalized = contingency.div(contingency.sum(axis=1), axis=0)
    normalized = normalized.sort_values(by=list(normalized.columns), ascending=False)

    plt.figure(figsize=(10, 6))
    normalized.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="Set2")
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel("Within-feature proportion")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Class")
    figure_path = output_dir / filename
    save_figure(figure_path)
    return figure_path


def plot_ranking_bar(
    ranking_df: pd.DataFrame,
    value_column: str,
    output_dir: Path,
    filename: str,
    title: str,
    top_n: int | None = None,
) -> Path:
    plot_df = ranking_df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)
    plot_df = plot_df.sort_values(value_column, ascending=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=plot_df, x=value_column, y="feature", hue="feature", dodge=False, palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel(value_column.replace("_", " ").title())
    plt.ylabel("Feature")
    figure_path = output_dir / filename
    save_figure(figure_path)
    return figure_path


def plot_cramers_v_heatmap(matrix_df: pd.DataFrame, output_dir: Path, filename: str, title: str) -> Path:
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title(title)
    figure_path = output_dir / filename
    save_figure(figure_path)
    return figure_path


def plot_numeric_by_class(
    train_df: pd.DataFrame,
    feature: str,
    output_dir: Path,
    filename: str,
    title: str,
) -> Path:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=train_df, x="class_label", y=feature, hue="class_label", dodge=False, palette="Set2", legend=False)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel(feature)
    figure_path = output_dir / filename
    save_figure(figure_path)
    return figure_path


def build_mutual_information_ranking(train_df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    encoded_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    discrete_mask: list[bool] = []

    if numeric_features:
        encoded_parts.append(train_df[numeric_features].to_numpy(dtype=float))
        feature_names.extend(numeric_features)
        discrete_mask.extend([False] * len(numeric_features))

    if categorical_features:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        categorical_array = encoder.fit_transform(train_df[categorical_features])
        encoded_parts.append(categorical_array)
        feature_names.extend(categorical_features)
        discrete_mask.extend([True] * len(categorical_features))

    feature_matrix = np.hstack(encoded_parts)
    scores = mutual_info_classif(
        feature_matrix,
        train_df[TARGET_COLUMN].to_numpy(),
        discrete_features=np.array(discrete_mask, dtype=bool),
        random_state=RANDOM_STATE,
    )

    ranking_df = pd.DataFrame({"feature": feature_names, "mutual_information": scores})
    return ranking_df.sort_values("mutual_information", ascending=False).reset_index(drop=True)


def run_uci_eda() -> dict[str, Any]:
    bundle = prepare_eda_dataset("uci")
    summary: dict[str, Any] = {
        "dataset_key": "uci",
        "dataset_name": "UCI Mushroom Dataset",
        "raw_path": str(DATASET_CONFIGS["uci"].raw_path.relative_to(PROJECT_ROOT)),
        "analysis_split": {
            "train_rows": int(bundle.train_df.shape[0]),
            "test_rows": int(bundle.test_df.shape[0]),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "duplicate_rows_removed": bundle.duplicate_rows_removed,
        "dropped_columns": bundle.dropped_columns,
        "tables": {},
        "figures": {},
    }

    raw_class_distribution = (
        bundle.raw_df[TARGET_COLUMN]
        .map(CLASS_LABELS)
        .value_counts()
        .rename_axis("class_label")
        .reset_index(name="count")
    )
    summary["tables"]["class_distribution"] = str(
        save_table(raw_class_distribution, "uci_class_distribution.csv").relative_to(PROJECT_ROOT)
    )

    raw_missing = (
        bundle.raw_df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("missing_count", ascending=False)
    )
    summary["tables"]["missing_values_before_cleaning"] = str(
        save_table(raw_missing, "uci_missing_values_before_cleaning.csv").relative_to(PROJECT_ROOT)
    )

    odor_table = pd.crosstab(bundle.train_df["odor"], bundle.train_df["class_label"]).reset_index()
    summary["tables"]["odor_vs_class"] = str(
        save_table(odor_table, "uci_odor_vs_class.csv").relative_to(PROJECT_ROOT)
    )

    gill_color_table = pd.crosstab(bundle.train_df["gill-color"], bundle.train_df["class_label"]).reset_index()
    summary["tables"]["gill_color_vs_class"] = str(
        save_table(gill_color_table, "uci_gill_color_vs_class.csv").relative_to(PROJECT_ROOT)
    )

    association_ranking = build_categorical_association_ranking(bundle.train_df, bundle.categorical_features)
    summary["tables"]["feature_association_ranking"] = str(
        save_table(association_ranking, "uci_feature_association_ranking.csv").relative_to(PROJECT_ROOT)
    )
    summary["top_ranked_features"] = association_ranking.head(5).to_dict(orient="records")

    selected_features = association_ranking.head(6)["feature"].tolist()
    cramers_matrix = build_cramers_v_matrix(bundle.train_df, selected_features)
    cramers_matrix_reset = cramers_matrix.reset_index().rename(columns={"index": "feature"})
    summary["tables"]["selected_cramers_v_matrix"] = str(
        save_table(cramers_matrix_reset, "uci_selected_cramers_v_matrix.csv").relative_to(PROJECT_ROOT)
    )
    summary["selected_heatmap_features"] = selected_features

    summary["figures"]["class_distribution"] = str(
        plot_class_distribution(
            bundle.raw_df,
            UCI_FIGURES_DIR,
            "class_distribution.png",
            "UCI Mushroom Class Distribution",
        ).relative_to(PROJECT_ROOT)
    )
    summary["figures"]["missing_values_before_cleaning"] = str(
        plot_missing_values(
            bundle.raw_df,
            UCI_FIGURES_DIR,
            "missing_values_before_cleaning.png",
            "UCI Missing Values Before Preprocessing",
        ).relative_to(PROJECT_ROOT)
    )
    summary["figures"]["odor_vs_class"] = str(
        plot_stacked_categorical_distribution(
            bundle.train_df,
            "odor",
            UCI_FIGURES_DIR,
            "odor_vs_class.png",
            "UCI Odor vs Class (Train Split)",
        ).relative_to(PROJECT_ROOT)
    )
    summary["figures"]["gill_color_vs_class"] = str(
        plot_stacked_categorical_distribution(
            bundle.train_df,
            "gill-color",
            UCI_FIGURES_DIR,
            "gill_color_vs_class.png",
            "UCI Gill Color vs Class (Train Split)",
            top_n=10,
        ).relative_to(PROJECT_ROOT)
    )
    summary["figures"]["selected_cramers_v_heatmap"] = str(
        plot_cramers_v_heatmap(
            cramers_matrix,
            UCI_FIGURES_DIR,
            "selected_cramers_v_heatmap.png",
            "UCI Cramer's V Heatmap for Selected Categorical Features",
        ).relative_to(PROJECT_ROOT)
    )
    summary["figures"]["feature_association_ranking"] = str(
        plot_ranking_bar(
            association_ranking,
            "cramers_v",
            UCI_FIGURES_DIR,
            "feature_association_ranking.png",
            "UCI Feature Association Ranking vs Class",
        ).relative_to(PROJECT_ROOT)
    )

    summary["table_json"] = str(save_json(summary, "uci_eda_summary.json").relative_to(PROJECT_ROOT))
    return summary


def run_secondary_eda() -> dict[str, Any]:
    bundle = prepare_eda_dataset("secondary")
    summary: dict[str, Any] = {
        "dataset_key": "secondary",
        "dataset_name": "Secondary Mushroom Dataset",
        "raw_path": str(DATASET_CONFIGS["secondary"].raw_path.relative_to(PROJECT_ROOT)),
        "analysis_split": {
            "train_rows": int(bundle.train_df.shape[0]),
            "test_rows": int(bundle.test_df.shape[0]),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "duplicate_rows_removed": bundle.duplicate_rows_removed,
        "dropped_columns": bundle.dropped_columns,
        "tables": {},
        "figures": {},
    }

    raw_class_distribution = (
        bundle.raw_df[TARGET_COLUMN]
        .map(CLASS_LABELS)
        .value_counts()
        .rename_axis("class_label")
        .reset_index(name="count")
    )
    summary["tables"]["class_distribution"] = str(
        save_table(raw_class_distribution, "secondary_class_distribution.csv").relative_to(PROJECT_ROOT)
    )

    raw_missing = (
        bundle.raw_df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("missing_count", ascending=False)
    )
    summary["tables"]["missing_values_before_cleaning"] = str(
        save_table(raw_missing, "secondary_missing_values_before_cleaning.csv").relative_to(PROJECT_ROOT)
    )

    numeric_describe = bundle.train_df[bundle.numeric_features].describe().transpose().reset_index().rename(columns={"index": "feature"})
    summary["tables"]["numeric_descriptive_stats"] = str(
        save_table(numeric_describe, "secondary_numeric_descriptive_stats.csv").relative_to(PROJECT_ROOT)
    )

    by_class_describe = (
        bundle.train_df.groupby("class_label")[bundle.numeric_features]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    by_class_describe.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else str(column)
        for column in by_class_describe.columns
    ]
    summary["tables"]["numeric_descriptive_by_class"] = str(
        save_table(by_class_describe, "secondary_numeric_descriptive_by_class.csv").relative_to(PROJECT_ROOT)
    )

    for feature, filename in [
        ("gill-color", "secondary_gill_color_vs_class.csv"),
        ("habitat", "secondary_habitat_vs_class.csv"),
        ("season", "secondary_season_vs_class.csv"),
    ]:
        contingency_df = pd.crosstab(bundle.train_df[feature], bundle.train_df["class_label"]).reset_index()
        summary["tables"][f"{feature.replace('-', '_')}_vs_class"] = str(
            save_table(contingency_df, filename).relative_to(PROJECT_ROOT)
        )

    numeric_correlation = (
        bundle.train_df[bundle.numeric_features]
        .corr(numeric_only=True)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    summary["tables"]["numeric_correlation"] = str(
        save_table(numeric_correlation, "secondary_numeric_correlation.csv").relative_to(PROJECT_ROOT)
    )

    mi_ranking = build_mutual_information_ranking(bundle.train_df, bundle.numeric_features, bundle.categorical_features)
    summary["tables"]["mutual_information_ranking"] = str(
        save_table(mi_ranking, "secondary_mutual_information_ranking.csv").relative_to(PROJECT_ROOT)
    )
    summary["top_ranked_features"] = mi_ranking.head(10).to_dict(orient="records")

    summary["figures"]["class_distribution"] = str(
        plot_class_distribution(
            bundle.raw_df,
            SECONDARY_FIGURES_DIR,
            "class_distribution.png",
            "Secondary Mushroom Class Distribution",
        ).relative_to(PROJECT_ROOT)
    )
    summary["figures"]["missing_values_before_cleaning"] = str(
        plot_missing_values(
            bundle.raw_df,
            SECONDARY_FIGURES_DIR,
            "missing_values_before_cleaning.png",
            "Secondary Missing Values Before Cleaning",
        ).relative_to(PROJECT_ROOT)
    )

    for feature, filename, title in [
        ("cap-diameter", "cap_diameter_by_class.png", "Secondary Cap Diameter by Class (Train Split)"),
        ("stem-height", "stem_height_by_class.png", "Secondary Stem Height by Class (Train Split)"),
        ("stem-width", "stem_width_by_class.png", "Secondary Stem Width by Class (Train Split)"),
    ]:
        summary["figures"][feature.replace("-", "_")] = str(
            plot_numeric_by_class(bundle.train_df, feature, SECONDARY_FIGURES_DIR, filename, title).relative_to(PROJECT_ROOT)
        )

    for feature, filename, title, top_n in [
        ("gill-color", "gill_color_vs_class.png", "Secondary Gill Color vs Class (Train Split)", 10),
        ("habitat", "habitat_vs_class.png", "Secondary Habitat vs Class (Train Split)", None),
        ("season", "season_vs_class.png", "Secondary Season vs Class (Train Split)", None),
    ]:
        summary["figures"][f"{feature.replace('-', '_')}_vs_class"] = str(
            plot_stacked_categorical_distribution(bundle.train_df, feature, SECONDARY_FIGURES_DIR, filename, title, top_n=top_n).relative_to(PROJECT_ROOT)
        )

    summary["figures"]["mutual_information_ranking"] = str(
        plot_ranking_bar(
            mi_ranking,
            "mutual_information",
            SECONDARY_FIGURES_DIR,
            "mutual_information_ranking.png",
            "Secondary Feature Ranking by Mutual Information",
            top_n=15,
        ).relative_to(PROJECT_ROOT)
    )

    summary["table_json"] = str(save_json(summary, "secondary_eda_summary.json").relative_to(PROJECT_ROOT))
    return summary


def run_all_eda() -> dict[str, dict[str, Any]]:
    ensure_report_directories()
    configure_plot_style()
    return {"uci": run_uci_eda(), "secondary": run_secondary_eda()}
