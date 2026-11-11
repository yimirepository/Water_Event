#!/usr/bin/env python3
"""Reproduce the Aqueduct-version and cutoff sensitivity check for R2-06.

Inputs:
  * clean_historical_panel.csv (Codex-cleaned panel used for the Figure 4 model)
  * aqueduct30.csv
  * aqueduct40.csv

The script compares Aqueduct categories for common basin IDs, identifies the
four affected basins in the study panel, assigns each basin to the country or
countries with the most panel records linked to it (ties retained), and
reruns the Figure 4 Random Forest under 1980, 1985, and 1990 cutoffs.

Install dependencies with:
  python -m pip install pandas numpy scikit-learn shap

Example:
  python r2_06_aqueduct_cutoff_sensitivity.py \
    --panel clean_historical_panel.csv \
    --aqueduct30 aqueduct30.csv --aqueduct40 aqueduct40.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


BASE_CUTOFF = 1985
ALTERNATIVE_CUTOFFS = (1980, 1985, 1990)
AFFECTED_BASINS = (125534, 452422, 142785, 452520)

# Predictor order follows the historical Figure 4 notebook.
MODE_COLUMNS = [
    "cross_border_basin_agreements",
    "collaborative_planning_and_adaptation_strategies",
    "joint_water_allocation_models",
    "joint_data_sharing_systems",
    "transboundary_water_quality_standards",
    "coordinated_hydropower_operations",
]
CONTEXT_COLUMNS = [
    "gdp_percapita",
    "export",
    "population_density",
    "rural_population",
    "precipitation",
    "bws_score",
]
FEATURES = ["conflict"] + CONTEXT_COLUMNS + MODE_COLUMNS


def load_aqueduct(path: Path) -> pd.DataFrame:
    """Load one Aqueduct extract, retaining one record per basin ID."""
    data = pd.read_csv(path, low_memory=False)
    required = {"pfaf_id", "bws_score", "bws_cat"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    data = data[["pfaf_id", "bws_score", "bws_cat"]].copy()
    data["pfaf_id"] = pd.to_numeric(data["pfaf_id"], errors="coerce")
    data["bws_score"] = pd.to_numeric(data["bws_score"], errors="coerce")
    data["bws_cat"] = pd.to_numeric(data["bws_cat"], errors="coerce")
    # The source extracts repeat the same basin-level value across many rows.
    return data.dropna(subset=["pfaf_id"]).drop_duplicates("pfaf_id").set_index("pfaf_id")


def compare_aqueduct(aq30: pd.DataFrame, aq40: pd.DataFrame) -> tuple[pd.DataFrame, int, float, int, float]:
    """Return common-basin comparison and exact/broad-category agreement."""
    common = aq30[["bws_cat", "bws_score"]].join(
        aq40[["bws_cat", "bws_score"]],
        how="inner",
        lsuffix="_30",
        rsuffix="_40",
    )
    valid = common[
        common["bws_cat_30"].between(0, 4)
        & common["bws_cat_40"].between(0, 4)
    ].copy()
    exact_n = int((valid["bws_cat_30"] == valid["bws_cat_40"]).sum())
    exact_pct = exact_n / len(valid) * 100
    high30 = valid["bws_cat_30"] >= 3
    high40 = valid["bws_cat_40"] >= 3
    broad_n = int((high30 == high40).sum())
    broad_pct = broad_n / len(valid) * 100
    return valid, exact_n, exact_pct, broad_n, broad_pct


def assign_countries(panel: pd.DataFrame) -> dict[int, dict[str, object]]:
    """Assign basin to modal panel country; keep all countries tied for first."""
    linked = panel.loc[
        panel["basin_pfaf_id"].isin(AFFECTED_BASINS)
        & panel["iso3"].notna()
        & panel["country"].notna(),
        ["basin_pfaf_id", "country", "iso3"],
    ].copy()
    linked["basin_pfaf_id"] = linked["basin_pfaf_id"].astype(int)
    # Count country-year panel rows. Ties at the largest count are retained.
    counts = (
        linked.groupby(["basin_pfaf_id", "country", "iso3"], dropna=False)
        .size()
        .rename("n_records")
        .reset_index()
    )
    assignments: dict[int, dict[str, object]] = {}
    for basin_id in AFFECTED_BASINS:
        rows = counts[counts["basin_pfaf_id"] == basin_id].copy()
        if rows.empty:
            raise ValueError(f"No country-linked panel records for basin {basin_id}")
        max_n = int(rows["n_records"].max())
        selected = rows[rows["n_records"] == max_n]
        assignments[basin_id] = {
            "all_counts": rows.sort_values(["n_records", "country"], ascending=[False, True]),
            "countries": selected["country"].tolist(),
            "iso3": selected["iso3"].astype(str).tolist(),
        }
    return assignments


def format_country_assignment(info: dict[str, object]) -> str:
    rows = info["all_counts"]
    selected_names = set(info["countries"])
    parts = []
    for row in rows.itertuples(index=False):
        name = row.country
        suffix = " (tie retained)" if name in selected_names and len(selected_names) > 1 else ""
        parts.append(f"{name}: {int(row.n_records)}{suffix}")
    return "; ".join(parts)


def prepare_model_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Apply the Figure 4 notebook's complete-case and scaling steps."""
    required = [
        "year", "iso3", "country", "country_count", "conflict", "both",
        "basin_pfaf_id", *CONTEXT_COLUMNS, *MODE_COLUMNS,
    ]
    missing = set(required).difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing columns: {sorted(missing)}")

    data = panel[required].copy()
    for column in ["year", "country_count", "conflict", "both", "basin_pfaf_id", *CONTEXT_COLUMNS, *MODE_COLUMNS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # The notebook treats observations coded as both cooperation and conflict
    # as conflict observations before fitting the regression.
    data.loc[data["both"] == 1, "conflict"] = 1
    data = data.dropna(subset=FEATURES).copy()
    # The outcome is required for fitting; in the cleaned panel all retained
    # model observations have an observed outcome.
    data = data.dropna(subset=["country_count"]).copy()

    scaler = StandardScaler()
    data[CONTEXT_COLUMNS] = scaler.fit_transform(data[CONTEXT_COLUMNS])
    data["basin_pfaf_id"] = data["basin_pfaf_id"].astype("Int64")
    return data, scaler


def make_cutoff_features(
    data: pd.DataFrame,
    base_features: pd.DataFrame,
    scaler: StandardScaler,
    aq30: pd.DataFrame,
    aq40: pd.DataFrame,
    assignments: dict[int, dict[str, object]],
    cutoff: int,
) -> tuple[pd.DataFrame, int]:
    """Replace only basin-linked BWS values whose source changes at a cutoff."""
    features = base_features.copy()
    if cutoff == BASE_CUTOFF:
        return features, 0

    if cutoff < BASE_CUTOFF:
        first_year, last_year = cutoff, BASE_CUTOFF - 1
        source = aq40
    else:
        first_year, last_year = BASE_CUTOFF, cutoff - 1
        source = aq30

    changed = pd.Series(False, index=data.index)
    for basin_id, info in assignments.items():
        mask = (
            data["basin_pfaf_id"].eq(basin_id)
            & data["iso3"].astype(str).isin(info["iso3"])
            & data["year"].between(first_year, last_year)
        )
        if not mask.any():
            continue
        score = source.loc[basin_id, "bws_score"]
        if pd.isna(score):
            raise ValueError(f"Missing BWS score for basin {basin_id} in alternative source")
        bws_index = CONTEXT_COLUMNS.index("bws_score")
        features.loc[mask, "bws_score"] = (
            float(score) - scaler.mean_[bws_index]
        ) / scaler.scale_[bws_index]
        changed |= mask
    return features, int(changed.sum())


def fit_and_summarize(
    data: pd.DataFrame,
    features: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    background_indices: pd.Index,
) -> dict[str, object]:
    """Fit the notebook's Random Forest and summarize error and SHAP ranks."""
    model = RandomForestRegressor(random_state=42)
    model.fit(features.loc[train_mask, FEATURES], data.loc[train_mask, "country_count"])
    prediction = model.predict(features.loc[test_mask, FEATURES])
    observed = data.loc[test_mask, "country_count"].to_numpy()

    # A fixed 100-row training background is reused across cutoffs to make
    # SHAP importance comparisons stable and computationally manageable.
    background = features.loc[background_indices, FEATURES]
    explainer = shap.Explainer(model, background)
    shap_values = explainer(features.loc[test_mask, FEATURES])
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(mean_abs, index=FEATURES)
    mode_order = importance[MODE_COLUMNS].sort_values(ascending=False).index.tolist()
    all_order = importance.sort_values(ascending=False).index.tolist()

    return {
        "mae": mean_absolute_error(observed, prediction),
        "rmse": mean_squared_error(observed, prediction) ** 0.5,
        "mode_order": mode_order,
        "all_order": all_order,
        "importance": importance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("clean_historical_panel.csv"))
    parser.add_argument("--aqueduct30", type=Path, default=Path("aqueduct30.csv"))
    parser.add_argument("--aqueduct40", type=Path, default=Path("aqueduct40.csv"))
    args = parser.parse_args()

    for path in (args.panel, args.aqueduct30, args.aqueduct40):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    panel = pd.read_csv(args.panel, low_memory=False)
    panel["basin_pfaf_id"] = pd.to_numeric(panel["basin_pfaf_id"], errors="coerce")
    aq30, aq40 = load_aqueduct(args.aqueduct30), load_aqueduct(args.aqueduct40)

    valid, exact_n, exact_pct, broad_n, broad_pct = compare_aqueduct(aq30, aq40)
    print("AQUEDUCT VERSION COMPARISON")
    print(f"Common basins with valid categories: {len(valid):,}")
    print(f"Exact category agreement: {exact_n:,}/{len(valid):,} ({exact_pct:.1f}%)")
    print(f"High/very high agreement (categories 3--4): {broad_n:,}/{len(valid):,} ({broad_pct:.1f}%)")

    assignments = assign_countries(panel)
    print("\nFOUR BASINS IN THE STUDY PANEL WITH CATEGORY CHANGES")
    for basin_id in AFFECTED_BASINS:
        row30, row40 = aq30.loc[basin_id], aq40.loc[basin_id]
        print(
            f"{basin_id}: {format_country_assignment(assignments[basin_id])}; "
            f"Aqueduct 3.0 category/score={row30['bws_cat']:.0f}/{row30['bws_score']:.6f}; "
            f"Aqueduct 4.0 category/score={row40['bws_cat']:.0f}/{row40['bws_score']:.6f}"
        )

    data, scaler = prepare_model_panel(panel)
    train_mask = data["year"] <= 2010
    test_mask = (data["year"] > 2010) & (data["conflict"] == 1)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Training or held-out sample is empty")
    background_indices = data.index[train_mask].to_series().sample(
        n=min(100, int(train_mask.sum())), random_state=42
    ).index

    print("\nMODEL SAMPLES")
    print(f"Training observations: {int(train_mask.sum()):,}")
    print(f"Held-out conflict observations after 2010: {int(test_mask.sum()):,}")
    print("\nCUTOFF SENSITIVITY")
    results: dict[int, dict[str, object]] = {}
    for cutoff in ALTERNATIVE_CUTOFFS:
        features, changed_n = make_cutoff_features(
            data, data[FEATURES], scaler, aq30, aq40, assignments, cutoff
        )
        result = fit_and_summarize(
            data, features, train_mask, test_mask, background_indices
        )
        result["changed_training_rows"] = int(changed_n)
        result["changed_heldout_rows"] = 0
        results[cutoff] = result
        print(
            f"{cutoff}: changed training rows={result['changed_training_rows']}; "
            f"changed held-out rows={result['changed_heldout_rows']}; "
            f"MAE={result['mae']:.3f}; RMSE={result['rmse']:.3f}"
        )
        print("  Mode SHAP order: " + " > ".join(result["mode_order"]))
        print("  All-feature SHAP order: " + " > ".join(result["all_order"]))

    baseline_order = results[BASE_CUTOFF]["mode_order"]
    same_modes = all(results[c]["mode_order"] == baseline_order for c in ALTERNATIVE_CUTOFFS)
    bws_ranks = {
        c: results[c]["all_order"].index("bws_score") + 1
        for c in ALTERNATIVE_CUTOFFS
    }
    print("\nCHECKS")
    print(f"Six-mode SHAP ranking identical across cutoffs: {same_modes}")
    print("BWS score rank among all predictors: " + ", ".join(f"{c}: {bws_ranks[c]}" for c in ALTERNATIVE_CUTOFFS))


if __name__ == "__main__":
    main()
