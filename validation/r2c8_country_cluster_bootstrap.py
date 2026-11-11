#!/usr/bin/env python3
"""Reproduce country-cluster bootstrap intervals for Figure 4 contrasts.

The model is fitted on observations through 2010. For the 2011-2019 test
period, each contrast is the mean difference between predictions for a
conflict-only coding and a hard-coded cooperation coding. The 95% interval is
obtained by resampling countries with replacement and retaining every eligible
row belonging to each sampled country.

Usage:
  python r2c8_country_cluster_bootstrap.py --panel path/to/03_Clustered_panel_1119.xlsx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

MODES = [
    "Cross-Border Basin Agreements",
    "Collaborative River Basin Planning",
    "Joint Water Allocation Models",
    "Joint Data-Sharing Systems",
    "Transboundary Water Quality Standards",
    "Coordinated Hydropower Operations",
]
CONTEXT = [
    "gdp_percapita", "export", "population_density", "rural_population",
    "precipitation", "bws_score",
]
FEATURES = ["Conflict", *CONTEXT, *MODES]
THREE_MODE_COMBINATION = [
    "Cross-Border Basin Agreements",
    "Collaborative River Basin Planning",
    "Joint Water Allocation Models",
]


def load_panel(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    if "Both" in raw.columns:
        raw.loc[raw["Both"] == 1, ["Conflict", "Cooperation"]] = 1
    raw["Potential_Conflict"] = raw["Conflict"]
    required = ["year", "ccode_iso3", "Country_count", "Potential_Conflict", *FEATURES]
    panel = raw[required].replace("..", np.nan).copy()
    panel = panel.dropna(subset=[c for c in required if c != "Country_count"])
    for col in ["year", "Country_count", "Potential_Conflict", *FEATURES]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.dropna(subset=["year", "Potential_Conflict", *FEATURES])
    panel["year"] = panel["year"].astype(int)
    return panel


def scenario_difference(model, cases: pd.DataFrame, active_modes: list[str]) -> np.ndarray:
    conflict_only = cases[FEATURES].copy()
    conflict_only["Conflict"] = 1
    conflict_only[MODES] = 0
    cooperation = cases[FEATURES].copy()
    cooperation["Conflict"] = 0
    cooperation[MODES] = 0
    cooperation[active_modes] = 1
    return model.predict(conflict_only) - model.predict(cooperation)


def cluster_bootstrap_interval(
    differences: np.ndarray,
    countries: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile interval; sampled country contributes all its rows each time."""
    frame = pd.DataFrame({"country": countries, "difference": differences})
    by_country = frame.groupby("country", sort=False)["difference"].agg(["sum", "count"])
    sums = by_country["sum"].to_numpy()
    counts = by_country["count"].to_numpy()
    n_countries = len(by_country)
    draws = rng.integers(0, n_countries, size=(n_boot, n_countries))
    boot_sums = sums[draws].sum(axis=1)
    boot_counts = counts[draws].sum(axis=1)
    boot_means = boot_sums / boot_counts
    return tuple(np.quantile(boot_means, [0.025, 0.975]))


def summarize(name, active_modes, cases, model, rng, n_boot):
    difference = scenario_difference(model, cases, active_modes)
    lo, hi = cluster_bootstrap_interval(
        difference, cases["ccode_iso3"].to_numpy(), n_boot, rng
    )
    return {
        "scenario": name,
        "mean_predicted_difference": float(difference.mean()),
        "ci_2.5": float(lo),
        "ci_97.5": float(hi),
        "n_observations": len(cases),
        "n_countries": cases["ccode_iso3"].nunique(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--reps", default=10_000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    panel = load_panel(args.panel)
    training = panel.loc[(panel["year"] <= 2010) & panel["Country_count"].notna()]
    test = panel.loc[panel["year"] > 2010]
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(training[FEATURES], training["Country_count"])

    cases = test.loc[test["Potential_Conflict"] == 1].copy()
    rng = np.random.default_rng(args.seed)
    results = [
        summarize(mode, [mode], cases, model, rng, args.reps) for mode in MODES
    ]

    # The originally reported three-mode contrast is limited to test cases
    # where all three modes were observed together.
    combo_cases = cases.loc[cases[THREE_MODE_COMBINATION].eq(1).all(axis=1)].copy()
    results.append(summarize(
        "Three-mode combination", THREE_MODE_COMBINATION,
        combo_cases, model, rng, args.reps
    ))

    output = pd.DataFrame(results)
    print(output.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if args.output:
        output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
