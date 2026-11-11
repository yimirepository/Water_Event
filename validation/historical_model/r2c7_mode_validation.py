#!/usr/bin/env python3
"""Reproduce the R2C7 prediction-accuracy and mode-scenario comparison.

Input: the Figure 4 country-year panel from the Water_Cooperation_and_Conflict
repository (03_Clustered_panel_1119.xlsx).

The script reports (1) observed-outcome prediction metrics for the original
2011-2019 split and for five chronological folds, and (2) hard-coded mean
prediction contrasts for six individual cooperation modes and the prespecified
three-mode combination. It does not treat these contrasts as causal effects.

Example:
  python r2c7_mode_validation.py --panel path/to/03_Clustered_panel_1119.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODES = [
    "Cross-Border Basin Agreements",
    "Collaborative River Basin Planning",
    "Joint Water Allocation Models",
    "Joint Data-Sharing Systems",
    "Transboundary Water Quality Standards",
    "Coordinated Hydropower Operations",
]
CONTEXT = [
    "gdp_percapita",
    "export",
    "population_density",
    "rural_population",
    "precipitation",
    "bws_score",
]
FEATURES = ["Conflict", *CONTEXT, *MODES]
THREE_MODE_COMBINATION = [
    "Cross-Border Basin Agreements",
    "Collaborative River Basin Planning",
    "Joint Water Allocation Models",
]
FOLD_ORIGINS = [1990, 1996, 2002, 2008, 2014]


def load_panel(path: Path) -> pd.DataFrame:
    """Read and prepare the Figure 4 panel using the notebook's input fields."""
    raw = pd.read_excel(path)

    # Match the Figure 4 notebook's treatment of events coded as both conflict
    # and cooperation, then create the potential-conflict test flag.
    if "Both" in raw.columns:
        raw.loc[raw["Both"] == 1, ["Conflict", "Cooperation"]] = 1
    raw["Cooperation"] = raw[MODES].eq(1).any(axis=1).astype(int)
    raw["Potential_Conflict"] = raw["Conflict"]

    required = ["year", "ccode_iso3", "Country_count", "Potential_Conflict", *FEATURES]
    panel = raw[required].replace("..", np.nan).copy()

    # The notebook removes rows missing predictors, while Country_count is the
    # stored five-year outcome and is not recomputed from the filtered panel.
    predictor_columns = [c for c in required if c != "Country_count"]
    panel = panel.dropna(subset=predictor_columns)
    numeric = ["year", "Country_count", "Potential_Conflict", *FEATURES]
    for column in numeric:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(subset=["year", "Potential_Conflict", *FEATURES])
    panel["year"] = panel["year"].astype(int)
    return panel


def fit_forest(training: pd.DataFrame) -> RandomForestRegressor:
    """Fit the same default Random Forest specification used in Figure 4."""
    fit_data = training.dropna(subset=["Country_count"])
    return RandomForestRegressor(random_state=42, n_jobs=-1).fit(
        fit_data[FEATURES], fit_data["Country_count"]
    )


def prediction_metrics(
    model: RandomForestRegressor, test: pd.DataFrame
) -> dict[str, float | int]:
    """Calculate metrics only for test rows with an observed outcome."""
    scored = test.dropna(subset=["Country_count"])
    observed = scored["Country_count"].to_numpy()
    predicted = model.predict(scored[FEATURES])
    return {
        "n_test": len(scored),
        "MAE": mean_absolute_error(observed, predicted),
        "RMSE": mean_squared_error(observed, predicted) ** 0.5,
        "R2": r2_score(observed, predicted),
    }


def scenario_contrasts(
    model: RandomForestRegressor, test: pd.DataFrame, scheme: str
) -> pd.DataFrame:
    """Apply the fixed hard-coded mode scenarios to conflict test cases.

    Each contrast is the mean of prediction(conflict-only) minus
    prediction(mode scenario); a positive value means the model predicts a
    lower five-year count under the mode scenario. The same eligible conflict
    cases are used for all seven scenarios within each validation scheme.
    """
    cases = test.loc[test["Potential_Conflict"] == 1].copy()
    if cases.empty:
        return pd.DataFrame(
            columns=["validation", "scenario", "n_conflict_cases", "n_countries",
                     "mean_predicted_difference", "positive_share"]
        )

    conflict_only = cases[FEATURES].copy()
    conflict_only["Conflict"] = 1
    conflict_only[MODES] = 0
    baseline_prediction = model.predict(conflict_only)

    scenarios: list[tuple[str, list[str]]] = [(mode, [mode]) for mode in MODES]
    scenarios.append(("Three-mode Integrative Governance combination",
                      THREE_MODE_COMBINATION))

    records = []
    for name, active_modes in scenarios:
        coded = cases[FEATURES].copy()
        coded["Conflict"] = 0
        coded[MODES] = 0
        coded[active_modes] = 1
        difference = baseline_prediction - model.predict(coded)
        records.append(
            {
                "validation": scheme,
                "scenario": name,
                "n_conflict_cases": len(cases),
                "n_countries": cases["ccode_iso3"].nunique(),
                "mean_predicted_difference": float(np.mean(difference)),
                "positive_share": float(np.mean(difference > 0)),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        required=True,
        help="Path to figure4/03_Clustered_panel_1119.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for CSV output tables.",
    )
    args = parser.parse_args()

    panel = load_panel(args.panel)

    # Original split: train through 2010; evaluate on observations after 2010.
    old_train = panel.loc[(panel["year"] <= 2010) & panel["Country_count"].notna()]
    old_test = panel.loc[panel["year"] > 2010]
    old_model = fit_forest(old_train)
    old_metrics = prediction_metrics(old_model, old_test)
    old_metrics["validation"] = "Original 2011-2019 split"
    old_contrasts = scenario_contrasts(
        old_model, old_test, "Original 2011-2019 split"
    )

    # Five expanding chronological folds. For test origin t, training ends at
    # t-6, leaving the five immediately preceding origin years out of training.
    fold_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    fold_contrasts: list[pd.DataFrame] = []
    fold_counts = []
    for origin in FOLD_ORIGINS:
        train = panel.loc[
            (panel["year"] <= origin - 6) & panel["Country_count"].notna()
        ]
        test = panel.loc[panel["year"] == origin]
        model = fit_forest(train)

        scored = test.dropna(subset=["Country_count"])
        fold_predictions.append(
            (
                scored["Country_count"].to_numpy(),
                model.predict(scored[FEATURES]),
            )
        )
        fold_contrasts.append(
            scenario_contrasts(model, test, f"Chronological origin {origin}")
        )
        fold_counts.append(
            {
                "test_origin": origin,
                "training_through": origin - 6,
                "outcome_start": origin + 1,
                "outcome_end": origin + 5,
                "n_test": len(scored),
                "n_conflict_cases": int((test["Potential_Conflict"] == 1).sum()),
            }
        )

    observed = np.concatenate([item[0] for item in fold_predictions])
    predicted = np.concatenate([item[1] for item in fold_predictions])
    blocked_metrics = {
        "validation": "Five chronological blocks",
        "n_test": len(observed),
        "MAE": mean_absolute_error(observed, predicted),
        "RMSE": mean_squared_error(observed, predicted) ** 0.5,
        "R2": r2_score(observed, predicted),
    }

    blocked_rows = []
    for scenario in [*MODES, "Three-mode Integrative Governance combination"]:
        rows = pd.concat(fold_contrasts, ignore_index=True)
        rows = rows.loc[rows["scenario"] == scenario]
        n = int(rows["n_conflict_cases"].sum())
        weighted_difference = np.average(
            rows["mean_predicted_difference"], weights=rows["n_conflict_cases"]
        ) if n else np.nan
        weighted_positive_share = np.average(
            rows["positive_share"], weights=rows["n_conflict_cases"]
        ) if n else np.nan
        # Country count is reported as the union across folds, not a sum.
        blocked_rows.append(
            {
                "validation": "Five chronological blocks",
                "scenario": scenario,
                "n_conflict_cases": n,
                "mean_predicted_difference": weighted_difference,
                "positive_share": weighted_positive_share,
            }
        )
    blocked_contrasts = pd.DataFrame(blocked_rows)

    accuracy = pd.DataFrame([old_metrics, blocked_metrics])
    contrasts = pd.concat([old_contrasts, blocked_contrasts], ignore_index=True)
    folds = pd.DataFrame(fold_counts)

    print("\nObserved-outcome prediction accuracy")
    print(accuracy[["validation", "n_test", "MAE", "RMSE", "R2"]].to_string(
        index=False, formatters={"MAE": "{:.3f}".format,
                                 "RMSE": "{:.3f}".format,
                                 "R2": "{:.3f}".format}
    ))
    print("\nChronological folds")
    print(folds.to_string(index=False))
    print("\nHard-coded scenario contrasts (positive = lower predicted count)")
    print(contrasts[["validation", "scenario", "n_conflict_cases",
                     "mean_predicted_difference", "positive_share"]].to_string(
        index=False,
        formatters={"mean_predicted_difference": "{:.3f}".format,
                    "positive_share": "{:.3f}".format},
    ))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        accuracy.to_csv(args.output_dir / "prediction_accuracy.csv", index=False)
        contrasts.to_csv(args.output_dir / "mode_scenario_contrasts.csv", index=False)
        folds.to_csv(args.output_dir / "chronological_folds.csv", index=False)
        print(f"\nCSV tables written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
