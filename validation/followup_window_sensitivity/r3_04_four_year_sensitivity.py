"""R3-04: direct four-year versus five-year cooperation-duration sensitivity.

The test uses the available human-coded cooperation-event workbook inside the
repository archive and the clean historical country-year panel. For each event,
it asks whether either participating country has a recorded water-related
conflict in the next four or five calendar years. The same event sample is used
for both definitions and has complete five-year follow-up.

Run from the project root:
    python r3_04_four_year_sensitivity.py \
        --panel audit_materials/clean_historical_panel.csv \
        --repo current_review/WaterConflict-main(2).zip \
        --output outputs/1965483311d0/r3_04_four_year_sensitivity.csv
"""
from __future__ import annotations
import argparse
import io
import zipfile
import unicodedata
from pathlib import Path

import pandas as pd

EVENT_MEMBER = (
    "WaterConflict-main/Data processing/"
    "20241107_One_hot_JB241107v2_02.xlsx"
)


def clean_iso3(value: object) -> str:
    """Normalize whitespace in the archived three-letter country codes."""
    if pd.isna(value):
        return ""
    return (
        unicodedata.normalize("NFKC", str(value))
        .replace("\u00a0", "")
        .replace("\u807d", "")
        .strip()
        .upper()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    panel["iso3"] = panel["iso3"].map(clean_iso3)
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel["conflict_count_yr"] = pd.to_numeric(panel["conflict_count_yr"], errors="coerce")
    panel = panel.loc[panel["iso3"].ne("")].copy()

    # The historical panel should have at most one row for each country and year.
    if panel.duplicated(["iso3", "year"]).any():
        duplicates = panel.loc[panel.duplicated(["iso3", "year"], keep=False), ["iso3", "year"]]
        raise ValueError(f"Duplicate country-year rows found:\n{duplicates.head()}")
    if panel["conflict_count_yr"].isna().any():
        raise ValueError("Missing conflict counts remain in the matched historical panel.")

    country_year_conflict = {
        (row.iso3, int(row.year)): int(row.conflict_count_yr > 0)
        for row in panel[["iso3", "year", "conflict_count_yr"]].itertuples(index=False)
    }
    panel_countries = set(panel["iso3"])
    last_year = int(panel["year"].max())

    with zipfile.ZipFile(args.repo) as repo:
        event_bytes = repo.read(EVENT_MEMBER)
    events = pd.read_excel(io.BytesIO(event_bytes), sheet_name=0)
    events["country1"] = events["country1"].map(clean_iso3)
    events["country2"] = events["country2"].map(clean_iso3)
    events["year"] = pd.to_numeric(events["year"], errors="coerce")

    # Keep bilateral events whose two participants can be checked in the panel
    # and whose full five-year follow-up falls inside the historical period.
    eligible = events.loc[
        events["country1"].isin(panel_countries)
        & events["country2"].isin(panel_countries)
        & events["year"].notna()
        & (events["year"] <= last_year - 5)
    ].copy()

    def classify(row: pd.Series, horizon: int) -> bool | None:
        participants = {row["country1"], row["country2"]}
        followup = [
            country_year_conflict.get((country, int(row["year"]) + year_offset))
            for country in participants
            for year_offset in range(1, horizon + 1)
        ]
        if any(value is None for value in followup):
            return None
        return not any(followup)

    eligible["no_conflict_4y"] = eligible.apply(lambda row: classify(row, 4), axis=1)
    eligible["no_conflict_5y"] = eligible.apply(lambda row: classify(row, 5), axis=1)
    common = eligible.dropna(subset=["no_conflict_4y", "no_conflict_5y"]).copy()
    common["no_conflict_4y"] = common["no_conflict_4y"].astype(bool)
    common["no_conflict_5y"] = common["no_conflict_5y"].astype(bool)

    # Both results must be based on the same records, and the five-year group
    # must be nested within the four-year group.
    assert len(common) > 0
    assert not (common["no_conflict_5y"] & ~common["no_conflict_4y"]).any()

    rows = []
    for label, column, years in [
        ("Four-year definition", "no_conflict_4y", 4),
        ("Five-year definition", "no_conflict_5y", 5),
    ]:
        n = int(common[column].sum())
        rows.append({
            "definition": label,
            "followup_years": years,
            "eligible_cooperation_events": int(len(common)),
            "events_followed_by_no_conflict": n,
            "share_percent": round(100 * n / len(common), 2),
            "event_year_start": int(common["year"].min()),
            "event_year_end": int(common["year"].max()),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)

    n4 = int(common["no_conflict_4y"].sum())
    n5 = int(common["no_conflict_5y"].sum())
    print(f"Common event sample: {len(common)} events, {int(common.year.min())}–{int(common.year.max())}")
    print(f"Four years: {n4}/{len(common)} ({100*n4/len(common):.2f}%)")
    print(f"Five years: {n5}/{len(common)} ({100*n5/len(common):.2f}%)")
    print(f"Events that change classification: {n4-n5} ({100*(n4-n5)/len(common):.2f} percentage points)")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
