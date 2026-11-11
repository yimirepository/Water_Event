#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Card-style circular radar panel inspired by ggradar/ggiraphExtra aesthetics.
Six circular radars are placed inside boxed cards with integrated title bands,
node values, and a shared A-F label key.

Recovered from cooperation_radar_v6.ipynb, code cell 7 (zero-based), in feedback.zip.
Only the offline plotting cell is retained; the fixed input scores are not recomputed.
"""

import argparse
import csv
import re
from pathlib import Path
from textwrap import fill
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

PLOT_RC_PARAMS = {
    "figure.figsize": (14.9, 10.8),
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 14.6,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#222222",
    "text.color": "#111111",
    "axes.labelcolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

BACKGROUND = "#FFFFFF"
CARD_EDGE = "#BDC9D5"
CARD_EDGE_SOFT = "#E6ECF2"
CARD_LINEWIDTH = 1.75
GRID_COLOR = "#DCE3EA"
OUTER_RING = "#AEB8C2"
TEXT_COLOR = "#111111"

PALETTE = {
    "Blue": {"line": "#1F4E79", "fill": "#AFC6DD"},
    "Orange": {"line": "#C75B12", "fill": "#EDC2A8"},
    "Green": {"line": "#1B7F6A", "fill": "#B8DDD4"},
    "Purple": {"line": "#8C3F74", "fill": "#DEC0D4"},
    "Gold": {"line": "#8B6A00", "fill": "#E6D8A3"},
    "Teal": {"line": "#007A8A", "fill": "#B7DBE0"},
}

MODE_ORDER = [
    "Collaborative Planning & Adaptation Strategies",
    "Cross-Border Basin Agreements",
    "Joint Water Allocation Models",
    "Joint Data-Sharing Systems",
    "Transboundary Water Quality Standards",
    "Coordinated Hydropower Operations",
]

MODE_ALIASES = {
    "Collaborative Planning and Adaptation Strategies": "Collaborative Planning & Adaptation Strategies",
    "Collaborative River Basin Planning": "Collaborative Planning & Adaptation Strategies",
}

MODE_STYLES = {
    "Cross-Border Basin Agreements": PALETTE["Blue"],
    "Collaborative Planning & Adaptation Strategies": PALETTE["Orange"],
    "Joint Water Allocation Models": PALETTE["Green"],
    "Joint Data-Sharing Systems": PALETTE["Purple"],
    "Transboundary Water Quality Standards": PALETTE["Gold"],
    "Coordinated Hydropower Operations": PALETTE["Teal"],
}

ASPECT_ORDER = ["Supportive", "Facilitative", "Stress-relieving", "Complementary", "Preventive", "Prescriptive"]
AXIS_TAGS = ["A", "B", "C", "D", "E", "F"]
AXIS_LEGEND = "A = Supportive   |   B = Facilitative   |   C = Stress-relieving   |   D = Complementary   |   E = Preventive   |   F = Prescriptive"

MODE_TITLES = {
    "Cross-Border Basin Agreements": "Cross-border basin agreements",
    "Collaborative Planning & Adaptation Strategies": "Collaborative planning and adaptation strategies",
    "Joint Water Allocation Models": "Joint water allocation models",
    "Joint Data-Sharing Systems": "Joint data-sharing systems",
    "Transboundary Water Quality Standards": "Transboundary water quality standards",
    "Coordinated Hydropower Operations": "Coordinated hydropower operations",
}

def read_scores_from_txt(file_path: Path):
    """Read the supplied score matrix, rejecting malformed or duplicate entries."""
    scores = {}
    current_mode = None
    for line_number, raw_line in enumerate(Path(file_path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        header = re.fullmatch(r"[1-6]\.\s+(.+)", line)
        if header:
            current_mode = MODE_ALIASES.get(header.group(1), header.group(1))
            if current_mode in scores:
                raise ValueError(f"Duplicate mode at line {line_number}: {current_mode}")
            scores[current_mode] = {}
        elif current_mode and ":" in line:
            aspect, value = (part.strip() for part in line.split(":", 1))
            if aspect in scores[current_mode]:
                raise ValueError(f"Duplicate aspect at line {line_number}: {aspect}")
            try:
                scores[current_mode][aspect] = float(value)
            except ValueError as error:
                raise ValueError(f"Invalid score at line {line_number}: {value}") from error
        else:
            raise ValueError(f"Invalid score-file line {line_number}: {line}")
    return scores


def clean_and_normalize_scores(scores: dict):
    """Retain the source naming aliases and require a complete 6-by-6 matrix."""
    normalized = {}
    for mode, sc in scores.items():
        canonical_mode = MODE_ALIASES.get(mode, mode)
        if canonical_mode in normalized:
            raise ValueError(f"Duplicate canonical mode: {canonical_mode}")
        normalized[canonical_mode] = dict(sc)
        if "Affordable" in normalized[canonical_mode] and "Supportive" not in normalized[canonical_mode]:
            normalized[canonical_mode]["Supportive"] = normalized[canonical_mode].pop("Affordable")
    if set(normalized) != set(MODE_ORDER):
        raise ValueError("The input must contain exactly the six published cooperation modes.")
    for mode, sc in normalized.items():
        if set(sc) != set(ASPECT_ORDER):
            raise ValueError(f"The input must contain exactly six published aspects for {mode}.")
        for aspect in ASPECT_ORDER:
            value = float(sc[aspect])
            if not np.isfinite(value) or not 0.0 <= value <= 6.0:
                raise ValueError(f"Score outside the finite [0, 6] range: {mode}, {aspect}")
            sc[aspect] = value
    return normalized


def make_angles(n_vars: int):
    base_angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    return base_angles, base_angles + base_angles[:1]

def draw_card_radar(card_ax, mode_name: str, score_dict: dict, style: dict):
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.axis("off")

    border_shadow = FancyBboxPatch((0.013, 0.013), 0.974, 0.974, boxstyle="round,pad=0.001,rounding_size=0.020", linewidth=0.85, edgecolor=CARD_EDGE_SOFT, facecolor="none", transform=card_ax.transAxes, zorder=0)
    card_ax.add_patch(border_shadow)
    border = FancyBboxPatch((0.012, 0.012), 0.976, 0.976, boxstyle="round,pad=0.001,rounding_size=0.020", linewidth=CARD_LINEWIDTH, edgecolor=CARD_EDGE, facecolor=BACKGROUND, transform=card_ax.transAxes, zorder=1)
    card_ax.add_patch(border)

    band = Rectangle((0.012, 0.908), 0.976, 0.082, linewidth=0, facecolor=style["fill"], transform=card_ax.transAxes, zorder=2)
    card_ax.add_patch(band)
    card_ax.plot([0.012, 0.988], [0.908, 0.908], color=CARD_EDGE, linewidth=CARD_LINEWIDTH, transform=card_ax.transAxes, zorder=3)

    title = fill(MODE_TITLES.get(mode_name, mode_name), width=28)
    card_ax.text(0.5, 0.949, title, ha="center", va="center", fontsize=14.7, color=TEXT_COLOR, fontweight="bold", transform=card_ax.transAxes, zorder=4)

    ax = card_ax.inset_axes([0.095, 0.105, 0.81, 0.735], polar=True)
    ax.set_facecolor(BACKGROUND)

    n_vars = len(ASPECT_ORDER)
    base_angles, closed_angles = make_angles(n_vars)
    values = [score_dict.get(a, 0.0) for a in ASPECT_ORDER]
    values_closed = values + values[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 6)
    ax.set_xticks(base_angles)
    ax.set_xticklabels(AXIS_TAGS, fontsize=18.2, fontweight="bold", color=TEXT_COLOR)
    ax.tick_params(axis="x", pad=12)
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.9)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.9)
    ax.spines["polar"].set_color(OUTER_RING)
    ax.spines["polar"].set_linewidth(1.15)

    ax.fill(closed_angles, values_closed, color=style["fill"], alpha=0.34, zorder=2)
    ax.plot(closed_angles, values_closed, color=style["line"], linewidth=3.05, solid_capstyle="round", zorder=3)
    ax.scatter(base_angles, values, s=46, color=style["line"], edgecolor="white", linewidth=1.05, zorder=4)

    for angle, value in zip(base_angles, values):
        ax.text(angle, min(value + 0.56, 6.20), f"{value:.1f}", fontsize=12.8, fontweight="bold", color=style["line"], ha="center", va="center", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.99), zorder=5)

def plot_card_radar_panel(scores: dict, output_dir: Path, filename_base="cooperation_modes_radar_cards_panel"):
    ordered_items = [(mode, scores[mode]) for mode in MODE_ORDER if mode in scores]
    if not ordered_items:
        print("[WARN] No scores found.")
        return

    fig = plt.figure(figsize=(14.9, 10.8), facecolor=BACKGROUND)
    gs = fig.add_gridspec(2, 3, left=0.018, right=0.982, top=0.946, bottom=0.122, wspace=0.09, hspace=0.16)

    for idx, (mode_name, score_dict) in enumerate(ordered_items):
        card_ax = fig.add_subplot(gs[idx // 3, idx % 3])
        draw_card_radar(card_ax, mode_name, score_dict, MODE_STYLES[mode_name])

    fig.text(0.5, 0.040, AXIS_LEGEND, ha="center", va="center", fontsize=14.9, color=TEXT_COLOR, fontweight="semibold")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{filename_base}.png"
    svg_path = output_dir / f"{filename_base}.svg"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=False)
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=False)

    plt.close(fig)
    print(f"[OK] Saved PNG: {png_path}")
    print(f"[OK] Saved SVG: {svg_path}")
    return {"png": png_path, "svg": svg_path}

def run_pipeline(output_dir: Path | None = None, score_file: Path | None = None):
    """Render the recovered Figure 5a source and export the actual plotted values."""
    module_dir = Path(__file__).resolve().parent
    input_file = Path(score_file) if score_file is not None else module_dir.parent / "figure5b" / "data" / "cooperation_mode_scores.txt"
    output_root = Path(output_dir) if output_dir is not None else module_dir / "outputs"
    scores = clean_and_normalize_scores(read_scores_from_txt(input_file))
    with mpl.rc_context(PLOT_RC_PARAMS):
        figures = plot_card_radar_panel(scores, output_root)
    csv_path = output_root / "plot_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", *ASPECT_ORDER])
        for mode in MODE_ORDER:
            writer.writerow([mode, *(scores[mode][aspect] for aspect in ASPECT_ORDER)])
    print(f"[OK] Saved plotted scores: {csv_path}")
    return {"scores": scores, "plot_scores_csv": csv_path, "figures": figures}


def main():
    parser = argparse.ArgumentParser(description="Render Figure 5a from the supplied fixed cooperation scores.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: figure5a/outputs).")
    parser.add_argument("--scores", type=Path, default=None, help="Optional score file; defaults to the shared Figure 5b input.")
    args = parser.parse_args()
    run_pipeline(output_dir=args.output_dir, score_file=args.scores)


if __name__ == "__main__":
    main()
