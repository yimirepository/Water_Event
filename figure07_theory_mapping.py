from __future__ import annotations

import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FIGURE_DATA_DIR", THIS_DIR / "data")) / "figure07"
THEORY_NAME_UPDATE_FILE = DATA_DIR / "theory_name_updates.csv"
THEORY_LENS_ORDER_FILE = DATA_DIR / "theory_lens_order.csv"

PALETTE = {
    "Blue": {"line": "#1F4E79", "fill": "#AFC6DD"},
    "Orange": {"line": "#C75B12", "fill": "#EDC2A8"},
    "Green": {"line": "#1B7F6A", "fill": "#B8DDD4"},
    "Purple": {"line": "#8C3F74", "fill": "#DEC0D4"},
    "Gold": {"line": "#8B6A00", "fill": "#E6D8A3"},
    "Teal": {"line": "#007A8A", "fill": "#B7DBE0"},
}

MODE_COLORS = {
    "Cross-Border Basin Agreements": PALETTE["Blue"]["line"],
    "Collaborative Planning and Adaptation Strategies": PALETTE["Orange"]["line"],
    "Joint Water Allocation Models": PALETTE["Green"]["line"],
    "Joint Data-Sharing Systems": PALETTE["Purple"]["line"],
    "Transboundary Water Quality Standards": PALETTE["Gold"]["line"],
    "Coordinated Hydropower Operations": PALETTE["Teal"]["line"],
}

MODE_LIGHT_COLORS = {
    "Cross-Border Basin Agreements": "#90B0CC",
    "Collaborative Planning and Adaptation Strategies": "#DEA989",
    "Joint Water Allocation Models": "#95CDBF",
    "Joint Data-Sharing Systems": "#C7A4BF",
    "Transboundary Water Quality Standards": "#D4C483",
    "Coordinated Hydropower Operations": "#97C8D1",
}

DISCIPLINE_COLORS = {
    "Hydrology": "#5E7F9F",
    "International Relations": "#E68613",
    "Sustainability Science": "#2A9D55",
}

HEATMAP_CMAP = sns.blend_palette(
    ["#FFFFFF", PALETTE["Blue"]["fill"], PALETTE["Blue"]["line"]],
    as_cmap=True,
)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "axes.titlesize": 17.5,
        "axes.labelsize": 14.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)
sns.set_theme(style="whitegrid", font="Arial")


def _save_figure(output_path: Path) -> None:
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


FINAL_MODE_CATALOG = [
    {
        "mode": "Cross-Border Basin Agreements",
        "description": "Formal treaties, protocols, basin committees, and rules for stable cooperation.",
        "keywords": ["treaty", "agreement", "protocol", "framework", "committee", "commission", "institution", "legal", "basin"],
    },
    {
        "mode": "Collaborative Planning and Adaptation Strategies",
        "description": "Joint planning, adaptation, restoration, and stakeholder coordination under uncertainty.",
        "keywords": ["planning", "adaptation", "restoration", "coordination", "stakeholder", "conservation", "adaptive", "resilience"],
    },
    {
        "mode": "Joint Water Allocation Models",
        "description": "Negotiated allocation, benefit-sharing, irrigation deals, and balancing asymmetric demands.",
        "keywords": ["allocation", "distribution", "sharing", "equitable", "benefit", "irrigation", "quota", "tradeoff"],
    },
    {
        "mode": "Joint Data-Sharing Systems",
        "description": "Shared monitoring, data exchange, technical investigations, and forecasting systems.",
        "keywords": ["data", "information", "monitoring", "observation", "investigation", "sharing", "forecast", "technical"],
    },
    {
        "mode": "Transboundary Water Quality Standards",
        "description": "Pollution control, ecological standards, environmental compliance, and quality protection.",
        "keywords": ["water quality", "pollution", "cleanup", "decontamination", "ecological", "environmental", "quality", "standards"],
    },
    {
        "mode": "Coordinated Hydropower Operations",
        "description": "Cooperative dam and hydropower operations for flood control and basin-scale optimization.",
        "keywords": ["hydropower", "hydroelectric", "dam", "reservoir", "power", "flood control", "operation", "infrastructure"],
    },
]

THEORY_CATALOG = [
    {
        "discipline": "Hydrology",
        "theory_name": "Hydro-Diplomacy",
        "source_anchor": "01_Institutional arrangements for beneficial regional cooperation on water.pdf",
        "summary": "Shared water systems create repeated technical interdependence and channels for cooperation.",
        "why_effective": "Technical coordination creates mutual gains and reduces operational uncertainty.",
        "conditions": "Best in interdependent basins requiring recurrent coordination.",
        "effects": "Supports agreements, routine cooperation, and diplomatic continuity.",
        "keywords": ["hydro-diplomacy", "transboundary", "shared water", "cooperation", "agreement", "river basin"],
    },
    {
        "discipline": "Hydrology",
        "theory_name": "Benefit-Sharing Allocation",
        "source_anchor": "01_Institutional arrangements for beneficial regional cooperation on water.pdf",
        "summary": "Durable cooperation is easier when states bargain over benefits, not only water volumes.",
        "why_effective": "Benefit-sharing expands the bargaining space beyond zero-sum allocation.",
        "conditions": "Best when riparians have asymmetric needs across water, energy, or irrigation.",
        "effects": "Improves fairness, flexibility, and durability of allocation deals.",
        "keywords": ["allocation", "benefit sharing", "distribution", "irrigation", "equitable", "tradeoff"],
    },
    {
        "discipline": "Hydrology",
        "theory_name": "instituitional resilience and capacity",
        "source_anchor": "01_Projecting conflict risk in transboundary river basins by 2050 following different ambition scenarios.pdf",
        "summary": "Basin rules should be iterative and responsive to climate variability and changing demand.",
        "why_effective": "Adaptive planning prevents brittle one-shot agreements under uncertainty.",
        "conditions": "Best under drought, flood, climate stress, or rapid development change.",
        "effects": "Improves preparedness, resilience, and stress reduction.",
        "keywords": ["adaptive", "planning", "climate", "uncertainty", "adaptation", "resilience"],
    },
    {
        "discipline": "Hydrology",
        "theory_name": "Coordinated Infrastructure Operations",
        "source_anchor": "Water Resources Research - 2018 - Wheeler - Exploring Cooperative Transboundary River Management Strategies for the Eastern Nile.pdf",
        "summary": "Joint dam and reservoir operations can outperform fragmented national operation.",
        "why_effective": "Synchronized operations reduce spillovers and optimize storage and power production.",
        "conditions": "Best in basins with cascaded infrastructure or strong seasonal variation.",
        "effects": "Raises efficiency and improves flood, drought, and energy-water management.",
        "keywords": ["hydropower", "dam", "reservoir", "operation", "infrastructure", "flood control"],
    },
    {
        "discipline": "International Relations",
        "theory_name": "Institutionalism and Water Regime Formation",
        "source_anchor": "02_A state-of-the-art review of water diplomacy.pdf",
        "summary": "Formal rules and institutions lower transaction costs and stabilize expectations.",
        "why_effective": "Institutions create procedures, credibility, and venues for dispute management.",
        "conditions": "Best when states need credible commitments over repeated interactions.",
        "effects": "Improves compliance, continuity, and predictability of cooperation.",
        "keywords": ["institution", "regime", "treaty", "committee", "commission", "framework", "governance"],
    },
    {
        "discipline": "International Relations",
        "theory_name": "Trust-Building and Confidence Building",
        "source_anchor": "01_Trust in Transboundary Waters.pdf",
        "summary": "Repeated transparency and low-risk cooperation can build confidence over time.",
        "why_effective": "Trust reduces fears of opportunism and makes coordination politically feasible.",
        "conditions": "Best when relations are fragile but repeated contact is still possible.",
        "effects": "Improves transparency, negotiation quality, and willingness to share data.",
        "keywords": ["trust", "confidence", "transparency", "data sharing", "credibility", "monitoring"],
    },
    {
        "discipline": "International Relations",
        "theory_name": "Negotiated Water Diplomacy",
        "source_anchor": "02_Contemporary Water Research - 2015 - Islam - Water Diplomacy  A Negotiated Approach to Manage Complex Water Problems.pdf",
        "summary": "Complex water disputes are managed through negotiation, mediation, and issue reframing.",
        "why_effective": "Negotiation broadens bargaining space under power asymmetry and uncertainty.",
        "conditions": "Best for contested, politically sensitive, cross-sector water problems.",
        "effects": "Unlocks deadlocks and improves dispute resolution.",
        "keywords": ["negotiation", "mediation", "dispute", "dialogue", "diplomacy", "bargaining"],
    },
    {
        "discipline": "International Relations",
        "theory_name": "Nexus Governance and Issue Linkage",
        "source_anchor": "01_Water diplomacy and nexus governance in a transboundary context.pdf",
        "summary": "Linking water to energy, food, trade, or security expands cooperation packages.",
        "why_effective": "Issue linkage creates extra bargaining chips beyond water volumes alone.",
        "conditions": "Best when water disputes sit inside wider economic or geopolitical relations.",
        "effects": "Improves political feasibility and regime durability.",
        "keywords": ["nexus", "issue linkage", "energy", "food", "trade", "security", "regional governance"],
    },
    {
        "discipline": "Sustainability Science",
        "theory_name": "Environmental Peacebuilding",
        "source_anchor": "01_dresse-et-al-2018-environmental-peacebuilding-towards-a-theoretical-framework.pdf",
        "summary": "Joint environmental action can reduce conflict risk and build cooperative routines.",
        "why_effective": "Ecological improvement and cooperation can reinforce one another.",
        "conditions": "Best where degradation and political tension overlap but shared dependence remains.",
        "effects": "Reduces conflict incentives and strengthens cooperation norms.",
        "keywords": ["environmental peacebuilding", "cooperation", "conflict reduction", "restoration", "shared environment"],
    },
    {
        "discipline": "Sustainability Science",
        "theory_name": "Adaptive and Polycentric Governance",
        "source_anchor": "01_dresse-et-al-2018-environmental-peacebuilding-towards-a-theoretical-framework.pdf",
        "summary": "Layered and flexible governance can manage cross-scale sustainability complexity better.",
        "why_effective": "Polycentric arrangements allow learning and adaptation across levels.",
        "conditions": "Best when governance spans multiple sectors, scales, and stakeholders.",
        "effects": "Improves adaptability, coordination, and resilience.",
        "keywords": ["adaptive governance", "polycentric", "multi-level", "stakeholder", "learning", "coordination"],
    },
    {
        "discipline": "Sustainability Science",
        "theory_name": "Socio-Ecological Resilience",
        "source_anchor": "01_Projecting conflict risk in transboundary river basins by 2050 following different ambition scenarios.pdf",
        "summary": "Resilient basins need ecological buffering and institutional flexibility under stress.",
        "why_effective": "Resilience-oriented strategies keep cooperation viable under shocks.",
        "conditions": "Best where drought, flood, and environmental stress are intensifying.",
        "effects": "Strengthens adaptive capacity and lowers cascading conflict risk.",
        "keywords": ["resilience", "adaptive capacity", "stress", "drought", "flood", "vulnerability"],
    },
    {
        "discipline": "Sustainability Science",
        "theory_name": "Ecosystem Stewardship and Restoration",
        "source_anchor": "01_Transboundary cooperation in establishing ecological networks.pdf",
        "summary": "Sustainable cooperation should protect ecological integrity through restoration and standards.",
        "why_effective": "Restoration creates visible ecological gains and shared protection goals.",
        "conditions": "Best when pollution or ecological degradation weakens basin outcomes.",
        "effects": "Improves water quality, ecosystem services, and legitimacy of cooperation.",
        "keywords": ["ecosystem", "restoration", "conservation", "water quality", "stewardship", "standards"],
    },
]

THEORY_LENS_ORDER_FALLBACK = [
    "Hydrology: Hydro-Diplomacy",
    "Hydrology: Benefit-Sharing Allocation",
    "Hydrology: instituitional resilience and capacity",
    "Hydrology: Coordinated Infrastructure Operations",
    "International Relations: Institutionalism and Water Regime Formation",
    "International Relations: Trust-Building and Confidence Building",
    "International Relations: Negotiated Water Diplomacy",
    "International Relations: Nexus Governance and Issue Linkage",
    "Sustainability Science: Environmental Peacebuilding",
    "Sustainability Science: Adaptive and Polycentric Governance",
    "Sustainability Science: Socio-Ecological Resilience",
    "Sustainability Science: Ecosystem Stewardship and Restoration",
]

CLUSTER_FAMILY_MAP = {
    "Transboundary Waterway Strategies": "Transboundary environmental protection & water quality",
    "Hydro-Diplomacy Alliances": "Hydropower & infrastructure coordination",
    "Hydraulic Diplomacy Ventures": "Hydropower & infrastructure coordination",
    "Diplomatic Watershed Accords": "Basin agreements & institutional frameworks",
    "Hydraulic Alliances": "Hydropower & infrastructure coordination",
    "Mekong Synergy Initiatives": "Regional coordination & diplomatic alignment",
    "Diplomatic Endorsement Dynamics": "Regional coordination & diplomatic alignment",
    "Diplomatic Water Initiatives": "Allocation, irrigation & negotiated sharing",
    "Transboundary Environmental Initiatives": "Transboundary environmental protection & water quality",
    "Hydroelectric Diplomacy Chronicles": "Hydropower & infrastructure coordination",
    "Transnational Water Management": "Basin agreements & institutional frameworks",
    "Hydroelectric Diplomacy": "Hydropower & infrastructure coordination",
    "Technical Hydro-Diplomacy": "Data, technical exchange & monitoring",
    "Hydrodiplomacy Frameworks": "Basin agreements & institutional frameworks",
    "Riparian Diplomatic Initiatives": "Regional coordination & diplomatic alignment",
    "Diplomatic Maneuverings Chronicle": "Regional coordination & diplomatic alignment",
    "Irrigation Diplomacy Initiative": "Allocation, irrigation & negotiated sharing",
    "Eco-Diplomacy Frameworks": "Transboundary environmental protection & water quality",
    "Hydro-Diplomacy Initiatives": "Hydropower & infrastructure coordination",
    "\"Hydro-Diplomacy and Restoration\"": "Planning, restoration & stewardship",
    "Hydro-Development Diplomacy": "Hydropower & infrastructure coordination",
    "Unified Hydro-Diplomacy Frameworks": "Basin agreements & institutional frameworks",
    "Hydraulic Diplomacy Initiatives": "Hydropower & infrastructure coordination",
    "Mekong Water Diplomacy Dynamics": "Regional coordination & diplomatic alignment",
    "Diplomatic Shifts and Alliances": "Regional coordination & diplomatic alignment",
    "Diplomatic Watershed Initiatives": "Allocation, irrigation & negotiated sharing",
    "Trans-Boundary Water Solutions": "Data, technical exchange & monitoring",
    "Hydro-Diplomacy Synergies": "Hydropower & infrastructure coordination",
}

CLUSTER_FAMILY_ORDER = [
    "Basin agreements & institutional frameworks",
    "Planning, restoration & stewardship",
    "Allocation, irrigation & negotiated sharing",
    "Data, technical exchange & monitoring",
    "Hydropower & infrastructure coordination",
    "Regional coordination & diplomatic alignment",
    "Transboundary environmental protection & water quality",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _keyword_score(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    normalized = _normalize_text(text)
    hits = sum(1 for keyword in keywords if _normalize_text(keyword) in normalized)
    return hits / len(keywords)


def _hybrid_similarity(
    source_texts: list[str],
    target_texts: list[str],
    source_keywords: list[list[str]] | None = None,
    target_keywords: list[list[str]] | None = None,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(source_texts + target_texts)
    cosine = cosine_similarity(matrix[: len(source_texts)], matrix[len(source_texts) :]) * 0.7
    hybrid = cosine.copy()

    if target_keywords:
        for i, text in enumerate(source_texts):
            for j, keywords in enumerate(target_keywords):
                hybrid[i, j] += 0.3 * _keyword_score(text, keywords)

    if source_keywords:
        for i, keywords in enumerate(source_keywords):
            for j, text in enumerate(target_texts):
                hybrid[i, j] += 0.3 * _keyword_score(text, keywords)

    return hybrid


def load_theory_name_updates(update_file: Path | None = None) -> dict[tuple[str, str], str]:
    update_file = update_file or THEORY_NAME_UPDATE_FILE
    if not update_file.exists():
        return {}
    try:
        df = pd.read_csv(update_file)
    except Exception:
        return {}
    required = {"discipline", "theory_name", "theory_name_news"}
    if not required.issubset(set(df.columns)):
        return {}
    updates: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        discipline = str(row.get("discipline", "")).strip()
        old_name = str(row.get("theory_name", "")).strip()
        new_name = str(row.get("theory_name_news", "")).strip()
        if not discipline or not old_name or not new_name or new_name.lower() == "nan":
            continue
        updates[(discipline, old_name)] = new_name
    return updates


def apply_theory_name_updates(theories: pd.DataFrame, updates: dict[tuple[str, str], str]) -> pd.DataFrame:
    if not updates:
        return theories
    updated = theories.copy()
    updated["theory_name_original"] = updated["theory_name"]
    updated["theory_name"] = updated.apply(
        lambda row: updates.get((row["discipline"], row["theory_name"]), row["theory_name"]),
        axis=1,
    )
    return updated


def build_dynamic_theory_lens_map(updates: dict[tuple[str, str], str]) -> dict[str, str]:
    # Legacy helper kept for compatibility with earlier notebooks.
    # Current alluvial theory lenses are built directly from discipline + theory_name.
    _ = updates
    return {}


def load_theory_lens_order_from_verification(verification_file: Path | None = None) -> list[str]:
    verification_file = verification_file or THEORY_NAME_UPDATE_FILE
    if not verification_file.exists():
        return THEORY_LENS_ORDER_FALLBACK
    try:
        df = pd.read_csv(THEORY_LENS_ORDER_FILE)
    except Exception:
        try:
            df = pd.read_csv(verification_file)
        except Exception:
            return THEORY_LENS_ORDER_FALLBACK

    required = {"discipline", "theory_name"}
    if not required.issubset(set(df.columns)):
        return THEORY_LENS_ORDER_FALLBACK

    ordered = []
    for _, row in df.iterrows():
        discipline = str(row.get("discipline", "")).strip()
        theory_name = str(row.get("theory_name", "")).strip()
        if not discipline or not theory_name or theory_name.lower() == "nan":
            continue
        lens = f"{discipline}: {theory_name}"
        if lens not in ordered:
            ordered.append(lens)
    return ordered if ordered else THEORY_LENS_ORDER_FALLBACK


def load_theory_lens_label_map_from_verification(verification_file: Path | None = None) -> dict[tuple[str, str], str]:
    verification_file = verification_file or THEORY_NAME_UPDATE_FILE
    label_map: dict[tuple[str, str], str] = {}
    if not verification_file.exists():
        return label_map

    sheet1 = None
    sheet2 = None
    try:
        sheet1 = pd.read_csv(THEORY_LENS_ORDER_FILE)
    except Exception:
        pass
    try:
        sheet2 = pd.read_csv(verification_file)
    except Exception:
        pass

    if sheet1 is not None and {"discipline", "theory_name"}.issubset(set(sheet1.columns)):
        for _, row in sheet1.iterrows():
            discipline = str(row.get("discipline", "")).strip()
            theory_name = str(row.get("theory_name", "")).strip()
            if discipline and theory_name and theory_name.lower() != "nan":
                label_map[(discipline, theory_name)] = f"{discipline}: {theory_name}"

    # If theory_name_news exists, keep display aligned with Sheet1 theory_name.
    if sheet2 is not None and {"discipline", "theory_name", "theory_name_news"}.issubset(set(sheet2.columns)):
        for _, row in sheet2.iterrows():
            discipline = str(row.get("discipline", "")).strip()
            old_name = str(row.get("theory_name", "")).strip()
            new_name = str(row.get("theory_name_news", "")).strip()
            if not discipline or not old_name or not new_name:
                continue
            if old_name.lower() == "nan" or new_name.lower() == "nan":
                continue
            # Display should follow Sheet1 theory_name wording.
            label_map[(discipline, new_name)] = f"{discipline}: {old_name}"

    return label_map


def load_final_modes() -> pd.DataFrame:
    df = pd.DataFrame(FINAL_MODE_CATALOG)
    df["mode_text"] = df.apply(
        lambda row: f"{row['mode']}. {row['description']}. Keywords: {' '.join(row['keywords'])}",
        axis=1,
    )
    return df


def load_theories() -> pd.DataFrame:
    df = pd.DataFrame(THEORY_CATALOG)
    updates = load_theory_name_updates()
    df = apply_theory_name_updates(df, updates)
    df["theory_text"] = df.apply(
        lambda row: " ".join(
            [
                row["theory_name"],
                row["summary"],
                row["why_effective"],
                row["conditions"],
                row["effects"],
                " ".join(row["keywords"]),
            ]
        ),
        axis=1,
    )
    return df


def load_discipline_clusters() -> pd.DataFrame:
    clusters = pd.read_csv(DATA_DIR / "discipline_clusters.csv")
    clusters["cluster_text"] = clusters["cluster_name"] + ". " + clusters["cluster_description"]
    return clusters


def parse_radar_scores(file_path: Path | None = None) -> pd.DataFrame:
    file_path = file_path or (THIS_DIR / "reference_inputs" / "figure07_cooperation_mode_scores.txt")
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []
    current_mode = None
    current_scores: dict[str, float] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d+\.\s+", stripped):
            if current_mode is not None:
                rows.append({"mode": current_mode, **current_scores})
            current_mode = re.sub(r"^\d+\.\s+", "", stripped)
            current_scores = {}
        elif ":" in stripped and current_mode is not None:
            aspect, value = stripped.split(":", 1)
            try:
                current_scores[aspect.strip()] = float(value.strip())
            except ValueError:
                pass

    if current_mode is not None:
        rows.append({"mode": current_mode, **current_scores})

    return pd.DataFrame(rows)


def build_cluster_mode_trace(clusters: pd.DataFrame, final_modes: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    scores = _hybrid_similarity(
        source_texts=clusters["cluster_text"].tolist(),
        target_texts=final_modes["mode_text"].tolist(),
        target_keywords=final_modes["keywords"].tolist(),
    )
    rows = []
    for i, cluster_row in clusters.reset_index(drop=True).iterrows():
        top_idx = np.argsort(scores[i])[::-1][:top_k]
        top_scores = scores[i][top_idx]
        total = top_scores.sum() if top_scores.sum() > 0 else 1.0
        for idx, raw_score in zip(top_idx, top_scores):
            mode_row = final_modes.iloc[idx]
            weight = raw_score / total
            rows.append(
                {
                    "discipline": cluster_row["discipline"],
                    "cluster": int(cluster_row["cluster"]),
                    "cluster_name": cluster_row["cluster_name"],
                    "event_count": int(cluster_row["event_count"]),
                    "mode": mode_row["mode"],
                    "raw_score": float(raw_score),
                    "normalized_weight": float(weight),
                    "weighted_event_support": float(cluster_row["event_count"] * weight),
                }
            )
    return pd.DataFrame(rows).sort_values(["discipline", "cluster", "normalized_weight"], ascending=[True, True, False])


def build_theory_cluster_trace(theories: pd.DataFrame, clusters: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    rows = []
    for discipline in theories["discipline"].unique():
        theory_subset = theories[theories["discipline"] == discipline].reset_index(drop=True)
        cluster_subset = clusters[clusters["discipline"] == discipline].reset_index(drop=True)
        scores = _hybrid_similarity(
            source_texts=theory_subset["theory_text"].tolist(),
            target_texts=cluster_subset["cluster_text"].tolist(),
            source_keywords=theory_subset["keywords"].tolist(),
        )
        for i, theory_row in theory_subset.iterrows():
            top_idx = np.argsort(scores[i])[::-1][:top_k]
            top_scores = scores[i][top_idx]
            total = top_scores.sum() if top_scores.sum() > 0 else 1.0
            for idx, raw_score in zip(top_idx, top_scores):
                cluster_row = cluster_subset.iloc[idx]
                rows.append(
                    {
                        "discipline": discipline,
                        "theory_name": theory_row["theory_name"],
                        "source_anchor": theory_row["source_anchor"],
                        "cluster": int(cluster_row["cluster"]),
                        "cluster_name": cluster_row["cluster_name"],
                        "cluster_event_count": int(cluster_row["event_count"]),
                        "raw_score": float(raw_score),
                        "normalized_weight": float(raw_score / total),
                    }
                )
    return pd.DataFrame(rows).sort_values(["discipline", "theory_name", "normalized_weight"], ascending=[True, True, False])


def build_theory_mode_trace(
    theories: pd.DataFrame,
    final_modes: pd.DataFrame,
    theory_cluster_trace: pd.DataFrame,
    cluster_mode_trace: pd.DataFrame,
) -> pd.DataFrame:
    merged = theory_cluster_trace.merge(
        cluster_mode_trace,
        on=["discipline", "cluster", "cluster_name"],
        how="inner",
        suffixes=("_theory_cluster", "_cluster_mode"),
    )
    merged["support_score"] = (
        merged["normalized_weight_theory_cluster"]
        * merged["normalized_weight_cluster_mode"]
        * merged["cluster_event_count"]
    )
    indirect = (
        merged.groupby(["discipline", "theory_name", "source_anchor", "mode"], as_index=False)
        .agg(
            support_score=("support_score", "sum"),
            mean_theory_cluster_score=("raw_score_theory_cluster", "mean"),
            mean_cluster_mode_score=("raw_score_cluster_mode", "mean"),
        )
    )
    direct_scores = _hybrid_similarity(
        source_texts=theories["theory_text"].tolist(),
        target_texts=final_modes["mode_text"].tolist(),
        source_keywords=theories["keywords"].tolist(),
        target_keywords=final_modes["keywords"].tolist(),
    )
    direct_rows = []
    for i, theory_row in theories.reset_index(drop=True).iterrows():
        for j, mode_row in final_modes.reset_index(drop=True).iterrows():
            direct_rows.append(
                {
                    "discipline": theory_row["discipline"],
                    "theory_name": theory_row["theory_name"],
                    "mode": mode_row["mode"],
                    "direct_mode_score": float(direct_scores[i, j]),
                }
            )
    direct = pd.DataFrame(direct_rows)
    combined = indirect.merge(direct, on=["discipline", "theory_name", "mode"], how="left")
    combined["combined_support_score"] = combined["support_score"] + combined["direct_mode_score"].fillna(0) * 75
    return combined.sort_values(["mode", "combined_support_score"], ascending=[True, False])


def build_mode_summary(
    theories: pd.DataFrame,
    cluster_mode_trace: pd.DataFrame,
    theory_mode_trace: pd.DataFrame,
    radar_scores: pd.DataFrame,
) -> pd.DataFrame:
    theory_lookup = theories.set_index(["discipline", "theory_name"])
    rows = []
    for mode in radar_scores["mode"].tolist():
        mode_theories = theory_mode_trace[theory_mode_trace["mode"] == mode].copy()
        mode_support = cluster_mode_trace[cluster_mode_trace["mode"] == mode].copy()
        top_theories = mode_theories.sort_values("combined_support_score", ascending=False).head(3)
        discipline_support = (
            mode_theories.groupby("discipline", as_index=False)["combined_support_score"]
            .sum()
            .sort_values("combined_support_score", ascending=False)
        )
        why_chunks, when_chunks, effect_chunks, theory_names = [], [], [], []
        for _, row in top_theories.iterrows():
            theory = theory_lookup.loc[(row["discipline"], row["theory_name"])]
            theory_names.append(f"{row['discipline']} - {row['theory_name']}")
            why_chunks.append(f"{row['theory_name']}: {theory['why_effective']}")
            when_chunks.append(f"{row['theory_name']}: {theory['conditions']}")
            effect_chunks.append(f"{row['theory_name']}: {theory['effects']}")
        radar_row = radar_scores[radar_scores["mode"] == mode].iloc[0].to_dict()
        rows.append(
            {
                "mode": mode,
                "top_disciplines": " | ".join(f"{x['discipline']} ({x['combined_support_score']:.1f})" for _, x in discipline_support.head(3).iterrows()),
                "top_theories": " | ".join(theory_names),
                "why_effective": " || ".join(why_chunks),
                "conditions": " || ".join(when_chunks),
                "effects": " || ".join(effect_chunks),
                "weighted_event_support": round(mode_support["weighted_event_support"].sum(), 2),
                **radar_row,
            }
        )
    return pd.DataFrame(rows)


def plot_cluster_mode_heatmap(cluster_mode_trace: pd.DataFrame, output_path: Path) -> None:
    plot_df = cluster_mode_trace.copy()
    plot_df["cluster_label"] = plot_df.apply(
        lambda row: f"{row['discipline']} | C{row['cluster']} | {row['cluster_name']}",
        axis=1,
    )
    pivot = plot_df.pivot_table(
        index="cluster_label",
        columns="mode",
        values="normalized_weight",
        aggfunc="sum",
        fill_value=0,
    )
    plt.figure(figsize=(12, 11))
    ax = sns.heatmap(
        pivot,
        cmap=HEATMAP_CMAP,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Normalized trace weight"},
    )
    plt.title("Cluster-to-mode trace across disciplinary trunks", pad=10, weight="bold")
    plt.xlabel("Final cooperation modes")
    plt.ylabel("Original discipline-specific clusters")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=10.5)
    plt.tight_layout()
    _save_figure(output_path)


def plot_theory_mode_heatmap(theory_mode_trace: pd.DataFrame, output_path: Path) -> None:
    plot_df = theory_mode_trace.copy()
    plot_df["theory_label"] = plot_df.apply(
        lambda row: f"{row['discipline']} | {row['theory_name']}",
        axis=1,
    )
    pivot = plot_df.pivot_table(
        index="theory_label",
        columns="mode",
        values="combined_support_score",
        aggfunc="sum",
        fill_value=0,
    )
    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(
        pivot,
        cmap=HEATMAP_CMAP,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Combined theory support"},
    )
    plt.title("Theory-to-mode support across disciplinary trunks", pad=10, weight="bold")
    plt.xlabel("Final cooperation modes")
    plt.ylabel("Disciplinary theories")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=10.5)
    plt.tight_layout()
    _save_figure(output_path)


def plot_mode_discipline_bar(theory_mode_trace: pd.DataFrame, output_path: Path) -> None:
    plot_df = (
        theory_mode_trace.groupby(["mode", "discipline"], as_index=False)["combined_support_score"]
        .sum()
        .sort_values(["mode", "combined_support_score"], ascending=[True, False])
    )
    plt.figure(figsize=(12.5, 6.2))
    ax = sns.barplot(
        data=plot_df,
        x="mode",
        y="combined_support_score",
        hue="discipline",
        palette=DISCIPLINE_COLORS,
    )
    plt.title("Disciplinary support composition of the six final modes", pad=10, weight="bold")
    plt.xlabel("")
    plt.ylabel("Aggregated support score")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for patch in ax.patches:
        patch.set_edgecolor("#FFFFFF")
        patch.set_linewidth(0.8)
        patch.set_alpha(0.95)
    ax.grid(axis="y", color="#D9E1E6", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    legend = ax.legend(title="Discipline", frameon=False, loc="upper right")
    if legend is not None:
        legend.get_title().set_fontsize(13)
        for text in legend.get_texts():
            text.set_fontsize(12)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    _save_figure(output_path)


def plot_mode_discipline_bubble(theory_mode_trace: pd.DataFrame, output_path: Path) -> None:
    disciplines = ["Hydrology", "International Relations", "Sustainability Science"]
    modes = [item["mode"] for item in FINAL_MODE_CATALOG]
    plot_df = (
        theory_mode_trace.groupby(["mode", "discipline"], as_index=False)["combined_support_score"]
        .sum()
    )
    full_index = pd.MultiIndex.from_product([modes, disciplines], names=["mode", "discipline"])
    plot_df = (
        plot_df.set_index(["mode", "discipline"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    max_score = plot_df["combined_support_score"].max() if not plot_df.empty else 1
    plot_df["size"] = plot_df["combined_support_score"].apply(
        lambda x: 60 + (x / max_score) * 1350 if x > 0 else 25
    )
    mode_pos = {mode: i for i, mode in enumerate(modes)}
    discipline_pos = {discipline: i for i, discipline in enumerate(disciplines)}

    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.set_facecolor("white")

    for _, row in plot_df.iterrows():
        color = DISCIPLINE_COLORS[row["discipline"]] if row["combined_support_score"] > 0 else "#D7DEE3"
        alpha = 0.9 if row["combined_support_score"] > 0 else 0.5
        ax.scatter(
            mode_pos[row["mode"]],
            discipline_pos[row["discipline"]],
            s=row["size"],
            color=color,
            alpha=alpha,
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
        if row["combined_support_score"] >= max_score * 0.22:
            ax.text(
                mode_pos[row["mode"]],
                discipline_pos[row["discipline"]],
                f"{row['combined_support_score']:.0f}",
                ha="center",
                va="center",
                fontsize=10.5,
                color="white",
                weight="bold",
                zorder=4,
            )

    for x in range(len(modes)):
        ax.axvline(x, color="#EEF2F5", linewidth=0.8, zorder=1)
    for y in range(len(disciplines)):
        ax.axhline(y, color="#EEF2F5", linewidth=0.8, zorder=1)

    ax.set_xlim(-0.5, len(modes) - 0.5)
    ax.set_ylim(-0.5, len(disciplines) - 0.5)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(disciplines)))
    ax.set_yticklabels(disciplines)
    ax.set_title("Mode-by-discipline support matrix", pad=10, weight="bold")
    ax.set_xlabel("Final cooperation modes")
    ax.set_ylabel("Discipline")
    sns.despine(ax=ax, left=False, bottom=False)

    legend_scores = [0.25, 0.55, 0.85]
    legend_handles = []
    legend_labels = []
    for frac in legend_scores:
        score = max_score * frac
        size = 60 + frac * 1350
        handle = ax.scatter([], [], s=size, color="#8FA3B5", alpha=0.65, edgecolors="white", linewidths=1.0)
        legend_handles.append(handle)
        legend_labels.append(f"{score:.0f}")
    ax.legend(
        legend_handles,
        legend_labels,
        title="Support score",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    plt.tight_layout()
    _save_figure(output_path)


def plot_mode_discipline_composition(theory_mode_trace: pd.DataFrame, output_path: Path) -> None:
    disciplines = ["Hydrology", "International Relations", "Sustainability Science"]
    modes = [item["mode"] for item in FINAL_MODE_CATALOG]

    plot_df = (
        theory_mode_trace.groupby(["mode", "discipline"], as_index=False)["combined_support_score"]
        .sum()
    )
    full_index = pd.MultiIndex.from_product([modes, disciplines], names=["mode", "discipline"])
    plot_df = (
        plot_df.set_index(["mode", "discipline"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    totals = (
        plot_df.groupby("mode", as_index=False)["combined_support_score"]
        .sum()
        .rename(columns={"combined_support_score": "total_support"})
    )
    plot_df = plot_df.merge(totals, on="mode", how="left")
    plot_df["share"] = np.where(
        plot_df["total_support"] > 0,
        plot_df["combined_support_score"] / plot_df["total_support"],
        0,
    )

    share_pivot = plot_df.pivot(index="mode", columns="discipline", values="share").reindex(modes)
    total_pivot = totals.set_index("mode").reindex(modes)

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(12.8, 6.1),
        gridspec_kw={"width_ratios": [2.6, 1.0]},
        sharey=True,
    )

    y = np.arange(len(modes))
    left = np.zeros(len(modes))
    for discipline in disciplines:
        values = share_pivot[discipline].fillna(0).values
        ax_left.barh(
            y,
            values,
            left=left,
            color=DISCIPLINE_COLORS[discipline],
            edgecolor="white",
            linewidth=1.0,
            height=0.72,
            label=discipline,
        )
        left += values

    ax_left.set_xlim(0, 1)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(modes)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Share of support within each mode")
    ax_left.set_title("Disciplinary composition", pad=10, weight="bold")
    ax_left.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(round(x * 100))}%"))
    ax_left.grid(axis="x", color="#D9E1E6", linewidth=0.8)
    ax_left.grid(axis="y", visible=False)
    sns.despine(ax=ax_left, top=True, right=True, left=False)

    totals_values = total_pivot["total_support"].fillna(0).values
    ax_right.hlines(y, 0, totals_values, color="#C9D3DB", linewidth=2.0, zorder=1)
    ax_right.scatter(
        totals_values,
        y,
        s=78,
        color=PALETTE["Blue"]["line"],
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )
    ax_right.set_xlabel("Total support score")
    ax_right.set_title("Overall magnitude", pad=10, weight="bold")
    ax_right.grid(axis="x", color="#D9E1E6", linewidth=0.8)
    ax_right.grid(axis="y", visible=False)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    sns.despine(ax=ax_right, top=True, right=True, left=True)

    fig.suptitle("Disciplinary structure and magnitude of support for the six final modes", y=1.02, fontsize=17.5, fontweight="bold")
    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Discipline",
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=12.5,
        title_fontsize=12.5,
        columnspacing=1.8,
        handlelength=2.2,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save_figure(output_path)


def build_radar_excel_summary(mode_summary: pd.DataFrame, theory_mode_trace: pd.DataFrame) -> pd.DataFrame:
    radar_rows = []
    for _, row in mode_summary.iterrows():
        mode_theories = (
            theory_mode_trace[theory_mode_trace["mode"] == row["mode"]]
            .sort_values("combined_support_score", ascending=False)
            .head(3)
            .reset_index(drop=True)
        )
        output_row = row.to_dict()
        for rank, (_, theory_row) in enumerate(mode_theories.iterrows(), start=1):
            output_row[f"theory_{rank}_discipline"] = theory_row["discipline"]
            output_row[f"theory_{rank}_name"] = theory_row["theory_name"]
            output_row[f"theory_{rank}_source"] = theory_row["source_anchor"]
            output_row[f"theory_{rank}_support_score"] = theory_row["combined_support_score"]
        radar_rows.append(output_row)
    return pd.DataFrame(radar_rows)


def build_alluvial_flows(theory_cluster_trace: pd.DataFrame, cluster_mode_trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = theory_cluster_trace.merge(
        cluster_mode_trace,
        on=["discipline", "cluster", "cluster_name"],
        how="inner",
        suffixes=("_theory_cluster", "_cluster_mode"),
    )
    merged["path_support"] = (
        merged["normalized_weight_theory_cluster"]
        * merged["normalized_weight_cluster_mode"]
        * merged["cluster_event_count"]
    )
    label_map = load_theory_lens_label_map_from_verification()
    merged["theory_lens"] = merged.apply(
        lambda row: label_map.get(
            (row["discipline"], row["theory_name"]),
            f"{row['discipline']}: {row['theory_name']}",
        ),
        axis=1,
    )
    merged["cluster_family"] = merged["cluster_name"].map(CLUSTER_FAMILY_MAP)
    merged = merged.dropna(subset=["theory_lens", "cluster_family"])

    left_mid = (
        merged.groupby(["discipline", "theory_lens", "cluster_family"], as_index=False)["path_support"]
        .sum()
        .sort_values("path_support", ascending=False)
    )
    mid_right = (
        merged.groupby(["discipline", "cluster_family", "mode"], as_index=False)["path_support"]
        .sum()
        .sort_values("path_support", ascending=False)
    )
    return left_mid, mid_right


def _discipline_from_lens(lens: str) -> str:
    if lens.startswith("Hydrology:"):
        return "Hydrology"
    if lens.startswith("IR:") or lens.startswith("International Relations:"):
        return "International Relations"
    return "Sustainability Science"


def _theory_name_from_lens(lens: str) -> str:
    if ":" in lens:
        return lens.split(":", 1)[1].strip()
    return lens


def _add_alluvial_band(ax, x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float, color: str, alpha: float = 0.6) -> None:
    c = 0.32 * (x1 - x0)
    verts = [
        (x0, y0a),
        (x0 + c, y0a),
        (x1 - c, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - c, y1b),
        (x0 + c, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1)
    ax.add_patch(patch)


def plot_theory_cluster_mode_alluvial(
    theory_cluster_trace: pd.DataFrame,
    cluster_mode_trace: pd.DataFrame,
    output_path: Path,
) -> None:
    left_mid, mid_right = build_alluvial_flows(theory_cluster_trace, cluster_mode_trace)
    mode_order = [item["mode"] for item in FINAL_MODE_CATALOG]

    left_totals = left_mid.groupby("theory_lens", as_index=False)["path_support"].sum()
    mid_in = left_mid.groupby("cluster_family", as_index=False)["path_support"].sum()
    mid_out = mid_right.groupby("cluster_family", as_index=False)["path_support"].sum()
    mid_totals = mid_in.merge(mid_out, on="cluster_family", how="outer", suffixes=("_in", "_out")).fillna(0)
    mid_totals["path_support"] = mid_totals[["path_support_in", "path_support_out"]].max(axis=1)
    right_totals = mid_right.groupby("mode", as_index=False)["path_support"].sum()

    left_available = set(left_totals["theory_lens"])
    left_nodes_order = load_theory_lens_order_from_verification()
    left_nodes = [x for x in left_nodes_order if x in left_available]
    for x in sorted(left_available):
        if x not in left_nodes:
            left_nodes.append(x)
    mid_nodes = [x for x in CLUSTER_FAMILY_ORDER if x in set(mid_totals["cluster_family"])]
    right_nodes = [x for x in mode_order if x in set(right_totals["mode"])]
    # Presentation tweak: place the two top policy modes in swapped order for better visual balance.
    mode_a = "Cross-Border Basin Agreements"
    mode_b = "Collaborative Planning and Adaptation Strategies"
    if mode_a in right_nodes and mode_b in right_nodes:
        idx_a, idx_b = right_nodes.index(mode_a), right_nodes.index(mode_b)
        right_nodes[idx_a], right_nodes[idx_b] = right_nodes[idx_b], right_nodes[idx_a]

    left_map = dict(zip(left_totals["theory_lens"], left_totals["path_support"]))
    mid_map = dict(zip(mid_totals["cluster_family"], mid_totals["path_support"]))
    right_map = dict(zip(right_totals["mode"], right_totals["path_support"]))

    columns = [
        {"nodes": left_nodes, "totals": left_map, "x": 0.08},
        {"nodes": mid_nodes, "totals": mid_map, "x": 0.52},
        {"nodes": right_nodes, "totals": right_map, "x": 0.94},
    ]

    top_margin, bottom_margin = 0.06, 0.06
    usable = 1 - top_margin - bottom_margin
    min_gap = 0.010
    scale_candidates = []
    for col in columns:
        n = len(col["nodes"])
        total = sum(col["totals"].get(node, 0) for node in col["nodes"])
        if n <= 1:
            scale_candidates.append(usable / total if total > 0 else 1.0)
        else:
            scale_candidates.append((usable - min_gap * (n - 1)) / total if total > 0 else 1.0)
    scale = min(scale_candidates)

    # Column-specific gaps to force equal top and bottom alignment.
    column_gaps = {}
    for idx, col in enumerate(columns):
        n = len(col["nodes"])
        total_h = sum(col["totals"].get(node, 0) for node in col["nodes"]) * scale
        if n <= 1:
            column_gaps[idx] = 0.0
        else:
            column_gaps[idx] = max((usable - total_h) / (n - 1), 0.0)

    node_boxes = {}
    for idx, col in enumerate(columns):
        y_top = 1 - top_margin
        gap = column_gaps[idx]
        for node in col["nodes"]:
            height = col["totals"].get(node, 0) * scale
            y_bottom = y_top - height
            node_boxes[node] = {"x": col["x"], "y0": y_bottom, "y1": y_top, "height": height}
            y_top = y_bottom - gap

    fig, ax = plt.subplots(figsize=(16.8, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left_mid_cursor_left = {node: node_boxes[node]["y1"] for node in left_nodes}
    left_mid_cursor_mid = {node: node_boxes[node]["y1"] for node in mid_nodes}
    left_mid_plot = left_mid.copy()
    left_mid_plot["theory_lens"] = pd.Categorical(left_mid_plot["theory_lens"], categories=left_nodes, ordered=True)
    left_mid_plot["cluster_family"] = pd.Categorical(left_mid_plot["cluster_family"], categories=mid_nodes, ordered=True)
    left_mid_plot = left_mid_plot.sort_values(["theory_lens", "cluster_family", "path_support"], ascending=[True, True, False])

    for _, row in left_mid_plot.iterrows():
        h = row["path_support"] * scale
        y0_top = left_mid_cursor_left[row["theory_lens"]]
        y0_bottom = y0_top - h
        y1_top = left_mid_cursor_mid[row["cluster_family"]]
        y1_bottom = y1_top - h
        _add_alluvial_band(
            ax,
            node_boxes[row["theory_lens"]]["x"] + 0.012,
            node_boxes[row["cluster_family"]]["x"] - 0.012,
            y0_bottom,
            y0_top,
            y1_bottom,
            y1_top,
            DISCIPLINE_COLORS[row["discipline"]],
            alpha=0.55,
        )
        left_mid_cursor_left[row["theory_lens"]] = y0_bottom
        left_mid_cursor_mid[row["cluster_family"]] = y1_bottom

    mid_right_cursor_mid = {node: node_boxes[node]["y1"] for node in mid_nodes}
    mid_right_cursor_right = {node: node_boxes[node]["y1"] for node in right_nodes}
    mid_right_plot = mid_right.copy()
    mid_right_plot["cluster_family"] = pd.Categorical(mid_right_plot["cluster_family"], categories=mid_nodes, ordered=True)
    mid_right_plot["mode"] = pd.Categorical(mid_right_plot["mode"], categories=right_nodes, ordered=True)
    mid_right_plot = mid_right_plot.sort_values(["cluster_family", "mode", "path_support"], ascending=[True, True, False])

    for _, row in mid_right_plot.iterrows():
        h = row["path_support"] * scale
        y0_top = mid_right_cursor_mid[row["cluster_family"]]
        y0_bottom = y0_top - h
        y1_top = mid_right_cursor_right[row["mode"]]
        y1_bottom = y1_top - h
        _add_alluvial_band(
            ax,
            node_boxes[row["cluster_family"]]["x"] + 0.012,
            node_boxes[row["mode"]]["x"] - 0.012,
            y0_bottom,
            y0_top,
            y1_bottom,
            y1_top,
            DISCIPLINE_COLORS[row["discipline"]],
            alpha=0.48,
        )
        mid_right_cursor_mid[row["cluster_family"]] = y0_bottom
        mid_right_cursor_right[row["mode"]] = y1_bottom

    def draw_nodes(nodes: list[str], facecolor_fn) -> None:
        for node in nodes:
            box = node_boxes[node]
            rect = Rectangle(
                (box["x"] - 0.012, box["y0"]),
                0.024,
                box["height"],
                facecolor=facecolor_fn(node),
                edgecolor="#FFFFFF",
                linewidth=1.2,
                zorder=3,
            )
            ax.add_patch(rect)

    draw_nodes(left_nodes, lambda node: DISCIPLINE_COLORS[_discipline_from_lens(node)])
    draw_nodes(mid_nodes, lambda node: "#D6E0E8")
    draw_nodes(right_nodes, lambda node: MODE_LIGHT_COLORS.get(node, "#C3D2DE"))

    for node in left_nodes:
        box = node_boxes[node]
        ax.text(
            box["x"] - 0.02,
            (box["y0"] + box["y1"]) / 2,
            _theory_name_from_lens(node),
            ha="right",
            va="center",
            fontsize=14.3,
            fontweight="semibold",
            color="#2F3B48",
        )
    for node in mid_nodes:
        box = node_boxes[node]
        ax.text(
            box["x"],
            (box["y0"] + box["y1"]) / 2,
            node,
            ha="center",
            va="center",
            fontsize=14.3,
            fontweight="semibold",
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#FFFFFF",
                "edgecolor": "#C8D2DC",
                "linewidth": 0.7,
                "alpha": 0.70,
            },
        )
    for node in right_nodes:
        box = node_boxes[node]
        ax.text(
            box["x"] + 0.02,
            (box["y0"] + box["y1"]) / 2,
            node,
            ha="left",
            va="center",
            fontsize=14.3,
            fontweight="semibold",
        )

    ax.text(columns[0]["x"], 0.985, "Theory lenses", ha="center", va="bottom", fontsize=17.4, fontweight="bold")
    ax.text(columns[1]["x"], 0.985, "Cluster families", ha="center", va="bottom", fontsize=17.4, fontweight="bold")
    ax.text(columns[2]["x"], 0.985, "Final cooperation modes", ha="center", va="bottom", fontsize=17.4, fontweight="bold")

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=DISCIPLINE_COLORS["Hydrology"], edgecolor="none"),
        Rectangle((0, 0), 1, 1, facecolor=DISCIPLINE_COLORS["International Relations"], edgecolor="none"),
        Rectangle((0, 0), 1, 1, facecolor=DISCIPLINE_COLORS["Sustainability Science"], edgecolor="none"),
    ]
    legend = ax.legend(
        legend_handles,
        ["Hydrology", "International Relations", "Sustainability Science"],
        title="Discipline",
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        fontsize=13.4,
        title_fontsize=14.2,
    )
    if legend is not None:
        legend.get_title().set_fontweight("bold")
        for text in legend.get_texts():
            text.set_fontweight("semibold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.995])
    _save_figure(output_path)


def save_outputs(
    clusters: pd.DataFrame,
    theories: pd.DataFrame,
    final_modes: pd.DataFrame,
    cluster_mode_trace: pd.DataFrame,
    theory_cluster_trace: pd.DataFrame,
    theory_mode_trace: pd.DataFrame,
    mode_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    clusters.to_csv(table_dir / "discipline_clusters.csv", index=False, encoding="utf-8-sig")
    theories.to_csv(table_dir / "theory_catalog.csv", index=False, encoding="utf-8-sig")
    final_modes.to_csv(table_dir / "final_mode_catalog.csv", index=False, encoding="utf-8-sig")
    cluster_mode_trace.to_csv(table_dir / "cluster_to_mode_trace.csv", index=False, encoding="utf-8-sig")
    theory_cluster_trace.to_csv(table_dir / "theory_to_cluster_trace.csv", index=False, encoding="utf-8-sig")
    theory_mode_trace.to_csv(table_dir / "theory_to_mode_trace.csv", index=False, encoding="utf-8-sig")
    mode_summary.to_csv(table_dir / "final_mode_theory_summary.csv", index=False, encoding="utf-8-sig")
    radar_excel_summary = build_radar_excel_summary(mode_summary, theory_mode_trace)
    radar_excel_summary.to_csv(table_dir / "radar_card_summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(table_dir / "theory_trace_outputs.xlsx") as writer:
        clusters.to_excel(writer, sheet_name="discipline_clusters", index=False)
        theories.to_excel(writer, sheet_name="theory_catalog", index=False)
        final_modes.to_excel(writer, sheet_name="final_mode_catalog", index=False)
        cluster_mode_trace.to_excel(writer, sheet_name="cluster_to_mode", index=False)
        theory_cluster_trace.to_excel(writer, sheet_name="theory_to_cluster", index=False)
        theory_mode_trace.to_excel(writer, sheet_name="theory_to_mode", index=False)
        mode_summary.to_excel(writer, sheet_name="mode_summary", index=False)
        radar_excel_summary.to_excel(writer, sheet_name="radar_card_summary", index=False)

    plot_cluster_mode_heatmap(cluster_mode_trace, figure_dir / "cluster_mode_heatmap.png")
    plot_theory_mode_heatmap(theory_mode_trace, figure_dir / "theory_mode_heatmap.png")
    plot_mode_discipline_composition(theory_mode_trace, figure_dir / "mode_discipline_support_composition.png")
    plot_theory_cluster_mode_alluvial(
        theory_cluster_trace,
        cluster_mode_trace,
        figure_dir / "theory_cluster_mode_alluvial.png",
    )


def run_pipeline(output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    output_dir = output_dir or (THIS_DIR / "outputs" / "figure07")
    clusters = load_discipline_clusters()
    theories = load_theories()
    final_modes = load_final_modes()
    radar_scores = parse_radar_scores()

    cluster_mode_trace = build_cluster_mode_trace(clusters, final_modes)
    theory_cluster_trace = build_theory_cluster_trace(theories, clusters)
    theory_mode_trace = build_theory_mode_trace(theories, final_modes, theory_cluster_trace, cluster_mode_trace)
    mode_summary = build_mode_summary(theories, cluster_mode_trace, theory_mode_trace, radar_scores)

    save_outputs(
        clusters=clusters,
        theories=theories,
        final_modes=final_modes,
        cluster_mode_trace=cluster_mode_trace,
        theory_cluster_trace=theory_cluster_trace,
        theory_mode_trace=theory_mode_trace,
        mode_summary=mode_summary,
        output_dir=output_dir,
    )

    return {
        "clusters": clusters,
        "theories": theories,
        "final_modes": final_modes,
        "cluster_mode_trace": cluster_mode_trace,
        "theory_cluster_trace": theory_cluster_trace,
        "theory_mode_trace": theory_mode_trace,
        "mode_summary": mode_summary,
        "output_dir": pd.DataFrame([{"output_dir": str(output_dir)}]),
    }


if __name__ == "__main__":
    results = run_pipeline()
    output_dir = Path(results["output_dir"].iloc[0, 0])
    print("Saved outputs to:")
    print(output_dir)
