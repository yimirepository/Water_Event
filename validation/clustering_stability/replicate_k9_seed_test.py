#!/usr/bin/env python3
"""Reproduce the k=9 event-clustering seed comparison and export theme-review inputs.

Required files under --repo:
  news_embeddings.pkl
  results/event_data_with_clusters.xlsx
  embeddingsLITERATURE.csv

This script reuses saved embeddings; it does not call Azure/OpenAI or regenerate PDFs.
The pickle is checked for executable opcodes before loading. Run against the archived
repository copy used for this analysis.
"""
from __future__ import annotations

import argparse
import json
import pickle
import pickletools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

K = 9
N_INIT = 10
REFERENCE_SEED = 42
ALTERNATIVE_SEEDS = list(range(1, 11))
THEME_SEED = 3  # illustrative alternative seed used for the semantic comparison
FORBIDDEN_PICKLE_OPS = {
    "GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "INST", "OBJ", "NEWOBJ",
    "NEWOBJ_EX", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID",
}


def load_data_only_pickle(path: Path):
    raw = path.read_bytes()
    opcodes = {op.name for op, _arg, _pos in pickletools.genops(raw)}
    bad = opcodes & FORBIDDEN_PICKLE_OPS
    if bad:
        raise ValueError(f"Refusing pickle with executable opcodes: {sorted(bad)}")
    return pickle.loads(raw)


def labels_for(seed: int, X_scaled: np.ndarray) -> np.ndarray:
    return KMeans(n_clusters=K, random_state=seed, n_init=N_INIT).fit_predict(X_scaled)


def aligned_pairs(reference: np.ndarray, alternative: np.ndarray):
    overlap = np.zeros((K, K), dtype=int)
    for a, b in zip(reference, alternative):
        overlap[int(a), int(b)] += 1
    ref_ids, alt_ids = linear_sum_assignment(-overlap)
    return [(int(r), int(a), int(overlap[r, a])) for r, a in zip(ref_ids, alt_ids)]


def build_theme_inputs(df, vectors, labels, lit_df, run_name: str):
    """Mirror the archived retrieval logic: five events, nearest literature, top 5 unique texts."""
    lit_cols = [f"embedding_{i}" for i in range(1536)]
    missing = [c for c in lit_cols if c not in lit_df.columns]
    if missing:
        raise ValueError("Literature embedding file lacks the expected first 1,536 dimensions")
    lit_vecs = lit_df[lit_cols].to_numpy(dtype=float)
    output = []
    for cluster in range(K):
        members = df.loc[labels == cluster]
        sampled = members.sample(n=min(5, len(members)), random_state=42)
        event_vecs = np.asarray([vectors[str(x)] for x in sampled["augment by llm"]], dtype=float)[:, :1536]
        sim = cosine_similarity(event_vecs, lit_vecs)
        picked = []
        for row in sim:
            for idx in np.argsort(-row)[:5]:
                idx = int(idx)
                if idx not in picked:
                    picked.append(idx)
                if len(picked) == 5:
                    break
            if len(picked) == 5:
                break
        events = []
        for _, r in sampled.iterrows():
            events.append({
                "event_code": str(r.get("event_code", "")),
                "event_text": str(r.get("event_text", "")),
                "embedding_text": str(r["augment by llm"]),
            })
        literature = []
        for idx in picked:
            literature.append({
                "source_row": idx,
                "text_excerpt": " ".join(str(lit_df.iloc[idx]["text"]).split())[:1600],
            })
        output.append({
            "run": run_name,
            "cluster": cluster,
            "cluster_size": int(len(members)),
            "sample_events": events,
            "retrieved_literature_excerpts": literature,
        })
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path, help="Path to extracted repository root")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    args = ap.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    events_path = repo / "results" / "event_data_with_clusters.xlsx"
    embeddings_path = repo / "news_embeddings.pkl"
    literature_path = repo / "embeddingsLITERATURE.csv"
    for p in (events_path, embeddings_path, literature_path):
        if not p.exists():
            raise FileNotFoundError(p)

    df = pd.read_excel(events_path)
    if "augment by llm" not in df:
        raise ValueError("Expected column 'augment by llm' in the event workbook")
    vectors = load_data_only_pickle(embeddings_path)
    missing_texts = [str(t) for t in df["augment by llm"] if str(t) not in vectors]
    if missing_texts:
        raise ValueError(f"{len(missing_texts)} event rows have no saved embedding")

    X = np.asarray([vectors[str(t)] for t in df["augment by llm"]], dtype=float)
    if X.ndim != 2 or X.shape[1] != 3072:
        raise ValueError(f"Expected 3,072-dimensional event vectors; received {X.shape}")
    X_scaled = StandardScaler().fit_transform(X)
    reference = labels_for(REFERENCE_SEED, X_scaled)
    all_labels = {REFERENCE_SEED: reference}
    for seed in ALTERNATIVE_SEEDS:
        all_labels[seed] = labels_for(seed, X_scaled)

    # Row-level assignments allow exact reruns and later inspection.
    assignment_df = pd.DataFrame({
        "event_code": df.get("event_code", pd.Series(range(len(df)))).astype(str),
        "event_text": df.get("event_text", pd.Series([""] * len(df))).astype(str),
        "seed42_cluster": reference,
    })
    for seed in ALTERNATIVE_SEEDS:
        assignment_df[f"seed{seed}_cluster"] = all_labels[seed]
    assignment_df.to_csv(out / "k9_event_cluster_assignments.csv", index=False)

    # Match clusters by maximum shared event count; report event overlap, not label-number agreement.
    summaries = []
    for seed in ALTERNATIVE_SEEDS:
        alt = all_labels[seed]
        for ref_c, alt_c, shared in aligned_pairs(reference, alt):
            ref_n = int(np.sum(reference == ref_c))
            alt_n = int(np.sum(alt == alt_c))
            summaries.append({
                "reference_seed": REFERENCE_SEED,
                "alternative_seed": seed,
                "reference_cluster": ref_c,
                "alternative_cluster": alt_c,
                "reference_events": ref_n,
                "alternative_events": alt_n,
                "shared_events": shared,
                "share_of_reference_cluster": shared / ref_n if ref_n else 0,
                "share_of_alternative_cluster": shared / alt_n if alt_n else 0,
            })
    pd.DataFrame(summaries).to_csv(out / "k9_cluster_event_overlaps.csv", index=False)

    lit_df = pd.read_csv(literature_path)
    theme_inputs = []
    theme_inputs += build_theme_inputs(df, vectors, reference, lit_df, f"seed{REFERENCE_SEED}")
    theme_inputs += build_theme_inputs(df, vectors, all_labels[THEME_SEED], lit_df, f"seed{THEME_SEED}")
    # Pair the alternative cluster with its maximum-overlap seed-42 cluster.
    mapping = {a: r for r, a, _n in aligned_pairs(reference, all_labels[THEME_SEED])}
    paired = []
    for alt in theme_inputs[K:]:
        alt_c = int(alt["cluster"])
        ref_c = int(mapping[alt_c])
        paired.append({
            "reference_cluster": next(x for x in theme_inputs[:K] if x["cluster"] == ref_c),
            "alternative_cluster": alt,
        })
    (out / "seed42_seed3_theme_review_inputs.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Events: {len(df)}; event-vector shape: {X.shape}; k={K}; n_init={N_INIT}")
    print("Wrote cluster assignments, matched event overlaps, and event/literature inputs for thematic review.")


if __name__ == "__main__":
    main()
