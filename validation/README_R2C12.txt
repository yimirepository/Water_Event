R2C12: k=9 clustering sensitivity and thematic review

What this package does
- Reuses the archived 3,072-dimensional event vectors and event descriptions.
- Recreates the k=9, n_init=10 K-means reference run (seed 42).
- Repeats the same clustering for seeds 1 through 10.
- Matches clusters by shared event records and exports counts and representative inputs.
- Retrieves the existing literature texts associated with five sampled events in each cluster, using the archived embedding columns used by the old retrieval code.
- Makes no Azure/OpenAI calls and does not regenerate the literature corpus or embedding files.

Run
  python replicate_k9_seed_test.py --repo /path/to/Water_Cooperation_and_Conflict-main --out /path/to/output

Outputs
- k9_event_cluster_assignments.csv: event-level assignments for seed 42 and seeds 1-10.
- k9_cluster_event_overlaps.csv: matched cluster sizes and shared-event counts; no Rand statistic is needed for the reported interpretation.
- seed42_seed3_theme_review_inputs.json: five event examples and five retrieved literature excerpts for each cluster in seed 42 and illustrative seed 3, aligned by shared events. This is the input for a Codex thematic comparison; the script does not claim a separate model call.

Recorded result for seed 42 versus seed 3
- The Aswan technical-assistance/hydropower group shares 113 of 117 seed-42 events with its matched seed-3 group.
- A broader basin-cooperation group shares 155 of 264 seed-42 events with its matched group.
- Reviewing matched event examples shows recurring themes involving hydropower cooperation, water-sharing agreements, and basin planning, even where broader groups shift.

Interpretation of the final modes
The nine K-means clusters were intermediate event groupings, not the final taxonomy. The author reports that event-level assignment and label refinement reduced the candidates to seven categories, which the expert workshops refined to six. The workshop materials document specific changes: data sharing was kept distinct under Joint Data-Sharing Systems; mode 6 was renamed Collaborative Planning and Adaptation Strategies to distinguish it from formal basin agreements and incorporate climate adaptation; and conflict resolution was not retained as a separate mode, with the Indus Waters Treaty added under Cross-Border Basin Agreements. Joint Climate Adaptation Strategies was considered as a standalone proposal but was not retained separately; climate-adaptation content appears in the revised planning mode.

The old Azure ChatGPT-4o generation cannot be replayed exactly from the available archive. This package tests initial-cluster sensitivity using saved representations and exports the inputs for a current-model thematic review. It is not a rerun of the complete RAG workflow or the expert workshops.
