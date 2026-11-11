R2-09 / R2-13 analysis package

Analysis performed
1. Historical chronological assessment of the binary conflict classifier in five held-out periods.
2. 500 country-cluster bootstrap refits of the conflict classifier, carrying probability variation through the existing suitability threshold. The mode-score table, historical cutoff and current high-risk group membership are held fixed.

Classifier validation (clean_historical_panel.csv)
- 3,180 held-out country-years, 205 conflict/both events, 183 countries.
- Pooled AUC 0.802; Brier score 0.055; average precision 0.224.
- Highest-scored 10%: 82/318 conflicts (25.8%), versus 6.4% across all held-out observations.

Classifier-probability bootstrap (500 draws)
- All score-eligible countries: 117/152 = 77.0%; 95% percentile interval 60.5%–82.2%.
- Existing high-risk group: 28/33 = 84.8%; 95% percentile interval 54.5%–84.8%.
- The high-risk interval is conditional on the current 33-country group; group membership is held fixed.
- For each draw, the change in classifier probability relative to the full-panel refit is applied to the probability estimates used in the current SSP5 file. The published point probabilities are therefore retained.
- The full-panel refit agrees with the supplied probabilities at r=0.804, MAE=0.0537.

Code inspection note
The supplied predict_conflict.ipynb does not contain exponential smoothing or Random Forest regression. It computes each country's cumulative average annual conflict count, carries the 2019 value to the 2050 rows, then estimates conflict probability with a Random Forest classifier. The manuscript should describe this exact workflow unless a separate production script is supplied.

Benchmark count note
The reconstructed historical benchmark contains 4,524 non-missing scores, while Supplement S5 currently reports 6,511 historical fit scores. Reconcile this count before treating the interval as final.

Files
- replicate_classifier_pipeline.py: validation and bootstrap code.
- historical_cv_folds.csv: fold metrics.
- historical_cv_summary.csv: pooled held-out metrics.
- historical_cv_predictions.csv: held-out scores and outcomes.
- classifier_bootstrap_replicates.csv: replicate-level all-country and high-risk proportions.
- classifier_bootstrap_summary.csv: point estimates and percentile intervals.
- point_reproduction_diagnostics.csv: classifier fit vs supplied probability check.
