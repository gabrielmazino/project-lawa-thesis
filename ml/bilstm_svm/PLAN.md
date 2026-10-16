# BiLSTM and SVR Soft Sensors

Build and compare regression models for BOD and fecal coliform using the LAWA monthly station data. Preserve measured-target integrity, use a chronological 70/30 evaluation, and report the requested 80% goal across regression, tolerance, and regulatory-class metrics without promising that threshold. This is an ML-only effort; the thesis manuscript and dashboard remain out of scope.

## Implementation

1. **Prepare consistent data.** Merge the v4 flagged BOD and FC station-month exports on month and station. Use the six approved predictors: DO, pH, Ammonia, Nitrate, Phosphate, and Temperature. Read predictor-imputation flags from the v4 workbook. Keep real flagged outliers unless review establishes an error. Exclude TSS and Stn. XXIV. Do not use the obsolete v3B BOD regression export as-is because it lacks month/station metadata and has a different outlier policy.
2. **Build sequences.** Form contiguous 3-, 6-, and 12-month windows separately per station. Exclude imputed target rows from both training and scoring, retain predictor-imputation indicators, and do not let windows cross station boundaries or contain future values.
3. **Split and tune.** Use the earliest 70% of available months for training and the latest 30% for the final holdout, keeping all stations for a month on the same side. Do not shuffle. Fit preprocessing on training data only. Tune SVR and BiLSTM candidates using expanding, time-ordered validation inside the training period; keep the final holdout untouched until final scoring.
4. **Fit models.** Train BiLSTM and scikit-learn SVR for both targets. Use raw BOD and log10 FC targets, then back-transform FC for original-scale reporting. Feed SVR and tabular baselines flattened windows from the same 12-month history used for the sequence models.
5. **Compare.** Report a training-mean baseline, linear regression, random forest, XGBoost, and FC persistence (last measured value within the existing three-month rule), in addition to the new models. Report row counts wherever persistence cannot cover every holdout row.
6. **Evaluate and save.** Report MAE, RMSE, and R²; FC metrics on both log10 and original scales; tolerance accuracy; exact and within-one DENR-class agreement; FC above-standard recall and false-alarm rate; and station-level results. Save predictions, parameters, model artifacts, preprocessing metadata, seeds, dates, and plots to `results_70_30/`.
7. **Review before integration.** Treat 80% as a goal, not a guarantee. Do not weaken the split or change metrics after seeing holdout results. Backend/dashboard integration and manuscript edits need separate approval.

## Predeclared Evaluation Choices

- Split: chronological 70/30 by available months; the current export puts the cutoff at March 2023.
- BOD tolerance accuracy: prediction within ±1 mg/L, matching the existing project discussion.
- FC tolerance accuracy: prediction within a factor of 2 of the measured concentration, equivalent to ±log10(2) on the log scale.
- FC target: log10 MPN/100 mL; report original-scale estimates and metrics as well.

## Repository Evidence and Risks

- The current BOD and FC v4 class datasets each contain 1,346 rows, 15 stations, and the same unique station-month keys from February 2016 through December 2025.
- Target imputation flags mark 36 BOD and 9 FC rows; predictor flags from the source workbook mark 358 imputed predictor cells.
- With a 12-month contiguous lookback and measured targets, data validation found 414 BOD training / 500 holdout targets and 424 FC training / 502 holdout targets.
- Existing time-holdout performance is weak for BOD regression (reported R² about 0.19). FC log regression is about 0.64 R², while persistence is about 0.66, and rolling performance varies. Reaching 80% is not assured.
- LLDA transcription spot-checking is still recommended by `ml/PROJECT_NOTES.md`.
- `ml/.venv` is machine-specific. The new `requirements.txt` pins Python package versions for this workflow; use the project Python 3.12 environment.

## Verification

1. Validate unique month/station keys, matching target exports, station/date coverage, imputation counts, finite values, and workbook imputation provenance.
2. Verify all sequence windows are contiguous, chronological, station-local, and score only measured targets.
3. Verify time-ordered train/validation folds precede the final holdout and all scaling is fitted on training data only.
4. Run the training workflow with a fixed seed and independently recompute saved metrics from holdout predictions.
5. Inspect per-station and time-series results for rare high-BOD misses and FC level-shift behavior.

Run from the repository root with:

```powershell
.\ml\.venv\Scripts\python.exe ml/bilstm_svm/train_models.py --validate-data
.\ml\.venv\Scripts\python.exe ml/bilstm_svm/train_models.py
```
