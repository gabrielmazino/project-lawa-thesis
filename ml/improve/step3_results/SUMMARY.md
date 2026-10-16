# Step 3 — More models (SVM, kNN, LSTM) and tuning (results)

Script: `ml/improve/step3_models.py` · Inputs from Step 1 · Imbalance choice from Step 2 (BOD classes: class weights;
kNN uses oversampling instead; FC classes: none).

- **Main test:** whole 2016–2025 dataset, one 70:30 split (stratified by station and year, so every station and year is
  in both parts; filled values never tested). Test size: BOD 386 rows, FC 394 rows.
- **Tuning:** grid search with 5-fold CV inside the 70% only (LSTM: 3-fold over hidden size 16/32 × sequence 3/6 months;
  final LSTM averages 3 networks). Chosen settings: `tuned_settings.json`.
- **Second check:** rolling year by year (each year 2018–2025 predicted from earlier years) with the tuned settings.
- Full tables: `regression_results.csv`, `classification_results.csv`. Charts: `regression_compare.png`,
  `classification_compare.png`, `best_regression_pred_vs_actual.png`, `best_classifier_confusion.png`.

## Regression
| Task | Best model (70:30) | R² | RMSE | Repeat last month R² | Best model (rolling) | Rolling R² |
|---|---|---|---|---|---|---|
| BOD (mg/L) | **Random Forest** | **0.35** | 1.12 | −0.23 | LSTM 0.16 / RF 0.15 | baseline −0.26 |
| FC log10 level | LSTM | 0.72 | 0.285 | 0.64 | none beats baseline | baseline 0.63 |
| FC log10 change | **Random Forest** | **0.71** | 0.291 | 0.64 | kNN 0.65 (RF 0.62) | baseline 0.63 |

- **BOD:** Random Forest is clearly best on the 70:30 split (R² 0.35 vs 0.21–0.27 for the others); all models beat both baselines.
  SVM is second (0.27). LSTM ties RF on the rolling check.
- **FC:** on the 70:30 split all models except kNN beat "repeat last month" (0.69–0.72 vs 0.64).
  On the rolling check, models predicting the FC *level* fall apart (the 2018 jump in FC levels), while the *change*
  models stay close to the baseline. Predicting the change remains the safer FC design.

## Classification (macro F1)
| Scheme | 70:30 best | Rolling best | "Same class as last month" (70:30 / rolling) |
|---|---|---|---|
| BOD 4-class | **Random Forest 0.37** (catches 70%, 18% false alarms) | **Random Forest 0.38** | 0.32 / 0.31 |
| BOD 2-class | **Random Forest 0.70** | **Random Forest 0.71** (46% caught, 7% false alarms) | 0.66 / 0.66 |
| FC 5-class | Random Forest 0.52 | baseline 0.50 (best model XGBoost 0.43) | 0.49 / 0.50 |
| FC 2-class | Logistic Regression 0.83 | baseline 0.79 (best model RF 0.79) | 0.79 / 0.79 |

- **BOD:** Random Forest + class weights is best on both tests and beats the baseline in both. XGBoost is second.
- **FC:** models beat the baseline on the 70:30 split but not on the rolling check; FC class mostly repeats last month.
- **SVM, kNN, LSTM** did not beat Random Forest anywhere for classification. kNN gives the most false alarms. LSTM is
  competitive only for BOD regression; with about 1,300 monthly rows it has too little data to learn more.

## Decision for the next steps
- **BOD regression:** Random Forest (tuned). **BOD classes:** Random Forest + class weights.
- **FC regression:** Random Forest predicting the change from last month. **FC classes:** Random Forest (or Logistic
  Regression for 2-class); report that persistence is a strong baseline.
- SVM, kNN and LSTM are kept in the comparison table (the adviser asked for them) but not used in the system.

## Notes
- 70:30 random-split scores are higher than rolling scores: the random test months sit between training months of the
  same station, which is easier than predicting a whole new year. Step 4 compares the splits (60-40, 70-30, 80-20,
  k-fold, rolling) properly.
- Tuning used the 70% part, which includes rows from every year, so the rolling check is slightly optimistic for all
  models equally.
