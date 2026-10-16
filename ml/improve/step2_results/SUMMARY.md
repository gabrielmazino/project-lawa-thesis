# Step 2 — Fixing the class imbalance (results)

Script: `ml/improve/step2_imbalance.py` · Inputs from Step 1 · Models with fixed settings (tuning is Step 3).
Main evaluation: rolling-origin CV over 2016–2025 (each year 2018–2025 predicted from earlier years only,
predictions pooled). Also the 2024–2025 holdout. Full tables: `results_all.csv`; confusion matrices: `cm_*.png`.

## BOD — imbalance techniques clearly help
| Scheme | Setup | Macro F1 | Problem months caught | False alarms |
|---|---|---|---|---|
| 4-class | Baseline: same class as last month | 0.31 | 46% | 13.8% |
| 4-class | XGBoost, no technique | 0.34 | 20% | 1.8% |
| 4-class | **XGBoost + class weights** | **0.37** | **52%** | **11.4%** |
| 4-class | XGBoost + oversampling | 0.37 | 48% | 12.1% |
| 2-class | Baseline: same class as last month | 0.66 | 46% | 13.8% |
| 2-class | Random Forest, no technique | 0.59 | 16% | 0.1% |
| 2-class | **Random Forest + class weights** | **0.71** | **49%** | **8.6%** |

- Without a technique the models almost never flag a polluted month (16–20% caught).
- **Class weights** lift that to about half of the polluted months, with fewer false alarms than "same class as last month".
- SMOTE did not beat class weights or simple oversampling.
- Early-warning setting (F2 tuning) catches 76–82% of polluted months but with 40–60% false alarms.

## Fecal coliform — the problem is not imbalance
| Scheme | Setup | Macro F1 | Above-standard caught | False alarms |
|---|---|---|---|---|
| 5-class | **Baseline: same class as last month** | **0.50** | 64% | 4.7% |
| 5-class | Best model (XGBoost + SMOTE) | 0.43 | 59% | 7.9% |
| 2-class | **Baseline: same class as last month** | **0.79** | 64% | 4.7% |
| 2-class | Best model (Logistic Regression, no technique) | 0.78 | 61% | 4.9% |
| 2-class | Random Forest + SMOTE | 0.77 | 74% | 9.2% |

- No technique beats "same class as last month" on macro F1. FC classes are already fairly balanced; what limits the models is that FC mostly repeats last month's level.
- Techniques trade accuracy for catching more above-standard months (up to 74–78%, with 9–12% false alarms).

## Decision for the next steps
- **BOD → class weights** (best or near-best in both schemes; simplest).
- **FC → no imbalance technique** for the balanced choice; class weights if the early-warning view is preferred.
  The FC regression model that predicts the change from last month remains the better route for FC.

## Notes
- Compared with the laptop's "Improved Logistic Regression" (BOD 4-class): rolling macro F1 0.37 here vs 0.34 there; on the
  2024–2025 holdout 0.46 here vs 0.48 there (different inputs and decision rule), so the two are close.
- Decision tuning (probability cut-off / class-share adjustment) is chosen on time-ordered CV of the training years only.
- Oversampling and SMOTE were applied to training data only, inside each fold.
