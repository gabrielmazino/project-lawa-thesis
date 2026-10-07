# BOD class model: status and to-do

Status (2026-10-07): paused, to be continued on the desktop PC.

## Current best
- Model: Improved Logistic Regression (ordinal "BOD > 3 / > 5 / > 7", class weights,
  oversampling of C/D in training only, thresholds tuned for balanced F1).
- Inputs: DO, pH, Ammonia, Nitrate, Phosphate, Temperature, station, month (sin/cos), last lab BOD (<= 3 months).
- Data: dataset/bod_class_dataset.csv (sheet "Preprocessing v4 (flagged)"), outliers kept and flagged.
- Results: ml/bod_class/results_beta1/ (holdout 2024-2025: polluted recall 60%, false alarms 18%, macro F1 0.48).

## TO DO: hyperparameter tuning (grid search), promised in the manuscript (Phase 4)
- Time-ordered 5-fold CV on the training years only (2016-2023); score = macro F1.
- Logistic Regression: C in {0.01, 0.1, 1, 10}
- Random Forest: n_estimators {200, 500}, max_depth {3, 5, 10, None}, min_samples_split {2, 5, 10}
- XGBoost: learning_rate {0.03, 0.1}, max_depth {2, 3, 4}, n_estimators {200, 500},
  subsample {0.8, 1.0}, colsample_bytree {0.8, 1.0}
- After choosing settings: re-tune thresholds, retrain on all training years, test once on 2024-2025
  and in the rolling 2018-2025 evaluation. Save to a NEW results folder (keep results_beta1).
- Expect ~10-30 min runtime.

## Setting up on the PC
- Needs Python 3.12 + packages: pandas numpy scikit-learn==1.8.0 xgboost matplotlib scipy openpyxl joblib
  (scikit-learn 1.9.x and the shap package were blocked by Windows Smart App Control on the laptop).
- Run: python ml/bod_class/train_bod_class.py 1
