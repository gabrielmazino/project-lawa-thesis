# Step 5 — Direct classifier vs regression -> class (results)

Script: `ml/improve/step5_class_route.py` · Main test: random 70-30, 10 seeds · Second check: rolling 2018–2025.
A = classifier trained on the classes. B = regression model predicts the value, DENR limits give the class.
Full tables: `SUMMARY_TABLES.md`, `summary.csv`, `all_runs.csv`, `direct_vs_regression_wins.csv`.

## BOD 4-class (random 70-30, mean of 10 seeds)
| Model | Route | Accuracy | Macro F1 | Polluted caught | False alarms |
|---|---|---|---|---|---|
| **Random Forest** | **A. direct** | 75.9% | **0.41** | **69%** | 18% |
| Random Forest | B. regression → class | 79.2% | 0.40 | 65% | 13% |
| XGBoost | A. direct | 71.2% | 0.39 | 72% | 22% |
| XGBoost | B. regression → class | 78.2% | 0.41 | 61% | 14% |
| SVM | A / B | 70.7% / 79.9% | 0.33 / 0.36 | 65% / 48% | 23% / 10% |
| Same class as last month | – | 76.0% | 0.32 | 47% | 14% |
Rolling: Random Forest direct 0.48 vs regression 0.48; direct catches 46% of polluted months vs 36%.
→ Near tie on macro F1; the direct classifier catches more polluted months (regression shrinks toward Class A).

## FC 5-class (random 70-30, mean of 10 seeds)
| Model | Route | Accuracy | Macro F1 | Above standard caught | False alarms |
|---|---|---|---|---|---|
| Random Forest | A. direct | 60.8% | 0.51 | 64% | 6% |
| Random Forest | B. regression → class | 62.2% | 0.53 | 58% | 3% |
| **SVM** | **B. regression → class** | **62.3%** | **0.53** | **65%** | **3%** |
| XGBoost | B. regression → class | 61.9% | 0.52 | 57% | 3% |
| Same class as last month | – | 60.4% | 0.51 | 70% | 7% |
Rolling: regression → class 0.45–0.49 vs direct 0.37–0.41 vs baseline 0.46. Regression route wins in 70–100% of repeats.
→ Regression → class is clearly better for FC, and is the first FC class approach that beats "same class as last month".

## Decision
- **BOD class: direct classifier (Random Forest + class weights).**
- **FC class: regression → class** using the FC change model (Random Forest, or SVM which is best on the rolling check).
