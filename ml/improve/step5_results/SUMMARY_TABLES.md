# Step 5 — Direct classifier vs regression -> class

Mean ± SD over 10 seeds (random 70-30) or 8 years (rolling). Settings tuned in Step 3.

## BOD 4-class · random 70-30

| Model | Route | Accuracy | Macro F1 | Balanced acc. | Problem months caught | False alarms |
|---|---|---|---|---|---|---|
| Baseline | same class as last month | 76.0% ± 1.8 | 0.31 ± 0.02 | 31.8% | 47% | 14% |
| Random Forest | A. direct classifier | 75.9% ± 1.7 | 0.41 ± 0.06 | 44.7% | 69% | 18% |
| Random Forest | B. regression -> class | 79.2% ± 1.8 | 0.40 ± 0.03 | 40.7% | 65% | 13% |
| XGBoost | A. direct classifier | 71.2% ± 1.8 | 0.38 ± 0.04 | 47.0% | 72% | 22% |
| XGBoost | B. regression -> class | 78.2% ± 2.0 | 0.40 ± 0.08 | 40.6% | 61% | 14% |
| SVM | A. direct classifier | 70.7% ± 1.6 | 0.33 ± 0.02 | 35.5% | 65% | 23% |
| SVM | B. regression -> class | 79.9% ± 1.6 | 0.36 ± 0.03 | 35.2% | 48% | 10% |
| MLR / Logistic Regression | A. direct classifier | 64.3% ± 1.2 | 0.32 ± 0.02 | 39.5% | 74% | 29% |
| MLR / Logistic Regression | B. regression -> class | 75.0% ± 2.0 | 0.36 ± 0.03 | 36.6% | 58% | 18% |

## BOD 4-class · rolling year-by-year

| Model | Route | Accuracy | Macro F1 | Balanced acc. | Problem months caught | False alarms |
|---|---|---|---|---|---|---|
| Baseline | same class as last month | 74.6% ± 10.7 | 0.42 ± 0.14 | 42.7% | 38% | 15% |
| Random Forest | A. direct classifier | 80.2% ± 9.3 | 0.48 ± 0.17 | 48.3% | 46% | 10% |
| Random Forest | B. regression -> class | 78.8% ± 8.2 | 0.48 ± 0.14 | 47.4% | 36% | 9% |
| XGBoost | A. direct classifier | 74.2% ± 9.8 | 0.47 ± 0.15 | 49.3% | 56% | 18% |
| XGBoost | B. regression -> class | 79.6% ± 6.9 | 0.47 ± 0.15 | 46.5% | 35% | 9% |
| SVM | A. direct classifier | 72.0% ± 7.5 | 0.44 ± 0.16 | 46.1% | 46% | 18% |
| SVM | B. regression -> class | 80.8% ± 6.5 | 0.40 ± 0.13 | 41.3% | 23% | 4% |
| MLR / Logistic Regression | A. direct classifier | 66.2% ± 6.5 | 0.42 ± 0.11 | 44.9% | 57% | 26% |
| MLR / Logistic Regression | B. regression -> class | 78.5% ± 4.4 | 0.44 ± 0.12 | 45.4% | 35% | 9% |

## FC 5-class · random 70-30

| Model | Route | Accuracy | Macro F1 | Balanced acc. | Problem months caught | False alarms |
|---|---|---|---|---|---|---|
| Baseline | same class as last month | 60.4% ± 1.7 | 0.51 ± 0.02 | 51.4% | 70% | 7% |
| Random Forest | A. direct classifier | 60.8% ± 2.0 | 0.50 ± 0.02 | 49.9% | 64% | 6% |
| Random Forest | B. regression -> class | 62.2% ± 2.4 | 0.53 ± 0.04 | 51.6% | 58% | 3% |
| XGBoost | A. direct classifier | 60.3% ± 2.2 | 0.49 ± 0.03 | 49.1% | 68% | 6% |
| XGBoost | B. regression -> class | 61.9% ± 2.5 | 0.52 ± 0.04 | 50.9% | 57% | 3% |
| SVM | A. direct classifier | 59.2% ± 2.5 | 0.48 ± 0.02 | 47.0% | 65% | 5% |
| SVM | B. regression -> class | 62.3% ± 1.9 | 0.53 ± 0.02 | 52.0% | 65% | 3% |
| MLR / Logistic Regression | A. direct classifier | 59.6% ± 2.8 | 0.48 ± 0.03 | 47.4% | 68% | 6% |
| MLR / Logistic Regression | B. regression -> class | 61.9% ± 2.0 | 0.52 ± 0.02 | 51.1% | 62% | 3% |

## FC 5-class · rolling year-by-year

| Model | Route | Accuracy | Macro F1 | Balanced acc. | Problem months caught | False alarms |
|---|---|---|---|---|---|---|
| Baseline | same class as last month | 66.3% ± 11.5 | 0.46 ± 0.10 | 45.8% | 48% | 4% |
| Random Forest | A. direct classifier | 55.9% ± 21.8 | 0.37 ± 0.14 | 44.3% | 40% | 4% |
| Random Forest | B. regression -> class | 65.3% ± 11.4 | 0.45 ± 0.09 | 45.0% | 35% | 2% |
| XGBoost | A. direct classifier | 61.4% ± 11.2 | 0.41 ± 0.09 | 45.3% | 37% | 5% |
| XGBoost | B. regression -> class | 64.1% ± 10.2 | 0.45 ± 0.09 | 48.0% | 34% | 2% |
| SVM | A. direct classifier | 59.8% ± 13.2 | 0.38 ± 0.09 | 41.3% | 38% | 6% |
| SVM | B. regression -> class | 66.3% ± 10.7 | 0.49 ± 0.09 | 52.3% | 46% | 3% |
| MLR / Logistic Regression | A. direct classifier | 59.5% ± 15.0 | 0.38 ± 0.10 | 41.7% | 42% | 6% |
| MLR / Logistic Regression | B. regression -> class | 66.7% ± 11.6 | 0.49 ± 0.09 | 52.6% | 47% | 3% |

## How often the direct classifier beats regression -> class (macro F1, same model and split)

| Scheme | Test | Model | Direct wins |
|---|---|---|---|
| BOD 4-class | random 70-30 | MLR / Logistic Regression | 10% |
| BOD 4-class | random 70-30 | Random Forest | 60% |
| BOD 4-class | random 70-30 | SVM | 20% |
| BOD 4-class | random 70-30 | XGBoost | 40% |
| BOD 4-class | rolling year-by-year | MLR / Logistic Regression | 38% |
| BOD 4-class | rolling year-by-year | Random Forest | 62% |
| BOD 4-class | rolling year-by-year | SVM | 50% |
| BOD 4-class | rolling year-by-year | XGBoost | 62% |
| FC 5-class | random 70-30 | MLR / Logistic Regression | 0% |
| FC 5-class | random 70-30 | Random Forest | 30% |
| FC 5-class | random 70-30 | SVM | 0% |
| FC 5-class | random 70-30 | XGBoost | 20% |
| FC 5-class | rolling year-by-year | MLR / Logistic Regression | 12% |
| FC 5-class | rolling year-by-year | Random Forest | 0% |
| FC 5-class | rolling year-by-year | SVM | 12% |
| FC 5-class | rolling year-by-year | XGBoost | 12% |