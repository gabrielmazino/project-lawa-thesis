# Step 4 — Split sensitivity (results)

Script: `ml/improve/step4_splits.py` · Data 2016–2025 · Settings tuned in Step 3 · Filled values never tested.

| Split method | Repeats | How |
|---|---|---|
| Random 60-40 / 70-30 / 80-20 | 10 seeds each | random months, stratified by station and year |
| 10-fold cross-validation | 10 folds | stratified by station |
| Chronological 60-40 / 70-30 / 80-20 | 1 each | oldest months train, newest test (cut-offs 2023-01/02, 2023-10/11, 2024-07) |
| Rolling year by year | 8 years | each year 2018–2025 predicted from earlier years |

Files: `all_runs.csv` (every repeat), `summary_by_split.csv` (average and standard deviation),
`top3_frequency.csv`, `SUMMARY_TABLES.md` (all scores incl. accuracy), `split_comparison.png`.
Rolling numbers here are the average of the 8 yearly scores (Step 3 pooled all years, so values differ).

## Main score (average over repeats)
**BOD value — R²**
| Model | 60-40 | 70-30 | 80-20 | 10-fold | Chrono 60-40 | Chrono 70-30 | Chrono 80-20 | Rolling |
|---|---|---|---|---|---|---|---|---|
| MLR | 0.08 | 0.09 | 0.11 | 0.11 | 0.19 | 0.18 | 0.08 | −0.17 |
| **Random Forest** | **0.25** | **0.28** | **0.27** | **0.34** | 0.22 | 0.22 | 0.06 | **0.03** |
| XGBoost | 0.24 | 0.25 | 0.24 | 0.31 | 0.21 | 0.22 | 0.10 | −0.06 |
| SVM | 0.19 | 0.22 | 0.19 | 0.26 | 0.22 | 0.22 | 0.11 | −0.16 |
| kNN | 0.15 | 0.16 | 0.14 | 0.20 | 0.16 | 0.16 | 0.04 | −0.17 |
| LSTM | 0.20 | 0.22 | 0.19 | 0.28 | **0.24** | **0.24** | **0.12** | −0.08 |
| Repeat last month | −2.03 | −1.83 | −1.38 | −2.32 | −0.17 | −0.21 | −0.34 | −0.49 |

**FC log10 (change model) — R²**
| Model | 60-40 | 70-30 | 80-20 | 10-fold | Chrono 60-40 | Chrono 70-30 | Chrono 80-20 | Rolling |
|---|---|---|---|---|---|---|---|---|
| MLR | 0.72 | 0.70 | 0.72 | 0.71 | 0.59 | 0.62 | 0.66 | 0.18 |
| **Random Forest** | **0.74** | **0.72** | **0.74** | **0.73** | 0.59 | 0.60 | 0.64 | 0.34 |
| XGBoost | 0.73 | 0.71 | 0.73 | 0.72 | 0.56 | 0.58 | 0.60 | 0.28 |
| SVM | 0.72 | 0.70 | 0.72 | 0.71 | 0.60 | 0.63 | 0.67 | 0.29 |
| kNN | 0.70 | 0.67 | 0.69 | 0.69 | **0.62** | **0.65** | **0.72** | **0.40** |
| LSTM | 0.72 | 0.70 | 0.72 | 0.72 | 0.53 | 0.53 | 0.59 | 0.25 |
| Repeat last month | 0.68 | 0.65 | 0.67 | 0.66 | 0.61 | 0.64 | 0.72 | 0.36 |

**BOD 4-class — macro F1 (accuracy in brackets)**
| Model | 60-40 | 70-30 | 80-20 | 10-fold | Chrono 60-40 | Chrono 70-30 | Chrono 80-20 | Rolling |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.32 (65%) | 0.32 (64%) | 0.30 (64%) | 0.36 (65%) | 0.43 (68%) | 0.46 (71%) | 0.43 (68%) | 0.42 (66%) |
| **Random Forest** | **0.41 (77%)** | **0.41 (76%)** | **0.40 (75%)** | **0.43 (75%)** | **0.47 (80%)** | 0.44 (76%) | 0.45 (74%) | **0.48 (80%)** |
| XGBoost | 0.37 (72%) | 0.39 (71%) | 0.39 (72%) | 0.41 (71%) | 0.47 (74%) | **0.48 (72%)** | **0.49 (71%)** | 0.47 (74%) |
| SVM | 0.32 (72%) | 0.33 (71%) | 0.33 (70%) | 0.39 (72%) | 0.39 (72%) | 0.42 (69%) | 0.40 (69%) | 0.44 (72%) |
| kNN | 0.30 (64%) | 0.32 (63%) | 0.32 (65%) | 0.39 (65%) | 0.41 (58%) | 0.43 (64%) | 0.47 (67%) | 0.40 (60%) |
| LSTM | 0.29 (62%) | 0.31 (63%) | 0.29 (61%) | 0.32 (58%) | 0.43 (71%) | 0.43 (68%) | 0.43 (72%) | 0.42 (67%) |
| Same class as last month | 0.32 (77%) | 0.32 (76%) | 0.31 (76%) | 0.36 (76%) | 0.42 (77%) | 0.41 (71%) | 0.41 (67%) | 0.42 (75%) |

**FC 5-class — macro F1 (accuracy in brackets)**
| Model | 60-40 | 70-30 | 80-20 | 10-fold | Chrono 60-40 | Chrono 70-30 | Chrono 80-20 | Rolling |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.48 (62%) | 0.48 (60%) | 0.48 (62%) | 0.47 (61%) | 0.40 (52%) | 0.43 (55%) | 0.42 (59%) | 0.38 (60%) |
| **Random Forest** | **0.51 (63%)** | **0.51 (61%)** | 0.51 (63%) | 0.50 (63%) | **0.44 (55%)** | **0.44 (57%)** | 0.49 (63%) | 0.37 (56%) |
| XGBoost | 0.51 (63%) | 0.49 (60%) | **0.51 (62%)** | **0.51 (63%)** | 0.42 (52%) | 0.43 (55%) | **0.51 (62%)** | **0.41 (61%)** |
| SVM | 0.47 (61%) | 0.48 (59%) | 0.48 (61%) | 0.46 (60%) | 0.39 (51%) | 0.40 (52%) | 0.42 (56%) | 0.38 (60%) |
| kNN | 0.41 (56%) | 0.43 (56%) | 0.43 (57%) | 0.42 (57%) | 0.28 (44%) | 0.29 (46%) | 0.36 (53%) | 0.32 (51%) |
| LSTM | 0.41 (57%) | 0.43 (57%) | 0.46 (59%) | 0.43 (57%) | 0.24 (41%) | 0.27 (44%) | 0.38 (52%) | 0.27 (49%) |
| Same class as last month | 0.52 (62%) | 0.51 (60%) | 0.51 (62%) | 0.50 (62%) | 0.53 (61%) | 0.53 (64%) | 0.55 (69%) | 0.46 (66%) |

## What the comparison shows
1. **The split ratio (60-40 vs 70-30 vs 80-20) barely matters.** Averages differ by about 0.01–0.03, which is smaller
   than the seed-to-seed spread (about ±0.02–0.15). So no ratio is "better"; 70-30 is a fine main choice.
2. **The split type matters much more.** Random splits and 10-fold give higher FC scores than chronological and rolling,
   because a random test month sits between known months of the same station. Predicting new years is harder.
   BOD 4-class goes the other way (higher on time-based splits), likely because the later test years contain more
   B–D months, which suits macro F1.
3. **One split can mislead.** Step 3's single 70:30 split gave BOD Random Forest R² 0.35; over 10 seeds the average is
   0.28 ± 0.10. Reporting the average ± spread is more honest.
4. **The 70 mg/L Sta. Rosa reading (Sep 2017)** makes random-split BOD scores jumpy: when the month after it is in the
   test part, "repeat last month" guesses 70 against an actual 4, so its R² falls to about −2.
5. **Random Forest and XGBoost are the most reliable models.** They are in the top 3 in 70–100% of random and
   10-fold repeats for every task, and in most time-based runs for the class tasks. The exception is FC values on
   time-based splits, where SVM, MLR and kNN do better. Third place varies: SVM or LSTM for BOD values, SVM for FC values,
   Logistic Regression or SVM for the classes.
6. **For FC, "same as last month" stays very hard to beat**, especially on time-based splits; models only beat it on
   random splits, and only slightly.

## Recommendation
- **Main results:** random 70-30, averaged over 10 seeds (mean ± standard deviation), as the adviser asked.
  Also report 10-fold CV (common in the literature) and the rolling year-by-year test (how the system is really used:
  predicting new months from past data).
- **Report all splits** in one table (`SUMMARY_TABLES.md`) to show the results are stable across ratios.
- **Top 3 models: Random Forest, XGBoost, SVM.** Keep Logistic Regression / MLR and "same as last month" as reference
  points; show kNN and LSTM in the full table only.
