# Step 1 — Correlation analysis and feature selection (results)

Script: `ml/improve/step1_correlation.py` · Data: 2016-02 to 2025-12, 15 lake stations.
Selection used training years 2016–2023 only; the heatmaps describe the whole dataset.

## Correlations (Spearman, measured values 2016–2025)
- **BOD**: weak correlations only. Strongest: temperature +0.31, pH +0.28, nitrate −0.26. DO, ammonia, phosphate ≈ 0.
- **Fecal coliform (log10)**: strongest reading is pH −0.30; everything else |ρ| ≤ 0.15.
  Last month's FC is by far the strongest input (r = 0.84).
- **Between inputs**: pH–DO +0.44, pH–nitrate −0.45, pH–temperature +0.39 (algal bloom pattern).
- **Multicollinearity**: every VIF is below 2 (threshold 5), so no input is redundant.

## Input ranking (composite of Pearson, Spearman, mutual information, RF importance)
| Rank | BOD | Fecal coliform |
|---|---|---|
| 1 | pH | Last month's FC |
| 2 | Month (season) | pH |
| 3 | Nitrate | Station |
| 4 | Temperature | DO |
| 5 | DO | Month (season) |
| 6 | Last month's BOD | Nitrate |
| 7 | Station | Ammonia |
| 8 | Phosphate | Temperature |
| 9 | Ammonia | Phosphate |

Rule A (manuscript: top half by composite, VIF check) and Rule B (adviser's 2020 paper: |r| ≥ 0.10):
| Target | Rule A keeps | Rule B keeps |
|---|---|---|
| BOD | pH, month, nitrate, temperature, DO | all except phosphate |
| FC | last month's FC, pH, station, DO, month | Rule A + nitrate |

## Does selection improve accuracy? (fixed settings, no tuning yet)
| Target | Inputs | Best model | Rolling 2018–2025 mean R² | Holdout 2024–2025 R² |
|---|---|---|---|---|
| BOD | all 9 | XGBoost | −0.17 | 0.21 |
| BOD | Rule A (5) | XGBoost | −0.12 | 0.09 |
| BOD | **Rule B (8)** | **XGBoost** | **−0.02** | **0.23** |
| FC (log10) | all 9 | MLR | 0.17 | 0.64 |
| FC (log10) | **Rule A (5)** | **MLR** | **0.18** | **0.64** |
| FC (log10) | Rule B (6) | MLR | 0.17 | 0.64 |

**Decision for the next steps**
- **BOD → Rule B inputs** (drop phosphate): best on both evaluations. Rule A drops last month's BOD and station and loses accuracy.
- **FC → Rule A inputs** (last month's FC, pH, station, DO, month): same accuracy as all 9 with fewer inputs; ammonia, nitrate, temperature and phosphate add nothing.

## Notes and limitations
- Random Forest permutation importance was measured on 2023, an unusual year for both targets; its values are near zero, so this method contributed little to the ranking.
- Rolling R² for BOD is negative mainly because of 2019 and 2023, when all models (and "repeat last month") do poorly.
- Station and month have no single Pearson r; their "r" is the correlation ratio (station) and the multiple correlation of sine + cosine (month).
