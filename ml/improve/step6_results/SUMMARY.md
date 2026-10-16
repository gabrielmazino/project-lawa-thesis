# Step 6 — Is last month's lab value needed? (results)

Script: `ml/improve/step6_last_month.py` · Same models, tuned settings and splits; only last month's value removed.
Main test: random 70-30, 10 seeds. Second check: rolling 2018–2025. Full numbers: `summary.csv`, `all_runs.csv`.

| Task (Random Forest) | With last month | Without last month | Change |
|---|---|---|---|
| BOD value R² (70-30) | 0.28 | 0.24 | small loss |
| BOD value R² (rolling) | 0.03 | −0.04 | small loss |
| BOD 4-class macro F1 (70-30) | 0.41 (69% polluted caught) | 0.40 (64% caught) | small loss |
| BOD 4-class macro F1 (rolling) | 0.48 | 0.45 | small loss |
| FC value R² (70-30) | 0.72 | 0.16 | **collapses** |
| FC value R² (rolling) | 0.33 | −1.25 | **collapses** |
| FC 5-class macro F1 (70-30) | 0.53 (59% above-standard caught) | 0.25 (14% caught) | **collapses** |
| FC 5-class macro F1 (rolling) | 0.45 | 0.23 | **collapses** |

XGBoost and SVM show the same pattern.

- **BOD:** the water readings do most of the work; last month's BOD adds a little (about +0.03 R², +0.01–0.04 macro F1).
  A readings-only BOD model is a true soft sensor and loses little.
- **FC:** the readings alone cannot predict fecal coliform (R² 0.16; catches only 14% of above-standard months).
  Last month's lab value is essential; the readings only adjust it.
- Note: the "without" models reuse settings tuned with last month included (not re-tuned), so they may be slightly
  under-rated, but not enough to change the FC conclusion.
