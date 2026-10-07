# Project LAWA: ML notes and handoff (last updated 2026-10-07)

Thesis: *Development of a Machine Learning Model for Soft Sensing of BOD and Fecal Coliform Estimation*
(manuscript: `thesis_manuscript/*.docx`). Data: LLDA Laguna Lake monitoring, 15 lake stations.

> **Rule from the user:** do NOT edit the thesis manuscript until they explicitly say so.

---

## 1. Decisions made so far

| Topic | Decision |
|---|---|
| Period | Feb 2016 – Dec 2025 (pH/nutrients not measured 2013–2015; no Q1 2026 report yet) |
| Predictors | DO, pH, Ammonia, Nitrate, Inorganic Phosphate, Temperature. **No TSS in the model** (no TSS for 2016–2023). TSS stays in the manuscript text for now. |
| Extra inputs | station (one-hot), month (sine/cosine), last month's lab value of the target (same station, within 3 months) |
| Stations | the 15 original lake stations; Stn. XXIV (Cardona North Central Bay, only from 2025) excluded |
| Below detection limit | "<x" stored as x/2 (same as the 2013–2023 data) |
| Gaps | linear interpolation within station, gaps ≤ 6 months (yellow cells) |
| Outliers | **flagged, not deleted** (sheet "Preprocessing v4 (flagged)"). Many high-BOD rows are algal blooms (high DO + high pH + high BOD) and are real. Sta. Rosa Sep 2017 BOD = 70 mg/L was confirmed in the LLDA report → kept. |
| Split | time-based: train 2016–2023, test 2024–2025 (+ rolling year-by-year 2018–2025). Imputed target values are never used as test targets. |
| Classes | user chose official DENR classes (see below) |

**BOD classes (DAO 2016-08), mg/L:** A ≤3, B >3–5, C >5–7, D >7 (the single value >15 is kept in D).
Counts: A 1,092 · B 210 · C 35 · D 9.

**Fecal coliform classes (DENR WQG 2021-19, as in the LLDA report legend), MPN/100 mL:**
A ≤50, B >50–100, C >100–200, D >200–400, Failed >400. Laguna Lake is Class C → above 200 = above standard.
Counts: A 662 · B 260 · C 182 · D 137 · Failed 105.

---

## 2. Files

**Workbook** `dataset/Copy of LLDA-Data-2013-2023-Compiled.xlsx` (backups: `... (backup before v3).xlsx`, `... (backup before v4).xlsx`).
Original sheets by the previous student are untouched. Added sheets:
- `Temperature (Raw)`, `Data Set 2 (Raw)` (Apr 2023–Dec 2025 from the PDFs), `Data Set 3 (Combined Raw)`
- `Preprocessing v3A` (gaps filled), `v3B` (±3SD outliers deleted, **superseded**), `Stats v3`, `Notes v3`
- `Preprocessing v4 (flagged)`: v3A rows, outliers flagged not deleted, BOD class. **Use this.**

**Important:** writing this workbook with the Node library exceljs corrupts it. Add sheets by building a fresh .xlsx
and copying it in with Excel: `dataset/preprocessing-v3/merge.ps1 -target <workbook> -newFile <new.xlsx>`.

**Transcribed PDF data** (read from scanned images by Claude, **needs spot-checking against the PDFs**):
`dataset/preprocessing-v3/wq.csv`, `temperature.csv`, plus `build.js` (rebuilds the v3 sheets).

**CSVs for modelling:**
- `dataset/bod_dataset.csv`: 6 readings + BOD, from v3B (old, outliers deleted)
- `dataset/bod_class_dataset.csv`: from v4, with Month, Station, BOD, flag, imputed, BOD_Class
- `dataset/fc_class_dataset.csv`: from v4, with fecal coliform and FC_Class

**Scripts** (run with `ml/.venv/Scripts/python.exe <script>`):
| Script | What it does | Results |
|---|---|---|
| `ml/bod/train_bod.py` | BOD regression, 6 readings, MLR/RF/XGB | `ml/bod/results/` |
| `ml/bod/compare_versions.py` | BOD regression versions A/B/C | `ml/bod/results_versions/` |
| `ml/bod_class/build_dataset.py` | builds v4 sheet + bod_class_dataset.csv | |
| `ml/bod_class/train_bod_class.py [beta]` | BOD 4-class model | `ml/bod_class/results_beta1/` (use), `results_beta2/` |
| `ml/fc_class/build_dataset.py` | builds fc_class_dataset.csv | |
| `ml/fc_class/train_fc_class.py [beta]` | fecal coliform 5-class model | `ml/fc_class/results_beta1/` (use), `results_beta2/` |
| `ml/fc_class/predict_one.py` | predict one reading with Improved XGBoost (edit `reading = {...}`) | prints class + probabilities |

`beta` = threshold setting: 1 = balanced (**use this**), 2 = favour catching pollution (too many false alarms).

---

## 3. Results so far (test 2024–2025 unless noted)

**BOD regression**
| Version | Best model | R² | RMSE |
|---|---|---|---|
| A: 6 readings | XGBoost | 0.05 | 1.28 |
| B: + station + month | MLR | 0.09 | 1.25 |
| C: + last month's BOD | **MLR** | **0.19** | 1.18 |
Benchmark 1 (R² ≥ 0.60) not met. Benchmark 3 (ensembles beat MLR) not met. ~65% of predictions within ±1 mg/L,
but "guess the average" gets 64%: BOD is mostly 1–3 mg/L.

**BOD class (4 classes)**: best = Improved Logistic Regression (ordinal + class weights + oversampling + tuned thresholds):
catches 60% of polluted (B+) months, 18% false alarms, macro F1 0.48 (rolling 2018–2025: 58% / 25% / 0.34).
Plain models mostly answer "A". C and D are rarely caught (too few examples).

**Fecal coliform class (5 classes)**:
| Model | Accuracy | Macro F1 | Above-standard (>200) caught | False alarms |
|---|---|---|---|---|
| Same class as last month (baseline) | 66% | **0.54** | 67% | **5%** |
| Plain XGBoost | 57% | 0.45 | 67% | 10% |
| Improved Random Forest | 60% | 0.44 | 75% | 9% |
| Improved XGBoost | 50% | 0.43 | **81%** | 18% |
→ no model beats "same as last month" overall; models only win at catching above-standard months.
The improved models use a "ladder" (go up a class when P(worse) ≥ tuned threshold), so they sometimes
predict a class above the most likely one (example: Stn. I Jan 2025, actual B, predicted C; San Pedro
Oct 2024, actual Failed 818, predicted D, still flagged above standard while "same as last month" said C).

**Main findings:** inputs explain little of BOD; fecal coliform depends mostly on last month's level
(consistent with the manuscript's idea that FC comes from waste discharge); the models' value is early warning.

---

## 4. TO DO: ideas to improve performance (in priority order)

1. **Fecal coliform regression on log10 values** (manuscript Benchmark 2: R² ≥ 0.55 on log FC).
   Same inputs + last month's log FC; MLR/RF/XGB; time split + rolling. Not done yet. Might reach the
   benchmark because last month's value is a strong predictor. Compare with "repeat last month".
2. **Grid-search tuning** (promised in manuscript Phase 4), for BOD class, FC class, and the regressions.
   Time-ordered 5-fold CV on training years only, score = macro F1 (classes) / RMSE (regression).
   - Logistic Regression: C {0.01, 0.1, 1, 10}
   - Random Forest: n_estimators {200, 500}, max_depth {3, 5, 10, None}, min_samples_split {2, 5, 10}
   - XGBoost: learning_rate {0.03, 0.1}, max_depth {2, 3, 4}, n_estimators {200, 500}, subsample {0.8, 1.0}, colsample_bytree {0.8, 1.0}
   - Re-tune thresholds after; save to NEW results folders. Details also in `ml/bod_class/NOTES_TODO.md`.
3. **Predict the change from last month** (up / same / down class) instead of the class from scratch,
   so the model can beat "same as last month" for fecal coliform.
4. **Hybrid**: default to last month's class, let the model raise it when it sees warning signs.
5. **Add the "most likely class" decision rule** to the comparison (vs the ladder rule); the dashboard could
   show both: "Most likely B (41%) · 37% chance of C or worse".
6. **Field-only model** (DO, pH, Temperature + station/month): what the IoT sensors alone can do, which supports
   the real-time argument (ammonia/nitrate/phosphate are lab tests).
7. **Spot-check the transcribed 2023–2025 and temperature data** against the PDFs (white-on-red/black cells).
   Suspicious temperatures: 36–38 °C May 2017 (all stations), 33–36 °C Jan 2021 (some stations).
8. Later: model per-station results, SHAP (XGBoost built-in `pred_contribs=True`; the shap package is blocked),
   save final models for the dashboard (frontend/ is React + Supabase).

## 5. Manuscript vs what was done (discuss with adviser; do NOT edit until told)
- TSS listed as a predictor (7 predictors) → model uses 6.
- Outliers: manuscript says IQR flag-and-review → v4 follows this idea (flagged, kept); v3B used ±3SD deletion.
- Imputation: manuscript says station-level mean fitted on training only → we used interpolation within station.
- Split: manuscript says random 70:30 stratified by station → we use time-based (random split inflates scores:
  BOD R² 0.25 random vs 0.05 time-based). "Stratified by station" does not apply to a numeric target.
- Period: manuscript says Jan 2013 – Mar 2026, ~159 months → data is Feb 2016 – Dec 2025.
- Manuscript says FC "1.8 to 918 after cleaning" → will change.
- Benchmarks are regression R²; the class models are not in the benchmarks yet.

## 6. Environment on a new PC
- Python 3.12 (`winget install Python.Python.3.12`), then:
  `python -m venv ml/.venv` and
  `ml/.venv/Scripts/python -m pip install pandas numpy scikit-learn==1.8.0 xgboost matplotlib scipy openpyxl joblib`
- On the laptop, Windows Smart App Control blocked scikit-learn 1.9.x and the `shap` package (numba/llvmlite);
  scikit-learn 1.8.0 worked. Do not turn Smart App Control off (it cannot be turned back on without a reset).
- Optional: Node.js (dataset/preprocessing-v3/build.js), Poppler (reading PDFs), Excel (merge.ps1).
- `ml/.venv` is machine-specific: recreate it on each PC, do not copy it.
