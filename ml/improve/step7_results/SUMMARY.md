# Step 7 — All six models WITHOUT last month's lab value (results)

Script: `ml/improve/step7_no_last_month.py` · Inputs = water parameters + station + month only · All models re-tuned
(`tuned_settings.json`) · Random 70-30 × 10 seeds (main) + rolling 2018–2025 · Full numbers: `summary.csv`, `all_runs.csv`.
"Reference: last month" rows use a lab value and are shown only for comparison.

## BOD (inputs: DO, pH, Ammonia, Nitrate, Temperature, month, station)
| Model | Value R² 70-30 | Value R² rolling | 4-class macro F1 70-30 (classifier + weights) | 4-class rolling |
|---|---|---|---|---|
| MLR / LR | 0.18 | −0.21 | 0.32 | – |
| **Random Forest** | **0.26** | **−0.01** | **0.39** (62% polluted caught, 19% false alarms) | **0.49** (41% caught, 11% false) |
| XGBoost | 0.21 | −0.15 | 0.35 | 0.43 |
| SVM | 0.19 | −0.27 | 0.32 | – |
| kNN | 0.15 | −0.19 | 0.33 | – |
| LSTM | 0.21 | −0.19 | 0.29 | – |
| With last month (Step 6, RF) | 0.28 | 0.03 | 0.41 | 0.48 |
→ BOD barely changes without last month. **Random Forest** stays best: value R² 0.26, class via a separate
classifier with class weights (regression → class catches far fewer polluted months: 27% vs 41% on rolling).

## Fecal coliform
| Setup (70-30) | Best model | Value R² | 5-class macro F1 | Above-standard caught |
|---|---|---|---|---|
| Step 1 inputs (pH, DO, month, station) | SVM | 0.20 | – | – |
| Step 1 inputs, classifier + class weights | RF / SVM / XGB | – | 0.32 | 47–57% (18–25% false alarms) |
| All 6 readings | XGBoost | 0.37 | 0.36 (reg → class) | 31% |
| Reference: repeat / same class as last month | – | 0.65 | 0.52 | 69% |
Rolling (new years): every model has negative R² for the FC value (2018 alone: about −9 because FC levels dropped
sharply that year); class macro F1 0.21–0.29.

**Warning about LSTM and "all 6 readings":** on the random split, the LSTM reached FC R² 0.71 (all readings) and the
tree models 0.34–0.37, but on the rolling test they are the *worst* (LSTM −2.1). The sequence of past readings
acts like a fingerprint of *when* a month is, so with neighbouring months in training the model recalls the FC level
of that period rather than learning it from the water chemistry. These scores are not trustworthy.

## Conclusion
- **BOD can be estimated from the parameters alone** with almost no loss → use Random Forest (value) + Random Forest
  classifier with class weights (class).
- **Fecal coliform cannot be estimated reliably from the parameters alone** (honest best: R² about 0.2, macro F1 about
  0.32 on 70-30; fails on new years). Last month's FC is what made the earlier FC models work (R² 0.72).
