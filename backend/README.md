# Project LAWA — Backend (Flask API)

Serves the monitoring data and the soft sensor predictions to the React site (`frontend/`).
Login is handled by Supabase: every request must carry the user's Supabase token, which this API checks.
Flask never handles passwords.

## Setup (once per PC)

```powershell
cd C:\THESIS
python -m venv backend\.venv
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt
copy backend\.env.example backend\.env      # then fill in the same Supabase values as frontend\.env
```

scikit-learn is pinned to **1.8.0** so the saved models load on every PC (the laptop's Smart App Control blocks 1.9.x).

## Run

Two terminals:

```powershell
backend\.venv\Scripts\python backend\app.py          # API on http://localhost:5000
cd frontend; npm run dev                              # website on http://localhost:5173
```

## Models

`backend/models/*.joblib` are created by:

```powershell
cd backend
.venv\Scripts\python train_models.py
```

Chosen in the model-improvement steps (`ml/improve/`, see each `step*_results/SUMMARY.md`). Scores: random 70-30
split of 2016–2025, repeated with 10 seeds (mean ± SD).

| Output | Model | Inputs | Random 70-30 (10 seeds) |
|---|---|---|---|
| BOD value | Random Forest, trained with BOD capped at 15 mg/L | DO, pH, Ammonia, Nitrate, Temperature + station + month + last month's BOD | R² 0.28 ± 0.11, MAE 0.83 mg/L |
| BOD class | separate Random Forest classifier with class weights | same | accuracy 76%, macro F1 0.42, catches 70% of polluted months |
| Fecal coliform value | Random Forest predicting the change in log10 FC from last month | DO, pH + station + month + last month's FC | R² 0.72 ± 0.02 (log10) |
| Fecal coliform class | from the estimated value (DENR limits) | – | accuracy 62%, macro F1 0.53 |

Inorganic phosphate is not used (it did not improve either model). Settings were tuned by grid search in
`ml/improve/step3_models.py`. scikit-learn must be 1.9.1, the same version as the experiments: the Random Forest
classifier with class weights gives different results on 1.8.

Note on the BOD training cap: the dataset keeps one real 70 mg/L value (Sta. Rosa, Sep 2017). Without a cap it
makes the tree models estimate about 60 mg/L for some ordinary months. For training only, BOD above 15 mg/L
(the Class D limit) is set to 15; test values are never capped. The saved models are refit on all years
(2016–2025) after testing.

## API

| Method | Path | What it returns |
|---|---|---|
| GET | `/api/health` | status (no login needed) |
| GET | `/api/stations` | station names and available months |
| GET | `/api/overview?month=YYYY-MM` | all stations for a month + summary numbers |
| GET | `/api/trend?station=...` | monthly history of one station |
| GET | `/api/last-values?station=...&month=YYYY-MM` | last lab BOD/FC before that month (form auto-fill) |
| GET | `/api/models` | model names, inputs and test scores (70-30 and rolling) |
| POST | `/api/predict` | BOD and fecal coliform estimates, DENR classes (BOD: classifier + confidence), typical ranges |
| GET | `/api/bod5/synthetic?month=YYYY-MM` | synthetic BOD₅ lab sheet rows for one month (demo data) |
| POST | `/api/bod5/compute` | unseeded BOD₅ = (D1 − D2) / P for a batch of samples, DENR class, validity check, summary |

Data comes from `dataset/bod_class_dataset.csv` and `dataset/fc_class_dataset.csv`.
