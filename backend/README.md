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

| Target | Model | Inputs | Test 2024–2025 (never seen) |
|---|---|---|---|
| BOD | XGBoost (tuned, time-ordered CV), trained with BOD capped at 15 mg/L | 6 readings + station + month + last month's BOD | R² 0.219, RMSE 1.18 mg/L, 37% better than "repeat last month" |
| Fecal coliform | Random Forest predicting the change in log10 FC | same + last month's FC | R² 0.679 (log), 5% better than "repeat last month" |

Note on the BOD training cap: the dataset keeps one real 70 mg/L value (Sta. Rosa, Sep 2017). Without a cap it
makes the tree models estimate about 60 mg/L for some ordinary months. For training only, BOD above 15 mg/L
(the Class D limit) is set to 15; test values are never capped. The saved models are refit on all years
(2016–2025) after testing. The same models, with every test, are in `ml/colab/` (Google Colab notebooks).

## API

| Method | Path | What it returns |
|---|---|---|
| GET | `/api/health` | status (no login needed) |
| GET | `/api/stations` | station names and available months |
| GET | `/api/overview?month=YYYY-MM` | all stations for a month + summary numbers |
| GET | `/api/trend?station=...` | monthly history of one station |
| GET | `/api/last-values?station=...&month=YYYY-MM` | last lab BOD/FC before that month (form auto-fill) |
| GET | `/api/models` | model names and test scores |
| POST | `/api/predict` | BOD and fecal coliform estimates, DENR classes, typical ranges |

Data comes from `dataset/bod_class_dataset.csv` and `dataset/fc_class_dataset.csv`.
