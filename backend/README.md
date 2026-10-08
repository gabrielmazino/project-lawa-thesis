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
| BOD | XGBoost (tuned, time-ordered CV) | 6 readings + station + month + last month's BOD | R² 0.225, RMSE 1.18 mg/L, 37% better than "repeat last month" |
| Fecal coliform | Random Forest predicting the change in log10 FC | same + last month's FC | R² 0.679 (log), 5% better than "repeat last month" |

Note: on the current v4 data (outliers kept) plain MLR scores R² −0.05 for BOD, because the single real
70 mg/L value distorts a straight line; tree models are not affected. The saved models are refit on all
years (2016–2025) after testing.

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
