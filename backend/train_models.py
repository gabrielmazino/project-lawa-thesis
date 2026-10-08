"""Train the two models used by the website and save them to backend/models/.

Chosen from the thesis experiments (see ml/PROJECT_NOTES.md):
  BOD            - XGBoost, inputs: 6 readings + station + month + last month's BOD (version C inputs).
                   On the current v4 data (outliers kept) XGBoost beat MLR (R² ~0.22 vs -0.05: the single
                   real 70 mg/L value distorts the straight-line MLR). Settings tuned here by grid search
                   with time-ordered 5-fold CV on the training years only.
  Fecal coliform - Random Forest predicting the CHANGE in log10 FC from last month
                   (ml/fc/train_fc_change.py, version D1: the only model that beat "repeat last month")
                   settings from its grid search: max_depth 3, min_samples_split 2, n_estimators 500

Each model is first checked on a holdout (train 2016-2023, test 2024-2025, measured values only)
and those scores are stored with the model, so the website can show them. The saved model is then
refit on ALL years, so it uses the most recent data.

Run:  backend/.venv/Scripts/python.exe backend/train_models.py
"""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from lawa.data import READINGS, add_last_values, build_features, load_records

OUT = Path(__file__).resolve().parent / "models"
OUT.mkdir(exist_ok=True)
SEED = 42
TEST_START = "2024-01-01"

df = add_last_values(load_records())
stations = sorted(df["Station"].unique())
input_ranges = {r: [float(df[r].min()), float(df[r].max())] for r in READINGS}


def holdout(data, imputed_col):
    train = data[data.Month < TEST_START]
    test = data[(data.Month >= TEST_START) & (~data[imputed_col])]
    return train, test


def scores(y, pred, last):
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "baseline_repeat_last_month_r2": float(r2_score(y, last)),
        "skill_vs_repeat_last_month": float(1 - ((y - pred) ** 2).sum() / ((y - last) ** 2).sum()),
    }


def save(name, model, features, meta):
    joblib.dump({"model": model, "features": features, "stations": stations, "input_ranges": input_ranges,
                 "trained_on": f"{df.Month.min():%Y-%m} to {df.Month.max():%Y-%m}",
                 "created": date.today().isoformat(), **meta}, OUT / f"{name}.joblib")


# ---------------- BOD: XGBoost + last month's BOD ----------------
bod = df.dropna(subset=["BOD_last"]).sort_values(["Month", "Station"])
train, test = holdout(bod, "BOD_imputed")
X_tr, X_te = build_features(train, stations, "BOD_last"), build_features(test, stations, "BOD_last")
grid = GridSearchCV(XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
                    {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, 4], "n_estimators": [200, 500],
                     "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]},
                    cv=TimeSeriesSplit(n_splits=5), scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_tr, train["BOD"])
bod_scores = scores(test["BOD"].values, grid.best_estimator_.predict(X_te), test["BOD_last"].values)
mlr = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, train["BOD"])
bod_scores["mlr_r2_same_data"] = float(r2_score(test["BOD"], mlr.predict(X_te)))
X_all = build_features(bod, stations, "BOD_last")
bod_model = XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0, **grid.best_params_).fit(X_all, bod["BOD"])
save("bod_model", bod_model, list(X_all.columns), {
    "target": "BOD (mg/L)", "algorithm": "XGBoost (gradient boosting)", "settings": grid.best_params_,
    "predicts": "level", "holdout": {"period": "trained 2016-2023, tested 2024-2025", "n_test": len(test), **bod_scores},
})
print(f"BOD  settings: {grid.best_params_}")
print(f"BOD  holdout (n={len(test)}): " + ", ".join(f"{k} {v:.3f}" for k, v in bod_scores.items()))

# ---------------- Fecal coliform: Random Forest on the change in log10 FC ----------------
def rf():
    return RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_split=2, random_state=SEED, n_jobs=-1)


fc = df.dropna(subset=["FC_last"]).copy()
fc["dlog"] = np.log10(fc["FC"]) - np.log10(fc["FC_last"])
train, test = holdout(fc, "FC_imputed")
X_tr, X_te = build_features(train, stations, "FC_last"), build_features(test, stations, "FC_last")
fc_model = rf().fit(X_tr, train["dlog"])
y_log, last_log = np.log10(test["FC"].values), np.log10(test["FC_last"].values)
fc_scores = scores(y_log, last_log + fc_model.predict(X_te), last_log)
X_all = build_features(fc, stations, "FC_last")
fc_model = rf().fit(X_all, fc["dlog"])
save("fc_model", fc_model, list(X_all.columns), {
    "target": "log10 fecal coliform (MPN/100 mL)", "algorithm": "Random Forest (predicts the change from last month)",
    "predicts": "change_log10", "holdout": {"period": "trained 2016-2023, tested 2024-2025", "n_test": len(test),
                                            "scale": "log10", **fc_scores},
})
print(f"FC   holdout (n={len(test)}, log10): " + ", ".join(f"{k} {v:.3f}" for k, v in fc_scores.items()))
print(f"saved to {OUT}")
