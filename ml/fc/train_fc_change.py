"""Fecal coliform: predict the CHANGE from last month (differencing / residual over persistence).

Target:      change = log10 FC this month - log10 FC last month (same station, within 3 months)
Prediction:  log10 FC = last month's log10 FC + predicted change
If a model learns nothing, it predicts a change near 0, i.e. "repeat last month".

Versions:
  D1: change target, inputs of version C (6 readings + station + month + last month's log10 FC)
  D2: D1 + change in each reading since last month (same station, same previous record)

Models, grids, split and metrics as in train_fc.py (time-ordered 5-fold CV grid search; holdout
2016-2023 -> 2024-2025; rolling 2018-2025; imputed FC never used as test target). New score:
  skill_vs_persistence = 1 - SSE(model) / SSE(repeat last month), on log10 FC
  (> 0 = beats "repeat last month", < 0 = worse)
Version C (level model) is retrained here with the same settings for a side-by-side comparison.
Results go to ml/fc/results_change/ (ml/fc/results/ is left untouched).

Run:  ml/.venv/Scripts/python.exe ml/fc/train_fc_change.py
"""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results_change"
OUT.mkdir(exist_ok=True)
SEED = 42
READINGS = ["DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"]
TEST_START = pd.Timestamp("2024-01-01")
MAX_LAG_MONTHS = 3
EDGES = [50, 100, 200, 400]  # DENR WQG 2021-19
STANDARD = 200
BENCH2_R2 = 0.55

# ---------------- data (same rows as train_fc.py) ----------------
df = pd.read_csv(ROOT / "dataset" / "fc_class_dataset.csv")
df["Month"] = pd.to_datetime(df["Month"])
df["logFC"] = np.log10(df["FC_MPN100mL"])
df["month_sin"] = np.sin(2 * np.pi * df.Month.dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.Month.dt.month / 12)
st = pd.get_dummies(df["Station"], prefix="stn", dtype=float)
STN = list(st.columns)
df = pd.concat([df, st], axis=1).sort_values(["Station", "Month"])
prev_m = df.groupby("Station")["Month"].shift(1)
gap = (df.Month.dt.year - prev_m.dt.year) * 12 + (df.Month.dt.month - prev_m.dt.month)
ok = gap <= MAX_LAG_MONTHS
df["logFC_last"] = df.groupby("Station")["logFC"].shift(1).where(ok)
DELTAS = []
for r in READINGS:
    df[f"d_{r}"] = (df[r] - df.groupby("Station")[r].shift(1)).where(ok)
    DELTAS.append(f"d_{r}")
df = df.dropna(subset=["logFC_last"]).sort_values(["Month", "Station"]).reset_index(drop=True)
df["dlogFC"] = df["logFC"] - df["logFC_last"]
print(f"rows used: {len(df)}")

BASE_FEATS = READINGS + STN + ["month_sin", "month_cos", "logFC_last"]
VERSIONS = {  # name: (features, predicts change?)
    "C: level (for comparison)": (BASE_FEATS, False),
    "D1: change": (BASE_FEATS, True),
    "D2: change + reading changes": (BASE_FEATS + DELTAS, True),
}
MODELS = {
    "MLR": (make_pipeline(StandardScaler(), LinearRegression()), {}),
    "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                      {"n_estimators": [200, 500], "max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}),
    "XGBoost": (XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
                {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, 4], "n_estimators": [200, 500],
                 "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}),
}
CV = TimeSeriesSplit(n_splits=5)


def fc_class(values):
    return np.digitize(values, EDGES, right=True)


def score(version, model, pred_log, data, params=""):
    y_log, y_mpn, last = data["logFC"].values, data["FC_MPN100mL"].values, data["logFC_last"].values
    pred = 10 ** pred_log
    exc_t, exc_p = y_mpn > STANDARD, pred > STANDARD
    sse_persist = ((y_log - last) ** 2).sum()
    true_up, pred_up = (y_log - last) > 0, (pred_log - last) > 0
    return {
        "version": version, "model": model,
        "R2_log": r2_score(y_log, pred_log),
        "skill_vs_persistence": 1 - ((y_log - pred_log) ** 2).sum() / sse_persist,
        "RMSE_log": np.sqrt(mean_squared_error(y_log, pred_log)), "MAE_log": mean_absolute_error(y_log, pred_log),
        "RMSE_MPN": np.sqrt(mean_squared_error(y_mpn, pred)),
        "direction_correct_%": 100 * (true_up == pred_up).mean(),
        "class_accuracy_%": 100 * (fc_class(y_mpn) == fc_class(pred)).mean(),
        "above_std_recall_%": 100 * (exc_p & exc_t).sum() / max(exc_t.sum(), 1),
        "false_alarm_%": 100 * (exc_p & ~exc_t).sum() / max((~exc_t).sum(), 1),
        "best_params": params,
    }


def fit(est, grid, X, y):
    if not grid:
        return clone(est).fit(X, y), ""
    gs = GridSearchCV(est, grid, cv=CV, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X, y)
    return gs.best_estimator_, json.dumps(gs.best_params_)


def predict_level(model, data, feats, is_change):
    p = model.predict(data[feats])
    return data["logFC_last"].values + p if is_change else p


# ---------------- holdout ----------------
train = df[df.Month < TEST_START]
test = df[(df.Month >= TEST_START) & (~df.FC_imputed)]
rows, preds, fitted = [], {}, {}
rows.append(score("Baseline", "Repeat last month", test["logFC_last"].values, test))
for vname, (feats, is_change) in VERSIONS.items():
    target = "dlogFC" if is_change else "logFC"
    for mname, (est, grid) in MODELS.items():
        model, params = fit(est, grid, train[feats], train[target].values)
        p = predict_level(model, test, feats, is_change)
        preds[(vname, mname)], fitted[(vname, mname)] = p, model
        rows.append(score(vname, mname, p, test, params))
        print(f"done {vname} / {mname}")
res = pd.DataFrame(rows)
res.round(4).to_csv(OUT / "metrics.csv", index=False)

# ---------------- rolling 2018-2025 (holdout settings refit each year) ----------------
roll = []
for yr in range(2018, 2026):
    tr = df[df.Month.dt.year < yr]
    te = df[(df.Month.dt.year == yr) & (~df.FC_imputed)]
    if len(te) == 0:
        continue
    roll.append({"year": yr, "n": len(te), **score("Baseline", "Repeat last month", te["logFC_last"].values, te)})
    for (vname, mname), model in fitted.items():
        feats, is_change = VERSIONS[vname]
        target = "dlogFC" if is_change else "logFC"
        m = clone(model).fit(tr[feats], tr[target].values)
        roll.append({"year": yr, "n": len(te), **score(vname, mname, predict_level(m, te, feats, is_change), te)})
    print("rolling year", yr, "test rows", len(te))
roll = pd.DataFrame(roll).drop(columns="best_params")
roll.round(4).to_csv(OUT / "rolling_by_year.csv", index=False)
roll_mean = roll.groupby(["version", "model"])[["R2_log", "skill_vs_persistence", "RMSE_log",
                                                  "above_std_recall_%", "false_alarm_%"]].mean()

# ---------------- predictions table ----------------
tab = test[["Month", "Station"] + READINGS + ["FC_MPN100mL", "FC_Class"]].copy()
tab["Month"] = tab.Month.dt.strftime("%Y-%m")
tab["FC_last_month"] = np.round(10 ** test["logFC_last"].values, 1)
for (v, m), p in preds.items():
    tab[f"pred {v.split(':')[0]} {m}"] = np.round(10 ** p, 1)
tab.to_csv(OUT / "predictions_holdout.csv", index=False)

# ---------------- plot ----------------
mcol = {"MLR": "#9e9e9e", "Random Forest": "#2e7d32", "XGBoost": "#1565c0"}
order = list(VERSIONS)
trained = res[res.version != "Baseline"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
panels = [("skill_vs_persistence", "Skill vs 'repeat last month' (holdout)\n> 0 = beats the baseline", trained),
          ("R2_log", "R² on log10 FC (holdout)", trained),
          ("skill_vs_persistence", "Skill vs 'repeat last month'\n(rolling 2018-2025, mean)", None)]
for ax, (met, ttl, data) in zip(axes, panels):
    if data is None:
        piv = roll_mean[met].drop(index="Baseline", level="version").unstack("model").loc[order]
    else:
        piv = data.pivot(index="version", columns="model", values=met).loc[order]
    piv.plot.bar(ax=ax, rot=0, color=[mcol[c] for c in piv.columns])
    ax.axhline(0 if met == "skill_vs_persistence" else res.loc[0, "R2_log"], ls="--", lw=1, color="red")
    if met == "R2_log":
        ax.axhline(BENCH2_R2, lw=1.5, color="orange")
    ax.set_title(ttl, fontsize=10); ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=7); ax.legend(fontsize=7)
axes[1].text(0.01, 0.03, "red dashed = repeat last month, orange = Benchmark 2", transform=axes[1].transAxes, fontsize=7)
fig.suptitle("Fecal coliform: predicting the change from last month vs predicting the level")
fig.tight_layout(); fig.savefig(OUT / "1_change_vs_level.png", dpi=150); plt.close(fig)

# ---------------- console summary ----------------
pd.set_option("display.width", 260)
cols = ["version", "model", "R2_log", "skill_vs_persistence", "RMSE_log", "direction_correct_%",
        "class_accuracy_%", "above_std_recall_%", "false_alarm_%"]
print(res[cols].round(3).to_string(index=False))
print("\nRolling 2018-2025 (mean over years):\n" + roll_mean.round(3).to_string())
print("\nRolling skill vs persistence per year:")
print(roll.pivot_table(index="year", columns=["version", "model"], values="skill_vs_persistence").round(2).to_string())
