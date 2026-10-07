"""Compare BOD input versions with the same time-based split.

  A: 6 readings (DO, pH, Ammonia, Nitrate, Phosphate, Temperature)
  B: A + station (one-hot) + month (sine/cosine)
  C: B + last available lab BOD of the same station (within 3 months)

Baselines: "guess the training average" and "repeat last month's BOD" (persistence).
All versions use the same rows, so the scores are directly comparable.
Train Feb 2016 - Dec 2023, test Jan 2024 - Dec 2025 (measured BOD only).

Run:  ml/.venv/Scripts/python.exe ml/bod/compare_versions.py
"""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, recall_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "dataset" / "bod_dataset.csv"
XLSX = ROOT / "dataset" / "Copy of LLDA-Data-2013-2023-Compiled.xlsx"
OUT = Path(__file__).resolve().parent / "results_versions"
OUT.mkdir(exist_ok=True)
SEED = 42
READINGS = ["DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"]
TEST_START = pd.Timestamp("2024-01-01")
MAX_LAG_MONTHS = 3
CLASS_EDGES = [3, 5, 7]  # DENR BOD classes: A <=3, B 3-5, C 5-7, >7 exceeds C

# ---------------- data ----------------
df = pd.read_csv(CSV)
ws = load_workbook(XLSX, data_only=True)["Preprocessing v3B"]
meta = []
for row in ws.iter_rows(min_row=2):
    if row[0].value is None:
        continue
    fill = row[2].fill.fgColor.rgb if row[2].fill and row[2].fill.fgColor is not None else None
    meta.append((pd.Timestamp(row[0].value), row[1].value, isinstance(fill, str) and fill.endswith("FFFF00"), row[2].value))
assert len(meta) == len(df)
df["Month"] = [m[0] for m in meta]
df["Station"] = [m[1] for m in meta]
df["BOD_imputed"] = [m[2] for m in meta]
assert np.allclose(df["BOD"], [m[3] for m in meta])

# month as a cycle
mnum = df["Month"].dt.month
df["month_sin"] = np.sin(2 * np.pi * mnum / 12)
df["month_cos"] = np.cos(2 * np.pi * mnum / 12)

# station one-hot
st = pd.get_dummies(df["Station"], prefix="stn", dtype=float)
STN = list(st.columns)
df = pd.concat([df, st], axis=1)

# last available BOD for the same station (previous row of that station, if within 3 months)
df = df.sort_values(["Station", "Month"])
prev_bod = df.groupby("Station")["BOD"].shift(1)
prev_month = df.groupby("Station")["Month"].shift(1)
gap = (df["Month"].dt.year - prev_month.dt.year) * 12 + (df["Month"].dt.month - prev_month.dt.month)
df["BOD_last"] = prev_bod.where(gap <= MAX_LAG_MONTHS)
df = df.sort_values(["Month", "Station"]).reset_index(drop=True)

rows_before = len(df)
df = df.dropna(subset=["BOD_last"]).reset_index(drop=True)  # same rows for every version
print(f"rows: {rows_before} -> {len(df)} after requiring a previous BOD within {MAX_LAG_MONTHS} months")

VERSIONS = {
    "A: 6 readings": READINGS,
    "B: + station + month": READINGS + STN + ["month_sin", "month_cos"],
    "C: + last month's BOD": READINGS + STN + ["month_sin", "month_cos", "BOD_last"],
}

train = df[df["Month"] < TEST_START]
test = df[(df["Month"] >= TEST_START) & (~df["BOD_imputed"])]
ytr, yte = train["BOD"], test["BOD"].values

MODELS = {
    "MLR": (make_pipeline(StandardScaler(), LinearRegression()), {}),
    "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                      {"n_estimators": [300], "max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10],
                       "min_samples_leaf": [1, 3]}),
    "XGBoost": (XGBRegressor(random_state=SEED, n_jobs=-1),
                {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, 4], "n_estimators": [200, 500],
                 "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]}),
}


def score(name, version, p, params=""):
    e = np.abs(p - yte)
    hi = yte >= 4
    ca = np.digitize(yte, CLASS_EDGES, right=True)
    cp = np.digitize(p, CLASS_EDGES, right=True)
    return {
        "version": version, "model": name,
        "R2": r2_score(yte, p), "RMSE": np.sqrt(mean_squared_error(yte, p)), "MAE": mean_absolute_error(yte, p),
        "within_0.5_%": 100 * (e <= 0.5).mean(), "within_1.0_%": 100 * (e <= 1.0).mean(),
        "within_1.0_when_BOD>=4_%": 100 * (e[hi] <= 1.0).mean(),
        "class_accuracy_%": 100 * (ca == cp).mean(),
        "recall_B_or_worse_%": 100 * recall_score(ca >= 1, cp >= 1, zero_division=0),
        "best_params": params,
    }


rows, preds = [], {}
rows.append(score("Guess the average", "Baseline", np.full(len(yte), ytr.mean())))
preds[("Baseline", "Repeat last month")] = test["BOD_last"].values
rows.append(score("Repeat last month's BOD", "Baseline", test["BOD_last"].values))

cv = TimeSeriesSplit(n_splits=5)
for vname, feats in VERSIONS.items():
    for mname, (est, grid) in MODELS.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(train[feats], ytr)
            model, params = gs.best_estimator_, json.dumps(gs.best_params_)
        else:
            model, params = clone(est).fit(train[feats], ytr), ""
        p = model.predict(test[feats])
        preds[(vname, mname)] = p
        rows.append(score(mname, vname, p, params))
        print(f"done {vname} / {mname}")

res = pd.DataFrame(rows).round(3)
res.to_csv(OUT / "version_comparison.csv", index=False)

# best = lowest RMSE among trained models
trained = res[res.version != "Baseline"]
best = trained.sort_values("RMSE").iloc[0]
bkey = (best.version, best.model)

# predictions table (test set)
tab = test[["Month", "Station"] + READINGS + ["BOD_last", "BOD"]].copy()
tab["Month"] = tab["Month"].dt.strftime("%Y-%m")
tab = tab.rename(columns={"BOD": "BOD_actual", "BOD_last": "BOD_last_month"})
for (v, m), p in preds.items():
    if v == "Baseline":
        continue
    tab[f"pred {v[0]} {m}"] = np.round(p, 3)
tab.to_csv(OUT / "predictions_all_versions.csv", index=False)

# ---------------- plots ----------------
order = list(VERSIONS)
mcol = {"MLR": "#9e9e9e", "Random Forest": "#2e7d32", "XGBoost": "#1565c0"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, met, ttl in zip(axes, ["R2", "RMSE", "within_1.0_when_BOD>=4_%"],
                        ["R² (higher is better)", "RMSE mg/L (lower is better)", "% within ±1 mg/L when BOD ≥ 4"]):
    piv = trained.pivot(index="version", columns="model", values=met).loc[order]
    piv.plot.bar(ax=ax, rot=0, color=[mcol[c] for c in piv.columns])
    for _, b in res[res.version == "Baseline"].iterrows():
        ax.axhline(b[met], ls="--", lw=1, color="red" if "last" in b.model else "black")
    ax.set_title(ttl); ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=8); ax.legend(fontsize=7)
axes[0].text(0.01, 0.97, "dashed: black = guess average, red = repeat last month", transform=axes[0].transAxes,
             fontsize=7, va="top")
fig.suptitle("BOD input versions compared (test 2024-2025)")
fig.tight_layout(); fig.savefig(OUT / "1_version_comparison.png", dpi=150); plt.close(fig)

p = preds[bkey]
fig, ax = plt.subplots(figsize=(6, 6))
jit = np.random.default_rng(SEED).uniform(-0.12, 0.12, len(yte))
ax.scatter(yte + jit, p, s=14, alpha=0.5, color=mcol[best.model])
lim = [0, max(yte.max(), p.max()) + 0.5]
ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual BOD (mg/L)"); ax.set_ylabel("Predicted BOD (mg/L)")
ax.set_title(f"Best: {best.version} / {best.model}\nR² {best.R2:.3f}  RMSE {best.RMSE:.3f}  MAE {best.MAE:.3f}")
fig.tight_layout(); fig.savefig(OUT / "2_best_actual_vs_predicted.png", dpi=150); plt.close(fig)

ts = tab.groupby("Month").agg(actual=("BOD_actual", "mean"), last=("BOD_last_month", "mean"))
for v in order:
    m = trained[trained.version == v].sort_values("RMSE").iloc[0].model
    ts[f"{v[0]} ({m})"] = tab.groupby("Month")[f"pred {v[0]} {m}"].mean()
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(ts.index, ts["actual"], "k-o", lw=2, label="Actual")
ax.plot(ts.index, ts["last"], ":", color="red", label="Repeat last month (baseline)")
for c, colr in zip(ts.columns[2:], ["#9e9e9e", "#2e7d32", "#1565c0"]):
    ax.plot(ts.index, ts[c], "-s", ms=4, color=colr, label=f"Version {c}")
ax.set_ylabel("BOD (mg/L), mean of stations"); ax.tick_params(axis="x", rotation=60)
ax.set_title("Monthly lake-wide BOD: actual vs best model of each version"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "3_monthly_by_version.png", dpi=150); plt.close(fig)

pd.set_option("display.width", 250)
cols = ["version", "model", "R2", "RMSE", "MAE", "within_1.0_%", "within_1.0_when_BOD>=4_%", "class_accuracy_%", "recall_B_or_worse_%"]
print(res[cols].to_string(index=False))
print(f"\ntrain rows {len(train)}, test rows {len(test)}; best: {bkey}")
