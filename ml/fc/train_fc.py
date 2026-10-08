"""Fecal coliform regression on log10 values (manuscript Benchmark 2: R² >= 0.55 on log FC).

Input versions (same rows for all, so scores are directly comparable):
  A: 6 readings (DO, pH, Ammonia, Nitrate, Phosphate, Temperature)
  B: A + station (one-hot) + month (sine/cosine)
  C: B + last available lab log10 FC of the same station (within 3 months)

Models: MLR, Random Forest, XGBoost, tuned by grid search with time-ordered 5-fold CV on the
training years only (grids from PROJECT_NOTES.md TO-DO #2). Baselines: "guess the training
average" and "repeat last month's value" (persistence).

Evaluation
  - holdout: train Feb 2016 - Dec 2023, test Jan 2024 - Dec 2025 (measured FC only, never imputed)
  - rolling: each year 2018-2025 tested on a model trained only on earlier years (holdout settings)
  - manuscript metrics: R² on the log10 scale (Benchmark 2), RMSE/MAE back-transformed to MPN/100 mL;
    log-scale RMSE/MAE are also reported because a few very high counts dominate original-scale RMSE
  - Benchmark 3: best ensemble vs MLR (>= 15% RMSE reduction AND >= 0.10 R² gain) plus a paired
    t-test on the per-fold CV errors
  - DENR class (WQG 2021-19) of the back-transformed prediction: accuracy, above-standard (>200) recall

Run:  ml/.venv/Scripts/python.exe ml/fc/train_fc.py
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
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
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SEED = 42
READINGS = ["DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"]
TEST_START = pd.Timestamp("2024-01-01")
MAX_LAG_MONTHS = 3
EDGES = [50, 100, 200, 400]  # DENR WQG 2021-19: A <=50, B 50-100, C 100-200, D 200-400, Failed >400
NAMES = ["A", "B", "C", "D", "Failed"]
STANDARD = 200  # Laguna Lake is Class C: above 200 MPN/100 mL = above standard
BENCH2_R2, BENCH3_RMSE_CUT, BENCH3_R2_GAIN = 0.55, 0.15, 0.10

# ---------------- data ----------------
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
df["logFC_last"] = df.groupby("Station")["logFC"].shift(1).where(gap <= MAX_LAG_MONTHS)
rows_before = len(df)
df = df.dropna(subset=["logFC_last"]).sort_values(["Month", "Station"]).reset_index(drop=True)
print(f"rows: {rows_before} -> {len(df)} after requiring a previous FC within {MAX_LAG_MONTHS} months")

VERSIONS = {
    "A: 6 readings": READINGS,
    "B: + station + month": READINGS + STN + ["month_sin", "month_cos"],
    "C: + last month's FC": READINGS + STN + ["month_sin", "month_cos", "logFC_last"],
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

train = df[df.Month < TEST_START]
test = df[(df.Month >= TEST_START) & (~df.FC_imputed)]
y_tr, y_te_log = train["logFC"].values, test["logFC"].values
y_te = test["FC_MPN100mL"].values


def fc_class(values):
    return np.digitize(values, EDGES, right=True)


def score(version, model, pred_log, y_log, y_mpn, params=""):
    pred = 10 ** pred_log
    ca, cp = fc_class(y_mpn), fc_class(pred)
    exc_t, exc_p = y_mpn > STANDARD, pred > STANDARD
    return {
        "version": version, "model": model,
        "R2_log": r2_score(y_log, pred_log),
        "RMSE_log": np.sqrt(mean_squared_error(y_log, pred_log)), "MAE_log": mean_absolute_error(y_log, pred_log),
        "RMSE_MPN": np.sqrt(mean_squared_error(y_mpn, pred)), "MAE_MPN": mean_absolute_error(y_mpn, pred),
        "within_one_class_%": 100 * (np.abs(ca - cp) <= 1).mean(), "class_accuracy_%": 100 * (ca == cp).mean(),
        "above_std_recall_%": 100 * (exc_p & exc_t).sum() / max(exc_t.sum(), 1),
        "false_alarm_%": 100 * (exc_p & ~exc_t).sum() / max((~exc_t).sum(), 1),
        "best_params": params,
    }


def cv_fold_rmse(model, X, y):
    """Per-fold RMSE (log scale) on the training years, for the paired Benchmark 3 test."""
    out = []
    for tr, va in CV.split(X):
        m = clone(model).fit(X.iloc[tr], y[tr])
        out.append(np.sqrt(mean_squared_error(y[va], m.predict(X.iloc[va]))))
    return np.array(out)


# ---------------- holdout 2024-2025 ----------------
rows, preds, fitted, folds = [], {}, {}, {}
rows.append(score("Baseline", "Guess the average", np.full(len(test), y_tr.mean()), y_te_log, y_te))
rows.append(score("Baseline", "Repeat last month", test["logFC_last"].values, y_te_log, y_te))
preds[("Baseline", "Repeat last month")] = test["logFC_last"].values

for vname, feats in VERSIONS.items():
    for mname, (est, grid) in MODELS.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=CV, scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(train[feats], y_tr)
            model, params = gs.best_estimator_, json.dumps(gs.best_params_)
        else:
            model, params = clone(est).fit(train[feats], y_tr), ""
        p = model.predict(test[feats])
        preds[(vname, mname)], fitted[(vname, mname)] = p, model
        folds[(vname, mname)] = cv_fold_rmse(model, train[feats], y_tr)
        rows.append(score(vname, mname, p, y_te_log, y_te, params))
        print(f"done {vname} / {mname}")

res = pd.DataFrame(rows)
res.round(3).to_csv(OUT / "metrics.csv", index=False)

# ---------------- benchmarks (per version) ----------------
bench = []
for vname in VERSIONS:
    r = res[res.version == vname].set_index("model")
    mlr = r.loc["MLR"]
    ens = r.loc[["Random Forest", "XGBoost"]].sort_values("R2_log", ascending=False)
    best_ens = ens.index[0]
    b = ens.loc[best_ens]
    t = ttest_rel(folds[(vname, "MLR")], folds[(vname, best_ens)])
    cut_mpn = (mlr.RMSE_MPN - b.RMSE_MPN) / mlr.RMSE_MPN
    cut_log = (mlr.RMSE_log - b.RMSE_log) / mlr.RMSE_log
    gain = b.R2_log - mlr.R2_log
    bench.append({
        "version": vname,
        "best_R2_log": r.R2_log.max(), "best_model": r.R2_log.idxmax(),
        "benchmark2_met (R2_log>=0.55)": r.R2_log.max() >= BENCH2_R2,
        "best_ensemble": best_ens,
        "RMSE_cut_vs_MLR_MPN_%": 100 * cut_mpn, "RMSE_cut_vs_MLR_log_%": 100 * cut_log, "R2_gain_vs_MLR": gain,
        "benchmark3_met (>=15% RMSE cut AND >=0.10 R2)": (cut_mpn >= BENCH3_RMSE_CUT) and (gain >= BENCH3_R2_GAIN),
        "paired_t_p (CV fold RMSE, MLR vs ensemble)": t.pvalue,
        "CV_RMSE_log_MLR": folds[(vname, "MLR")].mean(), "CV_RMSE_log_ensemble": folds[(vname, best_ens)].mean(),
    })
bench = pd.DataFrame(bench)
bench.round(4).to_csv(OUT / "benchmarks.csv", index=False)

# ---------------- rolling 2018-2025 (holdout settings refit each year) ----------------
roll = []
for yr in range(2018, 2026):
    tr = df[df.Month.dt.year < yr]
    te = df[(df.Month.dt.year == yr) & (~df.FC_imputed)]
    if len(te) == 0:
        continue
    yl, ym = te["logFC"].values, te["FC_MPN100mL"].values
    roll.append({"year": yr, "n": len(te),
                 **score("Baseline", "Repeat last month", te["logFC_last"].values, yl, ym)})
    for (vname, mname), model in fitted.items():
        if not vname.startswith("C"):
            continue
        feats = VERSIONS[vname]
        m = clone(model).fit(tr[feats], tr["logFC"].values)
        roll.append({"year": yr, "n": len(te), **score(vname, mname, m.predict(te[feats]), yl, ym)})
    print("rolling year", yr, "test rows", len(te))
roll = pd.DataFrame(roll)[["year", "n", "model", "R2_log", "RMSE_log", "RMSE_MPN", "class_accuracy_%",
                           "above_std_recall_%", "false_alarm_%"]]
roll.round(3).to_csv(OUT / "rolling_by_year.csv", index=False)
roll_mean = roll.groupby("model")[["R2_log", "RMSE_log", "class_accuracy_%", "above_std_recall_%", "false_alarm_%"]].mean()

# ---------------- best model: predictions table + saved models ----------------
trained = res[res.version != "Baseline"]
best = trained.sort_values("R2_log", ascending=False).iloc[0]
bkey = (best.version, best.model)

tab = test[["Month", "Station"] + READINGS + ["FC_MPN100mL", "FC_Class"]].copy()
tab["Month"] = tab.Month.dt.strftime("%Y-%m")
tab["FC_last_month"] = np.round(10 ** test["logFC_last"].values, 1)
for (v, m), p in preds.items():
    tag = "baseline repeat last month" if v == "Baseline" else f"{v[0]} {m}"
    tab[f"pred {tag}"] = np.round(10 ** p, 1)
tab[f"pred class {bkey[0][0]} {bkey[1]}"] = [NAMES[i] for i in fc_class(10 ** preds[bkey])]
tab.to_csv(OUT / "predictions_holdout.csv", index=False)

for mname in MODELS:
    joblib.dump({"model": fitted[("C: + last month's FC", mname)], "features": VERSIONS["C: + last month's FC"],
                 "target": "log10 fecal coliform (MPN/100 mL)", "trained_on": "2016-02 to 2023-12"},
                OUT / f"model_C_{mname.lower().replace(' ', '_')}.joblib")

# ---------------- plots ----------------
order = list(VERSIONS)
mcol = {"MLR": "#9e9e9e", "Random Forest": "#2e7d32", "XGBoost": "#1565c0"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, met, ttl in zip(axes, ["R2_log", "RMSE_log", "above_std_recall_%"],
                        ["R² on log10 FC (higher is better)", "RMSE on log10 FC (lower is better)",
                         "% of above-standard months caught (>200)"]):
    piv = trained.pivot(index="version", columns="model", values=met).loc[order]
    piv.plot.bar(ax=ax, rot=0, color=[mcol[c] for c in piv.columns])
    for _, b in res[res.version == "Baseline"].iterrows():
        ax.axhline(b[met], ls="--", lw=1, color="red" if "last" in b.model else "black")
    if met == "R2_log":
        ax.axhline(BENCH2_R2, ls="-", lw=1.5, color="orange")
    ax.set_title(ttl); ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=8); ax.legend(fontsize=7)
axes[0].text(0.01, 0.97, "dashed: black = guess average, red = repeat last month\norange = Benchmark 2 (0.55)",
             transform=axes[0].transAxes, fontsize=7, va="top")
fig.suptitle("Fecal coliform regression: input versions compared (test 2024-2025)")
fig.tight_layout(); fig.savefig(OUT / "1_version_comparison.png", dpi=150); plt.close(fig)

p = preds[bkey]
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_te_log, p, s=14, alpha=0.5, color=mcol[best.model])
lim = [min(y_te_log.min(), p.min()) - 0.2, max(y_te_log.max(), p.max()) + 0.2]
ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
for e in EDGES:
    ax.axvline(np.log10(e), color="0.85", lw=0.8); ax.axhline(np.log10(e), color="0.85", lw=0.8)
ax.set_xlabel("Actual log10 FC"); ax.set_ylabel("Predicted log10 FC")
ax.set_title(f"Best: {best.version} / {best.model}\nR² (log) {best.R2_log:.3f}  RMSE (log) {best.RMSE_log:.3f}")
fig.tight_layout(); fig.savefig(OUT / "2_best_actual_vs_predicted.png", dpi=150); plt.close(fig)

ts = pd.DataFrame({"Month": test.Month.values, "actual": y_te_log, "last": test["logFC_last"].values, "best": p})
ts = ts.groupby("Month").mean()
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(ts.index, ts["actual"], "k-o", lw=2, label="Actual")
ax.plot(ts.index, ts["last"], ":", color="red", label="Repeat last month (baseline)")
ax.plot(ts.index, ts["best"], "-s", ms=4, color=mcol[best.model], label=f"{best.version} / {best.model}")
ax.set_ylabel("log10 FC, mean of stations"); ax.tick_params(axis="x", rotation=60); ax.grid(alpha=0.3)
ax.set_title("Monthly lake-wide fecal coliform: actual vs best model"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "3_monthly_actual_vs_predicted.png", dpi=150); plt.close(fig)

# ---------------- console summary ----------------
pd.set_option("display.width", 260)
cols = ["version", "model", "R2_log", "RMSE_log", "RMSE_MPN", "MAE_MPN", "class_accuracy_%",
        "within_one_class_%", "above_std_recall_%", "false_alarm_%"]
print(res[cols].round(3).to_string(index=False))
print("\nBenchmarks:\n" + bench.round(3).to_string(index=False))
print("\nRolling 2018-2025 (mean over years, version C):\n" + roll_mean.round(3).to_string())
print(f"\ntrain rows {len(train)}, test rows {len(test)}; best: {bkey}")
