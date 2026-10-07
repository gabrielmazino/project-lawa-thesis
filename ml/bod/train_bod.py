"""BOD soft sensor: train and compare Multiple Linear Regression, Random Forest and XGBoost.

Data: dataset/bod_dataset.csv (DO, pH, Ammonia, Nitrate, Phosphate, Temperature -> BOD),
built from the "Preprocessing v3B" sheet. The workbook is read only to get each row's month
and to know which BOD values were imputed (yellow cells), so they can be kept out of testing.

Two evaluations:
  A. Time-based (main): train Feb 2016 - Dec 2023, test Jan 2024 - Dec 2025, tuned with TimeSeriesSplit.
  B. Random 70:30 (as written in the manuscript), tuned with shuffled 5-fold CV.

Run:  ml/.venv/Scripts/python.exe ml/bod/train_bod.py
"""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (classification_report, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "dataset" / "bod_dataset.csv"
XLSX = ROOT / "dataset" / "Copy of LLDA-Data-2013-2023-Compiled.xlsx"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SEED = 42
FEATURES = ["DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"]
TARGET = "BOD"
TEST_START = pd.Timestamp("2024-01-01")

# DENR DAO 2016-08 BOD classes (mg/L), as printed in the LLDA report legends
CLASS_EDGES = [3, 5, 7]
CLASS_NAMES = ["A (<=3)", "B (3-5]", "C (5-7]", "Exceeds C (>7)"]


def bod_class(v):
    return np.digitize(np.asarray(v), CLASS_EDGES, right=True)


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def metrics(y, p):
    return {"RMSE": rmse(y, p), "MAE": float(mean_absolute_error(y, p)), "R2": float(r2_score(y, p))}


# ---------------- data ----------------
df = pd.read_csv(CSV)

ws = load_workbook(XLSX, read_only=False, data_only=True)["Preprocessing v3B"]
months, stations, imputed, bod_check = [], [], [], []
for row in ws.iter_rows(min_row=2):
    if row[0].value is None:
        continue
    months.append(pd.Timestamp(row[0].value))
    stations.append(row[1].value)
    fill = row[2].fill.fgColor.rgb if row[2].fill and row[2].fill.fgColor is not None else None
    imputed.append(isinstance(fill, str) and fill.endswith("FFFF00"))
    bod_check.append(row[2].value)

assert len(months) == len(df), f"row mismatch: workbook {len(months)} vs csv {len(df)}"
assert np.allclose(df[TARGET].values, np.array(bod_check, dtype=float)), "BOD column does not match the workbook"
df["Month"] = months
df["Station"] = stations
df["BOD_imputed"] = imputed

X, y = df[FEATURES], df[TARGET]

# ---------------- models ----------------
MODELS = {
    "MLR": (make_pipeline(StandardScaler(), LinearRegression()), {}),
    "Random Forest": (
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
    ),
    "XGBoost": (
        XGBRegressor(random_state=SEED, n_jobs=-1, objective="reg:squarederror"),
        {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, 4], "n_estimators": [200, 500],
         "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]},
    ),
}


def run(split_name, train_idx, test_idx, cv):
    """Tune on the training part only, then score once on the held-out test part."""
    Xtr, ytr = X.loc[train_idx], y.loc[train_idx]
    # test only on measured BOD (imputed BOD values are estimates, not ground truth)
    test_idx = [i for i in test_idx if not df.at[i, "BOD_imputed"]]
    Xte, yte = X.loc[test_idx], y.loc[test_idx]

    rows, preds, fitted, fold_rmse = [], {}, {}, {}
    for name, (est, grid) in MODELS.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(Xtr, ytr)
            best, params = gs.best_estimator_, gs.best_params_
        else:
            best, params = clone(est).fit(Xtr, ytr), {}
        cvres = cross_validate(clone(best), Xtr, ytr, cv=cv,
                               scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"))
        fold_rmse[name] = -cvres["test_neg_root_mean_squared_error"]
        p = best.predict(Xte)
        preds[name], fitted[name] = p, best
        m = metrics(yte, p)
        rows.append({
            "split": split_name, "model": name,
            "CV_RMSE_mean": fold_rmse[name].mean(), "CV_RMSE_sd": fold_rmse[name].std(),
            "CV_MAE_mean": -cvres["test_neg_mean_absolute_error"].mean(),
            "CV_R2_mean": cvres["test_r2"].mean(), "CV_R2_sd": cvres["test_r2"].std(),
            "Test_RMSE": m["RMSE"], "Test_MAE": m["MAE"], "Test_R2": m["R2"],
            "n_train": len(Xtr), "n_test": len(Xte), "best_params": json.dumps(params),
        })
    return pd.DataFrame(rows), preds, fitted, fold_rmse, test_idx


# A. time-based split
train_A = df.index[df["Month"] < TEST_START].tolist()
test_A = df.index[df["Month"] >= TEST_START].tolist()
res_A, preds_A, fitted_A, folds_A, test_A = run("A: time-based (train 2016-2023, test 2024-2025)",
                                               train_A, test_A, TimeSeriesSplit(n_splits=5))

# B. random 70:30 as in the manuscript
train_B, test_B = train_test_split(df.index.tolist(), test_size=0.30, random_state=SEED, shuffle=True)
res_B, preds_B, fitted_B, folds_B, test_B = run("B: random 70:30 (manuscript)",
                                               sorted(train_B), sorted(test_B), KFold(5, shuffle=True, random_state=SEED))

results = pd.concat([res_A, res_B], ignore_index=True)

# baseline row: always predict the training mean (anything useful must beat this)
for label, tr, te in [(res_A.split[0], train_A, test_A), (res_B.split[0], train_B, test_B)]:
    m = metrics(y.loc[te], np.full(len(te), y.loc[tr].mean()))
    results.loc[len(results)] = {"split": label, "model": "Mean baseline (no model)",
                                 "Test_RMSE": m["RMSE"], "Test_MAE": m["MAE"], "Test_R2": m["R2"],
                                 "n_train": len(tr), "n_test": len(te), "best_params": ""}
results = results.round(4)
results.to_csv(OUT / "metrics.csv", index=False)

# ---------------- benchmarks (manuscript Phase 5) ----------------
bench = []
for label, res, folds in [("A", res_A, folds_A), ("B", res_B, folds_B)]:
    mlr = res[res.model == "MLR"].iloc[0]
    for ens in ["Random Forest", "XGBoost"]:
        r = res[res.model == ens].iloc[0]
        red = 100 * (mlr.Test_RMSE - r.Test_RMSE) / mlr.Test_RMSE
        gain = r.Test_R2 - mlr.Test_R2
        t_p = stats.ttest_rel(folds["MLR"], folds[ens]).pvalue
        w_p = stats.wilcoxon(folds["MLR"], folds[ens]).pvalue
        bench.append({"split": label, "model": ens,
                      "Benchmark1_R2>=0.60": bool(r.Test_R2 >= 0.60), "Test_R2": round(r.Test_R2, 4),
                      "RMSE_reduction_vs_MLR_%": round(red, 2), "R2_gain_vs_MLR": round(gain, 4),
                      "Benchmark3_met": bool(red >= 15 and gain >= 0.10),
                      "paired_t_p(CV folds)": round(t_p, 4), "wilcoxon_p(CV folds)": round(w_p, 4)})
bench = pd.DataFrame(bench)
bench.to_csv(OUT / "benchmarks.csv", index=False)

# ---------------- predictions table ----------------
pred_tab = df.loc[test_A, ["Month", "Station"] + FEATURES + [TARGET]].copy()
pred_tab = pred_tab.rename(columns={TARGET: "BOD_actual"})
for name, p in preds_A.items():
    pred_tab[f"pred_{name}"] = np.round(p, 3)
    pred_tab[f"error_{name}"] = np.round(p - pred_tab["BOD_actual"], 3)
pred_tab["Month"] = pred_tab["Month"].dt.strftime("%Y-%m")
pred_tab.to_csv(OUT / "predictions_test_timesplit.csv", index=False)

# ---------------- best model (time-based) ----------------
best_name = res_A.sort_values("Test_RMSE").iloc[0].model
best_pred = preds_A[best_name]
yA = y.loc[test_A].values

# per-station metrics
st_rows = []
for st, g in pred_tab.groupby("Station"):
    for name in MODELS:
        m = metrics(g["BOD_actual"], g[f"pred_{name}"]) if len(g) > 1 else {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
        st_rows.append({"Station": st, "model": name, "n": len(g), **{k: round(v, 3) for k, v in m.items()}})
pd.DataFrame(st_rows).to_csv(OUT / "per_station_metrics.csv", index=False)

# compliance classification (DENR classes)
ca, cp = bod_class(yA), bod_class(best_pred)
labels = list(range(len(CLASS_NAMES)))
cm = confusion_matrix(ca, cp, labels=labels)
report = classification_report(ca, cp, labels=labels, target_names=CLASS_NAMES, zero_division=0)
(OUT / "compliance_report.txt").write_text(
    f"Best model (time-based split): {best_name}\nRows: {len(yA)} (measured BOD only)\n\n"
    f"Confusion matrix (rows = actual class, cols = predicted class)\n"
    f"{pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string()}\n\n{report}")

# ---------------- plots ----------------
colors = {"MLR": "#7a7a7a", "Random Forest": "#2e7d32", "XGBoost": "#1565c0"}

# 1. actual vs predicted scatter, one panel per model
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
lim = [0, max(yA.max(), max(p.max() for p in preds_A.values())) + 0.5]
for ax, (name, p) in zip(axes, preds_A.items()):
    jitter = np.random.default_rng(SEED).uniform(-0.12, 0.12, len(yA))  # BOD is mostly whole numbers
    ax.scatter(yA + jitter, p, s=14, alpha=0.5, color=colors[name])
    ax.plot(lim, lim, "k--", lw=1, label="perfect prediction")
    m = metrics(yA, p)
    ax.set_title(f"{name}\nR² {m['R2']:.3f}  RMSE {m['RMSE']:.3f}  MAE {m['MAE']:.3f}")
    ax.set_xlabel("Actual BOD (mg/L)")
    ax.set_xlim(lim); ax.set_ylim(lim)
axes[0].set_ylabel("Predicted BOD (mg/L)")
axes[0].legend(loc="upper left")
fig.suptitle("Actual vs predicted BOD, test set 2024-2025 (time-based split)")
fig.tight_layout(); fig.savefig(OUT / "1_actual_vs_predicted.png", dpi=150); plt.close(fig)

# 2. lake-wide monthly mean: actual vs predicted over time
ts = pred_tab.groupby("Month")[["BOD_actual"] + [f"pred_{n}" for n in MODELS]].mean()
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(ts.index, ts["BOD_actual"], "k-o", lw=2, label="Actual (mean of stations)")
for n in MODELS:
    ax.plot(ts.index, ts[f"pred_{n}"], "-s", ms=4, color=colors[n], label=f"Predicted - {n}")
ax.set_ylabel("BOD (mg/L)"); ax.set_title("Monthly lake-wide BOD: actual vs predicted (test period)")
ax.tick_params(axis="x", rotation=60); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "2_monthly_actual_vs_predicted.png", dpi=150); plt.close(fig)

# 3. residuals of the best model
res_best = best_pred - yA
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].hist(res_best, bins=30, color=colors[best_name]); axes[0].axvline(0, color="k", ls="--")
axes[0].set_title(f"Residuals (predicted - actual), {best_name}"); axes[0].set_xlabel("mg/L")
axes[1].scatter(best_pred, res_best, s=12, alpha=0.5, color=colors[best_name]); axes[1].axhline(0, color="k", ls="--")
axes[1].set_xlabel("Predicted BOD (mg/L)"); axes[1].set_ylabel("Residual (mg/L)"); axes[1].set_title("Residuals vs predicted")
fig.tight_layout(); fig.savefig(OUT / "3_residuals_best_model.png", dpi=150); plt.close(fig)

# 4. model comparison (test RMSE / MAE / R2, both splits)
comp = results[results.model != "Mean baseline (no model)"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, met in zip(axes, ["Test_RMSE", "Test_MAE", "Test_R2"]):
    piv = comp.pivot(index="model", columns="split", values=met).loc[list(MODELS)]
    piv.plot.bar(ax=ax, rot=0, color=["#1565c0", "#90caf9"])
    ax.set_title(met.replace("_", " ")); ax.legend(fontsize=7)
    if met == "Test_R2":
        ax.axhline(0.60, color="red", ls="--", lw=1); ax.text(-0.4, 0.61, "Benchmark 1 (0.60)", color="red", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "4_model_comparison.png", dpi=150); plt.close(fig)

# 5. feature importance: permutation importance (all models) + XGBoost SHAP (built-in TreeSHAP)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
imp_rows = []
for ax, (name, model) in zip(axes, fitted_A.items()):
    pi = permutation_importance(model, X.loc[test_A], y.loc[test_A], n_repeats=30, random_state=SEED,
                                scoring="neg_root_mean_squared_error")
    order = np.argsort(pi.importances_mean)
    ax.barh(np.array(FEATURES)[order], pi.importances_mean[order], xerr=pi.importances_std[order], color=colors[name])
    ax.set_title(f"{name}: permutation importance\n(increase in RMSE when shuffled)")
    for f, mval, s in zip(FEATURES, pi.importances_mean, pi.importances_std):
        imp_rows.append({"model": name, "feature": f, "importance_mean": round(mval, 4), "importance_sd": round(s, 4)})
fig.tight_layout(); fig.savefig(OUT / "5_permutation_importance.png", dpi=150); plt.close(fig)
pd.DataFrame(imp_rows).to_csv(OUT / "permutation_importance.csv", index=False)

import xgboost as xgb
contrib = fitted_A["XGBoost"].get_booster().predict(xgb.DMatrix(X.loc[test_A]), pred_contribs=True)[:, :-1]
order = np.argsort(np.abs(contrib).mean(0))
fig, ax = plt.subplots(figsize=(8, 4.5))
for k, j in enumerate(order):
    vals = X.loc[test_A].iloc[:, j].values
    norm = (vals - vals.min()) / (np.ptp(vals) or 1)
    ax.scatter(contrib[:, j], np.full(len(vals), k) + np.random.default_rng(j).uniform(-0.2, 0.2, len(vals)),
               c=norm, cmap="coolwarm", s=10, alpha=0.7)
ax.set_yticks(range(len(order))); ax.set_yticklabels(np.array(FEATURES)[order])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("SHAP value (effect on predicted BOD, mg/L)  -  colour: low (blue) to high (red) feature value")
ax.set_title("XGBoost SHAP summary, test set")
fig.tight_layout(); fig.savefig(OUT / "6_xgboost_shap_summary.png", dpi=150); plt.close(fig)

# 7. confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(labels); ax.set_yticks(labels)
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right"); ax.set_yticklabels(CLASS_NAMES)
for i in labels:
    for j in labels:
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
ax.set_title(f"DENR BOD class, {best_name} (test 2024-2025)")
fig.tight_layout(); fig.savefig(OUT / "7_compliance_confusion_matrix.png", dpi=150); plt.close(fig)

# ---------------- save models (refit on all data for the dashboard later) ----------------
for name, model in fitted_A.items():
    joblib.dump(clone(model).fit(X, y), OUT / f"model_{name.replace(' ', '_').lower()}.joblib")

pd.set_option("display.width", 200)
print(results[["split", "model", "CV_RMSE_mean", "CV_R2_mean", "Test_RMSE", "Test_MAE", "Test_R2", "n_train", "n_test"]].to_string(index=False))
print("\n", bench.to_string(index=False))
print("\nBest (time-based):", best_name)
print((OUT / "compliance_report.txt").read_text())
print("BOD distribution (all rows):", y.describe().round(2).to_dict())
print("BOD value counts:", y.round(1).value_counts().sort_index().to_dict())
