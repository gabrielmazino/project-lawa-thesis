"""Step 1 — Correlation analysis and feature selection (BOD and fecal coliform).

Part A  Heatmaps (Pearson + Spearman, with p-values), whole dataset — DESCRIPTIVE only.
Part B  Multicollinearity: VIF of the numeric inputs (training years).
Part C  Non-linear relevance: mutual information + Random Forest permutation importance (training years).
Part D  Composite ranking (manuscript Ch. 3 protocol): rank every input under Pearson, Spearman,
        mutual information and RF importance, normalise, average. Keep the top half (Rule A), then
        drop inputs with VIF > 5 unless they rank in the top 3. Rule B (adviser's 2020 paper): keep
        inputs with |r| >= 0.10 with the target.
Proof   Does selection help? MLR / Random Forest / XGBoost with ALL vs Rule A vs Rule B inputs,
        evaluated by rolling-origin CV over 2016-2025 (each year 2018-2025 predicted from earlier
        years only) and by the 2024-2025 holdout.

Selection uses TRAINING years only (2016-2023) so the test years cannot influence which inputs are kept.

Candidate inputs per target: the 6 readings, last month's value of that target, month, station.
"Month" = sine + cosine together; "station" = categorical. For Pearson/Spearman these two use the
multiple correlation R (month) and the correlation ratio eta (station), both between 0 and 1.

BOD uses the training cap (values above 15 mg/L set to 15 for training/selection, as in the dashboard
model). Fecal coliform is analysed on log10. Models in the proof step predict the level (not the change)
so that dropping last month's value is a fair option.

Run:  ml/.venv/Scripts/python.exe ml/improve/step1_correlation.py
"""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "step1_results"
OUT.mkdir(exist_ok=True)
SEED = 42
READINGS = ["DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"]
TRAIN_END = "2024-01-01"
BOD_CAP = 15

# ---------------- data ----------------
bod = pd.read_csv(ROOT / "dataset" / "bod_class_dataset.csv")
fc = pd.read_csv(ROOT / "dataset" / "fc_class_dataset.csv")
df = bod[["Month", "Station"] + READINGS + ["BOD_mgL", "BOD_imputed"]].merge(
    fc[["Month", "Station", "FC_MPN100mL", "FC_imputed"]], on=["Month", "Station"])
df = df.rename(columns={"BOD_mgL": "BOD", "FC_MPN100mL": "FC"})
df["Month"] = pd.to_datetime(df["Month"])
df["logFC"] = np.log10(df["FC"])
df = df.sort_values(["Station", "Month"]).reset_index(drop=True)
prev = df.groupby("Station")["Month"].shift(1)
gap = (df.Month.dt.year - prev.dt.year) * 12 + (df.Month.dt.month - prev.dt.month)
df["BOD_last"] = df.groupby("Station")["BOD"].shift(1).where(gap <= 3)
df["logFC_last"] = df.groupby("Station")["logFC"].shift(1).where(gap <= 3)
df["month_sin"] = np.sin(2 * np.pi * df.Month.dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.Month.dt.month / 12)
df["station_code"] = df["Station"].astype("category").cat.codes
STN = pd.get_dummies(df["Station"], prefix="stn", dtype=float)
df = pd.concat([df, STN], axis=1)
STN_COLS = list(STN.columns)

TARGETS = {
    "BOD": {"y": "BOD", "lag": "BOD_last", "imputed": "BOD_imputed", "label": "BOD (mg/L)"},
    "FC": {"y": "logFC", "lag": "logFC_last", "imputed": "FC_imputed", "label": "log10 fecal coliform"},
}
LABELS = {"DO": "DO", "pH": "pH", "Ammonia": "Ammonia", "Nitrate": "Nitrate", "Phosphate": "Phosphate",
          "Temperature": "Temperature", "BOD_last": "Last month's BOD", "logFC_last": "Last month's FC (log10)",
          "month": "Month (season)", "station": "Station"}


def target_values(data, t):
    y = data[TARGETS[t]["y"]].astype(float)
    return y.clip(upper=BOD_CAP) if t == "BOD" else y


# ---------------- Part A: heatmaps (whole dataset, descriptive) ----------------
heat_cols = READINGS + ["BOD", "logFC"]
heat = df.dropna(subset=heat_cols)
heat = heat[~heat.BOD_imputed & ~heat.FC_imputed]
pair_rows = []
for method, fn in [("pearson", pearsonr), ("spearman", spearmanr)]:
    mat = heat[heat_cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
    names = [c if c != "logFC" else "log10 FC" for c in heat_cols]
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title(f"{method.capitalize()} correlation, measured values 2016–2025 (n={len(heat)})")
    fig.tight_layout(); fig.savefig(OUT / f"1_heatmap_{method}.png", dpi=150); plt.close(fig)
    for a in heat_cols:
        for b in heat_cols:
            if a < b:
                r, p = fn(heat[a], heat[b])
                pair_rows.append({"method": method, "var_1": a, "var_2": b, "r": r, "p_value": p,
                                  "significant_p<0.05": p < 0.05})
pd.DataFrame(pair_rows).round(4).to_csv(OUT / "correlations_all_pairs.csv", index=False)


# ---------------- Parts B-D per target (training years only) ----------------
def multiple_r(X, y):
    return float(np.sqrt(max(LinearRegression().fit(X, y).score(X, y), 0)))


def correlation_ratio(groups, y):
    overall = y.mean()
    between = sum(len(g) * (g.mean() - overall) ** 2 for _, g in y.groupby(groups))
    return float(np.sqrt(between / ((y - overall) ** 2).sum()))


def vif(X):
    out = {}
    for c in X.columns:
        others = X.drop(columns=c)
        r2 = LinearRegression().fit(others, X[c]).score(others, X[c])
        out[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return out


def model_columns(features, t):
    cols = []
    for f in features:
        if f == "month":
            cols += ["month_sin", "month_cos"]
        elif f == "station":
            cols += STN_COLS
        else:
            cols.append(f)
    return cols


selected, rankings, vifs = {}, {}, {}
for t, cfg in TARGETS.items():
    lag = cfg["lag"]
    candidates = READINGS + [lag, "month", "station"]
    tr = df[(df.Month < TRAIN_END) & (~df[cfg["imputed"]])].dropna(subset=[lag])
    y = target_values(tr, t)

    rows = []
    for f in candidates:
        if f == "month":
            pr = sr = multiple_r(tr[["month_sin", "month_cos"]], y)
            pr_p = sr_p = np.nan
            sr = multiple_r(tr[["month_sin", "month_cos"]], y.rank())
        elif f == "station":
            pr = correlation_ratio(tr["Station"], y)
            sr = correlation_ratio(tr["Station"], y.rank())
            pr_p = sr_p = np.nan
        else:
            pr, pr_p = pearsonr(tr[f], y)
            sr, sr_p = spearmanr(tr[f], y)
        rows.append({"input": f, "pearson_r": pr, "pearson_p": pr_p, "spearman_rho": sr, "spearman_p": sr_p})
    rank = pd.DataFrame(rows).set_index("input")

    # mutual information (month as two columns summed; station as a discrete code)
    mi_X = tr[READINGS + [lag, "month_sin", "month_cos", "station_code"]]
    mi = mutual_info_regression(mi_X, y, discrete_features=[False] * (len(READINGS) + 3) + [True], random_state=SEED)
    mi = pd.Series(mi, index=mi_X.columns)
    rank["mutual_info"] = [mi["month_sin"] + mi["month_cos"] if f == "month" else
                           mi["station_code"] if f == "station" else mi[f] for f in rank.index]

    # Random Forest permutation importance: fit on 2016-2022, measure on 2023 (both training years)
    fit, val = tr[tr.Month < "2023-01-01"], tr[tr.Month >= "2023-01-01"]
    rf_cols = READINGS + [lag, "month_sin", "month_cos", "station_code"]
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=SEED, n_jobs=-1).fit(fit[rf_cols], target_values(fit, t))
    pi = permutation_importance(rf, val[rf_cols], target_values(val, t), n_repeats=30, random_state=SEED,
                                scoring="neg_root_mean_squared_error")
    pi = pd.Series(pi.importances_mean, index=rf_cols)
    rank["rf_importance"] = [pi["month_sin"] + pi["month_cos"] if f == "month" else
                             pi["station_code"] if f == "station" else pi[f] for f in rank.index]

    # composite: rank each method by strength (|r| for correlations), normalise 1 (best) .. 0 (worst), average
    n = len(rank)
    scores = pd.DataFrame(index=rank.index)
    for col in ["pearson_r", "spearman_rho", "mutual_info", "rf_importance"]:
        strength = rank[col].abs()
        scores[col] = 1 - (strength.rank(ascending=False) - 1) / (n - 1)
    rank["composite_score"] = scores.mean(axis=1)
    rank = rank.sort_values("composite_score", ascending=False)

    # VIF of the numeric inputs (readings + lag + month sin/cos)
    v = vif(tr[READINGS + [lag, "month_sin", "month_cos"]])
    vifs[t] = v
    rank["VIF"] = [max(v["month_sin"], v["month_cos"]) if f == "month" else np.nan if f == "station" else v[f]
                   for f in rank.index]

    # Rule A: top half by composite, drop VIF > 5 unless in the top 3
    top = list(rank.index[: int(np.ceil(n / 2))])
    rule_a = [f for i, f in enumerate(top) if not (rank.loc[f, "VIF"] > 5 and i >= 3)]
    # Rule B: |Pearson r| >= 0.10 (adviser's 2020 paper)
    rule_b = [f for f in rank.index if abs(rank.loc[f, "pearson_r"]) >= 0.10]
    rank["rule_A_keep"] = rank.index.isin(rule_a)
    rank["rule_B_keep"] = rank.index.isin(rule_b)
    rankings[t] = rank
    selected[t] = {"all": candidates, "rule_A": rule_a, "rule_B": rule_b}
    rank.round(4).to_csv(OUT / f"ranking_{t}.csv")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = [LABELS[f] for f in rank.index]
    colors = ["#1565c0" if k else "#bbbbbb" for k in rank["rule_A_keep"]]
    ax.barh(names[::-1], rank["composite_score"][::-1], color=colors[::-1])
    ax.axvline(0.5, ls="--", color="k", lw=1)
    ax.set_xlabel("Composite relevance score (1 = most relevant under all 4 methods)")
    ax.set_title(f"{cfg['label']}: input ranking (blue = kept by Rule A)")
    fig.tight_layout(); fig.savefig(OUT / f"2_ranking_{t}.png", dpi=150); plt.close(fig)

json.dump(selected, open(OUT / "selected_inputs.json", "w"), indent=2)

# ---------------- Proof: all vs Rule A vs Rule B ----------------
MODELS = {
    "MLR": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=300, max_depth=6, random_state=SEED, n_jobs=-1),
    "XGBoost": lambda: XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                    random_state=SEED, n_jobs=-1, verbosity=0),
}
proof = []
for t, cfg in TARGETS.items():
    data = df.dropna(subset=[cfg["lag"]])
    for set_name, feats in selected[t].items():
        cols = model_columns(feats, t)
        for mname, make in MODELS.items():
            yearly = []
            for year in range(2018, 2026):
                tr = data[data.Month.dt.year < year]
                te = data[(data.Month.dt.year == year) & (~data[cfg["imputed"]])]
                p = make().fit(tr[cols], target_values(tr, t)).predict(te[cols])
                yearly.append((r2_score(te[cfg["y"]], p), np.sqrt(mean_squared_error(te[cfg["y"]], p))))
            tr = data[data.Month < TRAIN_END]
            te = data[(data.Month >= TRAIN_END) & (~data[cfg["imputed"]])]
            p = make().fit(tr[cols], target_values(tr, t)).predict(te[cols])
            proof.append({"target": t, "inputs": set_name, "n_inputs": len(feats), "model": mname,
                          "rolling_R2_mean": np.mean([a for a, _ in yearly]),
                          "rolling_RMSE_mean": np.mean([b for _, b in yearly]),
                          "holdout_2024_25_R2": r2_score(te[cfg["y"]], p),
                          "holdout_2024_25_RMSE": np.sqrt(mean_squared_error(te[cfg["y"]], p)),
                          "inputs_used": ", ".join(feats)})
proof = pd.DataFrame(proof)
proof.round(4).to_csv(OUT / "3_all_vs_selected.csv", index=False)

# ---------------- console summary ----------------
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
for t in TARGETS:
    print(f"\n===== {t}: ranking (training years 2016-2023) =====")
    print(rankings[t][["pearson_r", "pearson_p", "spearman_rho", "mutual_info", "rf_importance",
                       "composite_score", "VIF", "rule_A_keep", "rule_B_keep"]].round(3).to_string())
    print("Rule A keeps:", selected[t]["rule_A"])
    print("Rule B keeps:", selected[t]["rule_B"])
print("\n===== Proof: all vs selected inputs =====")
print(proof.drop(columns="inputs_used").round(3).to_string(index=False))
