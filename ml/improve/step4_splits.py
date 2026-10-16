"""Step 4 — Split sensitivity: do the results hold up under different ways of splitting the data?

Data:     whole 2016-2025 dataset. Filled (imputed) values are only ever used for training.
Tasks:    BOD value (mg/L), FC log10 (predicting the change from last month), BOD 4-class, FC 5-class.
Models:   the six from Step 3 with the settings tuned there (step3_results/tuned_settings.json) + baselines.
Splits:   random 60-40 / 70-30 / 80-20 (10 seeds each, stratified by station and year)
          10-fold cross-validation (stratified by station)
          chronological 60-40 / 70-30 / 80-20 (oldest months train, newest months test)
          rolling year by year (each year 2018-2025 predicted from earlier years)
Output:   mean and spread (standard deviation) of every score per split, and how often each model is in the top 3.

Run:  ml/.venv/Scripts/python.exe ml/improve/step4_splits.py
"""
from pathlib import Path
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from common import (CLASS_SCHEMES, IMPUTED, PROBLEM_FROM, SEED, SELECTED, class_metrics, columns, folds, load,
                    random_split, reg_metrics, to_class)
from models import (LSTMModel, WEIGHTED, clf_models, fit_clf, fit_target, level, persist, predict_clf, reg_models,
                    sequence_index)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "step4_results"
OUT.mkdir(exist_ok=True)
TUNED = json.loads((HERE / "step3_results" / "tuned_settings.json").read_text())
T0 = time.time()

TASKS = {   # name -> (kind, target, Step 3 key, FC change?)
    "BOD (mg/L)": ("reg", "BOD", "BOD (mg/L)", False),
    "FC log10 (change model)": ("reg", "FC", "FC log10 change", True),
    "BOD 4-class": ("clf", "BOD", "BOD 4-class", None),
    "FC 5-class": ("clf", "FC", "FC 5-class", None),
}
MODELS = ["MLR / Logistic Regression", "Random Forest", "XGBoost", "SVM", "kNN", "LSTM"]
METHODS = ["random 60-40", "random 70-30", "random 80-20", "10-fold CV",
           "chronological 60-40", "chronological 70-30", "chronological 80-20", "rolling year-by-year"]


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def splits(df, imputed_col):
    real, filled = df[~df[imputed_col]], df[df[imputed_col]]
    for test in (0.4, 0.3, 0.2):
        for s in range(10):
            tr, te = random_split(df, imputed_col, test, SEED + s)
            yield f"random {round(100 * (1 - test))}-{round(100 * test)}", s, tr, te
    for k, (a, b) in enumerate(StratifiedKFold(10, shuffle=True, random_state=SEED).split(real, real.Station)):
        yield "10-fold CV", k, pd.concat([real.iloc[a], filled]), real.iloc[b]
    share = real.groupby("Month").size().cumsum() / len(real)
    for test in (0.4, 0.3, 0.2):
        cut = share.index[np.searchsorted(share.to_numpy(), 1 - test)]
        yield (f"chronological {round(100 * (1 - test))}-{round(100 * test)}", str(cut.date())[:7],
               df[df.Month <= cut], real[real.Month > cut])
    for split, year, tr, te in folds(df, imputed_col):
        if split == "rolling":
            yield "rolling year-by-year", year, tr, te


def tuned_model(kind, key, name, weighted):
    sk_name = ("MLR" if kind == "reg" else "Logistic Regression") if name == "MLR / Logistic Regression" else name
    est, _ = (reg_models() if kind == "reg" else clf_models(weighted))[sk_name]
    est = clone(est).set_params(**TUNED.get(f"{key} | {sk_name}", {}))
    if name in ("Random Forest", "XGBoost"):
        est.set_params(n_jobs=-1)
    return est


df, STN = load()
SEQ = {L: sequence_index(df, L) for L in (3, 6)}
SEEDS3 = (SEED, SEED + 1, SEED + 2)
runs = []
for task, (kind, t, key, change) in TASKS.items():
    cols = columns(SELECTED[t], STN)
    F = df[cols].to_numpy(dtype=float)
    weighted = WEIGHTED[t]
    lcfg = TUNED[f"{key} | LSTM"]
    if kind == "clf":
        cs = CLASS_SCHEMES[key]
        lastcol, pf = ("BOD_last" if t == "BOD" else "FC_last"), PROBLEM_FROM[key]
    for method, rep, tr, te in splits(df, IMPUTED[t]):
        if kind == "reg":
            ytr, y, base = fit_target(tr, t, change), level(te, t), persist(te, t)
            back = (lambda p: p + base) if change else (lambda p: p)
            preds = {"Baseline: repeat last month": base,
                     "Baseline: training average": np.full(len(te), level(tr, t).mean())}
            for name in MODELS[:-1]:
                m = tuned_model(kind, key, name, weighted).fit(tr[cols], ytr)
                preds[name] = back(m.predict(te[cols]))
            lstm = LSTMModel(F, SEQ, "reg", lcfg["hidden"], lcfg["L"], seeds=SEEDS3).fit(tr.index.to_numpy(), ytr)
            preds["LSTM"] = back(lstm.predict(te.index.to_numpy()))
            for name, p in preds.items():
                runs.append({"task": task, "method": method, "rep": rep, "model": name, "n_train": len(tr),
                             "n_test": len(te), **reg_metrics(y, p, base)})
        else:
            ytr, y = to_class(tr[cs["value"]], cs["edges"]), to_class(te[cs["value"]], cs["edges"])
            preds = {"Baseline: same class as last month": to_class(te[lastcol], cs["edges"]),
                     "Baseline: always most common class": np.full(len(te), np.bincount(ytr).argmax())}
            for name in MODELS[:-1]:
                fitted = fit_clf(tuned_model(kind, key, name, weighted),
                                 "XGBoost" if name == "XGBoost" else name, weighted, tr[cols], ytr)
                preds[name] = predict_clf(fitted, te[cols])
            preds["LSTM"] = LSTMModel(F, SEQ, "clf", lcfg["hidden"], lcfg["L"], weighted, SEEDS3).fit(
                tr.index.to_numpy(), ytr).predict(te.index.to_numpy())
            for name, p in preds.items():
                runs.append({"task": task, "method": method, "rep": rep, "model": name, "n_train": len(tr),
                             "n_test": len(te), **class_metrics(y, p, pf)})
        if rep in (0, 2018) or method.startswith("chrono"):
            log(f"{task}: {method} {rep}")
    log(f"{task} done")

R = pd.DataFrame(runs)
R.round(4).to_csv(OUT / "all_runs.csv", index=False)

MAIN = {"reg": "R2", "clf": "macro_F1"}
SCORES = {"reg": ["R2", "RMSE", "MAE", "skill_vs_last_month"],
          "clf": ["accuracy", "macro_F1", "balanced_accuracy", "problem_caught_%", "false_alarms_%"]}
summary, top3 = [], []
for task, (kind, *_rest) in TASKS.items():
    r = R[R.task == task]
    g = r.groupby(["method", "model"])[SCORES[kind]]
    s = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(g.size().rename("repeats")).reset_index()
    s.insert(0, "task", task)
    summary.append(s)
    # how often each model is in the top 3 (models only, by the main score) within each split
    m = r[~r.model.str.startswith("Baseline")].copy()
    m["rank"] = m.groupby(["method", "rep"])[MAIN[kind]].rank(ascending=False, method="min")
    t3 = (m.assign(top3=m["rank"] <= 3).groupby(["method", "model"]).agg(top3_share=("top3", "mean"),
                                                                           mean_rank=("rank", "mean")).reset_index())
    t3.insert(0, "task", task)
    top3.append(t3)
S = pd.concat(summary)
T3 = pd.concat(top3)
S.round(4).to_csv(OUT / "summary_by_split.csv", index=False)
T3.round(3).to_csv(OUT / "top3_frequency.csv", index=False)

# ---------------- charts: main score per split, mean ± spread
COLORS = {"MLR / Logistic Regression": "#7f7f7f", "Random Forest": "#2e8b57", "XGBoost": "#1f77b4", "SVM": "#d62728",
          "kNN": "#9467bd", "LSTM": "#ff7f0e", "Baseline: repeat last month": "black",
          "Baseline: same class as last month": "black"}
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, (task, (kind, *_rest)) in zip(axes.ravel(), TASKS.items()):
    s = S[S.task == task]
    x = np.arange(len(METHODS))
    for k, (name, col) in enumerate(COLORS.items()):
        d = s[s.model == name].set_index("method").reindex(METHODS)
        if d[f"{MAIN[kind]}_mean"].isna().all():
            continue
        ax.errorbar(x + (k - 3) * 0.07, d[f"{MAIN[kind]}_mean"], yerr=d[f"{MAIN[kind]}_sd"].fillna(0), fmt="o",
                    color=col, ms=4, capsize=2, label=name, ls="--" if name.startswith("Baseline") else "none")
    ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=25, fontsize=8)
    ax.set_title(task); ax.set_ylabel("R²" if kind == "reg" else "macro F1"); ax.grid(axis="y", alpha=0.3)
    if task.startswith("BOD (mg/L)"):
        # 'repeat last month' falls to about -2 in random splits whenever Oct 2017 Sta. Rosa (last month 70 mg/L,
        # actual 4 mg/L) is in the test part; zoom in so the models can be compared
        ax.set_ylim(-0.8, 0.6)
        ax.set_title(f"{task}\n'repeat last month' is about −2 for random/10-fold (below the chart): "
                     "the month after the 70 mg/L reading", fontsize=10)
axes[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle("Step 4: score by splitting method (dot = average over repeats, bar = ± 1 standard deviation)")
fig.tight_layout(); fig.savefig(OUT / "split_comparison.png", dpi=140); plt.close(fig)

# ---------------- markdown tables
ORDER = MODELS + ["Baseline: repeat last month", "Baseline: training average",
                  "Baseline: same class as last month", "Baseline: always most common class"]
md = ["# Step 4 — Summary tables (split sensitivity)\n",
      "Each cell: average over repeats ± standard deviation (random splits: 10 seeds; 10-fold: 10 folds; "
      "rolling: 8 years; chronological: one split, so no ±). Settings tuned in Step 3.\n"]
for task, (kind, *_rest) in TASKS.items():
    s = S[S.task == task]
    for score in SCORES[kind]:
        label = {"R2": "R²", "RMSE": "RMSE", "MAE": "MAE", "skill_vs_last_month": "Skill vs repeat last month",
                 "accuracy": "Accuracy", "macro_F1": "Macro F1", "balanced_accuracy": "Balanced accuracy",
                 "problem_caught_%": "Problem months caught (%)", "false_alarms_%": "False alarms (%)"}[score]
        pct = score in ("accuracy", "balanced_accuracy")
        md.append(f"## {task} — {label}\n")
        md.append("| Model | " + " | ".join(METHODS) + " |")
        md.append("|---" * (len(METHODS) + 1) + "|")
        for name in [n for n in ORDER if n in set(s.model)]:
            cells = []
            for meth in METHODS:
                row = s[(s.model == name) & (s.method == meth)].iloc[0]
                mu, sd = row[f"{score}_mean"], row[f"{score}_sd"]
                f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.1f}") if score.endswith("%") else (
                    lambda v: f"{v:.3f}" if score in ("RMSE", "MAE") else f"{v:.2f}")
                cells.append(f(mu) + ("" if np.isnan(sd) else f" ± {f(sd).rstrip('%')}"))
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")
    t = T3[T3.task == task].pivot(index="model", columns="method", values="top3_share").reindex(columns=METHODS)
    md.append(f"## {task} — how often each model is in the top 3 (by {'R²' if kind == 'reg' else 'macro F1'})\n")
    md.append("| Model | " + " | ".join(METHODS) + " |")
    md.append("|---" * (len(METHODS) + 1) + "|")
    for name in [n for n in MODELS if n in t.index]:
        md.append(f"| {name} | " + " | ".join(f"{100 * v:.0f}%" for v in t.loc[name]) + " |")
    md.append("")
(OUT / "SUMMARY_TABLES.md").write_text("\n".join(md), encoding="utf-8")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 400)
for task, (kind, *_rest) in TASKS.items():
    main = MAIN[kind]
    p = S[S.task == task].pivot(index="model", columns="method", values=f"{main}_mean").reindex(columns=METHODS)
    q = S[S.task == task].pivot(index="model", columns="method", values=f"{main}_sd").reindex(columns=METHODS)
    print(f"\n===== {task}: {main} mean =====\n{p.round(3).to_string()}")
    print(f"----- {task}: {main} standard deviation -----\n{q.round(3).to_string()}")
    t = T3[T3.task == task].pivot(index="model", columns="method", values="top3_share").reindex(columns=METHODS)
    print(f"----- {task}: share of repeats in top 3 -----\n{(100 * t).round(0).to_string()}")
log("done")
