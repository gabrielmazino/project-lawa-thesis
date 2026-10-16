"""Step 3 — More models (SVM, kNN, LSTM) and tuning for every model.

Data:        whole 2016-2025 dataset. Main test: one 70:30 split drawn from it (stratified by station and year,
             so every station and year is in both parts). Filled (imputed) values are never tested.
Tuning:      grid search with 5-fold CV inside the 70% training part only (LSTM: 3-fold, small grid).
Second check: rolling year-by-year (each year 2018-2025 predicted from earlier years) with the tuned settings.
Inputs:      chosen in Step 1 (common.SELECTED).
Imbalance:   chosen in Step 2 - BOD classes use class weights (kNN has none, so it uses oversampling);
             FC classes use no technique.
Regression:  BOD (mg/L, training capped at 15), FC log10 level, FC log10 change from last month.
Classes:     BOD 4-class, BOD 2-class, FC 5-class, FC 2-class (DENR).
Models:      MLR / Logistic Regression, Random Forest, XGBoost, SVM, kNN, LSTM.

Run:  ml/.venv/Scripts/python.exe ml/improve/step3_models.py
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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from common import CLASS_SCHEMES, IMPUTED, PROBLEM_FROM, SEED, SELECTED, class_metrics, columns, folds, load, random_split, reg_metrics, to_class
from models import (LSTMModel, REG_TASKS, WEIGHTED, clf_models, fit_clf, fit_target, level, persist, predict_clf,
                    reg_models, sequence_index, tune_lstm, xgb_weights)

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "step3_results"
OUT.mkdir(exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- run
df, STN = load()
SEQ = {L: sequence_index(df, L) for L in (3, 6)}
FINAL_SEEDS = (SEED, SEED + 1, SEED + 2)     # LSTM final models average 3 networks (less luck in one start)
reg_rows, clf_rows, settings, keep = [], [], {}, {}

# ---------- regression
for task, (t, change) in REG_TASKS.items():
    cols = columns(SELECTED[t], STN)
    F = df[cols].to_numpy(dtype=float)
    tr, te = random_split(df, IMPUTED[t])
    ytr = fit_target(tr, t, change)
    y_true, base = level(te, t), persist(te, t)
    back = (lambda p, d: p + persist(d, t)) if change else (lambda p, d: p)
    preds = {"Baseline: repeat last month": base, "Baseline: training average": np.full(len(te), level(tr, t).mean())}
    tuned = {}
    for name, (est, grid) in reg_models().items():
        if grid:
            gs = GridSearchCV(est, grid, cv=KFold(5, shuffle=True, random_state=SEED),
                              scoring="neg_root_mean_squared_error", n_jobs=-1).fit(tr[cols], ytr)
            tuned[name], params = gs.best_estimator_, gs.best_params_
        else:
            tuned[name], params = clone(est).fit(tr[cols], ytr), {}
        preds[name] = back(tuned[name].predict(te[cols]), te)
        settings[f"{task} | {name}"] = params
        log(f"{task}: {name} tuned {params}")
    cfg = tune_lstm(F, SEQ, "reg", tr.index.to_numpy(), ytr, False)
    lstm = LSTMModel(F, SEQ, "reg", cfg["hidden"], cfg["L"], seeds=FINAL_SEEDS).fit(tr.index.to_numpy(), ytr)
    preds["LSTM"] = back(lstm.predict(te.index.to_numpy()), te)
    settings[f"{task} | LSTM"] = cfg
    log(f"{task}: LSTM tuned {cfg}")
    for name, p in preds.items():
        reg_rows.append({"task": task, "evaluation": "70:30 split 2016-2025", "model": name, "n_test": len(te),
                         **reg_metrics(y_true, p, base)})
    keep[("reg", task)] = (y_true, preds)

    # rolling check with the tuned settings
    roll = {name: [] for name in preds}
    for split, year, rtr, rte in folds(df, IMPUTED[t]):
        if split != "rolling":
            continue
        ry = fit_target(rtr, t, change)
        yt, b = level(rte, t), persist(rte, t)
        out = {"Baseline: repeat last month": b, "Baseline: training average": np.full(len(rte), level(rtr, t).mean())}
        for name, m in tuned.items():
            out[name] = back(clone(m).fit(rtr[cols], ry).predict(rte[cols]), rte)
        lm = LSTMModel(F, SEQ, "reg", cfg["hidden"], cfg["L"], seeds=FINAL_SEEDS).fit(rtr.index.to_numpy(), ry)
        out["LSTM"] = back(lm.predict(rte.index.to_numpy()), rte)
        for name, p in out.items():
            roll[name].append((yt, p, b))
    for name, parts in roll.items():
        y, p, b = (np.concatenate([x[i] for x in parts]) for i in range(3))
        reg_rows.append({"task": task, "evaluation": "rolling 2018-2025", "model": name, "n_test": len(y),
                         **reg_metrics(y, p, b)})
    log(f"{task}: rolling done")

# ---------- classification
for scheme, cs in CLASS_SCHEMES.items():
    t, K, pf, weighted = cs["target"], len(cs["names"]), PROBLEM_FROM[scheme], WEIGHTED[cs["target"]]
    cols = columns(SELECTED[t], STN)
    F = df[cols].to_numpy(dtype=float)
    lastcol = "BOD_last" if t == "BOD" else "FC_last"
    tr, te = random_split(df, IMPUTED[t])
    ytr, yte = to_class(tr[cs["value"]], cs["edges"]), to_class(te[cs["value"]], cs["edges"])
    preds = {"Baseline: same class as last month": to_class(te[lastcol], cs["edges"]),
             "Baseline: always most common class": np.full(len(te), np.bincount(ytr).argmax())}
    tuned = {}
    for name, (est, grid) in clf_models(weighted).items():
        present = np.unique(ytr)
        yy = np.searchsorted(present, ytr)
        gs = GridSearchCV(est, grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1_macro",
                          n_jobs=-1).fit(tr[cols], yy, **xgb_weights(name, weighted, yy))
        tuned[name] = gs.best_estimator_
        preds[name] = present[gs.best_estimator_.predict(te[cols])]
        settings[f"{scheme} | {name}"] = gs.best_params_
        log(f"{scheme}: {name} tuned {gs.best_params_}")
    rows_tr = tr.index.to_numpy()
    cfg = tune_lstm(F, SEQ, "clf", rows_tr, ytr, weighted)
    lstm = LSTMModel(F, SEQ, "clf", cfg["hidden"], cfg["L"], weighted, FINAL_SEEDS).fit(rows_tr, ytr)
    preds["LSTM"] = lstm.predict(te.index.to_numpy())
    settings[f"{scheme} | LSTM"] = cfg
    log(f"{scheme}: LSTM tuned {cfg}")
    for name, p in preds.items():
        clf_rows.append({"scheme": scheme, "evaluation": "70:30 split 2016-2025", "model": name, "n_test": len(te),
                         **class_metrics(yte, p, pf)})
    keep[("clf", scheme)] = (yte, preds)

    roll = {name: [] for name in preds}
    for split, year, rtr, rte in folds(df, IMPUTED[t]):
        if split != "rolling":
            continue
        ry, yt = to_class(rtr[cs["value"]], cs["edges"]), to_class(rte[cs["value"]], cs["edges"])
        out = {"Baseline: same class as last month": to_class(rte[lastcol], cs["edges"]),
               "Baseline: always most common class": np.full(len(rte), np.bincount(ry).argmax())}
        for name, m in tuned.items():
            out[name] = predict_clf(fit_clf(m, name, weighted, rtr[cols], ry), rte[cols])
        out["LSTM"] = LSTMModel(F, SEQ, "clf", cfg["hidden"], cfg["L"], weighted, FINAL_SEEDS).fit(
            rtr.index.to_numpy(), ry).predict(rte.index.to_numpy())
        for name, p in out.items():
            roll[name].append((yt, p))
    for name, parts in roll.items():
        y, p = np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])
        clf_rows.append({"scheme": scheme, "evaluation": "rolling 2018-2025", "model": name, "n_test": len(y),
                         **class_metrics(y, p, pf)})
    log(f"{scheme}: rolling done")

# ---------------------------------------------------------------- save
reg = pd.DataFrame(reg_rows)
clf = pd.DataFrame(clf_rows)
reg.round(4).to_csv(OUT / "regression_results.csv", index=False)
clf.round(4).to_csv(OUT / "classification_results.csv", index=False)
(OUT / "tuned_settings.json").write_text(json.dumps(settings, indent=2, default=str))

ORDER = ["MLR", "Logistic Regression", "Random Forest", "XGBoost", "SVM", "kNN", "LSTM"]

# regression comparison
fig, axes = plt.subplots(1, len(REG_TASKS), figsize=(16, 4.5))
for ax, task in zip(axes, REG_TASKS):
    r = reg[reg.task == task]
    names = [m for m in ORDER if m in set(r.model)]
    x = np.arange(len(names))
    for k, (ev, col) in enumerate([("70:30 split 2016-2025", "#2e6f9e"), ("rolling 2018-2025", "#e08a2e")]):
        vals = [r[(r.model == m) & (r.evaluation == ev)].R2.iloc[0] for m in names]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=ev, color=col)
        base = r[(r.model == "Baseline: repeat last month") & (r.evaluation == ev)].R2.iloc[0]
        ax.axhline(base, color=col, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=8); ax.set_title(task); ax.set_ylabel("R²")
    ax.axhline(0, color="black", lw=0.6)
axes[0].legend(fontsize=8)
fig.suptitle("Regression R² (dashed = 'repeat last month' baseline)")
fig.tight_layout(); fig.savefig(OUT / "regression_compare.png", dpi=140); plt.close(fig)

# classification comparison
fig, axes = plt.subplots(1, len(CLASS_SCHEMES), figsize=(18, 4.5))
for ax, scheme in zip(axes, CLASS_SCHEMES):
    r = clf[clf.scheme == scheme]
    names = [m for m in ORDER if m in set(r.model)]
    x = np.arange(len(names))
    for k, (ev, col) in enumerate([("70:30 split 2016-2025", "#2e6f9e"), ("rolling 2018-2025", "#e08a2e")]):
        vals = [r[(r.model == m) & (r.evaluation == ev)].macro_F1.iloc[0] for m in names]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=ev, color=col)
        base = r[(r.model == "Baseline: same class as last month") & (r.evaluation == ev)].macro_F1.iloc[0]
        ax.axhline(base, color=col, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=8); ax.set_title(scheme); ax.set_ylabel("macro F1")
axes[0].legend(fontsize=8)
fig.suptitle("Classification macro F1 (dashed = 'same class as last month' baseline)")
fig.tight_layout(); fig.savefig(OUT / "classification_compare.png", dpi=140); plt.close(fig)

# best regression model per task: predicted vs actual on the 30% test
fig, axes = plt.subplots(1, len(REG_TASKS), figsize=(15, 4.6))
for ax, task in zip(axes, REG_TASKS):
    r = reg[(reg.task == task) & (reg.evaluation == "70:30 split 2016-2025") & ~reg.model.str.startswith("Baseline")]
    best = r.sort_values("R2", ascending=False).iloc[0]
    y, preds = keep[("reg", task)]
    ax.scatter(y, preds[best.model], s=10, alpha=0.6)
    lo, hi = min(y.min(), preds[best.model].min()), max(y.max(), preds[best.model].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("actual"); ax.set_ylabel("predicted"); ax.set_title(f"{task}: {best.model} (R² {best.R2:.2f})", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "best_regression_pred_vs_actual.png", dpi=140); plt.close(fig)

# best classifier per scheme: confusion matrix on the 30% test
fig, axes = plt.subplots(1, len(CLASS_SCHEMES), figsize=(18, 4.4))
for ax, (scheme, cs) in zip(axes, CLASS_SCHEMES.items()):
    r = clf[(clf.scheme == scheme) & (clf.evaluation == "70:30 split 2016-2025") & ~clf.model.str.startswith("Baseline")]
    best = r.sort_values("macro_F1", ascending=False).iloc[0]
    y, preds = keep[("clf", scheme)]
    K = len(cs["names"])
    cm = confusion_matrix(y, preds[best.model], labels=list(range(K)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(cs["names"], fontsize=8); ax.set_yticklabels(cs["names"], fontsize=8)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"{scheme}: {best.model} (macro F1 {best.macro_F1:.2f})", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "best_classifier_confusion.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
for task in REG_TASKS:
    for ev in ["70:30 split 2016-2025", "rolling 2018-2025"]:
        r = reg[(reg.task == task) & (reg.evaluation == ev)].sort_values("R2", ascending=False)
        print(f"\n===== {task} · {ev} (n={r.n_test.iloc[0]}) =====")
        print(r[["model", "R2", "RMSE", "MAE", "skill_vs_last_month"]].round(3).to_string(index=False))
for scheme in CLASS_SCHEMES:
    for ev in ["70:30 split 2016-2025", "rolling 2018-2025"]:
        r = clf[(clf.scheme == scheme) & (clf.evaluation == ev)].sort_values("macro_F1", ascending=False)
        print(f"\n===== {scheme} · {ev} (n={r.n_test.iloc[0]}) =====")
        print(r[["model", "macro_F1", "balanced_accuracy", "accuracy", "problem_caught_%", "false_alarms_%",
                 "F2_problem"]].round(3).to_string(index=False))
log("done")
