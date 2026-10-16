"""Step 6 — Is last month's lab value needed?

Same models, same tuned settings (Step 3), same splits, with and without last month's lab value.
  BOD value:    with = Step 1 inputs; without = same inputs minus last month's BOD.
  BOD 4-class:  direct classifier (Step 5 decision), with / without last month's BOD.
  FC value:     with = change-from-last-month model (Step 1 inputs); without = predicts the log10 level from
                pH, station, DO, month (the change model cannot exist without last month).
  FC 5-class:   regression -> class (Step 5 decision), from the FC value above.
Baselines: training average / most common class (no lab value at all) and "repeat / same class as last month".
Main test: random 70-30 of 2016-2025, 10 seeds. Second check: rolling year by year 2018-2025.
Note: settings were tuned with last month included; the "without" models reuse them (no re-tuning).

Run:  ml/.venv/Scripts/python.exe ml/improve/step6_last_month.py
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone

from common import (BOD_CAP, CLASS_SCHEMES, IMPUTED, LAG, PROBLEM_FROM, SEED, SELECTED, class_metrics, columns, folds,
                    load, random_split, reg_metrics, to_class)
from models import WEIGHTED, clf_models, fit_clf, predict_clf, reg_models

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "step6_results"
OUT.mkdir(exist_ok=True)
TUNED = json.loads((HERE / "step3_results" / "tuned_settings.json").read_text())
MODELS = {"Random Forest": ("Random Forest", "Random Forest"), "XGBoost": ("XGBoost", "XGBoost"), "SVM": ("SVM", "SVM")}


def tuned(kind, key, name, weighted=False):
    est, _ = (reg_models() if kind == "reg" else clf_models(weighted))[name]
    est = clone(est).set_params(**TUNED.get(f"{key} | {name}", {}))
    if name in ("Random Forest", "XGBoost"):
        est.set_params(n_jobs=-1)
    return est


def inputs(t, with_last):
    feats = [f for f in SELECTED[t] if with_last or f != LAG[t]]
    return columns(feats, STN)


def evaluate(tr, te):
    rows = []
    # ---------- BOD value + BOD 4-class
    cs = CLASS_SCHEMES["BOD 4-class"]
    y_val, y_cls = te.BOD.to_numpy(), to_class(te.BOD, cs["edges"])
    ytr_val, ytr_cls = np.minimum(tr.BOD.to_numpy(), BOD_CAP), to_class(tr.BOD, cs["edges"])
    base_last = te.BOD_last.to_numpy()
    rows.append(("BOD value", "baseline", "training average (no lab value)",
                 reg_metrics(y_val, np.full(len(te), tr.BOD.mean()), base_last)))
    rows.append(("BOD value", "baseline", "repeat last month", reg_metrics(y_val, base_last, base_last)))
    rows.append(("BOD 4-class", "baseline", "always most common class (no lab value)",
                 class_metrics(y_cls, np.full(len(te), np.bincount(ytr_cls).argmax()), 1)))
    rows.append(("BOD 4-class", "baseline", "same class as last month",
                 class_metrics(y_cls, to_class(base_last, cs["edges"]), 1)))
    for model, (clf_name, reg_name) in MODELS.items():
        for with_last in (True, False):
            cols, label = inputs("BOD", with_last), "with last month" if with_last else "without last month"
            p = tuned("reg", "BOD (mg/L)", reg_name).fit(tr[cols], ytr_val).predict(te[cols])
            rows.append(("BOD value", model, label, reg_metrics(y_val, p, base_last)))
            c = predict_clf(fit_clf(tuned("clf", "BOD 4-class", clf_name, True), clf_name, True, tr[cols], ytr_cls),
                            te[cols])
            rows.append(("BOD 4-class", model, label, class_metrics(y_cls, c, PROBLEM_FROM["BOD 4-class"])))
    # ---------- FC value (log10) + FC 5-class via regression
    cs = CLASS_SCHEMES["FC 5-class"]
    te = te[~te.FC_imputed]          # filled FC values are never tested
    y_log, y_cls = te.logFC.to_numpy(), to_class(te.FC, cs["edges"])
    last_log = te.logFC_last.to_numpy()
    avg = np.full(len(te), tr.logFC.mean())
    rows.append(("FC value (log10)", "baseline", "training average (no lab value)", reg_metrics(y_log, avg, last_log)))
    rows.append(("FC value (log10)", "baseline", "repeat last month", reg_metrics(y_log, last_log, last_log)))
    rows.append(("FC 5-class", "baseline", "always most common class (no lab value)",
                 class_metrics(y_cls, np.full(len(te), np.bincount(to_class(tr.FC, cs["edges"])).argmax()), 3)))
    rows.append(("FC 5-class", "baseline", "same class as last month",
                 class_metrics(y_cls, to_class(te.FC_last, cs["edges"]), 3)))
    for model, (_c, reg_name) in MODELS.items():
        cols = inputs("FC", True)
        p = tuned("reg", "FC log10 change", reg_name).fit(tr[cols], tr.logFC - tr.logFC_last).predict(te[cols]) + last_log
        cols0 = inputs("FC", False)
        p0 = tuned("reg", "FC log10 level", reg_name).fit(tr[cols0], tr.logFC).predict(te[cols0])
        for label, pred in (("with last month", p), ("without last month", p0)):
            rows.append(("FC value (log10)", model, label, reg_metrics(y_log, pred, last_log)))
            rows.append(("FC 5-class", model, label,
                         class_metrics(y_cls, to_class(10 ** pred, cs["edges"]), PROBLEM_FROM["FC 5-class"])))
    return [{"task": a, "model": b, "inputs": c, **m} for a, b, c, m in rows]


df, STN = load()
runs = []
for s in range(10):
    tr, te = random_split(df, "BOD_imputed", 0.3, SEED + s)
    runs += [{"test": "random 70-30", "rep": s, **r} for r in evaluate(tr, te)]
for split, year, tr, te in folds(df, "BOD_imputed"):
    if split == "rolling":
        runs += [{"test": "rolling year-by-year", "rep": year, **r} for r in evaluate(tr, te)]
print("done", flush=True)

R = pd.DataFrame(runs)
R.round(4).to_csv(OUT / "all_runs.csv", index=False)
SC = ["R2", "RMSE", "MAE", "accuracy", "macro_F1", "balanced_accuracy", "problem_caught_%", "false_alarms_%"]
g = R.groupby(["test", "task", "model", "inputs"], sort=False)[SC]
S = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
S.round(4).to_csv(OUT / "summary.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
for task in ["BOD value", "FC value (log10)"]:
    print(f"\n===== {task} =====")
    print(S[S.task == task][["test", "model", "inputs", "R2_mean", "R2_sd", "RMSE_mean", "MAE_mean"]].round(3).to_string(index=False))
for task in ["BOD 4-class", "FC 5-class"]:
    print(f"\n===== {task} =====")
    print(S[S.task == task][["test", "model", "inputs", "accuracy_mean", "macro_F1_mean", "macro_F1_sd",
                             "problem_caught_%_mean", "false_alarms_%_mean"]].round(3).to_string(index=False))
