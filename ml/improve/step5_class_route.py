"""Step 5 — How should the DENR class be predicted?

Route A, direct classifier: a model trained on the class labels (BOD: class weights, FC: none).
Route B, regression -> class: a model predicts the value (BOD mg/L, FC log10 via the change from last month),
                              then the DENR limits turn that value into a class.
Same inputs, same tuned settings (Step 3), same test rows for both routes.
Main test: random 70-30 of 2016-2025, 10 seeds (mean ± SD). Second check: rolling year by year 2018-2025.
Class schemes: BOD 4-class (A/B/C/D+), FC 5-class (A/B/C/D/Failed).

Run:  ml/.venv/Scripts/python.exe ml/improve/step5_class_route.py
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone

from common import (CLASS_SCHEMES, IMPUTED, PROBLEM_FROM, SEED, SELECTED, class_metrics, columns, folds, load,
                    random_split, to_class)
from models import WEIGHTED, clf_models, fit_clf, fit_target, persist, predict_clf, reg_models

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "step5_results"
OUT.mkdir(exist_ok=True)
TUNED = json.loads((HERE / "step3_results" / "tuned_settings.json").read_text())

SCHEMES = {"BOD 4-class": ("BOD", "BOD (mg/L)", False), "FC 5-class": ("FC", "FC log10 change", True)}
MODELS = {"Random Forest": ("Random Forest", "Random Forest"), "XGBoost": ("XGBoost", "XGBoost"),
          "SVM": ("SVM", "SVM"), "MLR / Logistic Regression": ("Logistic Regression", "MLR")}
SCORES = ["accuracy", "macro_F1", "balanced_accuracy", "problem_caught_%", "false_alarms_%"]


def tuned(kind, key, name, weighted):
    est, _ = (reg_models() if kind == "reg" else clf_models(weighted))[name]
    est = clone(est).set_params(**TUNED.get(f"{key} | {name}", {}))
    if name in ("Random Forest", "XGBoost"):
        est.set_params(n_jobs=-1)
    return est


def evaluate(scheme, tr, te):
    t, reg_key, change = SCHEMES[scheme]
    cs, pf, weighted = CLASS_SCHEMES[scheme], PROBLEM_FROM[scheme], WEIGHTED[t]
    cols = columns(SELECTED[t], STN)
    ytr_c, y = to_class(tr[cs["value"]], cs["edges"]), to_class(te[cs["value"]], cs["edges"])
    ytr_r = fit_target(tr, t, change)
    lastcol = "BOD_last" if t == "BOD" else "FC_last"
    out = [("Baseline", "same class as last month", to_class(te[lastcol], cs["edges"]))]
    for label, (clf_name, reg_name) in MODELS.items():
        direct = predict_clf(fit_clf(tuned("clf", scheme, clf_name, weighted), clf_name, weighted, tr[cols], ytr_c),
                             te[cols])
        value = tuned("reg", reg_key, reg_name, False).fit(tr[cols], ytr_r).predict(te[cols])
        if t == "FC":
            value = 10 ** (value + persist(te, t))      # change in log10 -> log10 level -> MPN/100 mL
        out += [(label, "A. direct classifier", direct), (label, "B. regression -> class", to_class(value, cs["edges"]))]
    return [{"model": m, "route": r, **class_metrics(y, p, pf)} for m, r, p in out]


df, STN = load()
rows = []
for scheme, (t, *_r) in SCHEMES.items():
    for s in range(10):
        tr, te = random_split(df, IMPUTED[t], 0.3, SEED + s)
        rows += [{"scheme": scheme, "test": "random 70-30", "rep": s, **r} for r in evaluate(scheme, tr, te)]
    for split, year, tr, te in folds(df, IMPUTED[t]):
        if split == "rolling":
            rows += [{"scheme": scheme, "test": "rolling year-by-year", "rep": year, **r} for r in evaluate(scheme, tr, te)]
    print(f"{scheme} done", flush=True)

R = pd.DataFrame(rows)
R.round(4).to_csv(OUT / "all_runs.csv", index=False)
g = R.groupby(["scheme", "test", "model", "route"], sort=False)[SCORES]
S = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
S.round(4).to_csv(OUT / "summary.csv", index=False)

# how often route A beats route B (same model, same split) on macro F1
wins = (R[R.model != "Baseline"].pivot_table(index=["scheme", "test", "rep", "model"], columns="route", values="macro_F1")
        .assign(A_wins=lambda d: d["A. direct classifier"] > d["B. regression -> class"])
        .groupby(["scheme", "test", "model"]).A_wins.mean().mul(100).round(0).rename("A wins (% of repeats)"))
wins.to_csv(OUT / "direct_vs_regression_wins.csv")

md = ["# Step 5 — Direct classifier vs regression -> class\n",
      "Mean ± SD over 10 seeds (random 70-30) or 8 years (rolling). Settings tuned in Step 3.\n"]
for scheme in SCHEMES:
    for test in ["random 70-30", "rolling year-by-year"]:
        s = S[(S.scheme == scheme) & (S.test == test)]
        md += [f"## {scheme} · {test}\n", "| Model | Route | Accuracy | Macro F1 | Balanced acc. | Problem months caught | False alarms |",
               "|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            md.append(f"| {r.model} | {r.route} | {100 * r.accuracy_mean:.1f}% ± {100 * r.accuracy_sd:.1f} | "
                      f"{r.macro_F1_mean:.2f} ± {r.macro_F1_sd:.2f} | {100 * r.balanced_accuracy_mean:.1f}% | "
                      f"{r['problem_caught_%_mean']:.0f}% | {r['false_alarms_%_mean']:.0f}% |")
        md.append("")
md += ["## How often the direct classifier beats regression -> class (macro F1, same model and split)\n",
       "| Scheme | Test | Model | Direct wins |", "|---|---|---|---|"]
md += [f"| {k[0]} | {k[1]} | {k[2]} | {v:.0f}% |" for k, v in wins.items()]
(OUT / "SUMMARY_TABLES.md").write_text("\n".join(md), encoding="utf-8")

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(S[["scheme", "test", "model", "route", "accuracy_mean", "macro_F1_mean", "macro_F1_sd", "balanced_accuracy_mean",
         "problem_caught_%_mean", "false_alarms_%_mean"]].round(3).to_string(index=False))
print(wins.to_string())
