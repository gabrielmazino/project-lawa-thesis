"""Step 8b — Does Ammonia help the final BOD models WITH last month's BOD? (Random Forest, Step 3 settings, 70-30 x 10 seeds + rolling)"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from common import BOD_CAP, CLASS_SCHEMES, SEED, SELECTED, class_metrics, columns, folds, load, random_split, reg_metrics, to_class
from models import clf_models, fit_clf, predict_clf, reg_models

T = json.loads((Path(__file__).resolve().parent / "step3_results" / "tuned_settings.json").read_text())
df, STN = load()
edges = CLASS_SCHEMES["BOD 4-class"]["edges"]
rows = []


def run(test, rep, tr, te):
    for label, feats in [("with Ammonia", SELECTED["BOD"]), ("without Ammonia", [f for f in SELECTED["BOD"] if f != "Ammonia"])]:
        cols = columns(feats, STN)
        reg = clone(reg_models()["Random Forest"][0]).set_params(**T["BOD (mg/L) | Random Forest"], n_jobs=-1)
        p = reg.fit(tr[cols], np.minimum(tr.BOD, BOD_CAP)).predict(te[cols])
        clf = clone(clf_models(True)["Random Forest"][0]).set_params(**T["BOD 4-class | Random Forest"], n_jobs=-1)
        c = predict_clf(fit_clf(clf, "Random Forest", True, tr[cols], to_class(tr.BOD, edges)), te[cols])
        rows.append({"test": test, "rep": rep, "inputs": label, **reg_metrics(te.BOD, p, te.BOD_last),
                     **class_metrics(to_class(te.BOD, edges), c, 1)})


for s in range(10):
    run("random 70-30", s, *random_split(df, "BOD_imputed", 0.3, SEED + s))
for split, year, tr, te in folds(df, "BOD_imputed"):
    if split == "rolling":
        run("rolling", year, tr, te)
R = pd.DataFrame(rows)
print(R.groupby(["test", "inputs"])[["R2", "RMSE", "MAE", "accuracy", "macro_F1", "problem_caught_%", "false_alarms_%"]].mean().round(3).to_string())
