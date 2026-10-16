"""Step 8 — Does Ammonia help the final BOD models (parameters only, no last month)?
Random Forest value + Random Forest classifier (class weights), settings tuned in Step 7.
Random 70-30 x 10 seeds + rolling 2018-2025."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from common import BOD_CAP, CLASS_SCHEMES, SEED, class_metrics, columns, folds, load, random_split, reg_metrics, to_class
from models import clf_models, fit_clf, predict_clf, reg_models

T = json.loads((Path(__file__).resolve().parent / "step7_results" / "tuned_settings.json").read_text())
df, STN = load()
edges = CLASS_SCHEMES["BOD 4-class"]["edges"]
BASE = ["DO", "pH", "Ammonia", "Nitrate", "Temperature", "month", "station"]
rows = []


def run(test, rep, tr, te):
    te = te[~te.BOD_imputed]
    for label, feats in [("with Ammonia", BASE), ("without Ammonia", [f for f in BASE if f != "Ammonia"])]:
        cols = columns(feats, STN)
        reg = clone(reg_models()["Random Forest"][0]).set_params(**T["BOD value | Random Forest"], n_jobs=-1)
        p = reg.fit(tr[cols], np.minimum(tr.BOD, BOD_CAP)).predict(te[cols])
        clf = clone(clf_models(True)["Random Forest"][0]).set_params(**T["BOD 4-class | class weights | Random Forest"], n_jobs=-1)
        c = predict_clf(fit_clf(clf, "Random Forest", True, tr[cols], to_class(tr.BOD, edges)), te[cols])
        rows.append({"test": test, "rep": rep, "inputs": label, **reg_metrics(te.BOD, p, te.BOD_last),
                     **class_metrics(to_class(te.BOD, edges), c, 1)})


for s in range(10):
    run("random 70-30", s, *random_split(df, "BOD_imputed", 0.3, SEED + s))
for split, year, tr, te in folds(df, "BOD_imputed"):
    if split == "rolling":
        run("rolling", year, tr, te)
R = pd.DataFrame(rows)
R.to_csv(Path(__file__).resolve().parent / "step8_ammonia_results.csv", index=False)
print(R.groupby(["test", "inputs"])[["R2", "RMSE", "MAE", "accuracy", "macro_F1", "problem_caught_%", "false_alarms_%"]].mean().round(3).to_string())
