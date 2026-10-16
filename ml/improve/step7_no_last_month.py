"""Step 7 — All models WITHOUT last month's lab value (parameters + station + month only).

Inputs:  BOD: DO, pH, Ammonia, Nitrate, Temperature, month, station (Step 1 set minus last month's BOD)
         FC:  (a) Step 1 set minus last month: pH, DO, month, station
              (b) all 6 readings + month + station (Step 1 dropped some readings only because last month's FC
                  already covered them, so they are re-tested here)
Models:  MLR / Logistic Regression, Random Forest, XGBoost, SVM, kNN, LSTM (sequence of past readings only).
Tuning:  re-done without last month: grid search, 5-fold CV inside the 70% training part of the seed-42 split.
Classes: route A = classifier trained on the classes (with / without class weights);
         route B = regression value -> DENR limits.
Test:    random 70-30 of 2016-2025, 10 seeds (mean ± SD) + rolling year by year 2018-2025.
Reference only: "repeat / same class as last month" (uses a lab value, shown for comparison).

Run:  ml/.venv/Scripts/python.exe ml/improve/step7_no_last_month.py
"""
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from common import (BOD_CAP, CLASS_SCHEMES, IMPUTED, PROBLEM_FROM, READINGS, SEED, SELECTED, class_metrics, columns,
                    folds, load, random_split, reg_metrics, to_class)
from models import LSTMModel, clf_models, fit_clf, predict_clf, reg_models, sequence_index, tune_lstm, xgb_weights

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "step7_results"
OUT.mkdir(exist_ok=True)
T0 = time.time()
SEEDS3 = (SEED, SEED + 1, SEED + 2)
MODELS = ["MLR / Logistic Regression", "Random Forest", "XGBoost", "SVM", "kNN", "LSTM"]

INPUTS = {
    "BOD": [f for f in SELECTED["BOD"] if f != "BOD_last"],
    "FC Step 1 inputs": [f for f in SELECTED["FC"] if f != "logFC_last"],
    "FC all 6 readings": READINGS + ["month", "station"],
}
# regression: name -> (target, input set, class scheme for route B)
REG = {"BOD value": ("BOD", "BOD", "BOD 4-class"),
       "FC value | Step 1 inputs": ("FC", "FC Step 1 inputs", "FC 5-class"),
       "FC value | all 6 readings": ("FC", "FC all 6 readings", "FC 5-class")}
# classification: name -> (scheme, target, input set, class weights)
CLF = {"BOD 4-class | class weights": ("BOD 4-class", "BOD", "BOD", True),
       "FC 5-class | Step 1 inputs": ("FC 5-class", "FC", "FC Step 1 inputs", False),
       "FC 5-class | Step 1 inputs + class weights": ("FC 5-class", "FC", "FC Step 1 inputs", True),
       "FC 5-class | all 6 readings": ("FC 5-class", "FC", "FC all 6 readings", False),
       "FC 5-class | all 6 readings + class weights": ("FC 5-class", "FC", "FC all 6 readings", True)}


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def value(d, t):
    return (d["BOD"] if t == "BOD" else d["logFC"]).to_numpy()


def train_value(d, t):
    return np.minimum(value(d, t), BOD_CAP) if t == "BOD" else value(d, t)


def sk_name(name, kind):
    return ("MLR" if kind == "reg" else "Logistic Regression") if name == MODELS[0] else name


# ---------------------------------------------------------------- tuning (seed-42 70% training part)
df, STN = load()
SEQ = {L: sequence_index(df, L) for L in (3, 6)}
settings, lstm_cfg = {}, {}
for task, (t, inp, _s) in REG.items():
    cols = columns(INPUTS[inp], STN)
    tr, _ = random_split(df, IMPUTED[t])
    y = train_value(tr, t)
    for name in MODELS[:-1]:
        est, grid = reg_models()[sk_name(name, "reg")]
        settings[(task, name)] = (GridSearchCV(est, grid, cv=KFold(5, shuffle=True, random_state=SEED),
                                               scoring="neg_root_mean_squared_error", n_jobs=-1)
                                  .fit(tr[cols], y).best_params_ if grid else {})
    lstm_cfg[task] = tune_lstm(df[cols].to_numpy(dtype=float), SEQ, "reg", tr.index.to_numpy(), y, False)
    log(f"tuned {task}")
for task, (scheme, t, inp, w) in CLF.items():
    cols, cs = columns(INPUTS[inp], STN), CLASS_SCHEMES[scheme]
    tr, _ = random_split(df, IMPUTED[t])
    y = to_class(tr[cs["value"]], cs["edges"])
    for name in MODELS[:-1]:
        est, grid = clf_models(w)[sk_name(name, "clf")]
        settings[(task, name)] = GridSearchCV(est, grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                              scoring="f1_macro", n_jobs=-1).fit(
            tr[cols], y, **xgb_weights(sk_name(name, "clf"), w, y)).best_params_
    lstm_cfg[task] = tune_lstm(df[cols].to_numpy(dtype=float), SEQ, "clf", tr.index.to_numpy(), y, w)
    log(f"tuned {task}")
(OUT / "tuned_settings.json").write_text(json.dumps(
    {f"{a} | {b}": v for (a, b), v in settings.items()} | {f"{k} | LSTM": v for k, v in lstm_cfg.items()},
    indent=2, default=str))


def model(task, name, kind, weighted=False):
    est, _ = (reg_models() if kind == "reg" else clf_models(weighted))[sk_name(name, kind)]
    est = clone(est).set_params(**settings[(task, name)])
    if name in ("Random Forest", "XGBoost"):
        est.set_params(n_jobs=-1)
    return est


# ---------------------------------------------------------------- evaluation
def evaluate(tr_all, te_all):
    rows = []
    for task, (t, inp, scheme) in REG.items():
        te = te_all[~te_all[IMPUTED[t]]]
        cols, cs = columns(INPUTS[inp], STN), CLASS_SCHEMES[scheme]
        F = df[cols].to_numpy(dtype=float)
        y, last = value(te, t), te["BOD_last" if t == "BOD" else "logFC_last"].to_numpy()
        ytr = train_value(tr_all, t)
        preds = {"Baseline: training average": np.full(len(te), value(tr_all, t).mean()),
                 "Reference: repeat last month (lab value)": last}
        for name in MODELS[:-1]:
            preds[name] = model(task, name, "reg").fit(tr_all[cols], ytr).predict(te[cols])
        c = lstm_cfg[task]
        preds["LSTM"] = LSTMModel(F, SEQ, "reg", c["hidden"], c["L"], seeds=SEEDS3).fit(
            tr_all.index.to_numpy(), ytr).predict(te.index.to_numpy())
        y_cls = to_class(te[cs["value"]], cs["edges"])
        for name, p in preds.items():
            rows.append({"task": task, "route": "regression", "model": name, **reg_metrics(y, p, last)})
            as_class = to_class(p if t == "BOD" else 10 ** p, cs["edges"])
            rows.append({"task": f"{scheme} | {inp.replace('FC ', '') if t == 'FC' else 'BOD inputs'}",
                         "route": "B. regression -> class", "model": name,
                         **class_metrics(y_cls, as_class, PROBLEM_FROM[scheme])})
    for task, (scheme, t, inp, w) in CLF.items():
        te = te_all[~te_all[IMPUTED[t]]]
        cols, cs = columns(INPUTS[inp], STN), CLASS_SCHEMES[scheme]
        F = df[cols].to_numpy(dtype=float)
        ytr, y = to_class(tr_all[cs["value"]], cs["edges"]), to_class(te[cs["value"]], cs["edges"])
        preds = {"Baseline: always most common class": np.full(len(te), np.bincount(ytr).argmax()),
                 "Reference: same class as last month (lab value)":
                     to_class(te["BOD_last" if t == "BOD" else "FC_last"], cs["edges"])}
        for name in MODELS[:-1]:
            n = sk_name(name, "clf")
            preds[name] = predict_clf(fit_clf(model(task, name, "clf", w), n, w, tr_all[cols], ytr), te[cols])
        c = lstm_cfg[task]
        preds["LSTM"] = LSTMModel(F, SEQ, "clf", c["hidden"], c["L"], w, SEEDS3).fit(
            tr_all.index.to_numpy(), ytr).predict(te.index.to_numpy())
        group = f"{scheme} | {inp.replace('FC ', '') if t == 'FC' else 'BOD inputs'}"
        for name, p in preds.items():
            rows.append({"task": group, "route": "A. classifier" + (" + class weights" if w else ""), "model": name,
                         **class_metrics(y, p, PROBLEM_FROM[scheme])})
    return rows


runs = []
for s in range(10):
    tr, te = random_split(df, "BOD_imputed", 0.3, SEED + s)     # same rows for every task; filled values filtered per task
    runs += [{"test": "random 70-30", "rep": s, **r} for r in evaluate(tr, te)]
    log(f"seed {s} done")
for split, year, tr, te in folds(df, "BOD_imputed"):
    if split == "rolling":
        runs += [{"test": "rolling year-by-year", "rep": year, **r} for r in evaluate(tr, te)]
log("rolling done")

R = pd.DataFrame(runs)
R.round(4).to_csv(OUT / "all_runs.csv", index=False)
SC = ["R2", "RMSE", "MAE", "accuracy", "macro_F1", "balanced_accuracy", "problem_caught_%", "false_alarms_%"]
g = R.groupby(["test", "task", "route", "model"], sort=False)[SC]
S = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
S.round(4).to_csv(OUT / "summary.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
reg = S[S.route == "regression"]
print(reg[["test", "task", "model", "R2_mean", "R2_sd", "RMSE_mean", "MAE_mean"]].round(3).to_string(index=False))
cl = S[S.route != "regression"].sort_values(["test", "task", "macro_F1_mean"], ascending=[True, True, False])
print(cl[["test", "task", "route", "model", "accuracy_mean", "macro_F1_mean", "macro_F1_sd", "balanced_accuracy_mean",
          "problem_caught_%_mean", "false_alarms_%_mean"]].round(3).to_string(index=False))
log("done")
