"""Train the final models used by the website and save them to backend/models/.

Chosen in the model-improvement steps (ml/improve/, see each step*_results/SUMMARY.md):
  BOD value      - Random Forest regression. Inputs: DO, pH, Ammonia, Nitrate, Temperature + station + month
                   + last month's BOD (Phosphate dropped in Step 1; user's decision to use last month's lab value
                   for both BOD and fecal coliform). Settings tuned in Step 3.
                   Training cap: BOD above 15 mg/L is set to 15 for TRAINING only, so the single real 70 mg/L month
                   cannot make the model estimate very high values for ordinary months.
  BOD class      - separate Random Forest classifier with class weights (Step 5: catches more polluted months than
                   turning the BOD value into a class). Classes A, B, C, D-or-worse (DAO 2016-08). Same inputs.
                   Settings tuned in Step 3.
  Fecal coliform - Random Forest predicting the CHANGE in log10 FC from last month's lab value (Steps 3-6: the
                   parameters alone cannot estimate FC; last month's FC is needed). Inputs: pH, DO + station + month
                   + last month's FC (Step 1). The FC class comes from the estimated value and the DENR limits (Step 5).
                   Settings tuned in Step 3.

Scores stored with each model (shown on the website):
  same rows as the experiments (months with last month's BOD and FC available);
  random 70-30 split of 2016-2025 (stratified by station and year), repeated with 10 seeds -> mean and SD;
  rolling year by year 2018-2025 (each year predicted from earlier years) -> mean of the yearly scores.
  Filled (imputed) values are never used as test values. The saved model is then refit on ALL data.

Run:  backend/.venv/Scripts/python.exe backend/train_models.py
"""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from lawa.data import READINGS, add_last_values, build_features, load_records

OUT = Path(__file__).resolve().parent / "models"
OUT.mkdir(exist_ok=True)
SEED = 42
SEEDS = range(SEED, SEED + 10)
BOD_TRAIN_CAP = 15
BOD_READINGS = ["DO", "pH", "Ammonia", "Nitrate", "Temperature"]
FC_READINGS = ["DO", "pH"]
BOD_EDGES, BOD_NAMES = [3, 5, 7], ["A", "B", "C", "D"]   # "D" here means D or worse (> 7 mg/L)
FC_EDGES, FC_NAMES = [50, 100, 200, 400], ["A", "B", "C", "D", "Failed"]

df = add_last_values(load_records()).sort_values(["Month", "Station"]).reset_index(drop=True)
stations = sorted(df["Station"].unique())
input_ranges = {r: [float(df[r].min()), float(df[r].max())] for r in READINGS}


def bod_reg():
    return RandomForestRegressor(n_estimators=300, max_depth=None, max_features=0.5, min_samples_leaf=1,
                                 random_state=SEED, n_jobs=-1)


def bod_clf():
    return RandomForestClassifier(n_estimators=300, max_depth=8, max_features=0.5, min_samples_leaf=1,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def fc_reg():
    return RandomForestRegressor(n_estimators=300, max_depth=None, max_features=0.5, min_samples_leaf=5,
                                 random_state=SEED, n_jobs=-1)


def to_class(values, edges):
    return np.digitize(np.asarray(values, dtype=float), edges, right=True)


def random_split(data, imputed_col, seed):
    real = data[~data[imputed_col]]
    strata = real["Station"] + "_" + real.Month.dt.year.astype(str)
    tr, te = train_test_split(real.index, test_size=0.3, random_state=seed, stratify=strata)
    return pd.concat([data.loc[tr], data[data[imputed_col]]]), data.loc[te]


def rolling(data, imputed_col):
    for year in range(2018, 2026):
        yield data[data.Month.dt.year < year], data[(data.Month.dt.year == year) & ~data[imputed_col]]


def reg_scores(y, p):
    mse = np.mean((y - p) ** 2)
    return {"r2": 1 - mse / np.var(y), "rmse": np.sqrt(mse), "mae": np.mean(np.abs(y - p))}


def class_scores(y, p, problem_from):
    t, q = y >= problem_from, p >= problem_from
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, labels=np.unique(y), average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "problem_caught_pct": 100 * (t & q).sum() / max(t.sum(), 1),
            "false_alarm_pct": 100 * (~t & q).sum() / max((~t).sum(), 1)}


def summarise(runs):
    r = pd.DataFrame(runs)
    return {k: {"mean": float(r[k].mean()), "sd": float(r[k].std()), "median": float(r[k].median())} for k in r}


def evaluate(data, imputed_col, score_fn):
    out = {}
    for name, splits in [("random_70_30", (random_split(data, imputed_col, s) for s in SEEDS)),
                         ("rolling", rolling(data, imputed_col))]:
        out[name] = summarise([score_fn(tr, te) for tr, te in splits])
    out["random_70_30"]["repeats"], out["rolling"]["repeats"] = len(SEEDS), 8
    return out


def save(name, model, features, readings, meta):
    joblib.dump({"model": model, "features": features, "readings": readings, "stations": stations,
                 "input_ranges": input_ranges, "trained_on": f"{df.Month.min():%Y-%m} to {df.Month.max():%Y-%m}",
                 "created": date.today().isoformat(), **meta}, OUT / f"{name}.joblib")


def show(title, scores):
    for test, s in scores.items():
        print(f"{title} | {test}: " + ", ".join(f"{k} {v['mean']:.3f} ± {v['sd']:.3f}" for k, v in s.items()
                                                 if isinstance(v, dict)))


# ---------------- BOD value and BOD class ----------------
def bod_X(d):
    return build_features(d, stations, "BOD_last", BOD_READINGS)


def bod_value_scores(tr, te):
    p = bod_reg().fit(bod_X(tr), tr["BOD"].clip(upper=BOD_TRAIN_CAP)).predict(bod_X(te))
    return reg_scores(te["BOD"].to_numpy(), p)


def bod_class_scores(tr, te):
    m = bod_clf().fit(bod_X(tr), to_class(tr["BOD"], BOD_EDGES))
    return class_scores(to_class(te["BOD"], BOD_EDGES), m.predict(bod_X(te)), 1)


bod = df.dropna(subset=["BOD_last", "FC_last"])     # same rows as the experiments (ml/improve/common.py)
X_all = bod_X(bod)
value_scores = evaluate(bod, "BOD_imputed", bod_value_scores)
show("BOD value", value_scores)
save("bod_model", bod_reg().fit(X_all, bod["BOD"].clip(upper=BOD_TRAIN_CAP)), list(X_all.columns), BOD_READINGS, {
    "target": "BOD (mg/L)", "algorithm": "Random Forest", "predicts": "level", "uses_last_month": True,
    "training_cap_mgL": BOD_TRAIN_CAP, "scores": value_scores, "scale": "mg/L"})

class_scores_ = evaluate(bod, "BOD_imputed", bod_class_scores)
show("BOD class", class_scores_)
save("bod_class_model", bod_clf().fit(X_all, to_class(bod["BOD"], BOD_EDGES)), list(X_all.columns), BOD_READINGS, {
    "target": "BOD class (DAO 2016-08)", "algorithm": "Random Forest classifier (class weights)",
    "predicts": "class", "class_names": BOD_NAMES, "class_note": "D means D or worse (above 7 mg/L)",
    "uses_last_month": True, "scores": class_scores_})


# ---------------- Fecal coliform: change in log10 FC from last month ----------------
fc = df.dropna(subset=["BOD_last", "FC_last"]).copy()
fc["dlog"] = np.log10(fc["FC"]) - np.log10(fc["FC_last"])


def fc_X(d):
    return build_features(d, stations, "FC_last", FC_READINGS)


def fc_scores(tr, te):
    log_pred = np.log10(te["FC_last"].to_numpy()) + fc_reg().fit(fc_X(tr), tr["dlog"]).predict(fc_X(te))
    y_log = np.log10(te["FC"].to_numpy())
    return {**reg_scores(y_log, log_pred),
            **{f"class_{k}": v for k, v in class_scores(to_class(te["FC"], FC_EDGES),
                                                         to_class(10 ** log_pred, FC_EDGES), 3).items()}}


X_all = fc_X(fc)
scores = evaluate(fc, "FC_imputed", fc_scores)
show("FC", scores)
save("fc_model", fc_reg().fit(X_all, fc["dlog"]), list(X_all.columns), FC_READINGS, {
    "target": "log10 fecal coliform (MPN/100 mL)", "algorithm": "Random Forest (predicts the change from last month)",
    "predicts": "change_log10", "uses_last_month": True, "class_names": FC_NAMES, "scores": scores, "scale": "log10"})
print(f"saved to {OUT}")
