"""Shared data preparation for the model-improvement steps (Step 2 onward)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
READINGS = ["DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"]
TRAIN_END = "2024-01-01"      # holdout: train 2016-2023, test 2024-2025
ROLLING_YEARS = range(2018, 2026)
BOD_CAP = 15                  # training-only cap for BOD regression (see backend/train_models.py)

# Inputs chosen in Step 1 (ml/improve/step1_results/SUMMARY.md)
SELECTED = {
    "BOD": ["DO", "pH", "Ammonia", "Nitrate", "Temperature", "BOD_last", "month", "station"],   # Rule B
    "FC": ["logFC_last", "pH", "station", "DO", "month"],                                       # Rule A
}

# DENR classes: BOD DAO 2016-08 (mg/L), fecal coliform WQG 2021-19 (MPN/100 mL)
BOD_EDGES = [3, 5, 7]          # A <=3, B 3-5, C 5-7, D-or-worse >7
FC_EDGES = [50, 100, 200, 400]  # A, B, C, D, Failed
CLASS_SCHEMES = {
    "BOD 4-class": {"target": "BOD", "value": "BOD", "edges": BOD_EDGES, "names": ["A", "B", "C", "D+"]},
    "BOD 2-class": {"target": "BOD", "value": "BOD", "edges": [3], "names": ["A", "B or worse"]},
    "FC 5-class": {"target": "FC", "value": "FC", "edges": FC_EDGES, "names": ["A", "B", "C", "D", "Failed"]},
    "FC 2-class": {"target": "FC", "value": "FC", "edges": [200], "names": ["within standard", "above standard"]},
}
# which classes count as "a problem" for each scheme (index of the first problem class)
PROBLEM_FROM = {"BOD 4-class": 1, "BOD 2-class": 1, "FC 5-class": 3, "FC 2-class": 1}
IMPUTED = {"BOD": "BOD_imputed", "FC": "FC_imputed"}
LAG = {"BOD": "BOD_last", "FC": "logFC_last"}


def load():
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
    df["FC_last"] = df.groupby("Station")["FC"].shift(1).where(gap <= 3)
    df["logFC_last"] = np.log10(df["FC_last"])
    df["month_sin"] = np.sin(2 * np.pi * df.Month.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.Month.dt.month / 12)
    stn = pd.get_dummies(df["Station"], prefix="stn", dtype=float)
    df = pd.concat([df, stn], axis=1)
    df = df.dropna(subset=["BOD_last", "logFC_last"]).sort_values(["Month", "Station"]).reset_index(drop=True)
    return df, list(stn.columns)


def columns(features, stn_cols):
    out = []
    for f in features:
        out += ["month_sin", "month_cos"] if f == "month" else stn_cols if f == "station" else [f]
    return out


def to_class(values, edges):
    return np.digitize(np.asarray(values, dtype=float), edges, right=True)


def folds(df, imputed_col):
    """Rolling-origin folds over 2016-2025 + the 2024-2025 holdout. Test rows never use filled values."""
    for year in ROLLING_YEARS:
        yield "rolling", year, df[df.Month.dt.year < year], df[(df.Month.dt.year == year) & (~df[imputed_col])]
    yield "holdout", "2024-2025", df[df.Month < TRAIN_END], df[(df.Month >= TRAIN_END) & (~df[imputed_col])]


def random_split(df, imputed_col, test_size=0.3, seed=SEED):
    """One split drawn from the whole 2016-2025 dataset, stratified by station and year so every
    station and year appears in both parts. Filled (imputed) values only ever go to training."""
    from sklearn.model_selection import train_test_split
    real = df[~df[imputed_col]]
    strata = real["Station"] + "_" + real.Month.dt.year.astype(str)
    tr, te = train_test_split(real.index, test_size=test_size, random_state=seed, stratify=strata)
    train = pd.concat([df.loc[tr], df[df[imputed_col]]]).sort_values(["Month", "Station"])
    return train, df.loc[te].sort_values(["Month", "Station"])


def class_metrics(y, p, problem_from):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, fbeta_score
    y, p = np.asarray(y), np.asarray(p)
    prob_t, prob_p = y >= problem_from, p >= problem_from
    return {"macro_F1": f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "accuracy": accuracy_score(y, p),
            "problem_caught_%": 100 * (prob_p & prob_t).sum() / max(prob_t.sum(), 1),
            "false_alarms_%": 100 * (prob_p & ~prob_t).sum() / max((~prob_t).sum(), 1),
            "F2_problem": fbeta_score(prob_t, prob_p, beta=2, zero_division=0)}


def reg_metrics(y, p, persist):
    """R2, RMSE, MAE and skill against 'repeat last month' (1 = perfect, 0 = no better, < 0 = worse)."""
    y, p, persist = (np.asarray(a, dtype=float) for a in (y, p, persist))
    mse, mse_p = np.mean((y - p) ** 2), np.mean((y - persist) ** 2)
    return {"R2": 1 - mse / np.var(y), "RMSE": np.sqrt(mse), "MAE": np.mean(np.abs(y - p)),
            "skill_vs_last_month": 1 - mse / mse_p}
