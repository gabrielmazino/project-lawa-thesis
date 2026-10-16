"""Loads the LLDA monitoring records and builds model inputs.

Training (train_models.py) and the API (app.py) both use build_features(), so a prediction
on the website is computed from exactly the same inputs the model was trained on.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
READINGS = ["DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"]
MAX_LAG_MONTHS = 3  # "last month's value" = the station's previous record, if at most 3 months earlier


def load_records():
    """One row per station-month: readings, BOD, fecal coliform, and which values were filled in."""
    bod = pd.read_csv(DATASET / "bod_class_dataset.csv")
    fc = pd.read_csv(DATASET / "fc_class_dataset.csv")
    df = bod[["Month", "Station"] + READINGS + ["BOD_mgL", "BOD_imputed", "Outlier_flag"]].merge(
        fc[["Month", "Station", "FC_MPN100mL", "FC_imputed"]], on=["Month", "Station"], how="inner"
    )
    df = df.rename(columns={"BOD_mgL": "BOD", "FC_MPN100mL": "FC"})
    df["Month"] = pd.to_datetime(df["Month"])
    df["Outlier_flag"] = df["Outlier_flag"].fillna("")
    return df.sort_values(["Station", "Month"]).reset_index(drop=True)


def add_last_values(df):
    """Adds BOD_last and FC_last: the station's previous lab values (within MAX_LAG_MONTHS)."""
    df = df.sort_values(["Station", "Month"]).copy()
    prev_month = df.groupby("Station")["Month"].shift(1)
    gap = (df.Month.dt.year - prev_month.dt.year) * 12 + (df.Month.dt.month - prev_month.dt.month)
    ok = gap <= MAX_LAG_MONTHS
    df["BOD_last"] = df.groupby("Station")["BOD"].shift(1).where(ok)
    df["FC_last"] = df.groupby("Station")["FC"].shift(1).where(ok)
    return df


def build_features(df, stations, last_column=None, readings=READINGS):
    """Model inputs: readings + station (one-hot) + month (sine/cosine) [+ last month's value].

    readings:    which of the 6 readings the model uses.
    last_column: None (parameters only), "BOD_last" (used as is) or "FC_last" (converted to log10).
    """
    X = df[list(readings)].astype(float).copy()
    for s in stations:
        X[f"stn_{s}"] = (df["Station"] == s).astype(float)
    month = pd.to_datetime(df["Month"]).dt.month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    if last_column == "FC_last":
        X["logFC_last"] = np.log10(df["FC_last"].astype(float))
    elif last_column == "BOD_last":
        X["BOD_last"] = df["BOD_last"].astype(float)
    return X
