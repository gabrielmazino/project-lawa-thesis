"""BOD5 computational lab model (unseeded samples), same formulas as the Excel sheet "BOD5 Template".

    P      = sample volume / bottle volume
    BOD5   = (D1 - D2 - seed correction) / P        (unseeded: seed correction = 0)
    QC     = valid when DO depletion >= 2.0 mg/L and final DO (D2) >= 1.0 mg/L (Standard Methods rule)
    class  = DENR DAO 2016-08 class of BOD5 rounded to 1 decimal (as in the LLDA reports)
"""
from pathlib import Path

import pandas as pd

from .denr import above_standard, bod_class

SYNTHETIC_CSV = Path(__file__).resolve().parents[2] / "dataset" / "bod5" / "bod5_synthetic_lab.csv"
MIN_DEPLETION = 2.0   # mg/L
MIN_FINAL_DO = 1.0    # mg/L
LIMITS = {"sample_ml": (0, 1000), "bottle_ml": (1, 1000), "d1": (0, 20), "d2": (0, 20)}


def compute(sample_ml, bottle_ml, d1, d2, seed_correction=0.0):
    """One sample -> P, DO depletion, BOD5, DENR class, standard and QC check. Raises ValueError on bad input."""
    if sample_ml <= 0 or sample_ml > bottle_ml:
        raise ValueError("Sample volume must be more than 0 and at most the bottle volume.")
    if d2 > d1:
        raise ValueError("D2 (after 5 days) cannot be higher than D1 (initial DO).")
    p = sample_ml / bottle_ml
    depletion = d1 - d2
    bod = (depletion - seed_correction) / p
    reported = round(bod, 1)
    if d2 < MIN_FINAL_DO:
        qc, note = "Invalid", "Final DO below 1.0 mg/L: oxygen ran out. Repeat with more dilution."
    elif depletion < MIN_DEPLETION:
        qc, note = "Below range", "DO depletion below 2.0 mg/L: report with a qualifier (clean water)."
    else:
        qc, note = "Valid", "DO depletion ≥ 2.0 mg/L and final DO ≥ 1.0 mg/L."
    return {"P": round(p, 4), "depletion": round(depletion, 3), "BOD5": round(bod, 3), "BOD5_reported": reported,
            "class": bod_class(reported), "above_standard": above_standard("BOD", reported), "qc": qc, "qc_note": note}


def synthetic_samples():
    """The synthetic lab sheet (one row per measured LLDA station-month)."""
    df = pd.read_csv(SYNTHETIC_CSV)
    return df.rename(columns={"Sample ID": "sample_id", "Month": "month", "Station": "station",
                              "Sample volume (mL)": "sample_ml", "Bottle volume (mL)": "bottle_ml",
                              "D1": "d1", "D2": "d2", "LLDA BOD": "llda_bod"})
