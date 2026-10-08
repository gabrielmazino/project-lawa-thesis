"""DENR water quality classes, as printed in the latest LLDA quarterly report legend.

BOD: DAO 2016-08 (mg/L).  Fecal coliform: DENR WQG 2021-19 (MPN/100 mL).
Laguna Lake is classified Class C, so a value worse than Class C is "above standard".
"""
import numpy as np

# (class name, inclusive upper limit); anything above the last limit is "Failed"
BOD_LIMITS = [("A", 3), ("B", 5), ("C", 7), ("D", 15)]
FC_LIMITS = [("A", 50), ("B", 100), ("C", 200), ("D", 400)]
LAKE_STANDARD = {"BOD": 7, "FC": 200}  # Class C limits for Laguna Lake


def classify(value, limits):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    for name, upper in limits:
        if value <= upper:
            return name
    return "Failed"


def bod_class(value):
    return classify(value, BOD_LIMITS)


def fc_class(value):
    return classify(value, FC_LIMITS)


def above_standard(param, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return bool(value > LAKE_STANDARD[param])
