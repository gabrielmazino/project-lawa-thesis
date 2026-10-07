"""Predict the fecal coliform class for one set of readings with the Improved XGBoost (trained on 2016-2023).

Run:  ml/.venv/Scripts/python.exe ml/fc_class/predict_one.py
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], "1"]  # balanced thresholds (BETA = 1)
src = (HERE / "train_fc_class.py").read_text(encoding="utf-8")
exec(compile(src.split("\ntrain = df[")[0], "train_fc_class.py", "exec"))  # data prep + model classes only

reading = {"Station": "Stn. XV (San Pedro)", "Month": "2024-10",
           "DO": 6.0, "pH": 7.5, "Ammonia": 0.3, "Nitrate": 0.05, "Phosphate": 0.079, "Temperature": 29.0,
           "FC_last_month": 180.0}  # Sep 2024 lab value for San Pedro

row = {f: 0.0 for f in FEATS}
row.update({k: reading[k] for k in READINGS})
row["stn_" + reading["Station"]] = 1.0
m = int(reading["Month"][5:])
row["month_sin"], row["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
row["logFC_last"] = np.log10(reading["FC_last_month"])
x = np.array([[row[f] for f in FEATS]])

train = df[df.Month < "2024-01-01"]
model = OrdinalImproved("XGBoost").fit(train[FEATS].values, train["y"].values)
p = model.proba(x)[0]
print("predicted class:", NAMES[int(model.predict(x)[0])])
print("most likely class:", NAMES[int(np.argmax(p))])
print("probabilities:", {n: round(float(v), 3) for n, v in zip(NAMES, p)})
print("thresholds P(>A), P(>B), P(>C), P(>D):", [round(float(t), 2) for t in model.t])
