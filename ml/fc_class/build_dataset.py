"""Build the fecal coliform class dataset from the "Preprocessing v4 (flagged)" sheet.

Classes (DENR WQG 2021-19, as printed in the LLDA report legends), MPN/100 mL:
  A <=50, B >50-100, C >100-200, D >200-400, Failed >400
Writes dataset/fc_class_dataset.csv

Run:  ml/.venv/Scripts/python.exe ml/fc_class/build_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "dataset" / "Copy of LLDA-Data-2013-2023-Compiled.xlsx"
OUT = ROOT / "dataset" / "fc_class_dataset.csv"
EDGES = [50, 100, 200, 400]
NAMES = ["A", "B", "C", "D", "Failed"]

ws = load_workbook(XLSX, data_only=True)["Preprocessing v4 (flagged)"]
rows = []
for r in ws.iter_rows(min_row=2, max_col=12):
    if r[0].value is None:
        continue
    fill = r[4].fill.fgColor.rgb if r[4].fill and r[4].fill.fgColor is not None else ""
    rows.append({
        "Month": pd.Timestamp(r[0].value).strftime("%Y-%m"), "Station": r[1].value,
        "DO": r[3].value, "pH": r[5].value, "Ammonia": r[6].value, "Nitrate": r[7].value,
        "Phosphate": r[8].value, "Temperature": r[9].value, "FC_MPN100mL": r[4].value,
        "Outlier_flag": r[11].value or "",
        "FC_imputed": isinstance(fill, str) and fill.endswith("FFFF00"),
    })
df = pd.DataFrame(rows)
df["FC_Class"] = [NAMES[i] for i in np.digitize(df["FC_MPN100mL"], EDGES, right=True)]
df.to_csv(OUT, index=False)
print("rows:", len(df), "| imputed FC:", int(df.FC_imputed.sum()))
print(df.FC_Class.value_counts().reindex(NAMES).to_dict())
