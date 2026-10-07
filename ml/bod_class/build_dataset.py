"""Build the v4 dataset: same as Preprocessing v3A (gaps filled) but outliers are FLAGGED, not deleted,
and BOD gets its DENR class (DAO 2016-08). Writes:
  dataset/bod_class_dataset.csv
  ml/bod_class/v4_sheets.xlsx  (one sheet, copied into the workbook by merge.ps1)

Run:  ml/.venv/Scripts/python.exe ml/bod_class/build_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Font, PatternFill

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "dataset" / "Copy of LLDA-Data-2013-2023-Compiled.xlsx"
OUT_CSV = ROOT / "dataset" / "bod_class_dataset.csv"
OUT_XLSX = Path(__file__).resolve().parent / "v4_sheets.xlsx"

# DENR DAO 2016-08 BOD limits (mg/L): A <=3, B <=5, C <=7, D <=15; above 15 kept in D ("D or worse")
EDGES = [3, 5, 7]
NAMES = ["A", "B", "C", "D"]

# physically impossible = rogue data (manuscript step 1)
VALID = {"DO": (0, 25), "pH": (0, 14), "Ammonia": (0, None), "Nitrate": (0, None),
         "Phosphate": (0, None), "Temperature": (0, 45), "BOD": (0, None), "FC": (0, None)}

ws = load_workbook(XLSX, data_only=True)["Preprocessing v3A"]
rows = []
for r in ws.iter_rows(min_row=2):
    if r[0].value is None:
        continue
    fills = [(c.fill.fgColor.rgb if c.fill and c.fill.fgColor is not None else "") or "" for c in r[2:10]]
    imp = [isinstance(f, str) and f.endswith("FFFF00") for f in fills]
    rows.append({
        "Month": pd.Timestamp(r[0].value), "Station": r[1].value,
        "BOD": r[2].value, "DO": r[3].value, "FC": r[4].value, "pH": r[5].value, "Ammonia": r[6].value,
        "Nitrate": r[7].value, "Phosphate": r[8].value, "Temperature": r[9].value,
        "Outlier_flag": r[10].value or "",
        "BOD_imputed": imp[0], "imputed_cols": ",".join(n for n, f in zip(
            ["BOD", "DO", "FC", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"], imp) if f),
    })
df = pd.DataFrame(rows)

rogue = pd.Series(False, index=df.index)
for col, (lo, hi) in VALID.items():
    bad = (df[col] < lo) | ((df[col] > hi) if hi is not None else False)
    rogue |= bad
print("rogue (physically impossible) rows:", int(rogue.sum()))
df = df[~rogue].reset_index(drop=True)

df["BOD_Class"] = [NAMES[i] for i in np.digitize(df["BOD"], EDGES, right=True)]
df["Outlier_flag"] = df["Outlier_flag"].str.replace("TEMP", "Temperature").str.replace("NH3", "Ammonia") \
    .str.replace("NO3", "Nitrate").str.replace("PO4", "Phosphate")

out = df[["Month", "Station", "DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature", "BOD",
          "Outlier_flag", "BOD_imputed", "BOD_Class"]].copy()
out.insert(0, "Month_str", out.pop("Month").dt.strftime("%Y-%m"))
out = out.rename(columns={"Month_str": "Month", "BOD": "BOD_mgL"})
out.to_csv(OUT_CSV, index=False)
print("rows:", len(out))
print(out["BOD_Class"].value_counts().reindex(NAMES).to_dict(), "| BOD > 15:", int((df.BOD > 15).sum()))
print("flagged (kept):", int((out.Outlier_flag != "").sum()))

# ---- sheet for the workbook ----
wb = Workbook()
sh = wb.active
sh.title = "Preprocessing v4 (flagged)"
head = ["Month", "Stations", "BOD (mg/L)", "Dissolved Oxygen (mg/L)", "Fecal Coliform, MPN/100ml (Geomean)",
        "pH (units)", "Ammonia (mg/L)", "Nitrate (mg/L)", "Inorganic Phospate (mg/L)", "Temperature (°C)",
        "BOD class (DAO 2016-08)", "Outlier flag (kept, review)"]
sh.append(head)
for c in sh[1]:
    c.font = Font(name="Calibri", size=11, bold=True)
YELLOW = PatternFill("solid", fgColor="FFFF00")
ORANGE = PatternFill("solid", fgColor="F4B183")
cols = ["BOD", "DO", "FC", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"]
for _, r in df.iterrows():
    sh.append([r.Month.to_pydatetime(), r.Station] + [r[c] for c in cols] + [r.BOD_Class, r.Outlier_flag])
    i = sh.max_row
    sh.cell(i, 1).number_format = "mmm-yy"
    for j, c in enumerate(cols):
        if c in r.imputed_cols.split(","):
            sh.cell(i, 3 + j).fill = YELLOW
    if r.Outlier_flag:
        sh.cell(i, 12).fill = ORANGE
for col, w in zip("ABCDEFGHIJKL", [12.6, 40, 8.6, 12.7, 20, 10.7, 14, 14.7, 16, 14, 16, 26]):
    sh.column_dimensions[col].width = w
sh.cell(1, 14, "Same rows as Preprocessing v3A (gaps filled, yellow = imputed). Outliers are NOT removed: "
               "column L names the parameter(s) outside the v3 limits, for review (orange). "
               "Only physically impossible values would be removed (none found). "
               "BOD class: A <=3, B >3-5, C >5-7, D >7 mg/L (DAO 2016-08; the one value above 15, "
               "Sta. Rosa Sep 2017 = 70 mg/L, is kept as D and was confirmed in the LLDA report).")
wb.save(OUT_XLSX)
print("wrote", OUT_XLSX)
