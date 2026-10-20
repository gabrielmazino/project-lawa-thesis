"""BOD5 computational lab model: synthetic laboratory sheet + Excel template.

The adviser asked for a computational model of LLDA's BOD5 laboratory test, built in Excel first:
    BOD5 = (D1 - D2 - seed correction) / P          (unseeded lake samples: seed correction = 0)
    P    = sample volume / total bottle volume
LLDA's raw lab sheets (D1, D2, dilution) are not available, so a SYNTHETIC lab sheet is generated
that is consistent with LLDA's reported BOD (the adviser allowed synthetic data):

  rows       one per real station-month with a MEASURED BOD (interpolated values are skipped)
  bottle     300 mL standard BOD bottle
  D1         initial DO of the bottle, ~N(8.6, 0.2) mg/L, kept within 8.0-9.1 (aerated, 20 C)
  dilution   lake-type: the largest standard sample volume that leaves >= 1.5 mg/L expected final DO
             (clean lake water -> 300 mL, undiluted, P = 1; only very high BOD months are diluted)
  D2         D1 - BOD x P + DO-meter noise N(0, 0.03), both readings rounded to 0.01 mg/L
  validity   Standard Methods rule: DO depletion >= 2.0 mg/L and final DO >= 1.0 mg/L

Every computed column in the Excel sheets is a live formula, so the same sheet works as a template.
Output: dataset/bod5/bod5_lab_sheets.xlsx (sheets to copy into the LLDA workbook with
dataset/preprocessing-v3/merge.ps1) and dataset/bod5/bod5_synthetic_lab.csv.

Run:  ml/.venv/Scripts/python.exe ml/bod5/make_synthetic_lab.py
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "bod5"
OUT.mkdir(exist_ok=True)
SEED = 42
BOTTLE_ML = 300
STANDARD_VOLUMES_ML = [300, 250, 200, 150, 100, 75, 50, 30, 20, 15, 10, 5]
MIN_EXPECTED_FINAL_DO = 1.5      # margin above the 1.0 mg/L rule, so meter noise cannot break it
TEMPLATE_ROWS = 60

# ---------------------------------------------------------------- synthetic data
rng = np.random.default_rng(SEED)
src = pd.read_csv(ROOT / "dataset" / "bod_class_dataset.csv")
src = src[~src["BOD_imputed"]].sort_values(["Month", "Station"]).reset_index(drop=True)


def station_code(name):
    m = re.match(r"Stn\.\s*([IVXL]+)", name)
    return m.group(1) if m else name.split()[0]


rows = []
for _, r in src.iterrows():
    bod = float(r["BOD_mgL"])
    d1 = float(np.clip(rng.normal(8.6, 0.2), 8.0, 9.1))
    vol = next((v for v in STANDARD_VOLUMES_ML if d1 - bod * v / BOTTLE_ML >= MIN_EXPECTED_FINAL_DO),
               STANDARD_VOLUMES_ML[-1])
    p = vol / BOTTLE_ML
    d2 = d1 - bod * p + rng.normal(0, 0.03)
    d1, d2 = round(d1, 2), round(d2, 2)
    rows.append({"Sample ID": f"STN{station_code(r['Station'])}-{r['Month']}", "Month": r["Month"],
                 "Station": r["Station"], "Sample volume (mL)": vol, "Bottle volume (mL)": BOTTLE_ML,
                 "D1": d1, "D2": d2, "Seed correction": 0.0, "LLDA BOD": bod})
syn = pd.DataFrame(rows)
syn["P"] = syn["Sample volume (mL)"] / syn["Bottle volume (mL)"]
syn["BOD5 computed"] = (syn["D1"] - syn["D2"] - syn["Seed correction"]) / syn["P"]
syn.to_csv(OUT / "bod5_synthetic_lab.csv", index=False)

# ---------------------------------------------------------------- Excel sheets
HEAD = ["Sample ID", "Month", "Station", "Sample volume (mL)", "Bottle volume (mL)", "P (sample fraction)",
        "D1 initial DO (mg/L)", "D2 DO after 5 days (mg/L)", "DO depletion (mg/L)", "Seed correction (mg/L)",
        "BOD5 computed (mg/L)", "LLDA reported BOD (mg/L)", "Difference (mg/L)", "DENR class",
        "Above standard (Class C)", "QC check"]
WIDTH = [16, 9, 30, 11, 11, 11, 12, 13, 12, 12, 13, 13, 12, 9, 12, 30]
BLUE, LIGHT, GREEN, AMBER, RED = "1F5F8B", "EEF3F6", "E3F1E8", "FBF1E8", "F6DEDB"
thin = Side(style="thin", color="C9D3DA")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)


def formulas(r, guard):
    """Live formulas for row r. guard=True leaves the cell blank until D1/D2/volumes are entered."""
    f = {
        "F": f"=D{r}/E{r}",
        "I": f"=G{r}-H{r}",
        "K": f"=(G{r}-H{r}-J{r})/F{r}",
        "M": f"=IF(L{r}=\"\",\"\",K{r}-L{r})",
        "N": f"=IF(ROUND(K{r},1)<=3,\"A\",IF(ROUND(K{r},1)<=5,\"B\",IF(ROUND(K{r},1)<=7,\"C\","
             f"IF(ROUND(K{r},1)<=15,\"D\",\"Failed\"))))",
        "O": f"=IF(ROUND(K{r},1)>7,\"Yes\",\"No\")",
        "P": f"=IF(H{r}<1,\"Invalid: final DO below 1.0 mg/L\",IF(I{r}<2,\"Below range: depletion < 2.0 mg/L\","
             f"\"Valid\"))",
    }
    if guard:
        need = f"OR(D{r}=\"\",E{r}=\"\",G{r}=\"\",H{r}=\"\")"
        f = {k: f"=IF({need},\"\",{v[1:]})" for k, v in f.items()}
    return f


def header(ws):
    for c, (h, w) in enumerate(zip(HEAD, WIDTH), start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = BOX
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[1].height = 45
    ws.freeze_panes = "D2"


def style_rows(ws, n_rows, input_cols):
    for r in range(2, n_rows + 2):
        for c in range(1, len(HEAD) + 1):
            cell = ws.cell(row=r, column=c)
            cell.border = BOX
            letter = get_column_letter(c)
            if letter in input_cols:
                cell.fill = PatternFill("solid", fgColor="FFFFFF")
            elif letter in "FIKMNOP":
                cell.fill = PatternFill("solid", fgColor=LIGHT)
            if letter in "GHIJKLM":
                cell.number_format = "0.00"
            if letter == "F":
                cell.number_format = "0.000"
            if letter in "NO":
                cell.alignment = Alignment(horizontal="center")
    last = n_rows + 1
    ws.conditional_formatting.add(f"P2:P{last}", FormulaRule(formula=['LEFT($P2,7)="Invalid"'],
                                                            fill=PatternFill("solid", fgColor=RED)))
    ws.conditional_formatting.add(f"P2:P{last}", FormulaRule(formula=['LEFT($P2,5)="Below"'],
                                                            fill=PatternFill("solid", fgColor=AMBER)))
    ws.conditional_formatting.add(f"P2:P{last}", FormulaRule(formula=['$P2="Valid"'],
                                                            fill=PatternFill("solid", fgColor=GREEN)))
    ws.conditional_formatting.add(f"O2:O{last}", FormulaRule(formula=['$O2="Yes"'],
                                                            fill=PatternFill("solid", fgColor=RED)))


def panel(ws, col, synthetic):
    """Formula panel beside the table."""
    c1, c2 = get_column_letter(col), get_column_letter(col + 1)
    ws.column_dimensions[c1].width = 30
    ws.column_dimensions[c2].width = 100
    lines = [
        ("FORMULAS USED", None, "title"),
        ("BOD5 (5-day BOD, mg/L)", "BOD5 = (D1 − D2 − seed correction) / P", "formula"),
        ("  in this sheet", "K = (G − H − J) / F", "cell"),
        ("P (fraction of sample)", "P = sample volume / total bottle volume", "formula"),
        ("  in this sheet", "F = D / E", "cell"),
        ("DO depletion (mg/L)", "D1 − D2   (oxygen used by bacteria in 5 days)", "formula"),
        ("  in this sheet", "I = G − H", "cell"),
        ("Seed correction", "0 for unseeded samples (LLDA lake samples are unseeded)", "formula"),
        ("Difference", "M = K − L   (computed − LLDA reported)", "formula"),
        ("", "", None),
        ("WHAT EACH VALUE MEANS", None, "title"),
        ("D1", "DO of the bottle at the start (after filling, before incubation)", None),
        ("D2", "DO of the same bottle after 5 days in the dark at 20 °C", None),
        ("Sample volume", "mL of lake water put in the bottle", None),
        ("Bottle volume", "300 mL standard BOD bottle (rest = dilution water)", None),
        ("", "", None),
        ("WORKED EXAMPLE (lake water)", None, "title"),
        ("Inputs", "Sample 300 mL in a 300 mL bottle, D1 = 8.60, D2 = 6.00, unseeded", None),
        ("P", "300 / 300 = 1", None),
        ("BOD5", "(8.60 − 6.00 − 0) / 1 = 2.60 mg/L  →  Class A", None),
        ("Wastewater example", "15 mL in 300 mL: P = 0.05; D1 8.50, D2 6.00 → BOD5 = 50 mg/L", None),
        ("", "", None),
        ("VALIDITY RULES (QC check)", None, "title"),
        ("Valid", "DO depletion ≥ 2.0 mg/L AND final DO (D2) ≥ 1.0 mg/L", None),
        ("Below range", "Depletion < 2.0 mg/L: too little oxygen used to measure reliably", None),
        ("Invalid", "D2 < 1.0 mg/L: oxygen ran out; repeat with more dilution", None),
        ("Note", "Clean lake water often uses < 2 mg/L even undiluted; labs report such results with a qualifier", None),
        ("", "", None),
        ("DENR CLASS (DAO 2016-08, BOD mg/L)", None, "title"),
        ("A", "≤ 3", None), ("B", "> 3 – 5", None), ("C", "> 5 – 7", None), ("D", "> 7 – 15", None),
        ("Failed", "> 15", None),
        ("Above standard", "Laguna Lake is Class C: above standard = BOD > 7 mg/L", None),
        ("Rounding", "Class uses BOD5 rounded to 1 decimal, as in the LLDA reports", None),
    ]
    if synthetic:
        lines += [
            ("", "", None),
            ("HOW THIS SYNTHETIC SHEET WAS MADE", None, "title"),
            ("Why synthetic", "LLDA's raw lab sheets (D1, D2, dilution) are not available; the adviser allowed synthetic data", None),
            ("Rows", "One per station-month with a measured LLDA BOD (filled-in values skipped)", None),
            ("D1", "Random, about 8.6 mg/L (range 8.0–9.1): an aerated bottle at 20 °C", None),
            ("Sample volume", "Largest standard volume leaving ≥ 1.5 mg/L final DO (lake water: 300 mL, P = 1)", None),
            ("D2", "D1 − LLDA BOD × P + meter noise (± 0.03 mg/L), rounded to 0.01", None),
            ("Use", "To build and validate the BOD5 computation; not real laboratory measurements", None),
        ]
    else:
        lines += [
            ("", "", None),
            ("HOW TO USE", None, "title"),
            ("Enter", "White cells: Sample ID, Month, Station, volumes, D1, D2, seed correction (0 if unseeded), LLDA BOD (optional)", None),
            ("Automatic", "Blue cells compute P, depletion, BOD5, difference, DENR class, standard and QC check", None),
        ]
    for i, (a, b, kind) in enumerate(lines, start=1):
        ca, cb = ws.cell(row=i, column=col, value=a), ws.cell(row=i, column=col + 1, value=b)
        cb.alignment = Alignment(vertical="top")
        ca.alignment = Alignment(vertical="top")
        if kind == "title":
            for cell in (ca, cb):
                cell.fill = PatternFill("solid", fgColor=BLUE)
                cell.font = Font(bold=True, color="FFFFFF")
        elif kind == "formula":
            ca.font = Font(bold=True)
            cb.font = Font(bold=True, name="Consolas", color=BLUE)
            cb.fill = PatternFill("solid", fgColor=LIGHT)
        elif kind == "cell":
            ca.font = Font(italic=True, color="5B6770")
            cb.font = Font(name="Consolas", color="5B6770")
        elif a:
            ca.font = Font(bold=True)


wb = Workbook()
ws = wb.active
ws.title = "BOD5 Lab (Synthetic)"
header(ws)
for i, row in syn.iterrows():
    r = i + 2
    for c, v in zip("ABCDEGHJL", [row["Sample ID"], row["Month"], row["Station"], row["Sample volume (mL)"],
                                  row["Bottle volume (mL)"], row["D1"], row["D2"], row["Seed correction"],
                                  row["LLDA BOD"]]):
        ws[f"{c}{r}"] = v
    for c, f in formulas(r, guard=False).items():
        ws[f"{c}{r}"] = f
style_rows(ws, len(syn), input_cols="ABCDEGHJL")
ws.auto_filter.ref = f"A1:P{len(syn) + 1}"
panel(ws, len(HEAD) + 2, synthetic=True)

tp = wb.create_sheet("BOD5 Template")
header(tp)
for r in range(2, TEMPLATE_ROWS + 2):
    tp[f"E{r}"] = BOTTLE_ML
    tp[f"J{r}"] = 0
    for c, f in formulas(r, guard=True).items():
        tp[f"{c}{r}"] = f
style_rows(tp, TEMPLATE_ROWS, input_cols="ABCDEGHJL")
panel(tp, len(HEAD) + 2, synthetic=False)

path = OUT / "bod5_lab_sheets.xlsx"
wb.save(path)

# ---------------------------------------------------------------- summary
valid = (syn["D1"] - syn["D2"] >= 2) & (syn["D2"] >= 1)
below = (syn["D1"] - syn["D2"] < 2) & (syn["D2"] >= 1)
same_class = (pd.cut(syn["BOD5 computed"].round(1), [-np.inf, 3, 5, 7, 15, np.inf]).astype(str)
              == pd.cut(syn["LLDA BOD"], [-np.inf, 3, 5, 7, 15, np.inf]).astype(str))
print(f"rows: {len(syn)} (measured station-months)")
print("sample volumes used:", syn["Sample volume (mL)"].value_counts().sort_index(ascending=False).to_dict())
print(f"D1 range {syn.D1.min():.2f}-{syn.D1.max():.2f}, D2 range {syn.D2.min():.2f}-{syn.D2.max():.2f}")
print(f"QC: valid {valid.sum()}, below range {below.sum()}, invalid {(syn.D2 < 1).sum()}")
d = syn["BOD5 computed"] - syn["LLDA BOD"]
print(f"computed - LLDA: mean {d.mean():+.3f}, mean |diff| {d.abs().mean():.3f}, max |diff| {d.abs().max():.3f}")
print(f"same DENR class: {100 * same_class.mean():.1f}%")
print(f"saved {path} and {OUT / 'bod5_synthetic_lab.csv'}")
