# Project LAWA

Thesis project: ML soft sensor estimating BOD and fecal coliform for Laguna Lake (LLDA data), plus a React +
Supabase dashboard (`frontend/`). Manuscript: `thesis_manuscript/*.docx`.

**Read `ml/PROJECT_NOTES.md` first.** It records all decisions, files, results and the prioritized TO-DO list
from earlier sessions (done on another computer, so that conversation is not available here).

Rules from the user:
- Do NOT edit the thesis manuscript until the user explicitly says so.
- TSS is not used in the models (no TSS data for 2016–2023).
- Do not write the LLDA workbook with exceljs (it corrupts it); use `dataset/preprocessing-v3/merge.ps1` (Excel).
- Keep earlier results folders; save new experiments to new folders.
- Run Python with `ml/.venv/Scripts/python.exe` (create the venv first on a new machine; see notes section 6).
