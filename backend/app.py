"""Project LAWA API (Flask): monitoring data, dashboard summaries and soft sensor predictions.

Login is handled by Supabase. Every /api request (except /api/health) must send the user's Supabase
access token ("Authorization: Bearer <token>"); Flask asks Supabase whether it is valid.
Flask never sees passwords.

Run (development):  backend/.venv/Scripts/python.exe backend/app.py   -> http://localhost:5000
"""
import math
import os
import time
from functools import wraps
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, g, jsonify, request
from flask_cors import CORS

from lawa.data import MAX_LAG_MONTHS, READINGS, add_last_values, build_features, load_records
from lawa.denr import LAKE_STANDARD, above_standard, bod_class, fc_class

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": FRONTEND_ORIGIN}})

# Physically possible input values (anything outside is rejected as a typo)
VALID_RANGES = {"DO": (0, 25), "pH": (0, 14), "Ammonia": (0, 50), "Nitrate": (0, 100),
                "Phosphate": (0, 20), "Temperature": (0, 45)}

RECORDS = add_last_values(load_records())
MONTHS = sorted(RECORDS["Month"].dt.strftime("%Y-%m").unique())
STATIONS = sorted(RECORDS["Station"].unique())
BOD_MODEL = joblib.load(HERE / "models" / "bod_model.joblib")
FC_MODEL = joblib.load(HERE / "models" / "fc_model.joblib")


# ---------------- helpers ----------------
def clean(value):
    """JSON-safe value: numpy -> python, NaN -> None."""
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def record_json(row):
    out = {"month": row["Month"].strftime("%Y-%m"), "station": row["Station"]}
    out.update({r: clean(row[r]) for r in READINGS})
    out.update({
        "BOD": clean(row["BOD"]), "BOD_class": bod_class(row["BOD"]), "BOD_filled": bool(row["BOD_imputed"]),
        "BOD_above_standard": above_standard("BOD", row["BOD"]),
        "FC": clean(row["FC"]), "FC_class": fc_class(row["FC"]), "FC_filled": bool(row["FC_imputed"]),
        "FC_above_standard": above_standard("FC", row["FC"]),
        "outlier_flag": row["Outlier_flag"],
    })
    return out


def error(message, status=400):
    return jsonify({"error": message}), status


# ---------------- auth ----------------
_token_cache = {}  # token -> (user json, expiry time); avoids asking Supabase on every request


def require_login(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not SUPABASE_URL or not SUPABASE_KEY:
            return error("Server is missing SUPABASE_URL / SUPABASE_PUBLISHABLE_KEY in backend/.env", 500)
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        if not token:
            return error("Not logged in.", 401)
        cached = _token_cache.get(token)
        if cached and cached[1] > time.time():
            g.user = cached[0]
            return view(*args, **kwargs)
        try:
            resp = requests.get(f"{SUPABASE_URL}/auth/v1/user",
                                headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_KEY}, timeout=10)
        except requests.RequestException:
            return error("Cannot reach the login server.", 503)
        if resp.status_code != 200:
            return error("Your session has expired. Please sign in again.", 401)
        g.user = resp.json()
        _token_cache[token] = (g.user, time.time() + 60)
        return view(*args, **kwargs)
    return wrapper


# ---------------- routes ----------------
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "records": len(RECORDS), "latest_month": MONTHS[-1]})


@app.get("/api/stations")
@require_login
def stations():
    return jsonify({"stations": STATIONS, "months": MONTHS})


@app.get("/api/overview")
@require_login
def overview():
    """All stations for one month (default: latest), plus summary numbers for the dashboard cards."""
    month = request.args.get("month", MONTHS[-1])
    if month not in MONTHS:
        return error(f"No data for {month}.")
    rows = RECORDS[RECORDS["Month"].dt.strftime("%Y-%m") == month]
    prev = MONTHS[MONTHS.index(month) - 1] if MONTHS.index(month) > 0 else None
    prev_rows = RECORDS[RECORDS["Month"].dt.strftime("%Y-%m") == prev] if prev else rows.iloc[0:0]

    def summary(frame):
        if frame.empty:
            return None
        return {
            "stations_reported": int(len(frame)),
            "avg_BOD": clean(frame["BOD"].mean()),
            "geomean_FC": clean(10 ** np.log10(frame["FC"]).mean()),
            "BOD_above_standard": int((frame["BOD"] > LAKE_STANDARD["BOD"]).sum()),
            "FC_above_standard": int((frame["FC"] > LAKE_STANDARD["FC"]).sum()),
            "values_filled": int((frame["BOD_imputed"] | frame["FC_imputed"]).sum()),
        }

    return jsonify({
        "month": month, "previous_month": prev,
        "summary": summary(rows), "previous_summary": summary(prev_rows),
        "stations": [record_json(r) for _, r in rows.sort_values("Station").iterrows()],
        "standard": LAKE_STANDARD,
    })


@app.get("/api/trend")
@require_login
def trend():
    """Full monthly history of one station (BOD, fecal coliform and readings)."""
    station = request.args.get("station", "")
    if station not in STATIONS:
        return error("Unknown station.")
    rows = RECORDS[RECORDS["Station"] == station].sort_values("Month")
    return jsonify({"station": station, "records": [record_json(r) for _, r in rows.iterrows()],
                    "standard": LAKE_STANDARD})


@app.get("/api/last-values")
@require_login
def last_values():
    """The station's most recent lab BOD and fecal coliform BEFORE a month (auto-fills the soft sensor form)."""
    station, month = request.args.get("station", ""), request.args.get("month", "")
    if station not in STATIONS:
        return error("Unknown station.")
    try:
        target = pd.Timestamp(month + "-01")
    except ValueError:
        return error("Month must look like 2026-01.")
    earlier = RECORDS[(RECORDS["Station"] == station) & (RECORDS["Month"] < target)].sort_values("Month")
    if earlier.empty:
        return jsonify({"found": False})
    last = earlier.iloc[-1]
    gap = (target.year - last["Month"].year) * 12 + (target.month - last["Month"].month)
    same_month = RECORDS[(RECORDS["Station"] == station) & (RECORDS["Month"] == target)]
    return jsonify({
        "found": True, "from_month": last["Month"].strftime("%Y-%m"), "gap_months": int(gap),
        "too_old": bool(gap > MAX_LAG_MONTHS), "max_gap_months": MAX_LAG_MONTHS,
        "BOD": clean(last["BOD"]), "FC": clean(last["FC"]),
        "recorded_this_month": record_json(same_month.iloc[0]) if not same_month.empty else None,
    })


@app.get("/api/models")
@require_login
def models():
    def info(m):
        return {k: m[k] for k in ["target", "algorithm", "predicts", "trained_on", "created", "holdout"]} | {
            "inputs": READINGS + ["station", "month", "last month's lab value"]}
    return jsonify({"BOD": info(BOD_MODEL), "FC": info(FC_MODEL)})


@app.post("/api/predict")
@require_login
def predict():
    """Soft sensor: readings + station + month + last lab values -> estimated BOD and fecal coliform."""
    body = request.get_json(silent=True) or {}
    station, month = body.get("station"), body.get("month", "")
    if station not in STATIONS:
        return error("Choose a monitoring station.")
    try:
        month_ts = pd.Timestamp(month + "-01")
    except ValueError:
        return error("Month must look like 2026-01.")

    values, problems = {}, []
    for name in READINGS + ["BOD_last", "FC_last"]:
        try:
            values[name] = float(body.get(name))
        except (TypeError, ValueError):
            problems.append(name)
    if problems:
        return error("Missing or invalid: " + ", ".join(problems))
    for name, (low, high) in VALID_RANGES.items():
        if not low <= values[name] <= high:
            return error(f"{name} = {values[name]} is not physically possible (allowed {low}–{high}).")
    if values["BOD_last"] < 0 or values["FC_last"] <= 0:
        return error("Last month's BOD must be ≥ 0 and fecal coliform must be > 0.")

    warnings = [f"{name} = {values[name]} is outside the training data range "
                f"({low:g}–{high:g}); the estimate is less reliable."
                for name, (low, high) in BOD_MODEL["input_ranges"].items() if not low <= values[name] <= high]

    row = pd.DataFrame([{"Month": month_ts, "Station": station, **values}])
    bod = max(float(BOD_MODEL["model"].predict(build_features(row, BOD_MODEL["stations"], "BOD_last")[BOD_MODEL["features"]])[0]), 0.0)
    change = float(FC_MODEL["model"].predict(build_features(row, FC_MODEL["stations"], "FC_last")[FC_MODEL["features"]])[0])
    fc = 10 ** (math.log10(values["FC_last"]) + change)

    bod_err = BOD_MODEL["holdout"]["rmse"]  # typical error in mg/L
    fc_factor = 10 ** FC_MODEL["holdout"]["rmse"]  # typical error as a multiply/divide factor
    return jsonify({
        "station": station, "month": month, "warnings": warnings,
        "BOD": {"estimate": round(bod, 2), "class": bod_class(bod), "above_standard": above_standard("BOD", bod),
                "typical_range": [round(max(bod - bod_err, 0), 2), round(bod + bod_err, 2)],
                "last_month": values["BOD_last"], "algorithm": BOD_MODEL["algorithm"]},
        "FC": {"estimate": round(fc, 1), "class": fc_class(fc), "above_standard": above_standard("FC", fc),
               "typical_range": [round(fc / fc_factor, 1), round(fc * fc_factor, 1)],
               "last_month": values["FC_last"], "change_percent": round((10 ** change - 1) * 100, 1),
               "algorithm": FC_MODEL["algorithm"]},
        "standard": LAKE_STANDARD,
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
