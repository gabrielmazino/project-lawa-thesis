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

from lawa import bod5
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
BOD_MODEL = joblib.load(HERE / "models" / "bod_model.joblib")              # BOD value (mg/L)
BOD_CLASS_MODEL = joblib.load(HERE / "models" / "bod_class_model.joblib")  # BOD class, separate classifier
FC_MODEL = joblib.load(HERE / "models" / "fc_model.joblib")                # FC value; its class comes from the value
MODEL_READINGS = sorted(set(BOD_MODEL["readings"]) | set(BOD_CLASS_MODEL["readings"]) | set(FC_MODEL["readings"]),
                        key=READINGS.index)                                  # Phosphate is not used by any model


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


def mean_scores(scores):
    """{test: {score: {mean, sd}}} -> JSON-friendly, rounded."""
    return {test: {k: {"mean": round(v["mean"], 4), "sd": round(v["sd"], 4)} for k, v in s.items() if isinstance(v, dict)}
            | {"repeats": s["repeats"]} for test, s in scores.items()}


@app.get("/api/models")
@require_login
def models():
    def info(m, last_label):
        return {k: m[k] for k in ["target", "algorithm", "predicts", "trained_on", "created"]} | {
            "inputs": m["readings"] + ["station", "month"] + ([last_label] if m["uses_last_month"] else []),
            "scores": mean_scores(m["scores"])}
    return jsonify({"BOD": info(BOD_MODEL, "last month's BOD"), "BOD_class": info(BOD_CLASS_MODEL, "last month's BOD"),
                    "FC": info(FC_MODEL, "last month's fecal coliform"), "readings": MODEL_READINGS,
                    "evaluation": "random 70-30 split of 2016-2025 repeated with 10 seeds (mean ± SD); "
                                  "rolling = each year 2018-2025 predicted from earlier years"})


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
    for name in MODEL_READINGS + ["BOD_last", "FC_last"]:
        try:
            values[name] = float(body.get(name))
        except (TypeError, ValueError):
            problems.append(name)
    if problems:
        return error("Missing or invalid: " + ", ".join(problems))
    for name, (low, high) in VALID_RANGES.items():
        if name in values and not low <= values[name] <= high:
            return error(f"{name} = {values[name]} is not physically possible (allowed {low}–{high}).")
    if values["BOD_last"] < 0 or values["FC_last"] <= 0:
        return error("Last month's BOD must be ≥ 0 and fecal coliform must be > 0.")

    warnings = [f"{name} = {values[name]} is outside the training data range "
                f"({low:g}–{high:g}); the estimate is less reliable."
                for name, (low, high) in BOD_MODEL["input_ranges"].items()
                if name in values and not low <= values[name] <= high]

    row = pd.DataFrame([{"Month": month_ts, "Station": station, **values}])

    def inputs(m, last_column):
        return build_features(row, m["stations"], last_column, m["readings"])[m["features"]]

    bod = max(float(BOD_MODEL["model"].predict(inputs(BOD_MODEL, "BOD_last"))[0]), 0.0)
    proba = BOD_CLASS_MODEL["model"].predict_proba(inputs(BOD_CLASS_MODEL, "BOD_last"))[0]
    names = [BOD_CLASS_MODEL["class_names"][int(c)] for c in BOD_CLASS_MODEL["model"].classes_]
    best = int(np.argmax(proba))
    change = float(FC_MODEL["model"].predict(inputs(FC_MODEL, "FC_last"))[0])
    fc = 10 ** (math.log10(values["FC_last"]) + change)

    # typical error = median RMSE over the 10 random 70-30 splits (the median is not pulled up by the
    # few splits that test the single 70 mg/L month)
    bod_err = BOD_MODEL["scores"]["random_70_30"]["rmse"]["median"]            # mg/L
    fc_factor = 10 ** FC_MODEL["scores"]["random_70_30"]["rmse"]["median"]     # multiply/divide factor
    return jsonify({
        "station": station, "month": month, "warnings": warnings,
        "BOD": {"estimate": round(bod, 2), "class": names[best], "class_confidence": round(float(proba[best]), 3),
                "class_probabilities": {n: round(float(p), 3) for n, p in zip(names, proba)},
                "class_from_estimate": bod_class(bod),
                "above_standard": names[best] == "D",       # classifier's D = D or worse (above 7 mg/L)
                "typical_range": [round(max(bod - bod_err, 0), 2), round(bod + bod_err, 2)],
                "last_month": values["BOD_last"], "algorithm": BOD_MODEL["algorithm"],
                "class_algorithm": BOD_CLASS_MODEL["algorithm"]},
        "FC": {"estimate": round(fc, 1), "class": fc_class(fc), "above_standard": above_standard("FC", fc),
               "typical_range": [round(fc / fc_factor, 1), round(fc * fc_factor, 1)],
               "last_month": values["FC_last"], "change_percent": round((10 ** change - 1) * 100, 1),
               "algorithm": FC_MODEL["algorithm"]},
        "standard": LAKE_STANDARD,
    })


# ---------------- BOD5 computational lab model (demo, synthetic lab data) ----------------
BOD5_SAMPLES = bod5.synthetic_samples()


@app.get("/api/bod5/synthetic")
@require_login
def bod5_synthetic():
    """Synthetic lab sheet rows for one month (default: latest), to demo the BOD5 calculator."""
    months = sorted(BOD5_SAMPLES["month"].unique())
    month = request.args.get("month", months[-1])
    rows = BOD5_SAMPLES[BOD5_SAMPLES["month"] == month].sort_values("station")
    cols = ["sample_id", "month", "station", "sample_ml", "bottle_ml", "d1", "d2", "llda_bod"]
    return jsonify({"month": month, "months": months,
                    "samples": [{k: clean(r[k]) for k in cols} for _, r in rows.iterrows()]})


@app.post("/api/bod5/compute")
@require_login
def bod5_compute():
    """Unseeded BOD5 for a batch of samples: BOD5 = (D1 - D2) / P, P = sample volume / bottle volume."""
    samples = (request.get_json(silent=True) or {}).get("samples") or []
    if not samples:
        return error("Add at least one sample.")
    if len(samples) > 500:
        return error("At most 500 samples at a time.")
    results = []
    for i, s in enumerate(samples, start=1):
        try:
            values = {k: float(s.get(k)) for k in bod5.LIMITS}
        except (TypeError, ValueError):
            return error(f"Sample {i}: sample volume, bottle volume, D1 and D2 must be numbers.")
        for k, (low, high) in bod5.LIMITS.items():
            if not low <= values[k] <= high:
                return error(f"Sample {i}: {k} = {values[k]} is outside {low}–{high}.")
        try:
            out = bod5.compute(values["sample_ml"], values["bottle_ml"], values["d1"], values["d2"])
        except ValueError as e:
            return error(f"Sample {i}: {e}")
        llda = s.get("llda_bod")
        llda = float(llda) if llda not in (None, "") else None
        results.append({"sample_id": s.get("sample_id") or f"Sample {i}", "station": s.get("station"),
                        "month": s.get("month"), **values, **out, "llda_bod": llda,
                        "difference": round(out["BOD5"] - llda, 3) if llda is not None else None})
    summary = {"samples": len(results),
               "valid": sum(r["qc"] == "Valid" for r in results),
               "below_range": sum(r["qc"] == "Below range" for r in results),
               "invalid": sum(r["qc"] == "Invalid" for r in results),
               "above_standard": sum(bool(r["above_standard"]) for r in results)}
    return jsonify({"results": results, "summary": summary, "standard": LAKE_STANDARD["BOD"]})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
