"""
VitalSense AI — Flask Application (Railway Safe Version)
Run locally: python app.py
Deploy on Railway with: gunicorn app:app
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = "vitalsense-secret-key"

# ─────────────────────────────────────────────
# LOAD ML MODEL & DATA
# ─────────────────────────────────────────────

MODEL = None
DF    = None
hr_history = []

def load_resources():
    global MODEL, DF

    try:
        MODEL = pickle.load(open("health_model.pkl", "rb"))
    except Exception:
        print("⚠ Model not found — using dummy model")
        from sklearn.ensemble import RandomForestClassifier
        MODEL = RandomForestClassifier()
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        MODEL.fit(X, y)

    try:
        DF = pd.read_csv("health_data_labeled.csv")
    except Exception:
        print("⚠ Dataset not found — using dummy data")
        DF = pd.DataFrame(
            np.random.rand(100, 5),
            columns=["heart_rate", "spo2", "temperature", "activity", "hrv"]
        )

load_resources()

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def predict_risk(heart_rate, spo2, temperature, activity, hrv):
    data = pd.DataFrame([[heart_rate, spo2, temperature, activity, hrv]],
                        columns=["heart_rate", "spo2", "temperature", "activity", "hrv"])
    return int(MODEL.predict(data)[0])

# ─────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/dashboard")
def dashboard():
    module = request.args.get("module", "face")
    return render_template("dashboard.html", module=module)

# ─────────────────────────────────────────────
# API — FACE SCANNER (Railway Safe)
# ─────────────────────────────────────────────

@app.route("/api/face/frame")
def face_frame():

    stress  = int(np.random.randint(0, 100))
    fatigue = int(np.random.randint(0, 100))

    return jsonify({
        "image": None,
        "face_count": 1,
        "stress": stress,
        "fatigue": fatigue,
        "stress_cls":  "danger" if stress  > 70 else ("warn" if stress  > 45 else "ok"),
        "fatigue_cls": "danger" if fatigue > 70 else ("warn" if fatigue > 45 else "ok"),
    })

# ─────────────────────────────────────────────
# API — VOICE RISK (Railway Safe)
# ─────────────────────────────────────────────

@app.route("/api/voice/record", methods=["POST"])
def voice_record():

    anxiety = int(np.random.randint(0, 100))
    asthma  = int(np.random.randint(0, 100))

    return jsonify({
        "anxiety": anxiety,
        "asthma": asthma,
        "anxiety_cls":  "danger" if anxiety > 70 else ("warn" if anxiety > 45 else "ok"),
        "asthma_cls":   "danger" if asthma  > 70 else ("warn" if asthma  > 45 else "ok"),
        "anxiety_lbl":  "⚠ ELEVATED" if anxiety > 70 else ("△ MODERATE" if anxiety > 45 else "✓ NORMAL"),
        "asthma_lbl":   "⚠ ELEVATED" if asthma  > 70 else ("△ MODERATE" if asthma  > 45 else "✓ NORMAL"),
    })

# ─────────────────────────────────────────────
# API — HEART MONITOR (Railway Safe)
# ─────────────────────────────────────────────

@app.route("/api/heart/frame")
def heart_frame():

    global hr_history

    last_hr    = hr_history[-1] if hr_history else 75
    heart_rate = int(np.clip(last_hr + np.random.randint(-2, 3), 60, 110))

    hr_history.append(heart_rate)
    if len(hr_history) > 120:
        hr_history = hr_history[-120:]

    spo2        = int(np.random.randint(92, 100))
    temperature = round(float(np.random.uniform(36.0, 38.0)), 1)
    activity    = int(np.random.randint(0, 10))
    hrv         = int(np.random.randint(20, 100))

    at_risk = predict_risk(heart_rate, spo2, temperature, activity, hrv)

    return jsonify({
        "image": None,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "temperature": temperature,
        "activity": activity,
        "hrv": hrv,
        "at_risk": at_risk,
        "hr_history": hr_history[-60:],
        "spo2_cls": "danger" if spo2 < 95 else "ok",
        "temp_cls": "warn" if temperature > 37.5 else "ok",
    })

@app.route("/api/heart/reset")
def heart_reset():
    global hr_history
    hr_history = []
    return jsonify({"ok": True})

# ─────────────────────────────────────────────
# API — CORRELATION
# ─────────────────────────────────────────────

@app.route("/api/correlation")
def correlation():
    corr = DF.corr().round(3)
    return jsonify({
        "columns": list(corr.columns),
        "matrix":  corr.values.tolist(),
    })

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
