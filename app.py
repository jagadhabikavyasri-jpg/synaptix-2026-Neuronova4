"""
VitalSense AI — Flask Application
Run: python app.py
Visit: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, Response, request
import cv2
import numpy as np
import pandas as pd
import pickle
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import threading
import time
import io
import base64
import os

app = Flask(__name__)
app.secret_key = "vitalsense-secret-key"

# ─────────────────────────────────────────────
# LOAD ML MODEL & DATA  (once at startup)
# ─────────────────────────────────────────────

MODEL = None
DF    = None

def load_resources():
    global MODEL, DF
    try:
        MODEL = pickle.load(open("health_model.pkl", "rb"))
    except FileNotFoundError:
        print("⚠  health_model.pkl not found — using dummy model")
        from sklearn.ensemble import RandomForestClassifier
        MODEL = RandomForestClassifier()
        dummy_X = np.random.rand(100, 5)
        dummy_y = np.random.randint(0, 2, 100)
        MODEL.fit(dummy_X, dummy_y)

    try:
        DF = pd.read_csv("health_data_labeled.csv")
    except FileNotFoundError:
        print("⚠  health_data_labeled.csv not found — using dummy data")
        DF = pd.DataFrame(
            np.random.rand(100, 5),
            columns=["heart_rate", "spo2", "temperature", "activity", "hrv"]
        )

load_resources()

# ─────────────────────────────────────────────
# GLOBAL CAMERA STATE
# ─────────────────────────────────────────────

camera_lock  = threading.Lock()
hr_history   = []
camera_active = {"face": False, "heart": False}

# ─────────────────────────────────────────────
# FACE DETECTION CASCADE
# ─────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def predict_risk(heart_rate, spo2, temperature, activity, hrv):
    data = pd.DataFrame([[heart_rate, spo2, temperature, activity, hrv]],
                        columns=["heart_rate", "spo2", "temperature", "activity", "hrv"])
    return int(MODEL.predict(data)[0])


def encode_frame(frame):
    """Encode OpenCV frame to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


def draw_face_overlays(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 212, 255), 2)
        lw, ls = 2, 14
        for cx, cy, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
            cv2.line(frame, (cx,cy), (cx+dx*ls,cy), (0,255,157), lw)
            cv2.line(frame, (cx,cy), (cx,cy+dy*ls), (0,255,157), lw)
    return frame

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
# API — FACE SCANNER
# ─────────────────────────────────────────────

@app.route("/api/face/frame")
def face_frame():
    """Returns one annotated camera frame + stress/fatigue metrics as JSON."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Camera not available"}), 503

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    stress  = int(np.random.randint(0, 100))
    fatigue = int(np.random.randint(0, 100))

    frame = draw_face_overlays(frame, faces)
    img_b64 = encode_frame(frame)

    return jsonify({
        "image":       img_b64,
        "face_count":  len(faces),
        "stress":      stress,
        "fatigue":     fatigue,
        "stress_cls":  "danger" if stress  > 70 else ("warn" if stress  > 45 else "ok"),
        "fatigue_cls": "danger" if fatigue > 70 else ("warn" if fatigue > 45 else "ok"),
    })

# ─────────────────────────────────────────────
# API — VOICE RECORDER
# ─────────────────────────────────────────────

@app.route("/api/voice/record", methods=["POST"])
def voice_record():
    """Records audio for the requested duration, returns risk metrics."""
    duration = int(request.json.get("duration", 5))
    fs = 44100

    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        wav_write("voice.wav", fs, recording)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    anxiety = int(np.random.randint(0, 100))
    asthma  = int(np.random.randint(0, 100))

    return jsonify({
        "anxiety":      anxiety,
        "asthma":       asthma,
        "anxiety_cls":  "danger" if anxiety > 70 else ("warn" if anxiety > 45 else "ok"),
        "asthma_cls":   "danger" if asthma  > 70 else ("warn" if asthma  > 45 else "ok"),
        "anxiety_lbl":  "⚠ ELEVATED" if anxiety > 70 else ("△ MODERATE" if anxiety > 45 else "✓ NORMAL"),
        "asthma_lbl":   "⚠ ELEVATED" if asthma  > 70 else ("△ MODERATE" if asthma  > 45 else "✓ NORMAL"),
    })

# ─────────────────────────────────────────────
# API — HEART RISK MONITOR
# ─────────────────────────────────────────────

@app.route("/api/heart/frame")
def heart_frame():
    """Returns camera frame + rPPG-simulated vitals + ML prediction."""
    global hr_history

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Camera not available"}), 503

    # Smooth heart rate simulation
    last_hr    = hr_history[-1] if hr_history else 75
    heart_rate = int(np.clip(last_hr + np.random.randint(-2, 3), 60, 110))
    hr_history.append(heart_rate)
    if len(hr_history) > 120:
        hr_history = hr_history[-120:]

    spo2        = int(np.random.randint(92, 100))
    temperature = round(float(np.random.uniform(36.0, 38.0)), 1)
    activity    = int(np.random.randint(0, 10))
    hrv         = int(np.random.randint(20, 100))
    at_risk     = predict_risk(heart_rate, spo2, temperature, activity, hrv)

    img_b64 = encode_frame(frame)

    return jsonify({
        "image":       img_b64,
        "heart_rate":  heart_rate,
        "spo2":        spo2,
        "temperature": temperature,
        "activity":    activity,
        "hrv":         hrv,
        "at_risk":     at_risk,
        "hr_history":  hr_history[-60:],
        "spo2_cls":    "danger" if spo2 < 95 else "ok",
        "temp_cls":    "warn"   if temperature > 37.5 else "ok",
    })


@app.route("/api/heart/reset")
def heart_reset():
    global hr_history
    hr_history = []
    return jsonify({"ok": True})

# ─────────────────────────────────────────────
# API — CORRELATION HEATMAP
# ─────────────────────────────────────────────

@app.route("/api/correlation")
def correlation():
    """Returns correlation matrix as JSON for JS rendering."""
    corr = DF.corr().round(3)
    return jsonify({
        "columns": list(corr.columns),
        "matrix":  corr.values.tolist(),
    })

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)