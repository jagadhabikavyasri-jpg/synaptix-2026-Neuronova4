"""
VitalSense AI — Flask Application (Render-ready)
Run locally:  python app.py
Deploy:       gunicorn app:app
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import pickle
import os
import base64
import time

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "vitalsense-secret-key")

# ─────────────────────────────────────────────
# ENVIRONMENT DETECTION
# Render has no webcam or microphone.
# App falls back to a simulated demo mode automatically.
# ─────────────────────────────────────────────

RENDER_ENV = os.environ.get("RENDER", False)
HAS_CAMERA = False
HAS_AUDIO  = False

try:
    import cv2
    cap_test = cv2.VideoCapture(0)
    if cap_test.isOpened():
        HAS_CAMERA = True
        cap_test.release()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
except Exception:
    pass

try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
    sd.query_devices()
    HAS_AUDIO = True
except Exception:
    pass

# ─────────────────────────────────────────────
# LOAD ML MODEL & DATA
# ─────────────────────────────────────────────

MODEL = None
DF    = None

def load_resources():
    global MODEL, DF

    try:
        MODEL = pickle.load(open("health_model.pkl", "rb"))
    except FileNotFoundError:
        from sklearn.ensemble import RandomForestClassifier
        MODEL = RandomForestClassifier(n_estimators=10, random_state=42)
        dummy_X = np.random.rand(200, 5)
        dummy_y = (dummy_X[:, 0] + dummy_X[:, 2] > 1.1).astype(int)
        MODEL.fit(dummy_X, dummy_y)

    try:
        DF = pd.read_csv("health_data_labeled.csv")
    except FileNotFoundError:
        np.random.seed(42)
        n = 200
        DF = pd.DataFrame({
            "heart_rate":  np.random.normal(75, 12, n).clip(50, 120),
            "spo2":        np.random.normal(97, 2,  n).clip(88, 100),
            "temperature": np.random.normal(37, 0.5, n).clip(35, 40),
            "activity":    np.random.randint(0, 10, n).astype(float),
            "hrv":         np.random.normal(55, 20, n).clip(10, 120),
        })

load_resources()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

hr_history = []

def predict_risk(heart_rate, spo2, temperature, activity, hrv):
    data = pd.DataFrame([[heart_rate, spo2, temperature, activity, hrv]],
                        columns=["heart_rate","spo2","temperature","activity","hrv"])
    return int(MODEL.predict(data)[0])


def encode_frame(frame):
    import cv2
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


def make_simulated_frame(label="SIMULATED", width=640, height=480):
    """Animated synthetic camera frame for cloud / no-camera environments."""
    import cv2
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (8, 13, 20)

    # Grid
    for x in range(0, width, 52):
        cv2.line(frame, (x,0), (x,height), (0,40,60), 1)
    for y in range(0, height, 52):
        cv2.line(frame, (0,y), (width,y), (0,40,60), 1)

    # ECG wave
    t = time.time()
    pts = []
    for i in range(0, width, 3):
        phase = (i / width) * np.pi * 8 + t * 2
        p = phase % (2 * np.pi)
        if   p < 0.15: v = p / 0.15 * 60
        elif p < 0.25: v = 60 * (1 - (p - 0.15) / 0.1)
        elif p < 0.35: v = -15 * (1 - (p - 0.25) / 0.1)
        else:          v = np.sin(phase) * 6
        pts.append((i, int(height * 0.65 - v)))
    for i in range(len(pts)-1):
        cv2.line(frame, pts[i], pts[i+1], (0, 212, 255), 2)

    # Labels
    cv2.putText(frame, label, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 212, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "SIMULATION MODE - No hardware on cloud server",
                (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (58, 90, 120), 1, cv2.LINE_AA)

    # Simulated face bracket
    cx, cy, r = width//2, height//2, 80
    cv2.circle(frame, (cx, cy), r, (0, 212, 255), 1)
    ls = 14
    for px,py,dx,dy in [(cx-r,cy-r,1,1),(cx+r,cy-r,-1,1),(cx-r,cy+r,1,-1),(cx+r,cy+r,-1,-1)]:
        cv2.line(frame, (px,py), (px+dx*ls,py), (0,255,157), 2)
        cv2.line(frame, (px,py), (px,py+dy*ls), (0,255,157), 2)

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
    return render_template("dashboard.html", module=module,
                           has_camera=HAS_CAMERA, has_audio=HAS_AUDIO)

# ─────────────────────────────────────────────
# API — FACE SCANNER
# ─────────────────────────────────────────────

@app.route("/api/face/frame")
def face_frame():
    stress  = int(np.random.randint(0, 100))
    fatigue = int(np.random.randint(0, 100))

    if HAS_CAMERA:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,212,255),2)
                lw,ls=2,14
                for cx,cy,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
                    cv2.line(frame,(cx,cy),(cx+dx*ls,cy),(0,255,157),lw)
                    cv2.line(frame,(cx,cy),(cx,cy+dy*ls),(0,255,157),lw)
        else:
            frame = make_simulated_frame("FACE SCANNER")
    else:
        frame = make_simulated_frame("FACE SCANNER")

    def cls(v): return "danger" if v>70 else ("warn" if v>45 else "ok")

    return jsonify({
        "image":       encode_frame(frame),
        "stress":      stress,
        "fatigue":     fatigue,
        "stress_cls":  cls(stress),
        "fatigue_cls": cls(fatigue),
        "simulated":   not HAS_CAMERA,
    })

# ─────────────────────────────────────────────
# API — VOICE RISK
# ─────────────────────────────────────────────

@app.route("/api/voice/record", methods=["POST"])
def voice_record():
    duration = int(request.json.get("duration", 5))
    anxiety  = int(np.random.randint(0, 100))
    asthma   = int(np.random.randint(0, 100))

    if HAS_AUDIO:
        try:
            import sounddevice as sd
            from scipy.io.wavfile import write as wav_write
            fs  = 44100
            rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            wav_write("voice.wav", fs, rec)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        time.sleep(min(duration, 3))   # simulate recording delay

    def cls(v): return "danger" if v>70 else ("warn" if v>45 else "ok")
    def lbl(v): return "⚠ ELEVATED" if v>70 else ("△ MODERATE" if v>45 else "✓ NORMAL")

    return jsonify({
        "anxiety":     anxiety,
        "asthma":      asthma,
        "anxiety_cls": cls(anxiety),
        "asthma_cls":  cls(asthma),
        "anxiety_lbl": lbl(anxiety),
        "asthma_lbl":  lbl(asthma),
        "simulated":   not HAS_AUDIO,
    })

# ─────────────────────────────────────────────
# API — HEART MONITOR
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
    at_risk     = predict_risk(heart_rate, spo2, temperature, activity, hrv)

    if HAS_CAMERA:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            frame = make_simulated_frame("HEART MONITOR")
    else:
        frame = make_simulated_frame("HEART MONITOR")

    return jsonify({
        "image":       encode_frame(frame),
        "heart_rate":  heart_rate,
        "spo2":        spo2,
        "temperature": temperature,
        "activity":    activity,
        "hrv":         hrv,
        "at_risk":     at_risk,
        "hr_history":  hr_history[-60:],
        "spo2_cls":    "danger" if spo2 < 95 else "ok",
        "temp_cls":    "warn"   if temperature > 37.5 else "ok",
        "simulated":   not HAS_CAMERA,
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
    corr = DF.corr(numeric_only=True).round(3)
    return jsonify({
        "columns": list(corr.columns),
        "matrix":  corr.values.tolist(),
    })

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=not RENDER_ENV, host="0.0.0.0", port=port, threaded=True)
