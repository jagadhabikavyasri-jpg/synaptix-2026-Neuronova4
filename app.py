"""
VitalSense AI — Flask Application (Railway / Render ready)
Run locally:  python app.py
Deploy:       gunicorn app:app --timeout 120
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import pickle
import os
import io
import base64
import time
import math

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "vitalsense-secret-2024")

# ─────────────────────────────────────────────
# HARDWARE DETECTION  (safe — never crashes)
# ─────────────────────────────────────────────

HAS_CV2    = False
HAS_CAMERA = False
HAS_AUDIO  = False
face_cascade = None

try:
    import cv2 as _cv2
    HAS_CV2 = True
    _cap = _cv2.VideoCapture(0)
    if _cap.isOpened():
        HAS_CAMERA = True
        _cap.release()
    face_cascade = _cv2.CascadeClassifier(
        _cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
except Exception:
    pass

try:
    import sounddevice as _sd
    _sd.query_devices()
    HAS_AUDIO = True
except Exception:
    pass

print(f"[VitalSense] camera={HAS_CAMERA}  audio={HAS_AUDIO}  cv2={HAS_CV2}")

# ─────────────────────────────────────────────
# ML MODEL & DATA  (dummy fallbacks included)
# ─────────────────────────────────────────────

MODEL = None
DF    = None

def load_resources():
    global MODEL, DF
    try:
        MODEL = pickle.load(open("health_model.pkl", "rb"))
        print("[VitalSense] Loaded health_model.pkl")
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(200, 5)
        y = (X[:,0] + X[:,2] > 1.1).astype(int)
        clf.fit(X, y)
        MODEL = clf
        print("[VitalSense] Using dummy RandomForest model")

    try:
        DF = pd.read_csv("health_data_labeled.csv")
        print("[VitalSense] Loaded health_data_labeled.csv")
    except Exception:
        np.random.seed(42)
        n = 200
        DF = pd.DataFrame({
            "heart_rate":  np.random.normal(75, 12, n).clip(50, 120),
            "spo2":        np.random.normal(97,  2, n).clip(88, 100),
            "temperature": np.random.normal(37, .5, n).clip(35, 40),
            "activity":    np.random.randint(0, 10, n).astype(float),
            "hrv":         np.random.normal(55, 20, n).clip(10, 120),
        })
        print("[VitalSense] Using dummy dataset")

load_resources()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

hr_history = []

def predict_risk(heart_rate, spo2, temperature, activity, hrv):
    row = pd.DataFrame([[heart_rate, spo2, temperature, activity, hrv]],
                       columns=["heart_rate","spo2","temperature","activity","hrv"])
    return int(MODEL.predict(row)[0])


# ── Pillow-based simulated frame (no OpenCV needed) ──────────────────────────
def make_sim_frame_pil(label="SIMULATION", width=640, height=380):
    """
    Generates a synthetic biometric-style camera frame using only Pillow.
    Returns a base64-encoded JPEG string.
    """
    from PIL import Image, ImageDraw, ImageFont
    import math, time

    img  = Image.new("RGB", (width, height), (8, 13, 20))
    draw = ImageDraw.Draw(img)

    # Grid
    grid_col = (0, 35, 55)
    for x in range(0, width, 52):
        draw.line([(x,0),(x,height)], fill=grid_col, width=1)
    for y in range(0, height, 52):
        draw.line([(0,y),(width,y)], fill=grid_col, width=1)

    # ECG wave
    ecg_col = (0, 200, 240)
    t = time.time()
    pts = []
    for i in range(0, width, 2):
        phase = (i / width) * math.pi * 8 + t * 2
        p = phase % (2 * math.pi)
        if   p < 0.15: v = p / 0.15 * 55
        elif p < 0.25: v = 55 * (1 - (p - 0.15) / 0.1)
        elif p < 0.35: v = -14 * (1 - (p - 0.25) / 0.1)
        else:          v = math.sin(phase) * 5
        pts.append((i, int(height * 0.60 - v)))
    for i in range(len(pts)-1):
        draw.line([pts[i], pts[i+1]], fill=ecg_col, width=2)

    # Glow effect (rough) — draw same line slightly thicker + transparent
    for i in range(0, len(pts)-1, 4):
        draw.ellipse(
            [pts[i][0]-1, pts[i][1]-1, pts[i][0]+1, pts[i][1]+1],
            fill=(0, 100, 120)
        )

    # Face bracket (centre)
    cx, cy, r = width//2, height//2 - 20, 70
    # Circle
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0, 200, 240), width=1)
    # Corner accents
    ls = 12
    corners = [(cx-r, cy-r, 1, 1),(cx+r, cy-r,-1, 1),
               (cx-r, cy+r, 1,-1),(cx+r, cy+r,-1,-1)]
    for px, py, dx, dy in corners:
        draw.line([(px,py),(px+dx*ls,py)], fill=(0,255,140), width=2)
        draw.line([(px,py),(px,py+dy*ls)], fill=(0,255,140), width=2)

    # Scan line across face area (animated)
    scan_y = cy - r + int(((time.time()*60) % (r*2)))
    if cy-r < scan_y < cy+r:
        draw.line([(cx-r, scan_y),(cx+r, scan_y)], fill=(0,212,255,80), width=1)

    # Labels
    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = font_lg

    draw.text((16, 14), label, fill=(0,200,240), font=font_lg)
    draw.text((16, height-22), "SIMULATION MODE — no camera on cloud server",
              fill=(55, 85, 110), font=font_sm)

    # Encode to base64 JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_cv2_frame(frame):
    """Encode an OpenCV BGR frame to base64 JPEG."""
    import cv2
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode("utf-8")


def get_frame_b64(label="FEED"):
    """Return a base64 frame — real camera if available, else Pillow simulation."""
    if HAS_CAMERA and HAS_CV2:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return encode_cv2_frame(frame), True
    return make_sim_frame_pil(label), False


def classify(v):
    return "danger" if v > 70 else ("warn" if v > 45 else "ok")

def label_risk(v):
    return "⚠ ELEVATED" if v > 70 else ("△ MODERATE" if v > 45 else "✓ NORMAL")

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

    img_b64, is_real = get_frame_b64("FACE SCANNER")

    if is_real and HAS_CV2:
        import cv2
        # Re-read to annotate (small overhead, keeps code clean)
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
            img_b64 = encode_cv2_frame(frame)

    return jsonify({
        "image":       img_b64,
        "stress":      stress,
        "fatigue":     fatigue,
        "stress_cls":  classify(stress),
        "fatigue_cls": classify(fatigue),
        "simulated":   not is_real,
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
        # Simulate realistic delay without blocking gunicorn workers too long
        time.sleep(min(duration, 2))

    return jsonify({
        "anxiety":     anxiety,
        "asthma":      asthma,
        "anxiety_cls": classify(anxiety),
        "asthma_cls":  classify(asthma),
        "anxiety_lbl": label_risk(anxiety),
        "asthma_lbl":  label_risk(asthma),
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

    img_b64, is_real = get_frame_b64("HEART MONITOR")

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
        "simulated":   not is_real,
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
    corr = DF.corr(numeric_only=True).round(3)
    return jsonify({
        "columns": list(corr.columns),
        "matrix":  corr.values.tolist(),
    })

# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "camera":  HAS_CAMERA,
        "audio":   HAS_AUDIO,
        "cv2":     HAS_CV2,
        "model":   MODEL is not None,
    })

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)
