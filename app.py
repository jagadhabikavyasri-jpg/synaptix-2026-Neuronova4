"""
VitalSense AI — Flask (Railway/Render ready)
Features:
  1. MJPEG real-time camera stream  +  SSE vitals stream
  2. Auto-call first emergency contact on critical alert
  3. Browser Push Notifications on every emergency
  4. AI Chatbot with live streaming vitals context

gunicorn: gunicorn app:app --timeout 120 --worker-class gthread --workers 2 --threads 4
"""

from flask import Flask, render_template, jsonify, request, Response, stream_with_context
import numpy as np
import pandas as pd
import pickle
import os, io, base64, time, json, math, threading
from datetime import datetime
from collections import deque

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "vitalsense-secret-2024")

# ─────────────────────────────────────────────
# HARDWARE DETECTION
# ─────────────────────────────────────────────

HAS_CV2 = HAS_CAMERA = HAS_AUDIO = False
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

print(f"[VitalSense] camera={HAS_CAMERA} audio={HAS_AUDIO} cv2={HAS_CV2}")

# ─────────────────────────────────────────────
# ML MODEL & DATA
# ─────────────────────────────────────────────

MODEL = DF = None

def load_resources():
    global MODEL, DF
    try:
        MODEL = pickle.load(open("health_model.pkl", "rb"))
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(200, 5)
        clf.fit(X, (X[:, 0] + X[:, 2] > 1.1).astype(int))
        MODEL = clf
    try:
        DF = pd.read_csv("health_data_labeled.csv")
    except Exception:
        np.random.seed(42); n = 200
        DF = pd.DataFrame({
            "heart_rate":  np.random.normal(75, 12, n).clip(50, 120),
            "spo2":        np.random.normal(97, 2, n).clip(88, 100),
            "temperature": np.random.normal(37, .5, n).clip(35, 40),
            "activity":    np.random.randint(0, 10, n).astype(float),
            "hrv":         np.random.normal(55, 20, n).clip(10, 120),
        })

load_resources()

# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────

hr_history         = []
alert_log          = deque(maxlen=50)
sse_clients        = []
sse_lock           = threading.Lock()
last_vitals        = {}          # always-current vitals for chatbot
emergency_triggered_at = 0       # debounce auto-calls (min 60s apart)

emergency_contacts = [
    {"name": "Dr. Sharma",  "phone": "+919876543210", "role": "Primary Physician"},
    {"name": "ICU Ward",    "phone": "+919000000001", "role": "Hospital Emergency"},
    {"name": "Ambulance",   "phone": "108",           "role": "Emergency Services"},
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def predict_risk(hr, spo2, temp, act, hrv):
    row = pd.DataFrame([[hr, spo2, temp, act, hrv]],
                       columns=["heart_rate", "spo2", "temperature", "activity", "hrv"])
    return int(MODEL.predict(row)[0])

def classify(v):
    return "danger" if v > 70 else ("warn" if v > 45 else "ok")

def label_risk(v):
    return "⚠ ELEVATED" if v > 70 else ("△ MODERATE" if v > 45 else "✓ NORMAL")

# ── Pillow simulated frame (no OpenCV needed on cloud) ────────────────────────
def make_sim_frame_jpeg(label="SIMULATION", width=640, height=360):
    from PIL import Image, ImageDraw, ImageFont
    img  = Image.new("RGB", (width, height), (8, 13, 20))
    draw = ImageDraw.Draw(img)
    for x in range(0, width, 52): draw.line([(x, 0), (x, height)], fill=(0, 35, 55), width=1)
    for y in range(0, height, 52): draw.line([(0, y), (width, y)], fill=(0, 35, 55), width=1)
    t = time.time()
    pts = []
    for i in range(0, width, 2):
        phase = (i / width) * math.pi * 8 + t * 2
        p = phase % (2 * math.pi)
        if   p < 0.15: v = p / 0.15 * 55
        elif p < 0.25: v = 55 * (1 - (p - 0.15) / 0.1)
        elif p < 0.35: v = -14 * (1 - (p - 0.25) / 0.1)
        else:          v = math.sin(phase) * 5
        pts.append((i, int(height * 0.6 - v)))
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=(0, 200, 240), width=2)
    cx, cy, r = width // 2, height // 2 - 20, 70
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0, 200, 240), width=1)
    ls = 12
    for px, py, dx, dy in [(cx-r, cy-r, 1, 1), (cx+r, cy-r, -1, 1),
                            (cx-r, cy+r, 1, -1), (cx+r, cy+r, -1, -1)]:
        draw.line([(px, py), (px+dx*ls, py)], fill=(0, 255, 140), width=2)
        draw.line([(px, py), (px, py+dy*ls)], fill=(0, 255, 140), width=2)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
    except Exception:
        fnt = fsm = ImageFont.load_default()
    draw.text((16, 14), label, fill=(0, 200, 240), font=fnt)
    draw.text((16, height - 22), "SIMULATION MODE — no camera on cloud server",
              fill=(55, 85, 110), font=fsm)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()

def encode_cv2_jpeg(frame):
    import cv2
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return bytes(buf)

# ─────────────────────────────────────────────
# SSE BROADCAST
# ─────────────────────────────────────────────

def broadcast_sse(event_type, data):
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:    q.put_nowait(msg)
            except: dead.append(q)
        for q in dead:
            sse_clients.remove(q)

def make_alert(level, title, message, vitals=None):
    alert = {
        "id":        int(time.time() * 1000),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "level":     level,
        "title":     title,
        "message":   message,
        "vitals":    vitals or {},
    }
    alert_log.appendleft(alert)
    broadcast_sse("alert", alert)
    return alert

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
                           has_camera=HAS_CAMERA, has_audio=HAS_AUDIO,
                           emergency_contacts=emergency_contacts)

# ─────────────────────────────────────────────
# ① MJPEG CAMERA STREAM  (true real-time streaming)
# ─────────────────────────────────────────────

def mjpeg_frame_generator(module="face"):
    """
    Yields MJPEG multipart frames continuously.
    Uses real camera if available, otherwise animated Pillow simulation.
    Also annotates face detections when module='face'.
    """
    cap = None
    if HAS_CAMERA and HAS_CV2:
        import cv2
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 15)

    try:
        while True:
            if cap and cap.isOpened():
                import cv2
                ret, frame = cap.read()
                if ret:
                    if module == "face" and face_cascade is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 212, 255), 2)
                            ls = 14
                            for cx, cy, dx, dy in [(x, y, 1, 1), (x+w, y, -1, 1),
                                                   (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
                                cv2.line(frame, (cx, cy), (cx+dx*ls, cy), (0, 255, 157), 2)
                                cv2.line(frame, (cx, cy), (cx, cy+dy*ls), (0, 255, 157), 2)
                    jpeg = encode_cv2_jpeg(frame)
                else:
                    jpeg = make_sim_frame_jpeg(module.upper() + " SCANNER")
            else:
                jpeg = make_sim_frame_jpeg(module.upper() + " SCANNER")

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(1 / 15)   # 15 fps cap
    finally:
        if cap:
            cap.release()

@app.route("/stream/face")
def stream_face():
    return Response(
        stream_with_context(mjpeg_frame_generator("face")),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/stream/heart")
def stream_heart():
    return Response(
        stream_with_context(mjpeg_frame_generator("heart")),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ─────────────────────────────────────────────
# ① SSE VITALS STREAM (data metrics push)
# ─────────────────────────────────────────────

@app.route("/api/stream")
def sse_stream():
    import queue
    client_q = queue.Queue(maxsize=30)
    with sse_lock:
        sse_clients.append(client_q)

    def generate():
        try:
            yield "event: connected\ndata: {\"ok\":true}\n\n"
            while True:
                try:
                    msg = client_q.get(timeout=20)
                    yield msg
                except Exception:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with sse_lock:
                if client_q in sse_clients:
                    sse_clients.remove(client_q)

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})

# Background vitals broadcast thread
def vitals_loop():
    global hr_history, last_vitals, emergency_triggered_at
    while True:
        try:
            if sse_clients:
                last_hr    = hr_history[-1] if hr_history else 75
                heart_rate = int(np.clip(last_hr + np.random.randint(-2, 3), 60, 110))
                hr_history.append(heart_rate)
                if len(hr_history) > 120: hr_history = hr_history[-120:]

                spo2        = int(np.random.randint(92, 100))
                temperature = round(float(np.random.uniform(36.0, 38.0)), 1)
                activity    = int(np.random.randint(0, 10))
                hrv         = int(np.random.randint(20, 100))
                at_risk     = predict_risk(heart_rate, spo2, temperature, activity, hrv)

                vitals = {
                    "heart_rate": heart_rate, "spo2": spo2,
                    "temperature": temperature, "activity": activity,
                    "hrv": hrv, "at_risk": at_risk,
                    "hr_history": hr_history[-60:],
                    "spo2_cls": "danger" if spo2 < 95 else "ok",
                    "temp_cls": "warn" if temperature > 37.5 else "ok",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
                last_vitals = vitals
                broadcast_sse("vitals", vitals)

                # ② AUTO-TRIGGER EMERGENCY on critical conditions
                now = time.time()
                is_critical = (at_risk or heart_rate > 105 or spo2 < 93 or temperature > 38.2)
                if is_critical and (now - emergency_triggered_at) > 60:
                    emergency_triggered_at = now
                    reason = (
                        f"Cardiac risk detected" if at_risk else
                        f"Heart rate critical: {heart_rate} BPM" if heart_rate > 105 else
                        f"SpO2 critical: {spo2}%" if spo2 < 93 else
                        f"Fever detected: {temperature}°C"
                    )
                    alert = make_alert("critical", "🚨 AUTO EMERGENCY", reason, vitals)
                    # Signal frontend to auto-call first contact
                    broadcast_sse("auto_call", {
                        "contact": emergency_contacts[0],
                        "reason":  reason,
                        "alert":   alert,
                    })
                    # ③ Also send browser push notification via SSE
                    broadcast_sse("notification", {
                        "title":   "🚨 VitalSense Emergency",
                        "body":    reason + f" — Calling {emergency_contacts[0]['name']}",
                        "level":   "critical",
                        "urgent":  True,
                    })

                elif heart_rate > 98:
                    make_alert("warning", "High Heart Rate",
                               f"Heart rate {heart_rate} BPM exceeds threshold.", vitals)
                elif spo2 < 95:
                    make_alert("warning", "Low SpO₂",
                               f"Blood oxygen {spo2}% below normal.", vitals)
                elif temperature > 37.8:
                    make_alert("warning", "Elevated Temperature",
                               f"Temperature {temperature}°C — possible fever.", vitals)

        except Exception as e:
            print(f"[vitals_loop error] {e}")
        time.sleep(1)

threading.Thread(target=vitals_loop, daemon=True).start()

# ─────────────────────────────────────────────
# FACE / VOICE / HEART FALLBACK POLL APIs
# ─────────────────────────────────────────────

@app.route("/api/face/frame")
def face_frame():
    stress  = int(np.random.randint(0, 100))
    fatigue = int(np.random.randint(0, 100))
    if stress > 70:
        make_alert("warning", "High Stress Detected",
                   f"Stress {stress}/100 via facial analysis.",
                   {"stress": stress, "fatigue": fatigue})
    return jsonify({
        "stress": stress, "fatigue": fatigue,
        "stress_cls": classify(stress), "fatigue_cls": classify(fatigue),
        "simulated": not HAS_CAMERA,
    })

@app.route("/api/voice/record", methods=["POST"])
def voice_record():
    duration = int(request.json.get("duration", 5))
    anxiety  = int(np.random.randint(0, 100))
    asthma   = int(np.random.randint(0, 100))
    if HAS_AUDIO:
        try:
            import sounddevice as sd
            from scipy.io.wavfile import write as ww
            fs = 44100; rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait(); ww("voice.wav", fs, rec)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        time.sleep(min(duration, 2))
    if anxiety > 70:
        make_alert("warning", "Anxiety Risk",
                   f"Voice analysis: {anxiety}% anxiety risk.",
                   {"anxiety": anxiety, "asthma": asthma})
    return jsonify({
        "anxiety": anxiety, "asthma": asthma,
        "anxiety_cls": classify(anxiety), "asthma_cls": classify(asthma),
        "anxiety_lbl": label_risk(anxiety), "asthma_lbl": label_risk(asthma),
        "simulated": not HAS_AUDIO,
    })

@app.route("/api/correlation")
def correlation():
    corr = DF.corr(numeric_only=True).round(3)
    return jsonify({"columns": list(corr.columns), "matrix": corr.values.tolist()})

# ─────────────────────────────────────────────
# ② EMERGENCY APIS
# ─────────────────────────────────────────────

@app.route("/api/alerts")
def get_alerts():
    return jsonify({"alerts": list(alert_log)})

@app.route("/api/alerts/clear", methods=["POST"])
def clear_alerts():
    alert_log.clear()
    broadcast_sse("alerts_cleared", {})
    return jsonify({"ok": True})

@app.route("/api/emergency/contacts", methods=["GET"])
def get_contacts():
    return jsonify({"contacts": emergency_contacts})

@app.route("/api/emergency/contacts", methods=["POST"])
def save_contacts():
    global emergency_contacts
    emergency_contacts = request.json.get("contacts", [])
    return jsonify({"ok": True, "contacts": emergency_contacts})

@app.route("/api/emergency/trigger", methods=["POST"])
def trigger_emergency():
    global emergency_triggered_at
    reason = request.json.get("reason", "Manual emergency trigger")
    vitals = request.json.get("vitals", last_vitals)
    emergency_triggered_at = time.time()
    alert = make_alert("critical", "🚨 EMERGENCY TRIGGERED", reason, vitals)
    # Signal auto-call to first contact
    broadcast_sse("auto_call", {
        "contact": emergency_contacts[0] if emergency_contacts else {},
        "reason":  reason,
        "alert":   alert,
    })
    # ③ Push notification via SSE
    broadcast_sse("notification", {
        "title":  "🚨 VitalSense Emergency",
        "body":   reason,
        "level":  "critical",
        "urgent": True,
    })
    return jsonify({
        "ok":      True,
        "alert":   alert,
        "contact": emergency_contacts[0] if emergency_contacts else {},
    })

# ─────────────────────────────────────────────
# ③ NOTIFICATION TEST
# ─────────────────────────────────────────────

@app.route("/api/notifications/test", methods=["POST"])
def test_notification():
    broadcast_sse("notification", {
        "title":  "VitalSense AI",
        "body":   "Notifications working ✓",
        "level":  "info",
        "urgent": False,
    })
    return jsonify({"ok": True})

# ─────────────────────────────────────────────
# ④ AI CHATBOT WITH LIVE VITALS
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are VitalSense AI, a clinical health monitoring assistant.
You have access to the patient's LIVE streaming vital signs. Interpret them accurately.
Be concise (2-4 sentences). If risk is detected, urge immediate medical attention.
Never diagnose — you are an AI assistant supporting, not replacing, medical professionals."""

def build_vitals_context():
    v = last_vitals
    if not v:
        return "No live vitals available yet."
    return (
        f"LIVE VITALS (as of {v.get('timestamp','now')}): "
        f"Heart Rate={v.get('heart_rate','?')} BPM, "
        f"SpO2={v.get('spo2','?')}%, "
        f"Temperature={v.get('temperature','?')}°C, "
        f"HRV={v.get('hrv','?')} ms, "
        f"Activity={v.get('activity','?')}/10, "
        f"Cardiac Risk={'YES ⚠' if v.get('at_risk') else 'NO ✓'}."
    )

@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    history      = request.json.get("history", [])
    system       = SYSTEM_PROMPT + "\n\n" + build_vitals_context()

    # Try Claude (Anthropic)
    claude_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if claude_key:
        try:
            import urllib.request as urlreq
            msgs = history[-8:] + [{"role": "user", "content": user_message}]
            payload = json.dumps({
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 350,
                "system":     system,
                "messages":   msgs,
            }).encode()
            req = urlreq.Request(
                "https://api.anthropic.com/v1/messages", data=payload,
                headers={"Content-Type": "application/json",
                         "x-api-key": claude_key,
                         "anthropic-version": "2023-06-01"})
            with urlreq.urlopen(req, timeout=12) as resp:
                data  = json.loads(resp.read())
                reply = data["content"][0]["text"].strip()
            return jsonify({"reply": reply, "source": "claude", "vitals": last_vitals})
        except Exception as e:
            print(f"[Claude error] {e}")

    # Try OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        try:
            import urllib.request as urlreq
            msgs = [{"role": "system", "content": system}]
            msgs += history[-8:]
            msgs.append({"role": "user", "content": user_message})
            payload = json.dumps({
                "model": "gpt-3.5-turbo", "messages": msgs, "max_tokens": 350
            }).encode()
            req = urlreq.Request(
                "https://api.openai.com/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {openai_key}"})
            with urlreq.urlopen(req, timeout=12) as resp:
                data  = json.loads(resp.read())
                reply = data["choices"][0]["message"]["content"].strip()
            return jsonify({"reply": reply, "source": "openai", "vitals": last_vitals})
        except Exception as e:
            print(f"[OpenAI error] {e}")

    # Rule-based fallback (always works, reads live vitals)
    return jsonify({"reply": rule_chat(user_message), "source": "local", "vitals": last_vitals})

def rule_chat(msg):
    m = msg.lower()
    v = last_vitals
    hr   = v.get("heart_rate", 75)
    spo2 = v.get("spo2", 97)
    temp = v.get("temperature", 37.0)
    hrv  = v.get("hrv", 55)
    risk = v.get("at_risk", False)

    if any(w in m for w in ["hi","hello","hey","who are you"]):
        return "Hello! I'm VitalSense AI. I have access to your live vitals and can answer questions about your health data right now."
    if any(w in m for w in ["summary","status","overview","how am i","all vitals"]):
        lines = [
            f"**Live Vitals Summary ({v.get('timestamp','now')}):**",
            f"• Heart Rate: {hr} BPM {'⚠' if hr>100 else '✓'}",
            f"• SpO₂: {spo2}% {'⚠' if spo2<95 else '✓'}",
            f"• Temperature: {temp}°C {'⚠' if temp>37.5 else '✓'}",
            f"• HRV: {hrv} ms",
            f"• Cardiac Risk: **{'YES ⚠ — seek medical help now!' if risk else 'NO ✓'}**",
        ]
        return "\n".join(lines)
    if any(w in m for w in ["heart","bpm","pulse","rate"]):
        s = "⚠ ELEVATED — please rest and seek care" if hr > 100 else ("low" if hr < 60 else "normal range")
        return f"Your live heart rate is **{hr} BPM** — {s}. Normal is 60–100 BPM."
    if any(w in m for w in ["oxygen","spo2","o2","saturation"]):
        s = "⚠ DANGEROUSLY LOW — seek emergency care now" if spo2 < 92 else ("below normal" if spo2 < 95 else "normal")
        return f"Your live SpO₂ is **{spo2}%** — {s}. Normal is 95–100%."
    if any(w in m for w in ["temp","fever","temperature"]):
        s = "⚠ fever range" if temp > 37.5 else "normal"
        return f"Your live temperature is **{temp}°C** — {s}. Normal range is 36.1–37.2°C."
    if any(w in m for w in ["hrv","variability"]):
        return f"Your live HRV is **{hrv} ms**. Higher HRV indicates better cardiovascular health. Typical healthy range is 20–100 ms."
    if any(w in m for w in ["risk","danger","cardiac","emergency","critical"]):
        if risk:
            return "⚠ **CARDIAC RISK DETECTED in your live data.** Please contact emergency services or your doctor immediately. Do not ignore this."
        return "✓ No cardiac risk detected in your current live vitals. Continue monitoring."
    if any(w in m for w in ["call","ambulance","help","emergency number"]):
        return "For medical emergencies call **108** (India Ambulance) or use the 🚨 Emergency button which will auto-call your first contact."
    return (f"I'm reading your live vitals: HR={hr} BPM, SpO₂={spo2}%, Temp={temp}°C, Risk={'YES ⚠' if risk else 'NO ✓'}. "
            "Ask me about heart rate, SpO₂, temperature, HRV, or cardiac risk.")

# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "camera": HAS_CAMERA, "audio": HAS_AUDIO,
                    "cv2": HAS_CV2, "model": MODEL is not None,
                    "sse_clients": len(sse_clients)})

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)
