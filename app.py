"""
VitalSense AI — Flask Application (Railway / Render ready)
Modifications:
  1. SSE streaming for live data
  2. Emergency alerts — call log + emergency contact direct call
  3. Browser push notifications
  4. AI chatbot endpoint (Claude-compatible)

Run locally:  python app.py
Deploy:       gunicorn app:app --timeout 120 --worker-class gthread --workers 2 --threads 4
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
        MODEL = pickle.load(open("health_model.pkl","rb"))
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(200,5)
        clf.fit(X, (X[:,0]+X[:,2]>1.1).astype(int))
        MODEL = clf
    try:
        DF = pd.read_csv("health_data_labeled.csv")
    except Exception:
        np.random.seed(42); n=200
        DF = pd.DataFrame({
            "heart_rate":  np.random.normal(75,12,n).clip(50,120),
            "spo2":        np.random.normal(97,2,n).clip(88,100),
            "temperature": np.random.normal(37,.5,n).clip(35,40),
            "activity":    np.random.randint(0,10,n).astype(float),
            "hrv":         np.random.normal(55,20,n).clip(10,120),
        })

load_resources()

# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────

hr_history      = []
alert_log       = deque(maxlen=50)   # circular log of all alerts
sse_clients     = []                 # active SSE connections
sse_lock        = threading.Lock()

# Emergency contacts (configurable via API)
emergency_contacts = [
    {"name": "Dr. Sharma",    "phone": "+91-9876543210", "role": "Primary Physician"},
    {"name": "ICU Ward",      "phone": "+91-9000000001", "role": "Hospital Emergency"},
    {"name": "Ambulance",     "phone": "108",            "role": "Emergency Services"},
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def predict_risk(hr, spo2, temp, act, hrv):
    row = pd.DataFrame([[hr, spo2, temp, act, hrv]],
                       columns=["heart_rate","spo2","temperature","activity","hrv"])
    return int(MODEL.predict(row)[0])

def classify(v):
    return "danger" if v>70 else ("warn" if v>45 else "ok")

def label_risk(v):
    return "⚠ ELEVATED" if v>70 else ("△ MODERATE" if v>45 else "✓ NORMAL")

def make_sim_frame_pil(label="SIMULATION", width=640, height=360):
    from PIL import Image, ImageDraw, ImageFont
    img  = Image.new("RGB",(width,height),(8,13,20))
    draw = ImageDraw.Draw(img)
    for x in range(0,width,52): draw.line([(x,0),(x,height)],fill=(0,35,55),width=1)
    for y in range(0,height,52): draw.line([(0,y),(width,y)],fill=(0,35,55),width=1)
    t = time.time()
    pts = []
    for i in range(0,width,2):
        phase=(i/width)*math.pi*8+t*2; p=phase%(2*math.pi)
        if p<0.15:   v=p/0.15*55
        elif p<0.25: v=55*(1-(p-0.15)/0.1)
        elif p<0.35: v=-14*(1-(p-0.25)/0.1)
        else:        v=math.sin(phase)*5
        pts.append((i,int(height*0.6-v)))
    for i in range(len(pts)-1): draw.line([pts[i],pts[i+1]],fill=(0,200,240),width=2)
    cx,cy,r=width//2,height//2-20,70
    draw.ellipse([cx-r,cy-r,cx+r,cy+r],outline=(0,200,240),width=1)
    ls=12
    for px,py,dx,dy in [(cx-r,cy-r,1,1),(cx+r,cy-r,-1,1),(cx-r,cy+r,1,-1),(cx+r,cy+r,-1,-1)]:
        draw.line([(px,py),(px+dx*ls,py)],fill=(0,255,140),width=2)
        draw.line([(px,py),(px,py+dy*ls)],fill=(0,255,140),width=2)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",13)
        fs= ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",10)
    except: f=fs=ImageFont.load_default()
    draw.text((16,14),label,fill=(0,200,240),font=f)
    draw.text((16,height-22),"SIMULATION MODE",fill=(55,85,110),font=fs)
    buf=io.BytesIO(); img.save(buf,format="JPEG",quality=82)
    return base64.b64encode(buf.getvalue()).decode()

def encode_cv2_frame(frame):
    import cv2
    _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,82])
    return base64.b64encode(buf).decode()

def get_frame_b64(label="FEED"):
    if HAS_CAMERA and HAS_CV2:
        import cv2
        cap=cv2.VideoCapture(0); ret,frame=cap.read(); cap.release()
        if ret: return encode_cv2_frame(frame),True
    return make_sim_frame_pil(label),False

# ─────────────────────────────────────────────
# SSE BROADCAST HELPER
# ─────────────────────────────────────────────

def broadcast_sse(event_type, data):
    """Push an SSE event to all connected clients."""
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)

def make_alert(level, title, message, vitals=None):
    """Create an alert object, log it, and broadcast via SSE."""
    alert = {
        "id":        int(time.time()*1000),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "level":     level,       # "critical" | "warning" | "info"
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
    module = request.args.get("module","face")
    return render_template("dashboard.html", module=module,
                           has_camera=HAS_CAMERA, has_audio=HAS_AUDIO,
                           emergency_contacts=emergency_contacts)

# ─────────────────────────────────────────────
# ① SSE STREAM  — real-time data push
# ─────────────────────────────────────────────

@app.route("/api/stream")
def sse_stream():
    """
    Server-Sent Events endpoint.
    Pushes heart vitals every second to all connected clients.
    """
    import queue
    client_q = queue.Queue(maxsize=30)
    with sse_lock:
        sse_clients.append(client_q)

    def generate():
        try:
            # Send initial connection confirmation
            yield "event: connected\ndata: {\"ok\":true}\n\n"
            while True:
                try:
                    msg = client_q.get(timeout=25)
                    yield msg
                except Exception:
                    # Send heartbeat keep-alive
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with sse_lock:
                if client_q in sse_clients:
                    sse_clients.remove(client_q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",
            "Connection":       "keep-alive",
        }
    )

# Background thread: pushes vitals every second
def vitals_broadcast_loop():
    global hr_history
    while True:
        try:
            if sse_clients:  # only compute when someone is connected
                last_hr    = hr_history[-1] if hr_history else 75
                heart_rate = int(np.clip(last_hr+np.random.randint(-2,3),60,110))
                hr_history.append(heart_rate)
                if len(hr_history)>120: hr_history=hr_history[-120:]

                spo2        = int(np.random.randint(92,100))
                temperature = round(float(np.random.uniform(36.0,38.0)),1)
                activity    = int(np.random.randint(0,10))
                hrv         = int(np.random.randint(20,100))
                at_risk     = predict_risk(heart_rate,spo2,temperature,activity,hrv)

                vitals = {
                    "heart_rate":heart_rate,"spo2":spo2,
                    "temperature":temperature,"activity":activity,"hrv":hrv,
                    "at_risk":at_risk,"hr_history":hr_history[-60:],
                    "spo2_cls":"danger" if spo2<95 else "ok",
                    "temp_cls":"warn" if temperature>37.5 else "ok",
                    "timestamp":datetime.now().strftime("%H:%M:%S"),
                }
                broadcast_sse("vitals", vitals)

                # ② AUTO EMERGENCY ALERTS
                if at_risk:
                    make_alert("critical","⚠ Cardiac Risk Detected",
                        f"Elevated cardiac risk at {heart_rate} BPM. Immediate attention required.",
                        vitals)
                elif heart_rate > 100:
                    make_alert("warning","High Heart Rate",
                        f"Heart rate {heart_rate} BPM exceeds normal threshold.",vitals)
                elif spo2 < 94:
                    make_alert("critical","Low SpO₂ Alert",
                        f"Blood oxygen {spo2}% is dangerously low.",vitals)
                elif temperature > 37.8:
                    make_alert("warning","Elevated Temperature",
                        f"Body temperature {temperature}°C may indicate fever.",vitals)
        except Exception as e:
            print(f"[SSE loop error] {e}")
        time.sleep(1)

_broadcast_thread = threading.Thread(target=vitals_broadcast_loop, daemon=True)
_broadcast_thread.start()

# ─────────────────────────────────────────────
# EXISTING POLLING APIs (kept for fallback)
# ─────────────────────────────────────────────

@app.route("/api/face/frame")
def face_frame():
    stress=int(np.random.randint(0,100)); fatigue=int(np.random.randint(0,100))
    img_b64,is_real=get_frame_b64("FACE SCANNER")
    if is_real and HAS_CV2:
        import cv2
        cap=cv2.VideoCapture(0); ret,frame=cap.read(); cap.release()
        if ret:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,212,255),2)
                for cx,cy,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
                    cv2.line(frame,(cx,cy),(cx+dx*14,cy),(0,255,157),2)
                    cv2.line(frame,(cx,cy),(cx,cy+dy*14),(0,255,157),2)
            img_b64=encode_cv2_frame(frame)
    # Trigger stress alert if high
    if stress>70:
        make_alert("warning","High Stress Detected",
            f"Stress level {stress}/100 detected via facial analysis.",{"stress":stress,"fatigue":fatigue})
    return jsonify({"image":img_b64,"stress":stress,"fatigue":fatigue,
                    "stress_cls":classify(stress),"fatigue_cls":classify(fatigue),
                    "simulated":not is_real})

@app.route("/api/voice/record", methods=["POST"])
def voice_record():
    duration=int(request.json.get("duration",5))
    anxiety=int(np.random.randint(0,100)); asthma=int(np.random.randint(0,100))
    if HAS_AUDIO:
        try:
            import sounddevice as sd; from scipy.io.wavfile import write as ww
            fs=44100; rec=sd.rec(int(duration*fs),samplerate=fs,channels=1); sd.wait(); ww("voice.wav",fs,rec)
        except Exception as e:
            return jsonify({"error":str(e)}),500
    else:
        time.sleep(min(duration,2))
    if anxiety>70:
        make_alert("warning","Anxiety Risk Detected",
            f"Voice analysis indicates {anxiety}% anxiety risk.",{"anxiety":anxiety,"asthma":asthma})
    return jsonify({"anxiety":anxiety,"asthma":asthma,
                    "anxiety_cls":classify(anxiety),"asthma_cls":classify(asthma),
                    "anxiety_lbl":label_risk(anxiety),"asthma_lbl":label_risk(asthma),
                    "simulated":not HAS_AUDIO})

@app.route("/api/heart/frame")
def heart_frame():
    global hr_history
    last_hr=hr_history[-1] if hr_history else 75
    heart_rate=int(np.clip(last_hr+np.random.randint(-2,3),60,110))
    hr_history.append(heart_rate)
    if len(hr_history)>120: hr_history=hr_history[-120:]
    spo2=int(np.random.randint(92,100)); temperature=round(float(np.random.uniform(36,38)),1)
    activity=int(np.random.randint(0,10)); hrv=int(np.random.randint(20,100))
    at_risk=predict_risk(heart_rate,spo2,temperature,activity,hrv)
    img_b64,is_real=get_frame_b64("HEART MONITOR")
    return jsonify({"image":img_b64,"heart_rate":heart_rate,"spo2":spo2,
                    "temperature":temperature,"activity":activity,"hrv":hrv,
                    "at_risk":at_risk,"hr_history":hr_history[-60:],
                    "spo2_cls":"danger" if spo2<95 else "ok",
                    "temp_cls":"warn" if temperature>37.5 else "ok",
                    "simulated":not is_real})

@app.route("/api/heart/reset")
def heart_reset():
    global hr_history; hr_history=[]; return jsonify({"ok":True})

@app.route("/api/correlation")
def correlation():
    corr=DF.corr(numeric_only=True).round(3)
    return jsonify({"columns":list(corr.columns),"matrix":corr.values.tolist()})

# ─────────────────────────────────────────────
# ② EMERGENCY ALERTS API
# ─────────────────────────────────────────────

@app.route("/api/alerts")
def get_alerts():
    return jsonify({"alerts": list(alert_log)})

@app.route("/api/alerts/clear", methods=["POST"])
def clear_alerts():
    alert_log.clear()
    broadcast_sse("alerts_cleared", {})
    return jsonify({"ok": True})

@app.route("/api/emergency/contacts")
def get_contacts():
    return jsonify({"contacts": emergency_contacts})

@app.route("/api/emergency/contacts", methods=["POST"])
def save_contacts():
    global emergency_contacts
    data = request.json.get("contacts", [])
    emergency_contacts = data
    return jsonify({"ok": True, "contacts": emergency_contacts})

@app.route("/api/emergency/trigger", methods=["POST"])
def trigger_emergency():
    """
    Manually trigger an emergency alert.
    In production: integrate Twilio/SMS API here to actually call contacts.
    """
    reason  = request.json.get("reason","Manual emergency trigger")
    vitals  = request.json.get("vitals",{})
    alert   = make_alert("critical", "🚨 EMERGENCY TRIGGERED", reason, vitals)
    # Log the call attempt
    call_log_entry = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason":     reason,
        "contacts_notified": [c["name"] for c in emergency_contacts],
        "method":     "simulated"  # replace with "twilio" in production
    }
    return jsonify({
        "ok":       True,
        "alert":    alert,
        "call_log": call_log_entry,
        "contacts": emergency_contacts,
        "note":     "In production, integrate Twilio to auto-call contacts."
    })

# ─────────────────────────────────────────────
# ③ NOTIFICATIONS API
# ─────────────────────────────────────────────

@app.route("/api/notifications/push", methods=["POST"])
def push_notification():
    """
    Sends a notification via SSE to the dashboard.
    Frontend uses the Web Notifications API to show a browser notification.
    """
    title   = request.json.get("title","VitalSense Alert")
    body    = request.json.get("body","")
    level   = request.json.get("level","info")
    broadcast_sse("notification", {"title": title, "body": body, "level": level})
    return jsonify({"ok": True})

@app.route("/api/notifications/test", methods=["POST"])
def test_notification():
    broadcast_sse("notification", {
        "title": "VitalSense AI",
        "body":  "Notifications are working correctly ✓",
        "level": "info"
    })
    return jsonify({"ok": True})

# ─────────────────────────────────────────────
# ④ AI CHATBOT
# ─────────────────────────────────────────────

CHATBOT_SYSTEM = """You are VitalSense AI Assistant, a medical health monitoring chatbot.
You have access to the patient's current vital signs data and help interpret them.
Always remind users you are an AI assistant and not a substitute for medical advice.
Be concise, clear, and helpful. If vitals indicate risk, recommend consulting a doctor immediately.
"""

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    AI chatbot endpoint. Integrates with OpenAI or Claude API.
    Falls back to a rule-based system if no API key is configured.
    """
    user_message = request.json.get("message","")
    vitals       = request.json.get("vitals", {})
    history      = request.json.get("history", [])  # [{role, content}, ...]

    # Build context string with current vitals
    vitals_ctx = ""
    if vitals:
        vitals_ctx = (
            f"\nCurrent patient vitals: "
            f"HR={vitals.get('heart_rate','?')}bpm, "
            f"SpO2={vitals.get('spo2','?')}%, "
            f"Temp={vitals.get('temperature','?')}°C, "
            f"HRV={vitals.get('hrv','?')}ms, "
            f"Activity={vitals.get('activity','?')}/10, "
            f"Risk={'YES' if vitals.get('at_risk') else 'NO'}."
        )

    # Try OpenAI API
    openai_key = os.environ.get("OPENAI_API_KEY","")
    if openai_key:
        try:
            import urllib.request as urlreq
            messages = [{"role":"system","content": CHATBOT_SYSTEM + vitals_ctx}]
            messages += history[-6:]  # last 3 turns
            messages.append({"role":"user","content": user_message})
            payload = json.dumps({
                "model":"gpt-3.5-turbo","messages":messages,"max_tokens":300
            }).encode()
            req = urlreq.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={"Content-Type":"application/json",
                         "Authorization":f"Bearer {openai_key}"}
            )
            with urlreq.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                reply = data["choices"][0]["message"]["content"].strip()
            return jsonify({"reply": reply, "source": "openai"})
        except Exception as e:
            print(f"[Chat OpenAI error] {e}")

    # Try Anthropic Claude API
    claude_key = os.environ.get("ANTHROPIC_API_KEY","")
    if claude_key:
        try:
            import urllib.request as urlreq
            msgs = history[-6:] + [{"role":"user","content":user_message}]
            payload = json.dumps({
                "model":"claude-haiku-4-5-20251001",
                "max_tokens":300,
                "system": CHATBOT_SYSTEM + vitals_ctx,
                "messages": msgs
            }).encode()
            req = urlreq.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={"Content-Type":"application/json",
                         "x-api-key":claude_key,
                         "anthropic-version":"2023-06-01"}
            )
            with urlreq.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                reply = data["content"][0]["text"].strip()
            return jsonify({"reply": reply, "source": "claude"})
        except Exception as e:
            print(f"[Chat Claude error] {e}")

    # ── Rule-based fallback (no API key needed) ──────────────────────────────
    reply = rule_based_chat(user_message, vitals)
    return jsonify({"reply": reply, "source": "rule-based"})


def rule_based_chat(msg, vitals):
    """Simple keyword-based chatbot when no LLM API key is set."""
    m = msg.lower()
    hr   = vitals.get("heart_rate", 75)
    spo2 = vitals.get("spo2", 97)
    temp = vitals.get("temperature", 37.0)
    risk = vitals.get("at_risk", False)

    if any(w in m for w in ["hello","hi","hey"]):
        return "Hello! I'm VitalSense AI. I can help interpret your health data. What would you like to know?"
    if any(w in m for w in ["heart rate","bpm","pulse","cardiac"]):
        status = "elevated" if hr>100 else ("low" if hr<60 else "normal")
        return (f"Your current heart rate is **{hr} BPM** — {status}. "
                f"Normal range is 60–100 BPM. "
                + ("⚠ Please consult a doctor immediately." if hr>110 or risk else ""))
    if any(w in m for w in ["oxygen","spo2","o2","saturation"]):
        status = "dangerously low" if spo2<92 else ("below normal" if spo2<95 else "normal")
        return (f"Blood oxygen (SpO₂) is **{spo2}%** — {status}. "
                f"Normal is 95–100%. "
                + ("⚠ Seek emergency care!" if spo2<90 else ""))
    if any(w in m for w in ["temperature","temp","fever"]):
        status = "fever range" if temp>37.5 else "normal"
        return f"Body temperature is **{temp}°C** — {status}. Normal is 36.1–37.2°C."
    if any(w in m for w in ["risk","danger","warning","critical"]):
        if risk:
            return "⚠ **Cardiac risk has been detected.** Please contact your physician or emergency services immediately. Do not ignore this warning."
        return "✓ No immediate cardiac risk detected in the current reading. Continue monitoring."
    if any(w in m for w in ["emergency","call","help","ambulance"]):
        return "In a medical emergency, call **108** (India) or your local emergency number immediately. You can also trigger an emergency alert from the Emergency panel."
    if any(w in m for w in ["hrv","variability"]):
        hrv = vitals.get("hrv",55)
        return f"Heart Rate Variability (HRV) is **{hrv} ms**. Higher HRV generally indicates better cardiovascular fitness and stress resilience."
    if any(w in m for w in ["summary","status","overview","how am i"]):
        lines = [
            f"**Current Vitals Summary:**",
            f"• Heart Rate: {hr} BPM {'⚠' if hr>100 else '✓'}",
            f"• SpO₂: {spo2}% {'⚠' if spo2<95 else '✓'}",
            f"• Temperature: {temp}°C {'⚠' if temp>37.5 else '✓'}",
            f"• Cardiac Risk: {'YES ⚠' if risk else 'NO ✓'}",
        ]
        return "\n".join(lines)
    return ("I can answer questions about your heart rate, SpO₂, temperature, HRV, "
            "cardiac risk, or emergency procedures. What would you like to know?")


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status":"ok","camera":HAS_CAMERA,"audio":HAS_AUDIO,
                    "cv2":HAS_CV2,"model":MODEL is not None,
                    "sse_clients":len(sse_clients)})

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)
