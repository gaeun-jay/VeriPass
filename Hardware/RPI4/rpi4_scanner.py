import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import firebase_admin
from firebase_admin import credentials, db
from pyzbar import pyzbar
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2, json
import datetime, time, threading

# ── Firebase 초기화 ──────────────────────────────────────
if not firebase_admin._apps:
    cred = credentials.Certificate(
        "veripass-21ab9-firebase-adminsdk-fbsvc-d90d3f0e15.json"
    )
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://veripass-21ab9-default-rtdb.firebaseio.com"
    })

logs_ref    = db.reference("/veripass/logs")
counter_ref = db.reference("/veripass/session_counter")

print("📡 Firebase 연결 완료")

# ── 공유 상태 ────────────────────────────────────────────
latest_frame   = None
last_scan_info = {"card_id": None, "msg": "대기 중... QR을 비춰주세요"}
frame_lock     = threading.Lock()
pending_card   = None   # RPI5 응답 대기 중인 카드 ID
gate_listener  = None   # 최초 스캔 후 게이트 리스너 등록


def get_next_session_key():
    """Firebase 카운터로 session_N 키를 원자적으로 발급"""
    n = counter_ref.transaction(lambda cur: (cur or 0) + 1)
    return f"session_{n}"


def setup_gate_listener(gate):
    """RPI5가 scan_result/Status 를 채우면 pending 해제"""
    global gate_listener, pending_card
    if gate_listener is not None:
        return

    def on_result(event):
        global pending_card
        if not event.data or not isinstance(event.data, dict):
            return
        status = event.data.get("Status", "")
        if status and pending_card:
            print(f"[RPI5 응답] {pending_card} → {status} | 다음 스캔 대기 해제")
            pending_card = None

    gate_listener = db.reference(
        f"/veripass/gates/{gate}/scan_result"
    ).listen(on_result)
    print(f"게이트 {gate} 리스닝 시작")


# ── 카메라 + QR 스캔 스레드 ─────────────────────────────
def camera_loop():
    global latest_frame, pending_card, last_scan_info

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("📷 카메라 시작!")

    while True:
        frame_rgb = picam2.capture_array()
        frame     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        for obj in pyzbar.decode(frame):
            raw = obj.data.decode("utf-8").strip()

            try:
                info = json.loads(raw)
            except json.JSONDecodeError:
                print(f"QR 파싱 실패: {raw}")
                continue

            card_id = info.get("card_id", "")
            if not card_id:
                continue

            # RPI5 응답 대기 중 → 모든 스캔 차단
            if pending_card is not None:
                continue

            ts   = datetime.datetime.now().isoformat(timespec="seconds")
            gate = info.get("gate")

            log_entry = {
                "Timestamp": ts,
                "Gate"     : gate,
                "Card_ID"  : card_id,
                "Card_Type": info.get("card_type", ""),
                "Status"   : "",
                "Gender"   : info.get("gender", ""),
                "Age"      : info.get("age", 0),
                "Name"     : info.get("name", ""),
                "Count"    : info.get("count", 0),
            }

            session_key = get_next_session_key()
            logs_ref.child(session_key).set(log_entry)

            if gate is not None:
                db.reference(f"/veripass/gates/{gate}/scan_result").set(log_entry)
                setup_gate_listener(gate)

            pending_card = card_id

            print(f"[{ts}] {session_key} | {card_id} | {info.get('name','')} | "
                  f"{info.get('age','')}세 | {info.get('gender','')} | "
                  f"{info.get('card_type','')} | gate={gate} → Firebase")
            print(f"RPI5 응답 대기 중...")

            last_scan_info = {
                "card_id": card_id,
                "msg": f"[{ts}] {card_id} | {info.get('name','')} | "
                       f"{info.get('age','')}세 | {info.get('gender','')} | "
                       f"{info.get('card_type','')} | gate={gate}"
            }

            pts   = [(p.x, p.y) for p in obj.polygon]
            color = (0, 255, 0) if info.get("card_type") == "adult" else (0, 0, 255)
            for i in range(len(pts)):
                cv2.line(frame, pts[i], pts[(i+1) % len(pts)], color, 3)


        status_text = (f"RPI5 : {pending_card}" if pending_card
                       else last_scan_info["msg"])

        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.03)


# ── Flask ────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>VeriPass QR Scanner</title>
  <style>
    body  { background:#111; color:#eee; font-family:sans-serif; text-align:center; padding:20px; }
    h1    { color:#4fc3f7; }
    img   { border:2px solid #4fc3f7; border-radius:8px; max-width:100%; }
    #info { margin-top:14px; font-size:1.1em; background:#1e1e1e;
            padding:12px 24px; border-radius:8px; display:inline-block; }
  </style>
  <script>
    setInterval(() => {
      fetch('/scan_status').then(r => r.json()).then(d => {
        document.getElementById('info').innerText = d.msg;
      });
    }, 1000);
  </script>
</head>
<body>
  <h1>VeriPass QR Scanner</h1>
  <img src="/video_feed" width="640">
  <div id="info">{{ msg }}</div>
</body>
</html>
"""

def gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.05)

@app.route("/")
def index():
    return render_template_string(HTML, msg=last_scan_info["msg"])

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/scan_status")
def scan_status():
    return last_scan_info

# ── 실행 ─────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("Flask 서버: http://0.0.0.0:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True)
