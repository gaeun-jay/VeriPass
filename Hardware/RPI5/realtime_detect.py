"""
VeriPass RPI5 — 실시간 얼굴 감지 + 부정승차 판정 + Firebase 연동
InsightFace finetuned 모델 (buffalo_l backbone + MLP Head)
카메라: Raspberry Pi Camera v3  /  웹: http://0.0.0.0:5000

Firebase 경로 (읽기):
  /veripass/gates/{GATE_ID}/scan_result   ← RPI4 QR 스캔 이벤트 수신

Firebase 경로 (쓰기):
  /veripass/gates/{GATE_ID}/scan_result/Status
  /veripass/logs/{session_N}/Status        ← Card_ID + Timestamp 매칭

Status 값:
  "normal"  — 정상
  "suspect" — 부정승차 의심

DB 필드 규칙 (대문자):
  Timestamp / Gate / Card_ID / Card_Type / Status / Gender / Age / Name / Count

판정 기준:
  나이 범주: ≤19 → teen / 20~65 → adult / ≥66 → senior
  Gender + Card_Type 모두 일치 → "normal"
  하나라도 불일치              → "suspect"
"""

import os
import sys

# picamera2는 camera_server.py (system python3) subprocess 에서 돌림
# → 이 프로세스는 sys.path 수정 없이 veripass-env 패키지만 사용

import time
import threading
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, Response, render_template_string, jsonify

import firebase_admin
from firebase_admin import credentials, db as firebase_db

# ======================== CONFIG ========================
HEAD_PATH     = os.path.join(os.path.dirname(__file__), "veripass_insight_head.pt")
FIREBASE_CRED = os.path.join(os.path.dirname(__file__), "veripass-21ab9-firebase-adminsdk-fbsvc-d90d3f0e15.json")
FIREBASE_URL  = "https://veripass-21ab9-default-rtdb.firebaseio.com"

GATE_ID            = 1       # rpi5 담당 게이트 번호 (정수)
DET_SIZE           = (640, 640)
DEVICE             = "cpu"
INFERENCE_INTERVAL = 0.5     # 얼굴 추론 주기(초)
DETECT_WAIT_SEC    = 5.0     # QR 스캔 후 얼굴 감지 대기 최대(초)
CAMERA_W, CAMERA_H = 1280, 720
WEB_PORT           = 5000
FRAME_PATH         = "/dev/shm/vp_frame.raw"   # camera_server.py 와 공유하는 프레임 파일
CAMERA_SERVER      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_server.py")
# ========================================================

CLASSES = ["teen_13_18", "adult_19_64", "senior_65plus"]
CLASS_TO_CARD = {
    "teen_13_18":    "teen",
    "adult_19_64":   "adult",
    "senior_65plus": "senior",
}

GENDER_TO_CODE = {
    "female": "F", "male": "M",
    "여": "F",     "남": "M",
    "F": "F",      "M": "M",
}


class Head(nn.Module):
    def __init__(self, in_dim=512, hidden=256, n=len(CLASSES), p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, n),
        )

    def forward(self, x):
        return self.net(x)


# ── 전역 상태 ──────────────────────────────────────────
raw_frame = None
raw_lock  = threading.Lock()

latest_result = {
    "status":    "waiting",   # "waiting" | "detected" | "no_face"
    "age":       None,
    "gender":    None,        # "M" | "F"
    "age_class": None,        # "teen" | "adult" | "senior"
    "bbox":      None,        # [x1, y1, x2, y2]
    "timestamp": None,
}
result_lock = threading.Lock()

last_verdict = {}
verdict_lock = threading.Lock()

_verdict_written = False   # 한 번 기록되면 True → 이후 추가 기록 차단
# ──────────────────────────────────────────────────────


def age_to_class(age: int) -> str:
    """나이 → 카드 유형 문자열"""
    if age <= 19:
        return "teen"
    elif age <= 65:
        return "adult"
    return "senior"


def age_to_class_ko(age: int) -> str:
    return {"teen": "청소년", "adult": "성인", "senior": "경로"}[age_to_class(age)]


def judge(session: dict, detected: dict) -> str:
    """
    session  : /veripass/logs/session_N 데이터 (Gender, Card_Type 필드)
    detected : latest_result (gender="M"/"F", age=int)
    반환     : "normal" | "suspect"
    """
    if detected.get("status") != "detected":
        return "suspect"

    det_age    = detected.get("age")
    det_gender = detected.get("gender")   # "M" or "F"
    if det_age is None or det_gender is None:
        return "suspect"

    # Gender 비교 — DB 에 "female"/"male"/"F"/"M"/"여"/"남" 혼용 대응
    raw_g = (session.get("Gender") or session.get("gender") or "")
    card_gender_code = GENDER_TO_CODE.get(str(raw_g).strip().lower())
    gender_ok = (card_gender_code == det_gender) if card_gender_code else False

    # 나이 범주 비교 — DB 에 "adult"/"teen"/"senior" (또는 대소문자 혼용)
    raw_t = (session.get("Card_Type") or session.get("card_type") or "")
    card_type = str(raw_t).strip().lower()
    age_ok = (card_type == age_to_class(det_age))

    return "normal" if (gender_ok and age_ok) else "suspect"


# ── 모델 / Firebase 초기화 ────────────────────────────
def load_models():
    from insightface.app import FaceAnalysis
    print("[Model] InsightFace buffalo_l 로딩 중...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=DET_SIZE)

    print("[Model] finetuned Head 로딩 중...")
    ckpt = torch.load(HEAD_PATH, map_location="cpu")
    head = Head(in_dim=ckpt.get("in_dim", 512), hidden=ckpt.get("hidden", 256))
    head.load_state_dict(ckpt["state_dict"])
    head.to(DEVICE).eval()
    return face_app, head


def init_firebase():
    cred = credentials.Certificate(FIREBASE_CRED)
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
    firebase_db.reference("/veripass/rpi5").update({
        "connected":  True,
        "gate":       GATE_ID,
        "started_at": datetime.now().isoformat(),
    })
    print("[Firebase] 연결 완료")


# ── 마지막 세션 조회 + 판정 기록 (원샷) ─────────────
def find_last_session() -> tuple[str, dict] | tuple[None, None]:
    """
    /veripass/logs 에서 가장 최신 session_N 키와 데이터를 반환.
    """
    try:
        logs = firebase_db.reference("/veripass/logs").get() or {}
        keys = sorted(
            (k for k in logs if logs[k] and isinstance(logs[k], dict)),
            key=lambda k: int(k.replace("session_", "")) if k.startswith("session_") else 0,
            reverse=True,
        )
        if keys:
            return keys[0], logs[keys[0]]
    except Exception as e:
        print(f"[Firebase] 세션 조회 실패: {e}")
    return None, None


def oneshot_worker():
    """
    실행 후 최초로 얼굴이 감지되면 마지막 session 의 Status 를 한 번만 기록하고 종료.
    이후 재실행 전까지 추가 기록 없음.
    """
    global _verdict_written
    print("[OneShot] 얼굴 감지 대기 중... (최대 30초)")

    deadline = time.time() + 30
    while time.time() < deadline:
        with result_lock:
            res = dict(latest_result)
        if res["status"] == "detected":
            break
        time.sleep(0.3)
    else:
        print("[OneShot] 30초 내 얼굴 미감지 → 종료")
        return

    # 이미 기록된 경우 중복 방지
    with verdict_lock:
        if _verdict_written:
            return
        _verdict_written = True

    # 마지막 세션 조회
    log_key, session = find_last_session()
    if log_key is None:
        print("[OneShot] 기록할 세션 없음")
        return

    with result_lock:
        detected = dict(latest_result)

    verdict = judge(session, detected)

    det_age    = detected.get("age")
    det_gender = detected.get("gender")
    raw_g  = session.get("Gender") or session.get("gender") or "?"
    raw_t  = session.get("Card_Type") or session.get("card_type") or "?"
    name   = session.get("Name") or session.get("name") or "?"
    card_id = session.get("Card_ID") or session.get("card_id") or "?"

    print(f"[OneShot] {log_key} ({name} / {card_id})")
    print(f"  세션 정보: gender={raw_g}  card_type={raw_t}")
    print(f"  감지 결과: gender={det_gender}  age={det_age}y → {age_to_class(det_age) if det_age else '-'}")
    print(f"  판정: {verdict}")

    try:
        firebase_db.reference(f"/veripass/logs/{log_key}/Status").set(verdict)
        print(f"[Firebase] logs/{log_key}/Status = {verdict}")
    except Exception as e:
        print(f"[Firebase] 기록 실패: {e}")

    with verdict_lock:
        last_verdict.update({
            "verdict":         verdict,
            "card_id":         card_id,
            "name":            name,
            "card_type":       raw_t,
            "card_gender":     str(raw_g),
            "card_age":        session.get("Age") or session.get("age"),
            "detected_age":    det_age,
            "detected_gender": det_gender,
            "detected_class":  age_to_class(det_age) if det_age else None,
            "judged_at":       datetime.now().isoformat(),
        })


# ── Thread 1: 캡처 전용 ───────────────────────────────
# camera_server.py (system python3) 가 프레임을 /dev/shm/vp_frame.raw 에 기록.
# 이 함수는 해당 파일을 폴링해 numpy 배열로 읽어 raw_frame 을 갱신한다.
# 프레임 형식: (CAMERA_H, CAMERA_W, 3) uint8 BGR
FRAME_BYTES = CAMERA_H * CAMERA_W * 3

def capture_worker():
    global raw_frame
    print("[Capture] 공유 메모리 프레임 읽기 시작")
    last_mtime = 0.0
    while True:
        try:
            mtime = os.path.getmtime(FRAME_PATH)
            if mtime != last_mtime:
                data = np.fromfile(FRAME_PATH, dtype=np.uint8)
                if data.nbytes == FRAME_BYTES:
                    frame = data.reshape((CAMERA_H, CAMERA_W, 3))
                    with raw_lock:
                        raw_frame = frame
                    last_mtime = mtime
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(0.01)


# ── Thread 2: 추론 전용 ───────────────────────────────
def inference_worker(face_app, head):
    print("[Inference] 추론 스레드 시작")
    while True:
        t0 = time.time()

        with raw_lock:
            frame = raw_frame
        if frame is None:
            time.sleep(0.1)
            continue

        faces = face_app.get(frame)

        if faces:
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            backbone_age = float(face.age)
            gender = face.sex if hasattr(face, "sex") else ("F" if face.gender == 0 else "M")

            emb = torch.tensor(face.normed_embedding.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                pred_idx = int(head(emb).argmax(1).item())
            age_class = CLASS_TO_CARD[CLASSES[pred_idx]]   # "teen"|"adult"|"senior"

            with result_lock:
                latest_result.update({
                    "status":    "detected",
                    "age":       int(round(backbone_age)),
                    "gender":    gender,
                    "age_class": age_class,
                    "bbox":      [int(v) for v in face.bbox],
                    "timestamp": datetime.now().isoformat(),
                })
        else:
            with result_lock:
                latest_result.update({
                    "status":    "no_face",
                    "age":       None,
                    "gender":    None,
                    "age_class": None,
                    "bbox":      None,
                    "timestamp": datetime.now().isoformat(),
                })

        wait = max(0.0, INFERENCE_INTERVAL - (time.time() - t0))
        if wait:
            time.sleep(wait)


# ── Flask Web ──────────────────────────────────────────
flask_app = Flask(__name__)

_VERDICT_KO = {"normal": "정상", "suspect": "부정승차 의심"}
_GENDER_KO  = {"M": "남성", "F": "여성"}
_TYPE_KO    = {"teen": "청소년 (≤19)", "adult": "성인 (20-65)", "senior": "경로 (≥66)"}

WEB_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>VeriPass G{gate} — RPI5</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#0a0a0a;color:#e5e5e5;
         display:flex;flex-direction:column;align-items:center;padding:20px 14px}}
    h1{{color:#4ade80;font-size:1.45rem;margin-bottom:16px;letter-spacing:1px}}
    .layout{{display:grid;grid-template-columns:3fr 2fr;gap:14px;width:100%;max-width:1100px}}
    #feed{{border:2px solid #4ade80;border-radius:8px;width:100%;display:block}}
    .side{{display:flex;flex-direction:column;gap:12px}}
    .card{{background:#161616;border:1px solid #2a2a2a;border-radius:8px;padding:16px 18px}}
    .card-title{{font-size:0.72rem;color:#6b7280;text-transform:uppercase;
                 letter-spacing:1px;margin-bottom:10px}}

    /* 감지 상태 */
    .det-badge{{display:inline-block;padding:3px 14px;border-radius:20px;
               font-weight:bold;font-size:0.88rem;margin-bottom:8px}}
    .s-detected{{background:#166534;color:#bbf7d0}}
    .s-no_face{{background:#7f1d1d;color:#fecaca}}
    .s-waiting{{background:#374151;color:#d1d5db}}
    .metrics{{display:flex;gap:16px;flex-wrap:wrap;margin-top:4px}}
    .metric .val{{font-size:1.6rem;font-weight:bold;color:#4ade80}}
    .metric .lbl{{font-size:0.7rem;color:#9ca3af}}

    /* 판정 결과 */
    #verdict-main{{display:block;text-align:center;padding:10px;border-radius:8px;
                  font-size:1.4rem;font-weight:bold;margin-bottom:12px}}
    .v-normal{{background:#166534;color:#bbf7d0}}
    .v-suspect{{background:#7f1d1d;color:#fecaca}}
    .v-none{{background:#1f2937;color:#6b7280}}
    table{{width:100%;border-collapse:collapse;font-size:0.82rem}}
    td{{padding:5px 4px;border-bottom:1px solid #1f1f1f}}
    td:first-child{{color:#9ca3af;width:44%}}
    td:last-child{{color:#e5e5e5;font-weight:500}}
    tr:last-child td{{border-bottom:none}}

    /* 일치 여부 표시 */
    .ok{{color:#4ade80}}
    .ng{{color:#f87171}}

    .ts{{font-size:0.7rem;color:#4b5563;margin-top:8px;text-align:right}}

    @media(max-width:700px){{
      .layout{{grid-template-columns:1fr}}
    }}
  </style>
</head>
<body>
  <h1>VeriPass &nbsp;|&nbsp; GATE {gate} &nbsp;|&nbsp; RPI5</h1>
  <div class="layout">
    <div>
      <img id="feed" src="/video_feed" alt="camera" />
    </div>
    <div class="side">

      <!-- 실시간 얼굴 감지 -->
      <div class="card">
        <div class="card-title">실시간 얼굴 감지</div>
        <span id="det-badge" class="det-badge s-waiting">대기 중</span>
        <div id="det-metrics" class="metrics"></div>
        <div id="det-ts" class="ts"></div>
      </div>

      <!-- 판정 결과 -->
      <div class="card">
        <div class="card-title">최근 판정 결과</div>
        <span id="verdict-main" class="v-none">—</span>
        <table id="verdict-table"></table>
        <div id="verdict-ts" class="ts"></div>
      </div>

    </div>
  </div>

  <script>
    const SL = {{detected:'감지됨', no_face:'얼굴 없음', waiting:'대기 중'}};
    const SC = {{detected:'s-detected', no_face:'s-no_face', waiting:'s-waiting'}};
    const GK = {{M:'남성', F:'여성'}};
    const TK = {{teen:'청소년(≤19)', adult:'성인(20-65)', senior:'경로(≥66)'}};
    const VK = {{normal:'정상', suspect:'부정승차 의심'}};

    function matchIcon(ok) {{
      return ok ? '<span class="ok">✔ 일치</span>' : '<span class="ng">✘ 불일치</span>';
    }}

    async function update() {{
      try {{
        const d = await fetch('/status').then(r => r.json());

        // 얼굴 감지 상태
        const badge = document.getElementById('det-badge');
        badge.textContent = SL[d.status] || d.status;
        badge.className   = 'det-badge ' + (SC[d.status] || 's-waiting');
        const dm = document.getElementById('det-metrics');
        const dt = document.getElementById('det-ts');
        if (d.status === 'detected') {{
          const cls = TK[d.age_class] || d.age_class || '-';
          dm.innerHTML = `
            <div class="metric"><div class="val">${{d.age ?? '-'}}세</div><div class="lbl">나이</div></div>
            <div class="metric"><div class="val">${{GK[d.gender] ?? d.gender ?? '-'}}</div><div class="lbl">성별</div></div>
            <div class="metric"><div class="val" style="font-size:1rem">${{cls}}</div><div class="lbl">범주</div></div>`;
          dt.textContent = d.timestamp ? '감지: ' + d.timestamp : '';
        }} else {{ dm.innerHTML = ''; dt.textContent = ''; }}

        // 판정 결과
        const v = d.last_verdict;
        const vm = document.getElementById('verdict-main');
        const vt = document.getElementById('verdict-table');
        const vts = document.getElementById('verdict-ts');
        if (v && v.verdict) {{
          vm.textContent = VK[v.verdict] || v.verdict;
          vm.className   = 'v-' + v.verdict;

          const genderOk = (
            (v.card_gender === 'female' && v.detected_gender === 'F') ||
            (v.card_gender === 'male'   && v.detected_gender === 'M')
          );
          const detCls   = v.detected_class || '-';
          const cardCls  = v.card_type || '-';
          const classOk  = (detCls === cardCls);

          vt.innerHTML = `
            <tr><td>이름</td><td>${{v.name ?? '-'}}</td></tr>
            <tr><td>카드 ID</td><td>${{v.card_id ?? '-'}}</td></tr>
            <tr><td>카드 유형</td><td>${{TK[v.card_type] ?? v.card_type ?? '-'}}</td></tr>
            <tr><td>카드 성별</td><td>${{GK[v.card_gender === 'female' ? 'F' : 'M'] ?? v.card_gender ?? '-'}}</td></tr>
            <tr><td>카드 나이</td><td>${{v.card_age != null ? v.card_age + '세' : '-'}}</td></tr>
            <tr><td>감지 나이</td><td>${{v.detected_age != null ? v.detected_age + '세 (' + (TK[v.detected_class] || v.detected_class || '-') + ')' : '-'}}</td></tr>
            <tr><td>감지 성별</td><td>${{GK[v.detected_gender] ?? v.detected_gender ?? '-'}}</td></tr>
            <tr><td>성별 일치</td><td>${{matchIcon(genderOk)}}</td></tr>
            <tr><td>범주 일치</td><td>${{matchIcon(classOk)}}</td></tr>`;
          vts.textContent = v.judged_at ? '판정: ' + v.judged_at : '';
        }} else {{
          vm.textContent = '—'; vm.className = 'v-none';
          vt.innerHTML   = ''; vts.textContent = '';
        }}
      }} catch(e) {{}}
    }}

    update();
    setInterval(update, 1000);
  </script>
</body>
</html>""".format(gate=GATE_ID)


@flask_app.route("/")
def index():
    return WEB_HTML


@flask_app.route("/status")
def api_status():
    with result_lock:
        r = {k: v for k, v in latest_result.items() if k != "bbox"}
    with verdict_lock:
        r["last_verdict"] = dict(last_verdict)
    return jsonify(r)


def generate_frames():
    """raw_frame + 오버레이 → MJPEG (캡처 스레드와 분리되어 항상 최신 프레임 사용)"""
    while True:
        with raw_lock:
            frame = raw_frame
        if frame is None:
            time.sleep(0.05)
            continue

        draw = frame.copy()

        with result_lock:
            res = dict(latest_result)
        with verdict_lock:
            v = last_verdict.get("verdict")

        if res["status"] == "detected" and res.get("bbox"):
            x1, y1, x2, y2 = res["bbox"]
            if v == "normal":
                color = (0, 220, 90)
            elif v == "suspect":
                color = (50, 50, 220)
            else:
                color = (180, 180, 180)

            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)

            age_lbl = f"{res['gender']} / {res['age']}y / {res['age_class']}" if res.get("age") else ""
            cv2.putText(draw, age_lbl, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
            if v:
                verdict_lbl = "NORMAL" if v == "normal" else "SUSPECT"
                cv2.putText(draw, verdict_lbl, (x1, y2 + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        ret, buf = cv2.imencode(".jpg", draw, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


@flask_app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── 메인 ──────────────────────────────────────────────
def main():
    import subprocess, signal, atexit

    print("=" * 64)
    print("  VeriPass RPI5  |  GATE", GATE_ID, " |  실시간 감지 + 부정승차 판정")
    print("=" * 64)

    # camera_server.py 를 system python3 로 subprocess 실행
    # (picamera2 + numpy 1.24 호환 환경 — veripass-env 와 분리)
    cam_proc = subprocess.Popen(
        ["/usr/bin/python3", CAMERA_SERVER],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(f"[CamServer] PID={cam_proc.pid} 시작 ({CAMERA_SERVER})")

    def _relay_cam_log():
        for line in cam_proc.stdout:
            print("[CamServer]", line.decode(errors="replace").rstrip())

    threading.Thread(target=_relay_cam_log, daemon=True).start()

    def _kill_cam():
        if cam_proc.poll() is None:
            cam_proc.terminate()
            cam_proc.wait(timeout=3)
            print("[CamServer] 종료됨")

    atexit.register(_kill_cam)
    signal.signal(signal.SIGTERM, lambda *_: (_kill_cam(), sys.exit(0)))

    # camera_server 가 첫 프레임을 쓸 때까지 잠시 대기
    print("[Main] 카메라 서버 초기화 대기 중...")
    for _ in range(30):
        if os.path.exists(FRAME_PATH):
            break
        time.sleep(0.2)
    else:
        print("[Main] 경고: 프레임 파일이 아직 없음, 계속 진행...")

    face_app, head = load_models()
    init_firebase()

    threading.Thread(target=capture_worker, daemon=True).start()

    # 첫 프레임 대기
    for _ in range(50):
        with raw_lock:
            if raw_frame is not None:
                break
        time.sleep(0.1)

    threading.Thread(target=inference_worker, args=(face_app, head), daemon=True).start()

    # 얼굴 감지되면 마지막 세션에 한 번만 Status 기록
    threading.Thread(target=oneshot_worker, daemon=True).start()

    print(f"[App] 웹 서버 → http://0.0.0.0:{WEB_PORT}")
    flask_app.run(host="0.0.0.0", port=WEB_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
