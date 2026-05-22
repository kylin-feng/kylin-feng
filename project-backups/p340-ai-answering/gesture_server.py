"""
gesture_server.py — 手势控制机械臂后端（Python 端检测，无需 CDN）
用法: python3 gesture_server.py
浏览器打开: http://localhost:5001
"""
import math
import time
import threading
import queue
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from pymycobot.ultraArmP340 import ultraArmP340

# ── 配置 ──────────────────────────────────────────────
PORT     = "/dev/cu.wchusbserial1140"
BAUD     = 115200
SPEED    = 25
ALPHA    = 0.25
SEND_HZ  = 20
CAM_ID   = 0
MODEL    = "./hand_landmarker.task"

J_LIMITS = [(-150, 170), (-20, 90), (-5, 110), (-179, 179)]
# ──────────────────────────────────────────────────────

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

smoothed   = [0.0, 0.0, 0.0, 0.0]
robot_lock = threading.Lock()
angle_queue = queue.Queue(maxsize=1)   # 只保留最新一帧角度

# ── 机械臂初始化 ───────────────────────────────────────
print("正在连接机械臂...")
ua = ultraArmP340(PORT, BAUD)
print("机械臂回零中...")
ua.go_zero()
print("机械臂就绪")

# ── MediaPipe Hands (Tasks API) ────────────────────────
latest_landmarks = None
lm_lock = threading.Lock()

def on_result(result, output_image, timestamp_ms):
    global latest_landmarks
    with lm_lock:
        if result.hand_landmarks:
            latest_landmarks = result.hand_landmarks[0]
        else:
            latest_landmarks = None

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=on_result
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

# 画连接线用的骨架连接对
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ── 摄像头 ─────────────────────────────────────────────
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_lock   = threading.Lock()
latest_frame = None  # JPEG bytes


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def landmarks_to_angles(lm):
    def d2(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    j1 = lm[8].y * 320.0 - 150.0
    j2 = (1.0 - lm[8].x) * 110.0 - 20.0
    pinch = d2(lm[4], lm[8])
    palm  = d2(lm[0], lm[9]) + 1e-6
    j3 = min(pinch / palm, 1.0) * 115.0 - 5.0
    dx = lm[5].x - lm[0].x
    dy = lm[5].y - lm[0].y
    j4 = math.degrees(math.atan2(-dy, dx)) * (179.0 / 90.0)
    raw = [j1, j2, j3, j4]
    return [clamp(v, lo, hi) for v, (lo, hi) in zip(raw, J_LIMITS)]


def robot_loop():
    """独立线程：以固定频率消费角度队列，发给机械臂"""
    interval = 1.0 / SEND_HZ
    while True:
        try:
            angles = angle_queue.get(timeout=0.5)
            ua.set_angles(angles, SPEED)
        except queue.Empty:
            pass
        time.sleep(interval)


def camera_loop():
    global latest_frame, smoothed
    ts = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        ts += 33

        # 送给 MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_landmarker.detect_async(mp_img, ts)

        h, w = frame.shape[:2]
        with lm_lock:
            lm = latest_landmarks

        hand_detected = lm is not None
        if hand_detected and control_mode == "gesture":
            # 画骨架
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (60, 120, 255), 2)
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 4, (120, 180, 255), -1)
            # 高亮食指尖(8)和拇指尖(4)
            cv2.circle(frame, pts[8], 10, (80, 80, 255), -1)
            cv2.circle(frame, pts[4], 10, (50, 210, 255), -1)

            # 映射角度 + EMA 平滑
            raw = landmarks_to_angles(lm)
            for i in range(4):
                smoothed[i] = ALPHA * raw[i] + (1 - ALPHA) * smoothed[i]

            # 放入队列（丢弃旧帧，只保留最新）
            angles = [round(a, 1) for a in smoothed]
            try:
                angle_queue.put_nowait(angles)
            except queue.Full:
                try:
                    angle_queue.get_nowait()
                except queue.Empty:
                    pass
                angle_queue.put_nowait(angles)

        # 仅手势模式下推送角度更新到浏览器
        if control_mode == "gesture":
            socketio.emit("angle_update", {
                "angles": [round(a, 2) for a in smoothed],
                "hand": hand_detected
            })

        label = "Hand Detected" if hand_detected else "No Hand"
        color = (80, 200, 80) if hand_detected else (80, 80, 200)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_frame = buf.tobytes()
        # 不 sleep，让摄像头跑满帧率


def gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.02)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


control_mode = "keyboard"   # 'gesture' | 'keyboard'

@socketio.on("set_mode")
def handle_set_mode(data):
    global control_mode
    control_mode = data.get("mode", "keyboard")
    print(f"控制模式切换: {control_mode}")

@socketio.on("keyboard_cmd")
def handle_keyboard(data):
    cmd = data.get("cmd")
    step = data.get("step", 5)

    if cmd == "go_zero":
        threading.Thread(target=ua.go_zero, daemon=True).start()
        for i in range(4): smoothed[i] = 0.0
        emit("angle_update", {"angles": [0.0]*4, "hand": False})
        return

    if cmd == "set_angles":
        angles = data.get("angles")
        if angles:
            for i in range(min(len(angles), 4)): smoothed[i] = angles[i]
            threading.Thread(target=ua.set_angles, args=(angles, SPEED), daemon=True).start()
            emit("angle_update", {"angles": [round(v, 2) for v in angles], "hand": False})
        return

    # 步进控制：读当前平滑角度，微调后发出
    a = list(smoothed)
    mapping = {
        "j1+": (0, +step), "j1-": (0, -step),
        "j2+": (1, +step), "j2-": (1, -step),
        "j3+": (2, +step), "j3-": (2, -step),
        "j4+": (3, +step), "j4-": (3, -step),
    }
    if cmd in mapping:
        idx, delta = mapping[cmd]
        lo, hi = J_LIMITS[idx]
        a[idx] = clamp(a[idx] + delta, lo, hi)
        smoothed[idx] = a[idx]
        try:
            angle_queue.put_nowait([round(v, 1) for v in a])
        except queue.Full:
            try: angle_queue.get_nowait()
            except queue.Empty: pass
            angle_queue.put_nowait([round(v, 1) for v in a])
        emit("angle_update", {"angles": [round(v, 2) for v in a], "hand": False})

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return render_template("gesture.html")


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=robot_loop, daemon=True).start()
    print("浏览器打开: http://localhost:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)
