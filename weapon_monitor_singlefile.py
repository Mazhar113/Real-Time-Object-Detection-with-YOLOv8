"""
full_object_detection.py
Real-time detection of all objects using YOLOv8 and webcam.
Saves alert frames and prints object names with confidence.
"""

import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

# ------------------ CONFIG ------------------
MODEL_PATH = "yolov8n.pt"       # YOLOv8 model path
VIDEO_SRC = 0                   # 0 = laptop webcam
CONF = 0.35                     # YOLO confidence threshold
ALERT_COOLDOWN = 2.0            # seconds between alerts
ALERTS_DIR = "alerts"
SAVE_ALERTS = True
SHOW_OPENCV_WINDOW = True
FRAME_WIDTH = 640               # resize webcam frames
ALERT_SOUND = None              # path to alarm.wav or None to disable
# --------------------------------------------

os.makedirs(ALERTS_DIR, exist_ok=True)
_last_alert_time = 0.0

# Load YOLO model
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# ------------------ FUNCTIONS ------------------
def save_alert_frame(frame, prefix="alert"):
    if not SAVE_ALERTS:
        return
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.jpg"
    path = os.path.join(ALERTS_DIR, fname)
    cv2.imwrite(path, frame)
    print("Saved alert frame:", path)

def process_frame(frame):
    global _last_alert_time
    alerts = []

    if FRAME_WIDTH:
        scale = FRAME_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0]*scale)))

    results = model.predict(frame, conf=CONF, verbose=False)
    r = results[0]

    for box in r.boxes:
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names.get(cls, str(cls))

        now = time.time()
        if (now - _last_alert_time) > ALERT_COOLDOWN:
            _last_alert_time = now
            print(f"ALERT! Detected {name} (conf: {conf:.2f})")
            alerts.append({"type": name, "bbox": (x1,y1,x2,y2)})
            save_alert_frame(frame, prefix=name)

        # Draw bounding box
        color = (0,255,0)  # green box for all objects
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, alerts

def run_opencv_loop():
    cap = cv2.VideoCapture(VIDEO_SRC)
    if not cap.isOpened():
        print("Cannot open video source", VIDEO_SRC)
        return

    print("Starting detection. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        annotated, alerts = process_frame(frame)

        if SHOW_OPENCV_WINDOW:
            cv2.imshow("Full Object Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    run_opencv_loop()
