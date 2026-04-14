import cv2
import numpy as np
from flask import Flask, render_template_string
import threading
from datetime import datetime

app = Flask(__name__)
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

# --- SETTINGS & "INVENTION" LOGIC ---
ZONE_AREA = [100, 100, 400, 400] # The Restricted Area [x, y, w, h]
detection_log = [] # This stores our "AI Intelligence Feed"

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def run_ai_engine():
    global detection_log
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. HIGH-ACCURACY YOLOv8 PROCESSING
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()
        outputs = np.array([cv2.transpose(outputs[0])])
        
        boxes, confidences, class_ids = [], [], []
        for i in range(outputs[0].shape[0]):
            classes_scores = outputs[0][i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= 0.5:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[0][i][:4]
                left = int((x - w / 2) * (frame.shape[1] / 640))
                top = int((y - h / 2) * (frame.shape[0] / 640))
                boxes.append([left, top, int(w * (frame.shape[1] / 640)), int(h * (frame.shape[0] / 640))])
                confidences.append(float(max_score))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # 2. DRAW RESTRICTED ZONE
        cv2.rectangle(frame, (ZONE_AREA[0], ZONE_AREA[1]), (ZONE_AREA[0]+ZONE_AREA[2], ZONE_AREA[1]+ZONE_AREA[3]), (0, 0, 255), 2)

        # 3. PROCESS EACH DETECTION
        current_frame_objects = []
        for i in indices:
            box = boxes[i]
            label = CLASSES[class_ids[i]]
            conf = int(confidences[i] * 100)
            
            # Intersection Logic (The Invention)
            centerX, centerY = box[0] + box[2]//2, box[1] + box[3]//2
            in_zone = (ZONE_AREA[0] < centerX < ZONE_AREA[0]+ZONE_AREA[2]) and (ZONE_AREA[1] < centerY < ZONE_AREA[1]+ZONE_AREA[3])
            
            color = (0, 0, 255) if in_zone else (0, 255, 0)
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
            cv2.putText(frame, f"{label} {conf}%", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Log significant events
            if in_zone:
                msg = f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ INTRUDER: {label} in Restricted Zone!"
                if msg not in detection_log[-5:]: # Avoid spamming the same log
                    detection_log.append(msg)

        cv2.imwrite("web_stream.jpg", frame)

# --- THE UI/UX (INTEGRATED DASHBOARD) ---
@app.route('/')
def index():
    log_html = "".join([f"<li class='list-group-item bg-dark text-success border-secondary'>{entry}</li>" for entry in detection_log[-10:][::-1]])
    
    return render_template_string(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Master Hub</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <meta http-equiv="refresh" content="1">
        <style>
            body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .sidebar {{ background: #161b22; height: 100vh; padding: 25px; border-right: 1px solid #30363d; }}
            .video-box {{ border: 3px solid #30363d; border-radius: 15px; background: #000; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
            .log-container {{ height: 300px; overflow-y: auto; background: #010409; border-radius: 8px; padding: 10px; border: 1px solid #30363d; }}
            .brand-text {{ color: #58a6ff; font-weight: 800; letter-spacing: 1px; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="row">
                <div class="col-md-3 sidebar">
                    <h2 class="brand-text">AI SENTINEL v2.0</h2>
                    <hr class="text-secondary">
                    <div class="card bg-dark border-secondary mb-3">
                        <div class="card-body">
                            <h6 class="text-primary">System Health</h6>
                            <p class="small text-success m-0">● YOLOv8-Engine: Running</p>
                            <p class="small text-success m-0">● Threading: Active</p>
                        </div>
                    </div>
                    <h6 class="mt-4 text-primary">Intelligence Feed</h6>
                    <ul class="list-group small mt-2">
                        {log_html if log_html else "<li class='list-group-item bg-dark text-muted'>Waiting for detections...</li>"}
                    </ul>
                </div>
                <div class="col-md-9 p-5 text-center">
                    <h1 class="mb-4">Real-Time Spatial Analytics</h1>
                    <div class="video-box mx-auto">
                        <img src="/get_image" style="width: 100%; border-radius: 12px;">
                    </div>
                    <div class="mt-4">
                        <span class="badge bg-danger p-2">Restricted Zone Enabled</span>
                        <span class="badge bg-primary p-2">80 Classes Loaded</span>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """)

@app.route('/get_image')
def get_image():
    with open("web_stream.jpg", "rb") as f:
        return f.read(), 200, {'Content-Type': 'image/jpeg'}

if __name__ == "__main__":
    t = threading.Thread(target=run_ai_engine)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=8080)