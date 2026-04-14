import cv2
import numpy as np
from flask import Flask, Response, render_template_string
import threading

app = Flask(__name__)
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

# UI CONFIG
ZONE_COLOR = (0, 0, 255) # Red for the "Danger Zone"
ZONE_AREA = [100, 100, 400, 400] # [x, y, w, h] for our custom logic

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def run_ai():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()
        outputs = np.array([cv2.transpose(outputs[0])])
        
        boxes, confidences, class_ids = [], [], []
        for i in range(outputs[0].shape[0]):
            classes_scores = outputs[0][i][4:]
            if np.amax(classes_scores) >= 0.5:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[0][i][:4]
                left = int((x - w / 2) * (frame.shape[1] / 640))
                top = int((y - h / 2) * (frame.shape[0] / 640))
                boxes.append([left, top, int(w * (frame.shape[1] / 640)), int(h * (frame.shape[0] / 640))])
                confidences.append(float(np.amax(classes_scores)))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # --- THE "INVENTION": ZONE LOGIC ---
        cv2.rectangle(frame, (ZONE_AREA[0], ZONE_AREA[1]), (ZONE_AREA[0]+ZONE_AREA[2], ZONE_AREA[1]+ZONE_AREA[3]), ZONE_COLOR, 2)
        cv2.putText(frame, "RESTRICTED ZONE", (ZONE_AREA[0], ZONE_AREA[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_COLOR, 2)

        for i in indices:
            box = boxes[i]
            # Check if object center is inside the Danger Zone
            centerX, centerY = box[0] + box[2]//2, box[1] + box[3]//2
            
            in_zone = (ZONE_AREA[0] < centerX < ZONE_AREA[0]+ZONE_AREA[2]) and \
                      (ZONE_AREA[1] < centerY < ZONE_AREA[1]+ZONE_AREA[3])
            
            color = (0, 0, 255) if in_zone else (0, 255, 0)
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
            
        cv2.imwrite("stream.jpg", frame)

# --- THE UI/UX DESIGN (HTML/CSS) ---
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Smart Hub</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: #0b0e14; color: #e0e0e0; font-family: 'Inter', sans-serif; }
            .sidebar { height: 100vh; background: #161b22; border-right: 1px solid #30363d; padding: 20px; }
            .main-feed { padding: 30px; }
            .video-container { border-radius: 12px; border: 2px solid #30363d; overflow: hidden; background: #000; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
            .status-card { background: #1c2128; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
            .btn-capture { background: #238636; border: none; width: 100%; color: white; padding: 10px; border-radius: 6px; font-weight: bold; }
            .btn-capture:hover { background: #2ea043; }
            .alert-pill { color: #f85149; font-weight: bold; }
        </style>
        <meta http-equiv="refresh" content="1">
    </head>
    <body>
        <div class="container-fluid">
            <div class="row">
                <div class="col-md-3 sidebar">
                    <h3>🛡️ AI Sentinel</h3>
                    <hr>
                    <div class="status-card">
                        <small class="text-secondary">SYSTEM STATUS</small>
                        <div class="d-flex align-items-center">
                            <div style="width:10px; height:10px; background:#238636; border-radius:50%; margin-right:10px;"></div>
                            <span>Active & Identifying</span>
                        </div>
                    </div>
                    <div class="status-card">
                        <small class="text-secondary">ZONE PROTECTION</small>
                        <p class="m-0 alert-pill">Restricted Area: ON</p>
                    </div>
                    <button class="btn-capture">📸 Snapshot Frame</button>
                </div>
                <div class="col-md-9 main-feed text-center">
                    <h2 class="mb-4">Live Analytics Dashboard</h2>
                    <div class="video-container">
                        <img src="/static_image" style="width: 100%;">
                    </div>
                    <p class="mt-3 text-secondary">Invention: Hierarchical Spatial Intersection Logic for Restricted Zones.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """)

@app.route('/static_image')
def static_image():
    with open("stream.jpg", "rb") as f:
        return f.read(), 200, {'Content-Type': 'image/jpeg'}

if __name__ == "__main__":
    t = threading.Thread(target=run_ai)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=8080)