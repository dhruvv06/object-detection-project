import cv2
import numpy as np
from flask import Flask, Response, request
import os
from datetime import datetime

app = Flask(__name__)

# 1. LOAD MODEL
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

# 80 Objects List
CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# To store the latest frame for the "Capture" function
latest_frame = None

def generate_frames():
    global latest_frame
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success: break

        # YOLOv8 Detection Logic
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs[0].shape[0]

        boxes, confidences, class_ids = [], [], []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= 0.5:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[0][i][:4]
                left = int((x - w / 2) * (frame.shape[1] / 640))
                top = int((y - h / 2) * (frame.shape[0] / 640))
                width = int(w * (frame.shape[1] / 640))
                height = int(h * (frame.shape[0] / 640))
                boxes.append([left, top, width, height])
                confidences.append(float(max_score))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # --- UI: DRAW BOXES AND COUNT ---
        for i in indices:
            box = boxes[i]
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
            label = f"{CLASSES[class_ids[i]]}"
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- UI: BLACK COUNTER BOX ---
        count_text = f"Objects Detected: {len(indices)}"
        cv2.rectangle(frame, (10, 10), (320, 60), (0, 0, 0), -1)
        cv2.putText(frame, count_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        latest_frame = frame.copy() # Save for capture
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
        <body style="background:#1a1a1a; color:white; text-align:center; font-family:sans-serif;">
            <h1 style="color:#4CAF50;">AI Web Dashboard</h1>
            <div style="margin:20px;">
                <img src="/video_feed" style="width:80%; border:3px solid #333; border-radius:10px;">
            </div>
            <button onclick="fetch('/capture')" style="padding:15px 30px; font-size:18px; background:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer;">📸 Capture Photo</button>
            <p id="status" style="color:#888; margin-top:10px;">Ready</p>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    global latest_frame
    if latest_frame is not None:
        if not os.path.exists('captures'): os.makedirs('captures')
        filename = f"captures/detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, latest_frame)
        return f"Saved as {filename}"
    return "Error: No frame"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, threaded=True)