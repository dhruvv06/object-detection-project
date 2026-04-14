import sys
import os

# FORCE LOGGING: This will create a file named 'crash_log.txt' if it fails
try:
    import cv2
    import numpy as np
    from flask import Flask, Response
    print("--- LIBRARIES LOADED ---")
except Exception as e:
    with open("crash_log.txt", "w") as f:
        f.write(f"Library Error: {str(e)}")
    print(f"Library Error: {e}")
    sys.exit()

app = Flask(__name__)
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

def generate():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed to open")
        return
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        # Simple detection for testing stability
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return "<h1>AI Dashboard Running</h1><img src='/video'>"

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("--- ATTEMPTING TO START SERVER ON PORT 5005 ---")
    try:
        # Using port 5005 because 8080 might be blocked/used by another app
        app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)
    except Exception as e:
        with open("crash_log.txt", "w") as f:
            f.write(f"Server Error: {str(e)}")
        print(f"Server Error: {e}")