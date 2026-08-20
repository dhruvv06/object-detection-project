from flask import Flask, render_template, Response, jsonify
import cv2
from ml_engine import AsticaVisionEngine

app = Flask(__name__)
engine = AsticaVisionEngine()
camera = cv2.VideoCapture(0)

# Global variables to sync detections and surveillance state
latest_detections = []
surveillance_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_surveillance')
def toggle():
    global surveillance_active
    surveillance_active = not surveillance_active
    return jsonify({"status": surveillance_active})

def stream():
    global latest_detections
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # 1. Process the frame 
        # Now passing 'surveillance_active' to handle the Red/Green box logic
        processed_frame, detections = engine.process(frame, surveillance_active)
        
        # 2. Update the global detections so the webpage can see them
        latest_detections = detections
        
        # 3. Encode and send the frame to the webpage
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections')
def get_detections():
    return jsonify(latest_detections)

if __name__ == "__main__":
    print("🚀 AI Sentinel Server Starting on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, threaded=True)