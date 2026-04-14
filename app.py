from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import csv
import os
from ml_engine import AsticaVisionEngine # Matches the class name above

app = Flask(__name__)
engine = AsticaVisionEngine()
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def stream():
    while True:
        success, frame = camera.read()
        if not success: break
        # This draws boxes AND filters hallucinations
        processed_frame, _ = engine.process(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/metadata')
def metadata():
    success, frame = camera.read()
    if not success: return jsonify([])
    _, detections = engine.process(frame)
    return jsonify(detections)

@app.route('/export')
def export():
    path = "astica_analytics.csv"
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Service", "Event"])
        writer.writerow(["Vision AI", "Data Exported Successfully"])
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, threaded=True)