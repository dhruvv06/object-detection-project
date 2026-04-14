import cv2
import numpy as np
import os

def main():
    print("[CHECKPOINT 1] Loading Files...")
    weights = "yolov4-tiny.weights"
    cfg = "yolov4-tiny.cfg"
    
    # Check if files exist
    if not os.path.exists(weights) or not os.path.exists(cfg):
        print(f"[ERROR] Files missing! Weights: {os.path.exists(weights)}, CFG: {os.path.exists(cfg)}")
        return

    print("[CHECKPOINT 2] Loading YOLO Network...")
    try:
        # Load the network using Darknet framework
        net = cv2.dnn.readNetFromDarknet(cfg, weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    print("[CHECKPOINT 3] Setting up Layers...")
    try:
        ln = net.getLayerNames()
        # Handle OpenCV 4.x vs 3.x layer indexing differences
        unconnected = net.getUnconnectedOutLayers()
        if len(unconnected.shape) == 1:
            ln = [ln[i - 1] for i in unconnected]
        else:
            ln = [ln[i[0] - 1] for i in unconnected]
    except Exception as e:
        print(f"[ERROR] Layer setup failed: {e}")
        return

    print("[CHECKPOINT 4] Starting Camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        return

    # COCO 80 Classes
    classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    print("[CHECKPOINT 5] Detection Loop Running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(ln)

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    h, w = frame.shape[:2]
                    center_x, center_y = int(detection[0] * w), int(detection[1] * h)
                    width, height = int(detection[2] * w), int(detection[3] * h)
                    x, y = int(center_x - width/2), int(center_y - height/2)
                    
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()