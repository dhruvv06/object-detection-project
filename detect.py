import cv2
import numpy as np
import os
import sys

def main():
    # 1. FILE CHECK
    weights = "yolov4-tiny.weights"
    config = "yolov4-tiny.cfg"
    
    print("--- STEP 1: VERIFYING FILES ---")
    for f in [weights, config]:
        if not os.path.isfile(f):
            print(f"ERROR: {f} is missing from this folder!")
            return
        else:
            print(f"OK: {f} found ({os.path.getsize(f)} bytes)")

    # 2. LOAD NETWORK
    print("\n--- STEP 2: LOADING YOLO ---")
    try:
        net = cv2.dnn.readNet(weights, config)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("OK: Model loaded into memory.")
    except Exception as e:
        print(f"ERROR loading network: {e}")
        return

    # 3. CAMERA
    print("\n--- STEP 3: STARTING CAMERA ---")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera blocked. Close other apps or check Windows Privacy Settings.")
        return
    print("OK: Camera active.")

    # 4. CLASSES (80 COCO Classes)
    classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    print("\n--- STEP 4: RUNNING DETECTION (Press 'q' to quit) ---")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        height, width, _ = frame.shape

        # Blob & Forward Pass
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        
        ln = net.getLayerNames()
        out_layers = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(out_layers)

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w/2), int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                color = colors[class_ids[i]]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                
                # Improved Output: Thick boxes and clear text
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        cv2.imwrite("stream.jpg", frame) 
        cv2.imshow("YOLOv4-Tiny Project", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nCRITICAL CRASH: {e}")
    
    # This prevents the window from closing instantly!
    input("\n--- Script Finished. Press ENTER to close this window ---")