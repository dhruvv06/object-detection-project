import cv2
import numpy as np

# Load the YOLOv8 model you downloaded earlier
try:
    net = cv2.dnn.readNetFromONNX("yolov8n.onnx")
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # YOLOv8 processing
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
            if CLASSES[class_id] == "person":
                from datetime import datetime # Make sure this is at the top or here
                timestamp = datetime.now().strftime("%H-%M-%S")
                cv2.imwrite(f"alert_{timestamp}.jpg", frame)
                print(f"⚠️ ALERT: Person detected at {timestamp}!")
            x, y, w, h = outputs[0][i][:4]
            left = int((x - w / 2) * (frame.shape[1] / 640))
            top = int((y - h / 2) * (frame.shape[0] / 640))
            width = int(w * (frame.shape[1] / 640))
            height = int(h * (frame.shape[0] / 640))
            boxes.append([left, top, width, height])
            confidences.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw Boxes
    for i in indices:
        box = boxes[i]
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        label = f"{CLASSES[class_ids[i]]}: {int(confidences[i]*100)}%"
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow("Local YOLOv8 Test", frame)
    cv2.imwrite("stream.jpg", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
