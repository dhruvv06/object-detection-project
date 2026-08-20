import cv2
import numpy as np
import time  # NEW: Required for performance tracking

class AsticaVisionEngine:
    def __init__(self):
        self.net = cv2.dnn.readNetFromONNX("yolov8n.onnx")
        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def process(self, frame, surveillance_mode=False):
        # 1. Start the performance clock
        start_time = time.time() 
        
        h_orig, w_orig = frame.shape[:2]
        length = max(h_orig, w_orig)
        letterbox = np.zeros((length, length, 3), np.uint8)
        letterbox[0:h_orig, 0:w_orig] = frame
        scale = length / 640
        
        blob = cv2.dnn.blobFromImage(letterbox, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        outputs = np.array([cv2.transpose(outputs[0])])
        
        boxes, confs, ids = [], [], []
        
        # Using your 0.10 threshold as per previous preference
        for i in range(outputs[0].shape[0]):
            scores = outputs[0][i][4:]
            max_s = np.amax(scores)
            if max_s >= 0.10: 
                x, y, w, h = outputs[0][i][:4]
                left = int((x - w/2) * scale)
                top = int((y - h/2) * scale)
                width = int(w * scale)
                height = int(h * scale)
                
                boxes.append([left, top, width, height])
                confs.append(float(max_s))
                ids.append(np.argmax(scores))

        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.10, 0.45)
        
        detections = []
        display_frame = frame.copy()
        
        if len(indices) > 0:
            for i in indices.flatten():
                label = self.classes[ids[i]].upper()
                box, conf = boxes[i], confs[i]
                
                # --- SURVEILLANCE COLOR LOGIC ---
                color = (0, 255, 0) # Default Green
                
                if surveillance_mode and label == "PERSON":
                    color = (0, 0, 255) # Red in BGR
                
                cv2.rectangle(display_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
                cv2.putText(display_frame, f"{label} {int(conf*100)}%", (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                detections.append({"label": label, "accuracy": f"{int(conf*100)}%"})

        # 2. End clock and Calculate Speed
        # We do this at the very end to include the overhead of drawing boxes
        latency = (time.time() - start_time) * 1000 # in milliseconds
        fps = 1.0 / (time.time() - start_time)

        # 3. Draw Performance Overlay (Black bar background for high visibility)
        cv2.rectangle(display_frame, (0, 0), (300, 40), (0, 0, 0), -1) 
        cv2.putText(display_frame, f"Latency: {int(latency)}ms | FPS: {int(fps)}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return display_frame, detections