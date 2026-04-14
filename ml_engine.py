import cv2
import numpy as np

class AsticaVisionEngine:
    def __init__(self):
        # Load the model - ensure yolov8n.onnx is in the same folder
        self.net = cv2.dnn.readNetFromONNX("yolov8n.onnx")
        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def process(self, frame):
        h_orig, w_orig = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = np.array([cv2.transpose(self.net.forward()[0])])
        
        boxes, confs, ids = [], [], []
        for i in range(outputs[0].shape[0]):
            scores = outputs[0][i][4:]
            max_s = np.amax(scores)
            
            # HIGH THRESHOLD (0.65) to kill the "Bus/Horse" spam
            if max_s >= 0.65: 
                x, y, w, h = outputs[0][i][:4]
                boxes.append([int((x-w/2)*(w_orig/640)), int((y-h/2)*(h_orig/640)), int(w*(w_orig/640)), int(h*(h_orig/640))])
                confs.append(float(max_s))
                ids.append(np.argmax(scores))

        # NMS removes overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.65, 0.45)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                label = self.classes[ids[i]].upper()
                conf = confs[i]
                
                # Draw high-quality boxes only
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(conf*100)}%", (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                detections.append({"label": label, "conf": f"{int(conf*100)}%"})
            
        return frame, detections