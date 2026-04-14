import cv2
import numpy as np

class AsticaEngine:
    def __init__(self):
        self.net = cv2.dnn.readNetFromONNX("yolov8n.onnx")
        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def analyze(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = np.array([cv2.transpose(self.net.forward()[0])])
        
        results = []
        for i in range(preds[0].shape[0]):
            score = np.amax(preds[0][i][4:])
            if score > 0.45:
                results.append({
                    "label": self.classes[np.argmax(preds[0][i][4:])],
                    "confidence": float(score)
                })
        return results