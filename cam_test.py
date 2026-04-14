import cv2
print("Attempting to open camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Success! Press any key to close the window.")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) != -1:
            break
cap.release()
cv2.destroyAllWindows()