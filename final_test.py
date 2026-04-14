import sys
try:
    import cv2
    import flask
    print("LOG: Libraries imported successfully.")
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit()

print("LOG: Starting Camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("LOG: Camera Error.")
    sys.exit()

print("LOG: Script reached the end. If you see this, your environment is FIXED.")
cap.release()
