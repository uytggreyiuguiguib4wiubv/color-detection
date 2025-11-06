import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)

# Define HSV color ranges
colors = {
    "Red": [(np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255]))],
    "Green": [(np.array([40, 40, 40]), np.array([80, 255, 255]))],
    "Blue": [(np.array([100, 150, 50]), np.array([130, 255, 255]))],
    "Yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    "Orange": [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
    "Purple": [(np.array([130, 50, 50]), np.array([160, 255, 255]))],
    "Pink": [(np.array([160, 100, 100]), np.array([170, 255, 255]))],
    "Cyan": [(np.array([80, 100, 100]), np.array([100, 255, 255]))],
    "Brown": [(np.array([10, 100, 20]), np.array([20, 255, 200]))],
    "Gray": [(np.array([0, 0, 40]), np.array([180, 40, 200]))],
    "White": [(np.array([0, 0, 200]), np.array([180, 40, 255]))],
    "Black": [(np.array([0, 0, 0]), np.array([180, 255, 30]))],
}

# BGR colors for drawing boxes
color_bgr = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Orange": (0, 165, 255),
    "Purple": (255, 0, 255),
    "Pink": (203, 192, 255),
    "Cyan": (255, 255, 0),
    "Brown": (19, 69, 139),
    "Gray": (128, 128, 128),
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, ranges in colors.items():
        mask = None
        for lower, upper in ranges:
            if mask is None:
                mask = cv2.inRange(hsv, lower, upper)
            else:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:  # Ignore small detections
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr[color_name], 2)
                cv2.putText(frame, color_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr[color_name], 2)

    cv2.imshow("Extended Color Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
