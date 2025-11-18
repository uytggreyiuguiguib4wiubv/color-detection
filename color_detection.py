import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# Define HSV color ranges with high precision
colors = {
    # Red (split into two ranges for hue wrap-around)
    "Red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255]))
    ],
    # Green
    "Green": [
        (np.array([40, 60, 60]), np.array([80, 255, 255]))
    ],
    # Blue
    "Blue": [
        (np.array([100, 80, 80]), np.array([130, 255, 255]))
    ],
    # Yellow
    "Yellow": [
        (np.array([18, 100, 100]), np.array([35, 255, 255]))
    ],
    # Orange
    "Orange": [
        (np.array([5, 100, 100]), np.array([18, 255, 255]))
    ],
    # Purple
    "Purple": [
        (np.array([125, 50, 50]), np.array([165, 255, 255]))
    ],
    # Pink
    "Pink": [
        (np.array([140, 50, 100]), np.array([180, 255, 255]))
    ],
    # Cyan
    "Cyan": [
        (np.array([80, 100, 100]), np.array([100, 255, 255]))
    ],
    # Brown
    "Brown": [
        (np.array([5, 80, 40]), np.array([20, 200, 180]))
    ],
    # Gray
    "Gray": [
        (np.array([0, 0, 50]), np.array([180, 50, 200]))
    ],
    # White
    "White": [
        (np.array([0, 0, 180]), np.array([180, 50, 255]))
    ],
    # Black
    "Black": [
        (np.array([0, 0, 0]), np.array([180, 255, 50]))
    ]
}

# BGR colors for drawing boxes
color_bgr = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Orange": (0, 140, 255),
    "Purple": (128, 0, 128),
    "Pink": (203, 192, 255),
    "Cyan": (255, 255, 0),
    "Brown": (42, 42, 165),
    "Gray": (128, 128, 128),
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
}

def process_mask(mask):
    """Apply advanced morphological operations to clean up the mask"""
    # Use different kernel sizes for better noise reduction
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Apply morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    mask = cv2.dilate(mask, kernel_large, iterations=1)
    
    # Apply Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to HSV
    frame = cv2.flip(frame, 1)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2HSV)
    
    detected_colors = []
    
    for color_name, ranges in colors.items():
        color_mask = None
        
        # Create mask for current color range(s)
        for lower, upper in ranges:
            if color_mask is None:
                color_mask = cv2.inRange(hsv, lower, upper)
            else:
                color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))
        
        # Process the mask to reduce noise
        color_mask = process_mask(color_mask)
        
        # Apply threshold to get binary mask
        _, color_mask = cv2.threshold(color_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and labels for detected colors
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Calculate circularity and aspect ratio for better filtering
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                else:
                    circularity = 0
                
                aspect_ratio = float(w) / h if h > 0 else 0
                solidity = area / cv2.contourArea(cv2.convexHull(cnt)) if cv2.contourArea(cv2.convexHull(cnt)) > 0 else 0
                
                # Filter by multiple criteria for accuracy
                if 0.2 < aspect_ratio < 5.0 and solidity > 0.5:
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr[color_name], 2)
                    cv2.putText(frame, f"{color_name}", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color_name], 2)
                    detected_colors.append(color_name)
    
    # Display frame
    cv2.imshow("Color Detection", frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
