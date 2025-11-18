import cv2
import numpy as np
import json
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# File to store calibrated color ranges
CALIBRATION_FILE = "color_calibration.json"

# Default HSV color ranges
default_colors = {
    "Red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([170, 100, 100]), np.array([180, 255, 255]))],
    "Green": [(np.array([40, 60, 60]), np.array([80, 255, 255]))],
    "Blue": [(np.array([100, 80, 80]), np.array([130, 255, 255]))],
    "Yellow": [(np.array([18, 100, 100]), np.array([35, 255, 255]))],
    "Orange": [(np.array([5, 100, 100]), np.array([18, 255, 255]))],
    "Purple": [(np.array([125, 50, 50]), np.array([165, 255, 255]))],
    "Pink": [(np.array([140, 50, 100]), np.array([180, 255, 255]))],
    "Cyan": [(np.array([80, 100, 100]), np.array([100, 255, 255]))],
    "Brown": [(np.array([5, 80, 40]), np.array([20, 200, 180]))],
    "Gray": [(np.array([0, 0, 50]), np.array([180, 50, 200]))],
    "White": [(np.array([0, 0, 180]), np.array([180, 50, 255]))],
    "Black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))]
}

# BGR colors for drawing
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

def load_calibration():
    """Load calibrated color ranges from file"""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as f:
            data = json.load(f)
            colors = {}
            for color_name, ranges in data.items():
                colors[color_name] = [
                    (np.array(r[0], dtype=np.uint8), np.array(r[1], dtype=np.uint8)) for r in ranges
                ]
            return colors
    return default_colors

def save_calibration(colors):
    """Save calibrated color ranges to file"""
    data = {}
    for color_name, ranges in colors.items():
        data[color_name] = [
            (r[0].tolist(), r[1].tolist()) for r in ranges
        ]
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def calibrate_color(color_name, frame_hsv):
    """Calibrate a single color by sampling from the frame"""
    print(f"\n=== Calibrating {color_name} ===")
    print("Instructions:")
    print("1. Click on the color sample in the video window")
    print("2. A region around your click will be sampled")
    print("3. Press SPACE to confirm or ESC to skip")
    
    sample_region = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal sample_region
        if event == cv2.EVENT_LBUTTONDOWN:
            # Sample a 50x50 region around the click
            x1 = max(0, x - 25)
            y1 = max(0, y - 25)
            x2 = min(frame_hsv.shape[1], x + 25)
            y2 = min(frame_hsv.shape[0], y + 25)
            sample_region = (x1, y1, x2, y2)
    
    cv2.namedWindow(f"Calibrate {color_name}")
    cv2.setMouseCallback(f"Calibrate {color_name}", mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        display = frame.copy()
        
        if sample_region:
            x1, y1, x2, y2 = sample_region
            cv2.rectangle(display, (x1, y1), (x2, y2), color_bgr[color_name], 2)
            
            # Calculate HSV range from sampled region
            region = frame_hsv[y1:y2, x1:x2]
            h_min, s_min, v_min = region.min(axis=(0, 1))
            h_max, s_max, v_max = region.max(axis=(0, 1))
            
            # Add some tolerance
            h_min = max(0, h_min - 5)
            s_min = max(0, s_min - 20)
            v_min = max(0, v_min - 20)
            h_max = min(180, h_max + 5)
            s_max = min(255, s_max + 20)
            v_max = min(255, v_max + 20)
            
            cv2.putText(display, f"H: {h_min}-{h_max} S: {s_min}-{s_max} V: {v_min}-{v_max}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(display, "Click on color | SPACE=Confirm | ESC=Skip", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(f"Calibrate {color_name}", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32 and sample_region:  # SPACE
            x1, y1, x2, y2 = sample_region
            region = frame_hsv[y1:y2, x1:x2]
            h_min, s_min, v_min = region.min(axis=(0, 1))
            h_max, s_max, v_max = region.max(axis=(0, 1))
            
            h_min = max(0, h_min - 5)
            s_min = max(0, s_min - 20)
            v_min = max(0, v_min - 20)
            h_max = min(180, h_max + 5)
            s_max = min(255, s_max + 20)
            v_max = min(255, v_max + 20)
            
            cv2.destroyWindow(f"Calibrate {color_name}")
            return [(np.array([h_min, s_min, v_min], dtype=np.uint8), np.array([h_max, s_max, v_max], dtype=np.uint8))]
    
    cv2.destroyWindow(f"Calibrate {color_name}")
    return None

def process_mask(mask):
    """Apply advanced morphological operations"""
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    mask = cv2.dilate(mask, kernel_large, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

def detect_colors(frame, colors):
    """Detect colors in frame"""
    frame = cv2.flip(frame, 1)
    
    # Apply CLAHE for contrast enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2HSV)
    
    detected = []
    
    for color_name, ranges in colors.items():
        color_mask = None
        
        for lower, upper in ranges:
            if color_mask is None:
                color_mask = cv2.inRange(hsv, lower, upper)
            else:
                color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))
        
        color_mask = process_mask(color_mask)
        _, color_mask = cv2.threshold(color_mask, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                else:
                    circularity = 0
                
                aspect_ratio = float(w) / h if h > 0 else 0
                solidity = area / cv2.contourArea(cv2.convexHull(cnt)) if cv2.contourArea(cv2.convexHull(cnt)) > 0 else 0
                
                if 0.2 < aspect_ratio < 5.0 and solidity > 0.5:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr[color_name], 2)
                    cv2.putText(frame, f"{color_name}", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color_name], 2)
                    detected.append(color_name)
    
    return frame, detected

def main():
    print("\n" + "="*50)
    print("COLOR DETECTION WITH TRAINING")
    print("="*50)
    print("\nOptions:")
    print("1. Train/Calibrate colors")
    print("2. Run detection with existing calibration")
    print("3. Use default ranges")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        colors = load_calibration()
        print("\nStarting calibration mode...")
        print("Available colors to calibrate:")
        for i, color in enumerate(colors.keys(), 1):
            print(f"{i}. {color}")
        
        calibrate_choice = input("\nEnter color number to calibrate (or 'all' for all colors): ").strip().lower()
        
        if calibrate_choice == "all":
            for color_name in colors.keys():
                result = calibrate_color(color_name, None)
                if result:
                    colors[color_name] = result
        else:
            try:
                idx = int(calibrate_choice) - 1
                color_name = list(colors.keys())[idx]
                result = calibrate_color(color_name, None)
                if result:
                    colors[color_name] = result
            except:
                print("Invalid choice")
                return
        
        save_calibration(colors)
        print("\nCalibration saved!")
    else:
        colors = load_calibration() if choice == "2" else default_colors
    
    print("\nStarting color detection...")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, detected = detect_colors(frame, colors)
        
        cv2.imshow("Color Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
