import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# Load YOLOv8 model (nano for speed, small for accuracy)
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # nano model
model.to(device)

# Color definitions
color_names = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'orange': (0, 140, 255),
    'purple': (128, 0, 128),
    'pink': (203, 192, 255),
    'cyan': (255, 255, 0),
    'brown': (42, 42, 165),
    'gray': (128, 128, 128),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
}

def get_dominant_color(frame, bbox):
    """Extract dominant color from bounding box region"""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Ensure coordinates are within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    region = frame[y1:y2, x1:x2]
    
    if region.size == 0:
        return None
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    # Get average color
    h = int(np.mean(hsv[:, :, 0]))
    s = int(np.mean(hsv[:, :, 1]))
    v = int(np.mean(hsv[:, :, 2]))
    
    return classify_color(h, s, v)

def classify_color(h, s, v):
    """Classify HSV values to color names"""
    # Handle white and black first
    if v < 50:
        return 'black'
    if s < 50 and v > 200:
        return 'white'
    if s < 50:
        return 'gray'
    
    # Classify by hue
    if h < 10 or h > 170:
        return 'red'
    elif 10 <= h < 25:
        return 'orange'
    elif 25 <= h < 35:
        return 'yellow'
    elif 35 <= h < 77:
        return 'green'
    elif 77 <= h < 100:
        return 'cyan'
    elif 100 <= h < 125:
        return 'blue'
    elif 125 <= h < 145:
        return 'purple'
    elif 145 <= h < 170:
        return 'pink'
    else:
        return 'gray'

def detect_objects_with_colors(frame):
    """Detect objects and classify their colors"""
    # Run YOLO detection
    results = model(frame, conf=0.5, verbose=False)
    
    detected_objects = []
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Get dominant color in bounding box
                color_name = get_dominant_color(frame, (x1, y1, x2, y2))
                
                if color_name:
                    detected_objects.append({
                        'bbox': (x1, y1, x2, y2),
                        'color': color_name,
                        'confidence': conf
                    })
    
    return detected_objects

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    for obj in detections:
        x1, y1, x2, y2 = obj['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        color = color_names.get(obj['color'], (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{obj['color'].upper()} ({obj['confidence']:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def main():
    print("\n" + "="*50)
    print("YOLO-BASED COLOR DETECTION")
    print("="*50)
    print("\nPress 'q' to quit")
    print("Press 's' to save frame\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect objects and classify colors
        detections = detect_objects_with_colors(frame)
        
        # Draw detections
        frame = draw_detections(frame, detections)
        
        # Display info
        cv2.putText(frame, f"Objects detected: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if detections:
            colors_detected = set([obj['color'] for obj in detections])
            cv2.putText(frame, f"Colors: {', '.join(colors_detected)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("YOLO Color Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"detection_{frame_count}.jpg", frame)
            print(f"Frame saved: detection_{frame_count}.jpg")
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
