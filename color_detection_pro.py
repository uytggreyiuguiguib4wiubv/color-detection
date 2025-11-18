"""
Professional Color Detection System v1.0
Advanced real-time color detection with multiple algorithms and optimization
"""

import cv2
import numpy as np
import json
import os
import logging
from datetime import datetime
from collections import deque
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('color_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DetectionMode(Enum):
    """Detection modes"""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"

class ColorDetectionPro:
    """Professional color detection system with advanced features"""
    
    # HSV color ranges with high precision
    COLOR_RANGES = {
        'Red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([170, 100, 100]), np.array([180, 255, 255]))],
        'Green': [(np.array([40, 60, 60]), np.array([80, 255, 255]))],
        'Blue': [(np.array([100, 80, 80]), np.array([130, 255, 255]))],
        'Yellow': [(np.array([18, 100, 100]), np.array([35, 255, 255]))],
        'Orange': [(np.array([5, 100, 100]), np.array([18, 255, 255]))],
        'Purple': [(np.array([125, 50, 50]), np.array([165, 255, 255]))],
        'Pink': [(np.array([140, 50, 100]), np.array([180, 255, 255]))],
        'Cyan': [(np.array([80, 100, 100]), np.array([100, 255, 255]))],
        'Brown': [(np.array([5, 80, 40]), np.array([20, 200, 180]))],
        'Gray': [(np.array([0, 0, 50]), np.array([180, 50, 200]))],
        'White': [(np.array([0, 0, 180]), np.array([180, 50, 255]))],
        'Black': [(np.array([0, 0, 0]), np.array([180, 255, 50]))]
    }
    
    COLOR_BGR = {
        'Red': (0, 0, 255),
        'Green': (0, 255, 0),
        'Blue': (255, 0, 0),
        'Yellow': (0, 255, 255),
        'Orange': (0, 140, 255),
        'Purple': (128, 0, 128),
        'Pink': (203, 192, 255),
        'Cyan': (255, 255, 0),
        'Brown': (42, 42, 165),
        'Gray': (128, 128, 128),
        'White': (255, 255, 255),
        'Black': (0, 0, 0),
    }
    
    def __init__(self, camera_id=0, resolution=(1280, 720), mode=DetectionMode.BALANCED):
        """Initialize color detection system"""
        self.cap = cv2.VideoCapture(camera_id)
        self.resolution = resolution
        self.mode = mode
        self.detection_history = deque(maxlen=30)  # Track last 30 frames
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = datetime.now()
        
        # Configure camera
        self._configure_camera()
        logger.info(f"Initialized color detection system in {mode.value} mode")
    
    def _configure_camera(self):
        """Configure camera settings for optimal performance"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        logger.info(f"Camera configured: {self.resolution}")
    
    def _apply_clahe(self, frame):
        """Apply CLAHE for contrast enhancement"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _process_mask(self, mask, mode=DetectionMode.BALANCED):
        """Apply morphological operations based on detection mode"""
        if mode == DetectionMode.FAST:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        elif mode == DetectionMode.BALANCED:
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        else:  # ACCURATE
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
            mask = cv2.dilate(mask, kernel_large, iterations=1)
        
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask
    
    def _get_min_area_threshold(self):
        """Adaptive minimum area based on resolution and mode"""
        frame_area = self.resolution[0] * self.resolution[1]
        ratio_map = {
            DetectionMode.FAST: 0.0015,    # ~1400 px for 720p
            DetectionMode.BALANCED: 0.0025,
            DetectionMode.ACCURATE: 0.0035
        }
        ratio = ratio_map.get(self.mode, 0.0025)
        return max(750, int(frame_area * ratio))

    def _filter_contours(self, contour, area, aspect_ratio, solidity, min_area):
        """Advanced contour filtering"""
        # Area threshold
        if area < min_area:
            return False
        
        # Aspect ratio filtering
        if not (0.2 < aspect_ratio < 5.0):
            return False
        
        # Solidity filtering
        if solidity < 0.5:
            return False
        
        return True
    
    def _calculate_metrics(self, cnt):
        """Calculate shape metrics for filtering"""
        area = cv2.contourArea(cnt)
        if area == 0:
            return None
        
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            return None
        
        aspect_ratio = float(w) / h
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter == 0:
            return None
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'bbox': (x, y, w, h)
        }
    
    def detect_colors(self, frame):
        """Detect colors with advanced filtering"""
        frame = cv2.flip(frame, 1)
        
        # Enhance contrast
        frame_enhanced = self._apply_clahe(frame)
        hsv = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2HSV)
        
        detections = []
        
        min_area = self._get_min_area_threshold()

        for color_name, ranges in self.COLOR_RANGES.items():
            color_mask = None
            
            # Create mask
            for lower, upper in ranges:
                if color_mask is None:
                    color_mask = cv2.inRange(hsv, lower, upper)
                else:
                    color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))
            
            # Process mask
            color_mask = self._process_mask(color_mask, self.mode)
            _, color_mask = cv2.threshold(color_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_detection = None

            # Process contours
            for cnt in contours:
                metrics = self._calculate_metrics(cnt)
                if metrics is None:
                    continue

                if self._filter_contours(cnt,
                                         metrics['area'],
                                         metrics['aspect_ratio'],
                                         metrics['solidity'],
                                         min_area):
                    x, y, w, h = metrics['bbox']
                    candidate = {
                        'color': color_name,
                        'bbox': (x, y, w, h),
                        'area': metrics['area'],
                        'confidence': min(metrics['solidity'] * 100, 100)
                    }

                    if (best_detection is None or
                            candidate['area'] > best_detection['area']):
                        best_detection = candidate

            if best_detection:
                detections.append(best_detection)

        return frame, detections
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame with professional styling"""
        for det in detections:
            x, y, w, h = det['bbox']
            color = self.COLOR_BGR[det['color']]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with background
            label = f"{det['color']} ({det['confidence']:.0f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x
            text_y = y - 10
            
            # Background rectangle
            cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 2, text_y + 2), color, -1)
            
            # Text
            cv2.putText(frame, label, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def draw_stats(self, frame, detections, fps):
        """Draw statistics on frame"""
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detection count
        cv2.putText(frame, f"Objects: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mode
        cv2.putText(frame, f"Mode: {self.mode.value.upper()}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detected colors
        if detections:
            colors_detected = list(set([d['color'] for d in detections]))
            color_text = ", ".join(colors_detected[:5])
            if len(colors_detected) > 5:
                color_text += f" +{len(colors_detected) - 5}"
            cv2.putText(frame, f"Colors: {color_text}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def calculate_fps(self):
        """Calculate frames per second"""
        if len(self.fps_counter) > 1:
            return len(self.fps_counter) / (sum(self.fps_counter) / 1000.0)
        return 0
    
    def run(self):
        """Main detection loop"""
        logger.info("Starting color detection...")
        print("\n" + "="*60)
        print("PROFESSIONAL COLOR DETECTION SYSTEM v1.0")
        print("="*60)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("  'f' - Toggle FAST mode")
        print("  'b' - Toggle BALANCED mode")
        print("  'a' - Toggle ACCURATE mode")
        print("  'r' - Reset statistics")
        print("\n" + "="*60 + "\n")
        
        saved_frames = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            frame_start = cv2.getTickCount()
            
            # Detect colors
            frame, detections = self.detect_colors(frame)
            
            # Draw results
            frame = self.draw_detections(frame, detections)
            
            # Calculate and draw FPS
            frame_time = (cv2.getTickCount() - frame_start) / cv2.getTickFrequency() * 1000
            self.fps_counter.append(frame_time)
            fps = self.calculate_fps()
            
            frame = self.draw_stats(frame, detections, fps)
            
            # Display
            cv2.imshow("Professional Color Detection", frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User quit application")
                break
            elif key == ord('s'):
                filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                saved_frames += 1
                logger.info(f"Frame saved: {filename}")
                print(f"✓ Frame saved: {filename}")
            elif key == ord('f'):
                self.mode = DetectionMode.FAST
                logger.info("Switched to FAST mode")
                print("✓ Switched to FAST mode")
            elif key == ord('b'):
                self.mode = DetectionMode.BALANCED
                logger.info("Switched to BALANCED mode")
                print("✓ Switched to BALANCED mode")
            elif key == ord('a'):
                self.mode = DetectionMode.ACCURATE
                logger.info("Switched to ACCURATE mode")
                print("✓ Switched to ACCURATE mode")
            elif key == ord('r'):
                self.fps_counter.clear()
                logger.info("Statistics reset")
                print("✓ Statistics reset")
            
            self.frame_count += 1
        
        self.cleanup()
        logger.info(f"Application closed. Total frames: {self.frame_count}, Saved: {saved_frames}")
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources cleaned up")

def main():
    """Main entry point"""
    print("\nSelect detection mode:")
    print("1. FAST (Lowest latency, basic accuracy)")
    print("2. BALANCED (Good balance, recommended)")
    print("3. ACCURATE (Highest accuracy, more processing)")
    
    choice = input("\nEnter choice (1-3, default=2): ").strip() or "2"
    
    mode_map = {
        "1": DetectionMode.FAST,
        "2": DetectionMode.BALANCED,
        "3": DetectionMode.ACCURATE
    }
    
    mode = mode_map.get(choice, DetectionMode.BALANCED)
    
    detector = ColorDetectionPro(mode=mode)
    detector.run()

if __name__ == "__main__":
    main()
