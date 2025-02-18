from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
from heapq import heappush, heappushpop
from collections import Counter
from typing import List, Tuple, Optional

# Load the YOLO model
MODEL_PATH = r'../models/PlateYOLO.pt'
model = YOLO(MODEL_PATH)

class PlateDetector:
    def __init__(self, conf_threshold: float = 0.85, cooldown_frames: int = 15, top_k: int = 7): # Experiment with the values
        self.conf_threshold = conf_threshold
        self.cooldown_frames = cooldown_frames
        self.current_cooldown = 0
        self.top_k = top_k
        self.top_detections = []  # Store (confidence, processed_plate)
    
    def _process_plate_region(self, plate_region: np.ndarray) -> Optional[np.ndarray]:
        if plate_region.size == 0:
            return None
            
        plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        _, plate_thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return plate_thresh
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        results = model(frame)
        processed_frame = frame.copy()
        stop_detection = False
        
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:  # License plate class
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    color = (0, 255, 0) if conf >= self.conf_threshold else (0, 165, 255)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(processed_frame, f'Conf: {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if conf >= self.conf_threshold and self.current_cooldown == 0:
                        plate_region = frame[y1:y2, x1:x2]
                        processed_plate = self._process_plate_region(plate_region)
                        
                        if processed_plate is not None:
                            if len(self.top_detections) < self.top_k:
                                heappush(self.top_detections, (-conf, processed_plate))
                            else:
                                heappushpop(self.top_detections, (-conf, processed_plate))
                            
                            self.current_cooldown = self.cooldown_frames

                            # Stop detection if we have collected enough frames
                            if len(self.top_detections) >= self.top_k:
                                stop_detection = True
        
        cv2.putText(processed_frame, f'Threshold: {self.conf_threshold}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, f'Top Detections: {len(self.top_detections)}/{self.top_k}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return processed_frame, stop_detection
    
    def get_top_plates(self) -> List[Tuple[float, np.ndarray]]:
        return [(-(conf), plate) for conf, plate in sorted(self.top_detections)]
    
    def clear_detections(self):
        self.top_detections = []

def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None
    
    detector = PlateDetector(conf_threshold=0.85)
    print(f"Starting processing... Press 'q' to stop")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, stop_detection = detector.process_frame(frame)
            cv2.imshow('License Plate Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or stop_detection:
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Get top 5 plates and send to OCR
    top_plates = detector.get_top_plates()
    if top_plates:
        # Import OCR model
        from OCR_model import PlateOCR
        ocr_model = PlateOCR()
        # Process each plate through OCR
        ocr_results = []
        for conf, plate_img in top_plates:
            try:
                # Pass the plate image directly to OCR model
                result = ocr_model.extract_text(plate_img)  # Make sure OCR model accepts numpy array
                if result:
                    ocr_results.append(result)
                    print(f"Plate with confidence {conf:.2f} -> OCR result: {result}")
            except Exception as e:
                print(f"OCR Error on plate with confidence {conf:.2f}: {str(e)}")
        
        # Get the most common OCR result
        if ocr_results:
            final_result = Counter(ocr_results).most_common(1)[0][0]
            print(f"\nFinal plate number (from {len(ocr_results)} successful reads): {final_result}")
            
            detector.clear_detections()
            return final_result
    
    detector.clear_detections()
    return None

if __name__ == "__main__":
    video_path = r'./data/car2.mp4'
    result = process_video(video_path)
