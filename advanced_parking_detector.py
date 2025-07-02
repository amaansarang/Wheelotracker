import cv2
import numpy as np
import json
import os
import pickle
from datetime import datetime
from skimage.transform import resize
import csv

class AdvancedParkingDetector:
    def __init__(self, mask_path='mask_1920_1080.png', model_path='model.p', config_file='parking_config.json'):
        """Initialize the advanced parking detector with ML model"""
        self.mask_path = mask_path
        self.model_path = model_path
        self.config_file = config_file
        
        # Load mask and model
        self.mask = self._load_mask()
        self.model = self._load_model()
        
        # Get parking spots from mask
        self.parking_spots = self._get_parking_spots_from_mask()
        
        # Initialize tracking
        self.previous_frame = None
        self.frame_count = 0
        self.step = 30  # Process every 30th frame for efficiency
        self.spots_status = [False] * len(self.parking_spots)
        self.diffs = np.zeros(len(self.parking_spots), dtype=float)
        
        # CSV logging
        self.log_file = None
        self.csv_writer = None
        self._init_csv_logging()
        
        print(f"Initialized parking detector with {len(self.parking_spots)} parking spots")

    def _load_mask(self):
        """Load parking lot mask"""
        try:
            mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not load mask from {self.mask_path}, creating default mask")
                return self._create_default_mask()
            return mask
        except Exception as e:
            print(f"Error loading mask: {str(e)}, creating default mask")
            return self._create_default_mask()

    def _create_default_mask(self):
        """Create a default mask if file is not available"""
        # Create a simple grid-based mask
        mask = np.zeros((600, 800), dtype=np.uint8)
        
        # Create parking spots in a grid pattern
        for row in range(4):
            for col in range(10):
                x = 50 + col * 70
                y = 75 + row * 120
                cv2.rectangle(mask, (x, y), (x + 50, y + 100), 255, -1)
        
        return mask

    def _load_model(self):
        """Load the trained ML model"""
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            print("ML model loaded successfully")
            return model
        except Exception as e:
            print(f"Warning: Could not load ML model from {self.model_path}: {str(e)}")
            return None

    def _get_parking_spots_from_mask(self):
        """Extract parking spot coordinates from mask using connected components"""
        try:
            connected = cv2.connectedComponentsWithStats(self.mask, connectivity=4, ltype=cv2.CV_32S)
            (total_labels, label_ids, values, centroid) = connected
            
            slots = []
            for i in range(1, total_labels):
                x1 = int(values[i, cv2.CC_STAT_LEFT])
                y1 = int(values[i, cv2.CC_STAT_TOP])
                w = int(values[i, cv2.CC_STAT_WIDTH])
                h = int(values[i, cv2.CC_STAT_HEIGHT])
                
                # Filter out very small or very large components
                if 500 < w * h < 50000:
                    slots.append([x1, y1, w, h])
            
            return slots
        except Exception as e:
            print(f"Error extracting parking spots: {str(e)}")
            return self._create_default_spots()

    def _create_default_spots(self):
        """Create default parking spots if mask processing fails"""
        spots = []
        for row in range(4):
            for col in range(10):
                x = 50 + col * 70
                y = 75 + row * 120
                spots.append([x, y, 50, 100])
        return spots

    def _init_csv_logging(self):
        """Initialize CSV logging"""
        try:
            self.log_file = open("parking_log.csv", "w", newline="")
            self.csv_writer = csv.writer(self.log_file)
            
            # Create header
            header = ["Frame", "Timestamp", "FreeSpots", "TotalSpots"] + [f"Spot{i+1}" for i in range(len(self.parking_spots))]
            self.csv_writer.writerow(header)
        except Exception as e:
            print(f"Warning: Could not initialize CSV logging: {str(e)}")

    def _calc_diff(self, im1, im2):
        """Calculate absolute difference of mean pixel intensities"""
        return abs(float(np.mean(im1)) - float(np.mean(im2)))

    def _empty_or_not_ml(self, spot_bgr):
        """Use ML model to determine if parking spot is empty"""
        if self.model is None:
            # Fallback to simple threshold method
            gray = cv2.cvtColor(spot_bgr, cv2.COLOR_BGR2GRAY) if len(spot_bgr.shape) == 3 else spot_bgr
            return np.mean(gray) > 100  # Simple brightness threshold
        
        try:
            # Resize image to model input size (15x15x3)
            img_resized = resize(spot_bgr, (15, 15, 3))
            flat_data = np.array([img_resized.flatten()])
            
            # Predict: 0 = empty, 1 = occupied
            prediction = self.model.predict(flat_data)
            return prediction[0] == 0  # True if empty
        except Exception as e:
            print(f"Error in ML prediction: {str(e)}")
            # Fallback to simple method
            gray = cv2.cvtColor(spot_bgr, cv2.COLOR_BGR2GRAY) if len(spot_bgr.shape) == 3 else spot_bgr
            return np.mean(gray) > 100

    def detect_parking_spaces(self, frame=None):
        """
        Main detection function
        Returns processed frame and parking status dictionary
        """
        try:
            if frame is None:
                return None, {}
            
            processed_frame = frame.copy()
            self.frame_count += 1
            
            # Difference calculation step
            if self.frame_count % self.step == 0 and self.previous_frame is not None:
                for i, (x, y, w, h) in enumerate(self.parking_spots):
                    current_roi = frame[y:y+h, x:x+w]
                    previous_roi = self.previous_frame[y:y+h, x:x+w]
                    self.diffs[i] = self._calc_diff(current_roi, previous_roi)

            # Classification step
            if self.frame_count % self.step == 0:
                if self.previous_frame is None:
                    idxs_to_check = range(len(self.parking_spots))
                else:
                    thresh = 0.4 * self.diffs.max() if self.diffs.max() else 0
                    idxs_to_check = [i for i, d in enumerate(self.diffs) if d > thresh]

                for i in idxs_to_check:
                    x, y, w, h = self.parking_spots[i]
                    spot_roi = frame[y:y+h, x:x+w]
                    self.spots_status[i] = self._empty_or_not_ml(spot_roi)

                self.previous_frame = frame.copy()

            # Drawing step
            parking_status = {}
            for i, ((x, y, w, h), is_free) in enumerate(zip(self.parking_spots, self.spots_status)):
                color = (0, 255, 0) if is_free else (0, 0, 255)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
                
                # Add spot number
                cv2.putText(processed_frame, f"P{i+1}", 
                           (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Store status
                parking_status[i+1] = 'free' if is_free else 'occupied'

            # Add summary text
            free_count = sum(self.spots_status)
            total_spots = len(self.spots_status)
            
            cv2.rectangle(processed_frame, (10, 10), (400, 60), (0, 0, 0), -1)
            cv2.putText(processed_frame, f"Available spots: {free_count} / {total_spots}",
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(processed_frame, timestamp, (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Log to CSV
            self._log_to_csv(free_count, total_spots, timestamp)

            return processed_frame, parking_status

        except Exception as e:
            print(f"Error in parking detection: {str(e)}")
            demo_frame = self._create_demo_frame()
            return demo_frame, {}

    def _create_demo_frame(self):
        """Create a demo frame for visualization"""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Add background
        cv2.rectangle(frame, (0, 0), (800, 600), (50, 50, 50), -1)
        
        # Add parking lot lines
        for i in range(4):
            for j in range(10):
                x = 50 + j * 70
                y = 75 + i * 120
                # Simulate some occupied spots
                is_occupied = (i + j) % 3 == 0
                color = (100, 100, 100) if is_occupied else (150, 150, 150)
                cv2.rectangle(frame, (x, y), (x + 50, y + 100), color, -1)
                cv2.rectangle(frame, (x, y), (x + 50, y + 100), (255, 255, 255), 2)
        
        return frame

    def _log_to_csv(self, free_count, total_spots, timestamp):
        """Log current state to CSV"""
        if self.csv_writer:
            try:
                log_row = [self.frame_count, timestamp, free_count, total_spots] + [int(s) for s in self.spots_status]
                self.csv_writer.writerow(log_row)
                self.log_file.flush()  # Ensure data is written
            except Exception as e:
                print(f"Error logging to CSV: {str(e)}")

    def get_parking_statistics(self):
        """Get comprehensive parking statistics"""
        total_spots = len(self.spots_status)
        free_spots = sum(self.spots_status)
        occupied_spots = total_spots - free_spots
        occupancy_rate = (occupied_spots / total_spots) * 100 if total_spots > 0 else 0

        return {
            'total_spaces': total_spots,
            'free': free_spots,
            'occupied': occupied_spots,
            'occupancy_rate': occupancy_rate,
            'frame_count': self.frame_count
        }

    def __del__(self):
        """Cleanup resources"""
        if self.log_file:
            self.log_file.close()