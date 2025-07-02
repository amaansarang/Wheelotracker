import cv2
import numpy as np
import json
import os
from datetime import datetime

class ParkingDetector:
    def __init__(self, config_file='parking_config.json'):
        """Initialize the parking space detector"""
        self.config_file = config_file
        self.parking_spaces = []
        self.cap = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Load parking space configuration
        self.load_parking_config()
        
        # Try to initialize video capture
        self.initialize_camera()
    
    def load_parking_config(self):
        """Load parking space coordinates from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.parking_spaces = data.get('parking_spaces', [])
            else:
                # Create default parking spaces if config doesn't exist
                self.create_default_config()
        except Exception as e:
            print(f"Error loading parking config: {str(e)}")
            self.create_default_config()
    
    def create_default_config(self):
        """Create default parking space configuration"""
        # Default parking spaces (you would adjust these based on your specific parking lot)
        self.parking_spaces = [
            {'id': 1, 'coordinates': [(100, 200), (200, 200), (200, 300), (100, 300)]},
            {'id': 2, 'coordinates': [(220, 200), (320, 200), (320, 300), (220, 300)]},
            {'id': 3, 'coordinates': [(340, 200), (440, 200), (440, 300), (340, 300)]},
            {'id': 4, 'coordinates': [(460, 200), (560, 200), (560, 300), (460, 300)]},
            {'id': 5, 'coordinates': [(100, 320), (200, 320), (200, 420), (100, 420)]},
            {'id': 6, 'coordinates': [(220, 320), (320, 320), (320, 420), (220, 420)]},
            {'id': 7, 'coordinates': [(340, 320), (440, 320), (440, 420), (340, 420)]},
            {'id': 8, 'coordinates': [(460, 320), (560, 320), (560, 420), (460, 420)]},
        ]
        
        # Save default configuration
        self.save_parking_config()
    
    def save_parking_config(self):
        """Save parking space configuration to file"""
        try:
            config_data = {
                'parking_spaces': self.parking_spaces,
                'created': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving parking config: {str(e)}")
    
    def initialize_camera(self):
        """Initialize camera for parking detection"""
        try:
            # Try different camera indices
            for camera_index in range(3):
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, frame = cap.read()
                    if ret:
                        self.cap = cap
                        print(f"Parking camera initialized on index {camera_index}")
                        return True
                    cap.release()
            
            print("Warning: Could not initialize parking camera, using demo mode")
            return False
            
        except Exception as e:
            print(f"Error initializing parking camera: {str(e)}")
            return False
    
    def detect_parking_spaces(self):
        """
        Detect parking space occupancy
        Returns processed frame and parking status dictionary
        """
        try:
            # Get frame from camera or create demo frame
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    frame = self.create_demo_frame()
            else:
                frame = self.create_demo_frame()
            
            # Process frame for parking detection
            processed_frame, parking_status = self._process_parking_frame(frame)
            
            return processed_frame, parking_status
            
        except Exception as e:
            print(f"Error in parking detection: {str(e)}")
            demo_frame = self.create_demo_frame()
            return demo_frame, {}
    
    def create_demo_frame(self):
        """Create a demo frame for parking visualization"""
        # Create a blank frame
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Add some background elements
        cv2.rectangle(frame, (50, 50), (750, 550), (50, 50, 50), -1)  # Parking lot background
        
        # Add lane markings
        for i in range(100, 700, 60):
            cv2.line(frame, (i, 100), (i, 500), (255, 255, 255), 2)
        
        cv2.line(frame, (80, 200), (720, 200), (255, 255, 255), 2)
        cv2.line(frame, (80, 320), (720, 320), (255, 255, 255), 2)
        cv2.line(frame, (80, 440), (720, 440), (255, 255, 255), 2)
        
        return frame
    
    def _process_parking_frame(self, frame):
        """Process frame to detect parking space occupancy"""
        try:
            processed_frame = frame.copy()
            parking_status = {}
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Process each parking space
            for space in self.parking_spaces:
                space_id = space['id']
                coordinates = space['coordinates']
                
                # Create mask for this parking space
                mask = np.zeros(gray.shape, dtype=np.uint8)
                pts = np.array(coordinates, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
                
                # Calculate occupancy based on motion and edges
                occupancy_score = self._calculate_occupancy(gray, fg_mask, mask)
                
                # Determine if space is occupied
                is_occupied = occupancy_score > 0.3  # Threshold for occupancy
                
                # Draw parking space
                color = (0, 0, 255) if is_occupied else (0, 255, 0)  # Red if occupied, Green if free
                cv2.polylines(processed_frame, [pts], True, color, 3)
                
                # Add space number
                center_x = int(np.mean([p[0] for p in coordinates]))
                center_y = int(np.mean([p[1] for p in coordinates]))
                
                status_text = "OCCUPIED" if is_occupied else "FREE"
                cv2.putText(processed_frame, f"P{space_id}", 
                           (center_x - 15, center_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(processed_frame, status_text, 
                           (center_x - 25, center_y + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Store status
                parking_status[space_id] = 'occupied' if is_occupied else 'free'
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(processed_frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return processed_frame, parking_status
            
        except Exception as e:
            print(f"Error processing parking frame: {str(e)}")
            return frame, {}
    
    def _calculate_occupancy(self, gray_frame, fg_mask, space_mask):
        """Calculate occupancy score for a parking space"""
        try:
            # Apply space mask to the foreground mask
            space_fg = cv2.bitwise_and(fg_mask, space_mask)
            
            # Calculate the ratio of foreground pixels in the space
            total_pixels = np.sum(space_mask > 0)
            fg_pixels = np.sum(space_fg > 0)
            
            if total_pixels == 0:
                return 0.0
            
            motion_score = fg_pixels / total_pixels
            
            # Also check for edge density (cars have more edges)
            space_gray = cv2.bitwise_and(gray_frame, space_mask)
            edges = cv2.Canny(space_gray, 50, 150)
            space_edges = cv2.bitwise_and(edges, space_mask)
            edge_pixels = np.sum(space_edges > 0)
            edge_score = edge_pixels / total_pixels
            
            # Combine motion and edge scores
            occupancy_score = (motion_score * 0.3) + (edge_score * 0.7)
            
            return min(occupancy_score, 1.0)
            
        except Exception as e:
            print(f"Error calculating occupancy: {str(e)}")
            return 0.0
    
    def add_parking_space(self, coordinates):
        """Add a new parking space"""
        new_id = max([space['id'] for space in self.parking_spaces], default=0) + 1
        new_space = {
            'id': new_id,
            'coordinates': coordinates
        }
        self.parking_spaces.append(new_space)
        self.save_parking_config()
        return new_id
    
    def remove_parking_space(self, space_id):
        """Remove a parking space"""
        self.parking_spaces = [space for space in self.parking_spaces if space['id'] != space_id]
        self.save_parking_config()
    
    def get_parking_statistics(self, parking_status):
        """Get parking statistics"""
        if not parking_status:
            return {
                'total_spaces': len(self.parking_spaces),
                'occupied': 0,
                'free': len(self.parking_spaces),
                'occupancy_rate': 0.0
            }
        
        total_spaces = len(parking_status)
        occupied = sum(1 for status in parking_status.values() if status == 'occupied')
        free = total_spaces - occupied
        occupancy_rate = (occupied / total_spaces) * 100 if total_spaces > 0 else 0
        
        return {
            'total_spaces': total_spaces,
            'occupied': occupied,
            'free': free,
            'occupancy_rate': occupancy_rate
        }
    
    def __del__(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
