import os
import pandas as pd
import cv2
import numpy as np
from datetime import datetime
import streamlit as st

def ensure_directory_exists(directory):
    """Ensure a directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return True
    return False

def validate_license_plate(plate_text):
    """Validate if the detected text looks like a license plate"""
    if not plate_text or len(plate_text) < 3:
        return False
    
    # Remove spaces and special characters
    clean_plate = ''.join(c for c in plate_text if c.isalnum())
    
    # Check length
    if len(clean_plate) < 3 or len(clean_plate) > 10:
        return False
    
    # Check if it has both letters and numbers (common pattern)
    has_letter = any(c.isalpha() for c in clean_plate)
    has_number = any(c.isdigit() for c in clean_plate)
    
    return has_letter or has_number

def format_confidence_score(confidence):
    """Format confidence score for display"""
    return f"{confidence * 100:.1f}%"

def create_excel_report(detected_plates, filename=None):
    """Create an Excel report from detected plates"""
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'license_plates_report_{timestamp}.xlsx'
    
    try:
        df = pd.DataFrame(detected_plates)
        
        if not df.empty:
            # Add additional columns
            df['Date'] = pd.to_datetime(df['timestamp']).dt.date
            df['Time'] = pd.to_datetime(df['timestamp']).dt.time
            df['Confidence (%)'] = df['confidence'].apply(lambda x: f"{x*100:.1f}%")
            
            # Reorder columns
            column_order = ['Date', 'Time', 'plate_number', 'Confidence (%)', 'timestamp']
            df = df.reindex(columns=[col for col in column_order if col in df.columns])
            
        df.to_excel(filename, index=False)
        return filename
        
    except Exception as e:
        st.error(f"Error creating Excel report: {str(e)}")
        return None

def resize_image_for_display(image, max_width=800, max_height=600):
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    return image

def calculate_processing_fps(start_time, frame_count):
    """Calculate processing FPS"""
    elapsed_time = datetime.now() - start_time
    if elapsed_time.total_seconds() > 0:
        fps = frame_count / elapsed_time.total_seconds()
        return fps
    return 0.0

def format_parking_statistics(stats):
    """Format parking statistics for display"""
    if not stats:
        return "No parking data available"
    
    total = stats.get('total_spaces', 0)
    occupied = stats.get('occupied', 0)
    free = stats.get('free', 0)
    occupancy_rate = stats.get('occupancy_rate', 0)
    
    return {
        'summary': f"{free}/{total} spaces available",
        'occupancy': f"{occupancy_rate:.1f}% occupied",
        'details': f"Free: {free}, Occupied: {occupied}, Total: {total}"
    }

def save_detection_log(detection_data, log_file='detection_log.txt'):
    """Save detection events to a log file"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(log_file, 'a', encoding='utf-8') as f:
            if 'plate_number' in detection_data:
                f.write(f"[{timestamp}] LICENSE PLATE: {detection_data['plate_number']} "
                       f"(Confidence: {detection_data.get('confidence', 0):.2f})\n")
            
            if 'parking_status' in detection_data:
                status_summary = format_parking_statistics(detection_data['parking_status'])
                f.write(f"[{timestamp}] PARKING: {status_summary['summary']}\n")
                
    except Exception as e:
        print(f"Error saving to log: {str(e)}")

def cleanup_old_files(directory, max_age_days=7):
    """Clean up old files in a directory"""
    try:
        if not os.path.exists(directory):
            return
        
        current_time = datetime.now()
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                age_days = (current_time - file_time).days
                
                if age_days > max_age_days:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename}")
                    
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def validate_camera_access():
    """Check if camera is accessible"""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        return False
    except Exception:
        return False

def get_system_info():
    """Get system information for debugging"""
    try:
        info = {
            'opencv_version': cv2.__version__,
            'python_version': __import__('sys').version,
            'camera_available': validate_camera_access(),
            'timestamp': datetime.now().isoformat()
        }
        return info
    except Exception as e:
        return {'error': str(e)}

class PerformanceMonitor:
    """Monitor application performance"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.frame_count = 0
        self.detection_count = 0
        
    def update_frame_count(self):
        """Update frame processing count"""
        self.frame_count += 1
        
    def update_detection_count(self):
        """Update detection count"""
        self.detection_count += 1
        
    def get_fps(self):
        """Get current FPS"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.frame_count / elapsed if elapsed > 0 else 0
        
    def get_detection_rate(self):
        """Get detection rate per minute"""
        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        return self.detection_count / elapsed_minutes if elapsed_minutes > 0 else 0
        
    def get_stats(self):
        """Get performance statistics"""
        return {
            'fps': self.get_fps(),
            'detection_rate': self.get_detection_rate(),
            'total_frames': self.frame_count,
            'total_detections': self.detection_count,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }
