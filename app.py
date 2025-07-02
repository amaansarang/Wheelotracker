import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from license_plate_detector import LicensePlateDetector
from advanced_parking_detector import AdvancedParkingDetector
import tempfile
import threading
import time

# Page configuration
st.set_page_config(
    page_title="Unified Vehicle Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'license_detector' not in st.session_state:
    st.session_state.license_detector = LicensePlateDetector()
if 'parking_detector' not in st.session_state:
    st.session_state.parking_detector = AdvancedParkingDetector()
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'parking_running' not in st.session_state:
    st.session_state.parking_running = False
if 'detected_plates' not in st.session_state:
    st.session_state.detected_plates = []
if 'parking_status' not in st.session_state:
    st.session_state.parking_status = {}
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {'frames_processed': 0, 'plates_detected': 0}

def main():
    st.title("🚗 Unified Vehicle Detection System")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("System Controls")
        
        # Real-time detection controls
        st.subheader("Live Detection")
        webcam_start = st.button("Start Webcam Detection", key="start_webcam")
        webcam_stop = st.button("Stop Webcam Detection", key="stop_webcam")
        parking_start = st.button("Start Parking Detection", key="start_parking")
        parking_stop = st.button("Stop Parking Detection", key="stop_parking")
        
        # File upload for video processing
        st.subheader("Video Processing")
        uploaded_video = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])
        if uploaded_video:
            if st.button("Process Video"):
                process_uploaded_video(uploaded_video)
        
        # Settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("License Plate Confidence", 0.3, 0.9, 0.6, 0.1)
        st.session_state.confidence_threshold = confidence_threshold
        
        # File management
        st.subheader("Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download Report"):
                download_excel_report()
        with col2:
            if st.button("Clear Data"):
                clear_all_data()
    
    # Handle button clicks
    if webcam_start:
        st.session_state.webcam_running = True
    if webcam_stop:
        st.session_state.webcam_running = False
    if parking_start:
        st.session_state.parking_running = True
    if parking_stop:
        st.session_state.parking_running = False
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 License Plate Recognition")
        webcam_placeholder = st.empty()
        plate_info_placeholder = st.empty()
        
    with col2:
        st.subheader("🅿️ Parking Space Monitor")
        parking_placeholder = st.empty()
        parking_stats_placeholder = st.empty()
    
    # Status indicators
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if st.session_state.webcam_running:
            st.success("🟢 Webcam Detection: ACTIVE")
        else:
            st.error("🔴 Webcam Detection: INACTIVE")
    
    with status_col2:
        if st.session_state.parking_running:
            st.success("🟢 Parking Detection: ACTIVE")
        else:
            st.error("🔴 Parking Detection: INACTIVE")
    
    # Main processing loop
    while st.session_state.webcam_running or st.session_state.parking_running:
        try:
            # Process webcam for license plates
            if st.session_state.webcam_running:
                process_webcam_feed(webcam_placeholder, plate_info_placeholder)
            
            # Process parking detection
            if st.session_state.parking_running:
                process_parking_detection(parking_placeholder, parking_stats_placeholder)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            break
    
    # Analytics Dashboard
    st.markdown("---")
    display_analytics_dashboard()
    
    # Display recent detections
    display_recent_detections()

def process_demo_detection(placeholder, info_placeholder):
    """Process demo license plate detection since webcam is not available"""
    try:
        info_placeholder.info("Webcam not available in this environment. Running demo detection...")
        
        # Create demo frame with license plate patterns
        demo_frame = create_demo_license_plate_frame()
        
        if demo_frame is not None:
            # Process demo frame
            processed_frame, plates = st.session_state.license_detector.detect_plates(demo_frame)
            
            # Display processed frame
            placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            
            # Store demo detections
            if plates:
                for plate_data in plates:
                    confidence = plate_data.get('confidence', 0)
                    if confidence >= 0.3:  # Lower threshold for demo
                        plate_info = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'plate_number': plate_data.get('text', ''),
                            'confidence': confidence,
                            'source': 'demo'
                        }
                        st.session_state.detected_plates.append(plate_info)
                        update_excel_file(plate_info)
            
            # Show demo results
            reliable_plates = st.session_state.license_detector.get_reliable_plates()
            info_placeholder.success(
                f"Demo complete: Detected {len(plates)} plate(s) | "
                f"Reliable: {len(reliable_plates)} plates"
            )
        else:
            placeholder.error("Could not create demo frame")
            
    except Exception as e:
        placeholder.error(f"Demo detection error: {str(e)}")

def create_demo_license_plate_frame():
    """Create a demo frame with sample license plates"""
    try:
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.fill(40)  # Dark gray background
        
        # Add realistic license plate rectangles
        plates = [
            {'pos': (100, 150), 'size': (200, 60), 'text': 'ABC123'},
            {'pos': (350, 250), 'size': (180, 50), 'text': 'XYZ789'},
            {'pos': (150, 350), 'size': (190, 55), 'text': 'DEF456'}
        ]
        
        for plate in plates:
            x, y = plate['pos']
            w, h = plate['size']
            text = plate['text']
            
            # White rectangle for plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
            
            # Add text
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        return frame
        
    except Exception as e:
        st.error(f"Error creating demo frame: {e}")
        return None

def process_webcam_feed(placeholder, info_placeholder):
    """Process webcam feed for license plate detection"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            placeholder.error("❌ Cannot access webcam")
            return
        
        ret, frame = cap.read()
        if ret:
            # Process frame for license plates
            processed_frame, plates = st.session_state.license_detector.detect_plates(frame)
            
            # Display processed frame
            placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            
            # Handle detected plates
            if plates:
                st.session_state.processing_stats['frames_processed'] += 1
                
                for plate_data in plates:
                    plate_text = plate_data.get('text', '')
                    confidence = plate_data.get('confidence', 0)
                    
                    confidence_thresh = st.session_state.get('confidence_threshold', 0.6)
                    if confidence > confidence_thresh:
                        # Update plate tracking in detector
                        st.session_state.license_detector.update_plate_tracking(plate_text, confidence)
                        
                        # Save plate data
                        plate_info = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'plate_number': plate_text,
                            'confidence': confidence
                        }
                        st.session_state.detected_plates.append(plate_info)
                        st.session_state.processing_stats['plates_detected'] += 1
                        
                        # Save cropped image
                        if 'cropped_image' in plate_data:
                            save_cropped_plate(plate_data['cropped_image'], plate_text)
                        
                        # Update Excel
                        update_excel_file(plate_info)
                
                # Display current detection info with enhanced statistics
                reliable_plates = st.session_state.license_detector.get_reliable_plates()
                info_placeholder.info(
                    f"Detected {len(plates)} plate(s) | "
                    f"Reliable: {len(reliable_plates)} | "
                    f"Total processed: {st.session_state.processing_stats['plates_detected']}"
                )
            else:
                info_placeholder.info("Scanning for license plates...")
        
        cap.release()
        
    except Exception as e:
        placeholder.error(f"Webcam error: {str(e)}")

def process_demo_parking_detection(placeholder, stats_placeholder):
    """Process demo parking space detection"""
    try:
        stats_placeholder.info("Running demo parking detection with your trained ML model...")
        
        # Get parking status using advanced ML-based detector
        parking_frame, parking_data = st.session_state.parking_detector.detect_parking_spaces()
        
        if parking_frame is not None:
            # Display parking visualization
            placeholder.image(parking_frame, channels="BGR", use_container_width=True)
            
            # Update parking status
            st.session_state.parking_status = parking_data
            
            # Get comprehensive statistics
            stats = st.session_state.parking_detector.get_parking_statistics()
            
            # Display statistics
            if stats:
                stats_text = f"""
**Demo Parking Detection Results:**
- Total spaces: {stats.get('total_spots', 0)}
- Free spaces: {stats.get('free_spots', 0)}
- Occupied spaces: {stats.get('occupied_spots', 0)}
- Occupancy rate: {stats.get('occupancy_rate', 0):.1f}%
- Using your trained ML model from attached_assets/model_1750850768456.p
                """
                stats_placeholder.success(stats_text)
        else:
            placeholder.warning("Demo parking detection not available")
            
    except Exception as e:
        placeholder.error(f"Demo parking detection error: {str(e)}")
        stats_placeholder.error("Check that your ML model file is accessible")

def save_cropped_plate(cropped_image, plate_text):
    """Save cropped license plate image"""
    try:
        if not os.path.exists('license_plates'):
            os.makedirs('license_plates')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"license_plates/plate_{plate_text}_{timestamp}.jpg"
        cv2.imwrite(filename, cropped_image)
        
    except Exception as e:
        st.error(f"Error saving plate image: {str(e)}")

def update_excel_file(plate_info):
    """Update Excel file with new plate detection"""
    try:
        excel_file = 'detected_plates.xlsx'
        
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
        else:
            df = pd.DataFrame(columns=['Timestamp', 'Plate Number', 'Confidence'])
        
        new_row = pd.DataFrame([{
            'Timestamp': plate_info['timestamp'],
            'Plate Number': plate_info['plate_number'],
            'Confidence': plate_info['confidence'],
            'Source': plate_info.get('source', 'video')
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_file, index=False)
        
    except Exception as e:
        st.error(f"Error updating Excel file: {str(e)}")

def download_excel_report():
    """Generate and provide download link for Excel report"""
    try:
        excel_file = 'detected_plates.xlsx'
        if os.path.exists(excel_file):
            with open(excel_file, 'rb') as file:
                st.download_button(
                    label="📥 Download Excel Report",
                    data=file.read(),
                    file_name=f"license_plates_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("No Excel report available yet.")
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

def clear_all_data():
    """Clear all stored data"""
    try:
        st.session_state.detected_plates = []
        st.session_state.parking_status = {}
        
        # Remove Excel file
        if os.path.exists('detected_plates.xlsx'):
            os.remove('detected_plates.xlsx')
        
        # Clear license plate images
        if os.path.exists('license_plates'):
            import shutil
            shutil.rmtree('license_plates')
        
        st.success("✅ All data cleared successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error clearing data: {str(e)}")

def display_recent_detections():
    """Display recent license plate detections"""
    st.markdown("---")
    st.subheader("📋 Recent Detections")
    
    if st.session_state.detected_plates:
        # Show last 10 detections
        recent_plates = st.session_state.detected_plates[-10:]
        df = pd.DataFrame(recent_plates)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No license plates detected yet.")
    
    # Parking status summary
    if st.session_state.parking_status:
        st.subheader("🅿️ Current Parking Status")
        
        # Create visual representation of parking spaces
        cols = st.columns(min(len(st.session_state.parking_status), 10))
        for i, (space_id, status) in enumerate(st.session_state.parking_status.items()):
            if i < len(cols):
                with cols[i]:
                    if status == 'free':
                        st.success(f"Space {space_id}\n🟢 FREE")
                    else:
                        st.error(f"Space {space_id}\n🔴 OCCUPIED")

def process_uploaded_video(uploaded_video):
    """Process uploaded video file for license plate detection"""
    if uploaded_video is None:
        return
        
    try:
        st.info(f"Processing video: {uploaded_video.name}")
        
        # Save uploaded video temporarily
        temp_video_path = f"temp_video_{int(time.time())}.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
        
        # Validate video file
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Could not open video file. Please ensure it's a valid format (MP4, AVI, MOV)")
            os.remove(temp_video_path)
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0 or fps == 0:
            st.error("Invalid video file or corrupted video")
            cap.release()
            os.remove(temp_video_path)
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_count = 0
        video_plates = []
        
        st.info(f"Video info: {total_frames} frames at {fps:.1f} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 15th frame for efficiency
            if frame_count % 15 == 0:
                try:
                    processed_frame, plates = st.session_state.license_detector.detect_plates(frame)
                    
                    for plate_data in plates:
                        confidence = plate_data.get('confidence', 0)
                        if confidence > 0.5:  # Reasonable threshold
                            video_plates.append({
                                'frame': frame_count,
                                'video_time': f"{frame_count / fps:.1f}s",
                                'plate_number': plate_data.get('text', ''),
                                'confidence': confidence,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                except Exception as frame_error:
                    st.warning(f"Error processing frame {frame_count}: {frame_error}")
            
            # Update progress every 50 frames
            if frame_count % 50 == 0:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing: {frame_count}/{total_frames} frames - Found {len(video_plates)} plates")
        
        cap.release()
        os.remove(temp_video_path)
        
        # Final progress
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Display results
        if video_plates:
            st.success(f"Successfully detected {len(video_plates)} license plates")
            df = pd.DataFrame(video_plates)
            st.dataframe(df, use_container_width=True)
            
            # Add to session state and Excel
            st.session_state.detected_plates.extend(video_plates)
            for plate in video_plates:
                update_excel_file(plate)
                
        else:
            st.warning("No license plates detected in this video")
            
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

def create_original_files_zip():
    """Create a ZIP file of original project files"""
    try:
        import zipfile
        
        zip_filename = f"original_project_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Add all files from attached_assets
            for root, dirs, files in os.walk('attached_assets'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, 'attached_assets')
                    zipf.write(file_path, arcname)
        
        # Provide download
        with open(zip_filename, 'rb') as f:
            st.download_button(
                label="Download Original Files ZIP",
                data=f.read(),
                file_name=zip_filename,
                mime="application/zip"
            )
        
        # Clean up
        os.remove(zip_filename)
        st.success("ZIP file created successfully!")
        
    except Exception as e:
        st.error(f"Error creating ZIP file: {str(e)}")

def display_analytics_dashboard():
    """Display comprehensive analytics dashboard"""
    st.subheader("📊 Analytics Dashboard")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_plates = len(st.session_state.detected_plates)
        st.metric("Total Plates Detected", total_plates)
    
    with col2:
        unique_plates = len(set(p['plate_number'] for p in st.session_state.detected_plates))
        st.metric("Unique Plates", unique_plates)
    
    with col3:
        frames_processed = st.session_state.processing_stats['frames_processed']
        st.metric("Frames Processed", frames_processed)
    
    with col4:
        if st.session_state.parking_status:
            parking_stats = st.session_state.parking_detector.get_parking_statistics()
            st.metric("Parking Occupancy", f"{parking_stats['occupancy_rate']:.1f}%")
        else:
            st.metric("Parking Occupancy", "N/A")
    
    # Recent activity chart
    if st.session_state.detected_plates:
        st.subheader("Recent Detection Activity")
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame(st.session_state.detected_plates)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.floor('H')
            
            # Group by hour and count detections
            hourly_counts = df.groupby('hour').size().reset_index(name='detections')
            
            if len(hourly_counts) > 1:
                st.line_chart(hourly_counts.set_index('hour')['detections'])
    
    # Top detected plates
    if st.session_state.detected_plates:
        st.subheader("Most Frequently Detected Plates")
        
        plate_counts = {}
        for plate in st.session_state.detected_plates:
            plate_num = plate['plate_number']
            if plate_num in plate_counts:
                plate_counts[plate_num] += 1
            else:
                plate_counts[plate_num] = 1
        
        # Sort by frequency
        sorted_plates = sorted(plate_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if sorted_plates:
            freq_df = pd.DataFrame(sorted_plates, columns=['Plate Number', 'Frequency'])
            st.dataframe(freq_df, use_container_width=True)

if __name__ == "__main__":
    main()
