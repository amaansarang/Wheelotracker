import streamlit as st
import streamlit.components.v1 as components
import base64
import numpy as np
import cv2
from PIL import Image
import io

def camera_input_live():
    """Create a live camera input component using browser's camera API"""
    
    camera_html = """
    <div id="camera-container">
        <video id="video" width="640" height="480" autoplay style="border: 2px solid #ddd; border-radius: 8px;"></video>
        <br><br>
        <button id="start-camera" onclick="startCamera()" style="padding: 10px 20px; margin: 5px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">Start Camera</button>
        <button id="stop-camera" onclick="stopCamera()" style="padding: 10px 20px; margin: 5px; background: #f44336; color: white; border: none; border-radius: 5px; cursor: pointer;">Stop Camera</button>
        <button id="capture" onclick="captureFrame()" style="padding: 10px 20px; margin: 5px; background: #008CBA; color: white; border: none; border-radius: 5px; cursor: pointer;">Capture Frame</button>
        <canvas id="canvas" width="640" height="480" style="display: none;"></canvas>
        <br><br>
        <div id="status" style="margin: 10px 0; font-weight: bold;"></div>
    </div>

    <script>
    let video = document.getElementById('video');
    let canvas = document.getElementById('canvas');
    let context = canvas.getContext('2d');
    let stream = null;
    let isCapturing = false;

    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640, 
                    height: 480,
                    facingMode: 'environment' // Try to use back camera
                } 
            });
            video.srcObject = stream;
            document.getElementById('status').innerHTML = '📹 Camera is ON';
            document.getElementById('status').style.color = 'green';
            
            // Auto-capture frames for live processing
            if (!isCapturing) {
                isCapturing = true;
                captureLoop();
            }
        } catch (err) {
            console.error('Error accessing camera:', err);
            document.getElementById('status').innerHTML = '❌ Camera access denied or not available';
            document.getElementById('status').style.color = 'red';
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            stream = null;
            isCapturing = false;
            document.getElementById('status').innerHTML = '📹 Camera is OFF';
            document.getElementById('status').style.color = 'gray';
        }
    }

    function captureFrame() {
        if (video.srcObject) {
            context.drawImage(video, 0, 0, 640, 480);
            let imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send frame to Streamlit
            window.parent.postMessage({
                type: 'camera_frame',
                data: imageData
            }, '*');
        }
    }

    function captureLoop() {
        if (isCapturing && stream) {
            captureFrame();
            setTimeout(captureLoop, 500); // Capture every 500ms
        }
    }

    // Handle camera permissions
    navigator.permissions.query({name: 'camera'}).then(function(result) {
        if (result.state === 'granted') {
            document.getElementById('status').innerHTML = '📹 Camera permission granted';
        } else if (result.state === 'prompt') {
            document.getElementById('status').innerHTML = '📹 Click "Start Camera" to enable camera access';
        } else {
            document.getElementById('status').innerHTML = '❌ Camera permission denied';
        }
    });
    </script>
    """
    
    # Create the component
    component_value = components.html(camera_html, height=600)
    return component_value

def decode_camera_frame(image_data):
    """Decode base64 image data to OpenCV format"""
    try:
        # Remove data URL prefix
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return cv_image
    except Exception as e:
        st.error(f"Error decoding camera frame: {e}")
        return None

def camera_input_upload():
    """Fallback camera input using file upload"""
    st.markdown("### 📷 Camera Upload (Fallback)")
    st.info("If live camera doesn't work, use this to upload images from your camera")
    
    uploaded_file = st.file_uploader(
        "Take a photo or upload image",
        type=['jpg', 'jpeg', 'png'],
        help="Use your device's camera app to take a photo, then upload it here"
    )
    
    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        cv_image = cv2.imdecode(file_bytes, 1)
        return cv_image
    
    return None

def mobile_camera_component():
    """Mobile-optimized camera component"""
    
    mobile_html = """
    <div style="text-align: center;">
        <div id="mobile-camera" style="max-width: 100%; margin: 0 auto;">
            <video id="mobile-video" autoplay playsinline style="width: 100%; max-width: 400px; border: 2px solid #ddd; border-radius: 8px;"></video>
            <br><br>
            <div style="margin: 10px;">
                <button onclick="startMobileCamera()" style="padding: 12px 24px; margin: 5px; background: #4CAF50; color: white; border: none; border-radius: 5px; font-size: 16px;">📹 Start</button>
                <button onclick="stopMobileCamera()" style="padding: 12px 24px; margin: 5px; background: #f44336; color: white; border: none; border-radius: 5px; font-size: 16px;">⏹️ Stop</button>
            </div>
            <canvas id="mobile-canvas" style="display: none;"></canvas>
            <div id="mobile-status" style="margin: 15px 0; font-weight: bold; font-size: 16px;"></div>
        </div>
    </div>

    <script>
    let mobileVideo = document.getElementById('mobile-video');
    let mobileCanvas = document.getElementById('mobile-canvas');
    let mobileContext = mobileCanvas.getContext('2d');
    let mobileStream = null;
    let mobileCapturing = false;

    async function startMobileCamera() {
        try {
            // Try different camera configurations for mobile
            const constraints = {
                video: {
                    facingMode: { ideal: 'environment' }, // Prefer back camera
                    width: { ideal: 1280, max: 1920 },
                    height: { ideal: 720, max: 1080 }
                }
            };

            mobileStream = await navigator.mediaDevices.getUserMedia(constraints);
            mobileVideo.srcObject = mobileStream;
            
            mobileVideo.onloadedmetadata = function() {
                mobileCanvas.width = mobileVideo.videoWidth;
                mobileCanvas.height = mobileVideo.videoHeight;
            };
            
            document.getElementById('mobile-status').innerHTML = '📹 Mobile camera is ON';
            document.getElementById('mobile-status').style.color = 'green';
            
            if (!mobileCapturing) {
                mobileCapturing = true;
                mobileCaptureLoop();
            }
        } catch (err) {
            console.error('Mobile camera error:', err);
            document.getElementById('mobile-status').innerHTML = '❌ Camera not available on this device';
            document.getElementById('mobile-status').style.color = 'red';
            
            // Fallback message for mobile
            document.getElementById('mobile-status').innerHTML += '<br>Try using the file upload option below';
        }
    }

    function stopMobileCamera() {
        if (mobileStream) {
            mobileStream.getTracks().forEach(track => track.stop());
            mobileVideo.srcObject = null;
            mobileStream = null;
            mobileCapturing = false;
            document.getElementById('mobile-status').innerHTML = '📹 Mobile camera is OFF';
            document.getElementById('mobile-status').style.color = 'gray';
        }
    }

    function mobileCaptureLoop() {
        if (mobileCapturing && mobileStream) {
            if (mobileVideo.readyState === mobileVideo.HAVE_ENOUGH_DATA) {
                mobileContext.drawImage(mobileVideo, 0, 0);
                let imageData = mobileCanvas.toDataURL('image/jpeg', 0.8);
                
                window.parent.postMessage({
                    type: 'mobile_camera_frame',
                    data: imageData
                }, '*');
            }
            setTimeout(mobileCaptureLoop, 1000); // Capture every 1 second for mobile
        }
    }

    // Check if mobile device
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    if (isMobile) {
        document.getElementById('mobile-status').innerHTML = '📱 Mobile device detected - Optimized camera controls';
    } else {
        document.getElementById('mobile-status').innerHTML = '💻 Desktop device - Full camera functionality available';
    }
    </script>
    """
    
    return components.html(mobile_html, height=500)