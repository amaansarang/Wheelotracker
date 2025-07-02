# Unified Vehicle Detection System

## Overview

This is a unified vehicle detection system built with Python and Streamlit that combines license plate detection and parking space monitoring capabilities. The application uses computer vision techniques with OpenCV for real-time detection and OCR (Optical Character Recognition) with Tesseract for license plate text extraction.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

**Frontend**: Streamlit web application providing an interactive dashboard
**Backend**: Python-based computer vision processing modules
**Detection Engine**: Separate modules for license plate and parking space detection
**Data Processing**: Utility functions for validation, reporting, and file management

## Key Components

### Core Modules

1. **app.py** - Main Streamlit application
   - Serves as the central control hub
   - Manages UI components and user interactions
   - Coordinates between detection modules
   - Handles session state management

2. **license_plate_detector.py** - License plate detection engine
   - Uses Haar cascades or contour-based detection
   - Integrates Tesseract OCR for text extraction
   - Implements confidence scoring for detected plates
   - Handles preprocessing for better OCR accuracy

3. **parking_detector.py** - Parking space monitoring system
   - Implements background subtraction for motion detection
   - Manages parking space configuration via JSON
   - Provides real-time parking occupancy status
   - Supports configurable parking space coordinates

4. **utils.py** - Utility functions
   - File and directory management
   - License plate validation logic
   - Excel report generation
   - Data formatting helpers

### Configuration Files

- **.streamlit/config.toml** - Streamlit server configuration for headless deployment
- **pyproject.toml** - Python project dependencies and metadata
- **.replit** - Replit environment configuration with system packages

## Data Flow

1. **Input Sources**: Webcam feed or uploaded images/videos
2. **Processing Pipeline**: 
   - Frame capture and preprocessing
   - Detection algorithm application
   - OCR processing (for license plates)
   - Result validation and filtering
3. **Output**: 
   - Real-time detection visualization
   - Structured data storage in session state
   - Excel report generation capability

## External Dependencies

### Core Libraries
- **OpenCV**: Computer vision and image processing
- **Tesseract/pytesseract**: OCR text extraction
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Pillow**: Image processing utilities

### System Packages (via Nix)
- Graphics libraries: OpenGL, FreeType, LCMS2
- Image format support: JPEG, TIFF, WebP, PNG
- OCR support: Tesseract
- GUI toolkit: Tcl/Tk

## Deployment Strategy

**Platform**: Replit with autoscale deployment target
**Runtime**: Python 3.11 environment
**Port Configuration**: Application runs on port 5000
**Process Management**: Streamlit server with headless configuration
**Scalability**: Configured for autoscale deployment to handle variable load

The deployment uses parallel workflow execution, allowing the Streamlit application to run efficiently in the Replit environment with proper port forwarding and process management.

## Recent Changes

```
Recent Changes:
- July 2, 2025: Removed mobile optimization and desktop camera naming for CCTV demonstration
- July 2, 2025: Implemented real-time CCTV functionality from old codes for live camera processing
- July 2, 2025: Created professional CCTV interface with automatic frame processing
- July 2, 2025: Added real-time license plate detection with 2-second intervals
- July 2, 2025: Integrated real-time parking monitoring with 5-second ML analysis cycles
- July 2, 2025: Enhanced system to present desktop camera as professional CCTV system
```

## Changelog

```
Changelog:
- June 24, 2025: Initial setup and project merge
- June 25, 2025: Major fixes for deployment environment compatibility
- June 25, 2025: Supabase database integration and GitHub preparation
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```