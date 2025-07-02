# Unified Vehicle Detection System

A real-time vehicle detection system that combines license plate recognition and parking space monitoring using computer vision and machine learning.

## Features

- **License Plate Detection**: OCR-based license plate recognition with confidence scoring
- **Parking Space Monitoring**: ML-based parking occupancy detection with 396 parking spots
- **Real-time Analytics**: Live statistics and detection tracking
- **Video Processing**: Batch processing of uploaded video files
- **Database Integration**: Supabase PostgreSQL backend for data persistence
- **Web Interface**: Interactive Streamlit dashboard

## Technology Stack

- **Frontend**: Streamlit
- **Computer Vision**: OpenCV, Tesseract OCR
- **Machine Learning**: scikit-learn, scikit-image
- **Database**: Supabase (PostgreSQL)
- **Language**: Python 3.11+

## Setup

### Prerequisites

1. Python 3.11 or higher
2. Supabase account and project
3. Required system packages (automatically handled in Replit)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd unified-vehicle-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Database Configuration

1. Create a Supabase project at [supabase.com](https://supabase.com)
2. Get your database URL from the project settings
3. Set the `DATABASE_URL` environment variable:
```bash
export DATABASE_URL="postgresql://user:password@host:port/database"
```

### Running the Application

```bash
streamlit run app_fixed.py --server.port 5000
```

## Usage

### Live Detection
- Run demo license plate detection to see detection capabilities
- Execute parking space detection using the trained ML model
- View real-time statistics and confidence scores

### Video Processing
- Upload MP4, AVI, MOV, or MKV video files
- Process videos for license plate detection
- Results are automatically saved to the database

### Analytics
- View detection statistics and metrics
- Download comprehensive Excel reports
- Monitor system performance

## Database Schema

### License Plates Table
```sql
CREATE TABLE license_plates (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    plate_number VARCHAR(20) NOT NULL,
    confidence DECIMAL(3,2),
    source VARCHAR(20) DEFAULT 'video',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Parking Events Table
```sql
CREATE TABLE parking_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_spots INTEGER,
    free_spots INTEGER,
    occupied_spots INTEGER,
    occupancy_rate DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Project Structure

```
unified-vehicle-detection/
├── app_fixed.py                    # Main Streamlit application
├── license_plate_detector.py       # License plate detection module
├── advanced_parking_detector.py    # ML-based parking detection
├── utils.py                        # Utility functions
├── attached_assets/                # Original project files and models
│   ├── model_1750850768456.p      # Trained ML model
│   ├── mask_1920_1080_1750850768455.png
│   └── parking coordinates_1750850768457.txt
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                     # Git ignore rules
```

## Configuration

The application uses environment variables for configuration:

- `DATABASE_URL`: Supabase PostgreSQL connection string
- `STREAMLIT_SERVER_PORT`: Server port (default: 5000)

## Features in Detail

### License Plate Detection
- Contour-based detection algorithm
- Tesseract OCR for text extraction
- Confidence scoring and validation
- Multiple detection attempts for accuracy

### Parking Space Detection
- Pre-trained SVM model for classification
- 396 parking spots from mask analysis
- Real-time occupancy calculation
- Historical tracking and analytics

### Database Integration
- Automatic table creation and migration
- Real-time data persistence
- Excel report generation from database
- Data clearing and management tools

## Deployment

This application is designed for deployment on Replit with autoscale capabilities. The system automatically handles:

- Environment setup and dependencies
- Database connections and migrations
- Port configuration and networking
- Static file serving

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please create an issue in the GitHub repository.