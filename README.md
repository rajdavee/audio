# Audio Transcription App

A Streamlit application for transcribing audio files with speaker diarization using Whisper.

## Setup Instructions

1. Create a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install FFmpeg:

- Windows: Download from https://ffmpeg.org/download.html and add to PATH
- Linux: `sudo apt-get install ffmpeg`
- Mac: `brew install ffmpeg`

4. Run the application:

```bash
streamlit run streamlit_app.py
```

## Usage

1. Select Whisper model size (larger models are more accurate but slower)
2. Upload an audio file
3. Click "Start Transcription"
4. Download the results

## Model Sizes

- tiny: Fastest, least accurate
- base: Good balance for most uses
- small: Better accuracy, reasonable speed
- medium: High accuracy, slower
- large: Best accuracy, slowest

## Features

- Local transcription using Whisper
- Multiple audio format support
- Speaker diarization
- Downloadable transcript and JSON report

## Docker Instructions

### Build and Run with Docker

1. Build the Docker image:

```bash
docker build -t audio-transcription-app .
```

2. Run the Docker container:

```bash
docker run -d \
  --name audio-transcription-app \
  -p 8501:8501 \
  -v $(pwd)/assets:/app/assets:ro \
  -v cache-data:/app/.cache \
  audio-transcription-app
```

### Using Docker Compose

1. Build and start the application with Docker Compose:

```bash
docker-compose up --build
```

2. Access the application in your browser:

```bash
http://localhost:8501
```

### Docker Volumes

- `cache-data`: Used to persist Whisper model cache between container restarts.

### Health Check

The Docker container includes a health check to ensure the application is running correctly. It checks the health endpoint at `/app/.cache` every 30 seconds.

```bash
docker inspect audio-transcription-app | grep -A 10 "Health"
```

# Audio Transcription App Deployment Guide

## Deploying to Infomaniak

### Prerequisites
- An Infomaniak account with Container App service
- Docker installed on your local machine
- Git (optional)

### Deployment Steps

1. **Login to Infomaniak Registry**
```bash
docker login registry.infomaniak.com
```

2. **Build the Docker Image**
```bash
docker build -t registry.infomaniak.com/[your-namespace]/audio-transcription:latest .
```

3. **Push the Image**
```bash
docker push registry.infomaniak.com/[your-namespace]/audio-transcription:latest
```

4. **Deploy on Infomaniak**
- Go to https://manager.infomaniak.com/
- Navigate to "Container Apps"
- Click "Create a new app"
- Select "Custom Docker image"
- Enter your image URL: `registry.infomaniak.com/[your-namespace]/audio-transcription:latest`
- Configure the following settings:
  - Port: 8501
  - Environment Variables:
    - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
    - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    - XDG_CACHE_HOME=/app/.cache
    - PYTHONPATH=/app

### Environment Configuration
The app requires the following environment variables:
- `STREAMLIT_SERVER_MAX_UPLOAD_SIZE`: Maximum upload size in MB
- `STREAMLIT_SERVER_ADDRESS`: Server address (0.0.0.0)
- `XDG_CACHE_HOME`: Cache directory for Whisper models
- `PYTHONPATH`: Python path

### Health Check
The application includes a health check endpoint at:
`http://[your-app-url]/_stcore/health`

### Troubleshooting
1. If the app fails to start, check the logs in Infomaniak dashboard
2. Ensure all environment variables are properly set
3. Verify that the port 8501 is correctly exposed
4. Check if the image was pushed successfully to the registry

### Support
For any issues:
- Check Infomaniak's [container documentation](https://www.infomaniak.com/en/support/faq/2438/container-app-prerequisites-and-limitations)
- Review application logs in the Infomaniak dashboard
