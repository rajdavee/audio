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

```

```
