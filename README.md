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
