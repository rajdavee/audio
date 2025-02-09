# Stage 1: Base image with Python and FFmpeg
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Install dependencies
FROM base as dependencies

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final stage
FROM dependencies as final

# Set work directory
WORKDIR /app

# Create cache directory with proper permissions
RUN mkdir -p /app/.cache && \
    chmod 777 /app/.cache

# Copy application code and assets
COPY ./streamlit_app.py .
COPY ./assets ./assets

# Create necessary directories with proper permissions
RUN mkdir -p /app/temp_audio && \
    chmod 777 /app/temp_audio

# Add non-root user and set permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Set cache directory for Whisper models
ENV XDG_CACHE_HOME=/app/.cache \
    PYTHONPATH=/app

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
