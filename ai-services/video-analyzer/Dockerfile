FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgthread-2.0-0 \
        libmagic1 \
        cmake \
        libopenblas-dev \
        liblapack-dev \
        libasound2-dev \
        portaudio19-dev \
        libportaudio2 \
        libportaudiocpp0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download required models
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN python -c "import whisper; whisper.load_model('base')"

# Copy source code
COPY src/ .

# Create directories
RUN mkdir -p /app/models /app/logs /app/temp

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=60s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Default command
CMD ["python", "main.py"]
