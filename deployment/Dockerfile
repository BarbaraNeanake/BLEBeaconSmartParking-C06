FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV (updated for Debian Trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories with proper permissions
RUN mkdir -p /app/models_cache /app/hf_cache && chmod 777 /app/models_cache /app/hf_cache

# Set Hugging Face cache directory
ENV HF_HOME=/app/hf_cache

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]