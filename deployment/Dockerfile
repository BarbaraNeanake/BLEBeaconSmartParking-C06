FROM python:3.10-slim

# Install basic build tools and OpenGL/graphics libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create models cache directory with proper permissions
RUN mkdir -p /app/models_cache && chmod 777 /app/models_cache

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]