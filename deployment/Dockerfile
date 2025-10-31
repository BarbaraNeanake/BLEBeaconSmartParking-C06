# Start with a smaller base image
FROM python:3.9-slim

# Create a non-root user and set up the working directory
RUN useradd -m -u 1000 user
USER user
WORKDIR /app

# Set PATH environment variable
ENV PATH="/home/user/.local/bin:${PATH}"

# 1. Copy only the requirements file first
COPY --chown=user ./requirements.txt .

# 2. Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 3. Copy the rest of the application code
COPY --chown=user . .

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]