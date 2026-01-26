# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the Flask port
EXPOSE 7860

# Set environment variable for Flask
ENV FLASK_APP=web/app.py
ENV PORT=7860

# Run the application
# We use host 0.0.0.0 and port 7860 as required by Hugging Face Spaces
CMD ["python", "web/app.py"]
