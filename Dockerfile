# Use Python 3.11 Slim as the base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements or pyproject.toml first to leverage Docker cache
COPY pyproject.toml .
COPY requirements.txt* ./

# Install Python dependencies
# Note: Using pip install . assumes current directory structure
RUN pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Create the .refinery directory for artifacts
RUN mkdir -p .refinery/profiles .refinery/pageindex .refinery/vector_store

# Command to run the refinery (e.g., triage a document)
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
