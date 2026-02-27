FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for wfdb, pyedflib, etc.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libhdf5-dev \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p datasets logs ml_models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
