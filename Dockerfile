FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    tensorflow==2.16.1 \
    scikit-learn \
    protobuf==3.20.3

# Set environment variables
ENV PYTHONPATH=/app
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Make run.sh executable
RUN chmod +x run.sh

# Default command
CMD ["python3", "AMEVulDetector.py"]