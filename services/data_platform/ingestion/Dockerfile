FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY services/data_platform/ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schemas and utils
COPY shared /app/shared

# Copy application code
COPY services/data_platform/ingestion /app/ingestion

# Set Python path to include app directory
ENV PYTHONPATH=/app

# Default command
CMD ["python", "/app/ingestion/main.py"]
