FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY services/data_platform/simulator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules (schemas and utils)
COPY shared /app/shared

# Copy application code
COPY services/data_platform/simulator /app/simulator

# Set Python path to include shared modules
ENV PYTHONPATH=/app

# Default command (can be overridden in docker-compose)
CMD ["python", "/app/simulator/main.py", "--machines", "5", "--interval", "1", "--anomaly-rate", "0.05"]
