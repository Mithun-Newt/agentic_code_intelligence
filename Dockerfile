# Minimal reproducible Docker environment for Samsung PRISM Hackathon Theme 01
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code, demo data, and configuration
COPY src/ ./src/
COPY data/ ./data/
COPY configs/ ./configs/
COPY eval/ ./eval/
COPY tests/ ./tests/
COPY demo.py pytest.ini appsretrieval_results.json README.md ./

# Default entrypoint runs the interactive code retrieval CLI demo
ENTRYPOINT ["python", "demo.py"]
CMD ["--query", "binary search on sorted array", "--top-k", "5"]
