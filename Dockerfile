FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ backend/
COPY crawlers/ crawlers/
COPY data/ data/
COPY alphazero_model.pth alphazero_model.pth

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.simple_server:app", "--host", "0.0.0.0", "--port", "8000"]
