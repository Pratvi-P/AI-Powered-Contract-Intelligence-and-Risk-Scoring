# Base image: Python 3.11 slim (smaller than full Python image)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by PyMuPDF and spaCy
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (so Docker caches this layer)
# If requirements don't change, Docker skips reinstalling on rebuild
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the project files
COPY src/ ./src/
COPY evaluation/ ./evaluation/
COPY vector_db/ ./vector_db/

# Create an empty data directory (real data stays local/on Drive)
RUN mkdir -p data

# Expose port 8000
EXPOSE 8000

# Start the API when the container runs
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
