FROM python:3.11-slim

# System deps: ffmpeg (video encoding), libreoffice (PPTX -> PDF conversion)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure runtime directories exist
RUN mkdir -p data/generated_code data/slides data/assets logs

EXPOSE 8000

CMD ["uvicorn", "demo_front_end:app", "--host", "0.0.0.0", "--port", "8000"]
