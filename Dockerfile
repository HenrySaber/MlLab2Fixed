FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Needed for pip dependencies installed directly from Git repositories
RUN apt-get update \
	&& apt-get install -y --no-install-recommends git \
	&& rm -rf /var/lib/apt/lists/*

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

ENV MODEL_CHECKPOINT=/app/checkpoints/house_segmentation.pth

# Flask app listens on 5000
EXPOSE 5000

# Run with Waitress (production server)
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]
