FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports
EXPOSE 8000 8501

# Copy supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

