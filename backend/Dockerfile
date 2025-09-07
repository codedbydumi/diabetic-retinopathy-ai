# backend/Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
# Use root to create directories initially
RUN mkdir -p /app/ml-pipeline/models /app/logs && \
    chmod -R 755 /app/ml-pipeline && \
    chmod -R 755 /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Create a startup script to handle permissions after volume mounts
RUN echo '#!/bin/bash\n\
# Fix permissions after volume mount\n\
if [ -d /app/ml-pipeline ]; then\n\
    echo "Setting permissions for ml-pipeline directory..."\n\
    chmod -R 755 /app/ml-pipeline 2>/dev/null || true\n\
    chown -R app:app /app/ml-pipeline 2>/dev/null || true\n\
fi\n\
\n\
if [ -d /app/logs ]; then\n\
    chown -R app:app /app/logs 2>/dev/null || true\n\
fi\n\
\n\
# Switch to app user and start the application\n\
exec gosu app "$@"\n' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Install gosu for better user switching
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Change ownership of app directory
RUN chown -R app:app /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]