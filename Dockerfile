FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir .

# Application code
COPY prompts/ prompts/
COPY policy/ policy/
COPY src/ src/

# Data directory (persisted via volume)
RUN mkdir -p data

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Default: run the API server
CMD ["python", "-m", "collections_agent.api"]
