FROM python:3.13-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY . .

# Remove any local .venv that might have been copied
RUN rm -rf .venv

# Install dependencies
RUN uv pip install --system -e ".[postgres,langgraph,claude]"

EXPOSE 8000

CMD ["/bin/sh", "-c", "alembic upgrade head && uvicorn src.memu.main:app --host 0.0.0.0 --port 8000"]