FROM python:3.13-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv using pip
RUN pip install uv

# Copy project files
COPY . .

# Remove any local .venv that might have been copied (safe even if it doesn't exist)
RUN rm -rf .venv

# Install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e ".[postgres,langgraph,claude]"

# Expose port (if needed for API)
EXPOSE 8000

# Keep container running without executing anything
CMD ["tail", "-f", "/dev/null"]