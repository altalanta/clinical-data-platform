FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.6.1
RUN pip install poetry==$POETRY_VERSION

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/opt/poetry-cache

# Copy dependency files
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install dependencies including ML/data processing libraries
RUN poetry install --only=main --extras=ml && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r worker && useradd -r -g worker worker

# Copy virtual environment from builder
COPY --from=builder --chown=worker:worker /app/.venv /app/.venv

# Set up application
WORKDIR /app
COPY --chown=worker:worker . .

# Make sure virtual environment is in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set up proper permissions
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R worker:worker /app

# Switch to non-root user
USER worker

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Default command for batch processing
CMD ["python", "-m", "clinical_data_platform.ml.train"]