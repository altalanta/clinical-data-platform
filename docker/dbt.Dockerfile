FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
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

# Install dependencies including dbt
RUN poetry install --only=main --extras=dbt && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r dbtuser && useradd -r -g dbtuser dbtuser

# Copy virtual environment from builder
COPY --from=builder --chown=dbtuser:dbtuser /app/.venv /app/.venv

# Set up application
WORKDIR /app
COPY --chown=dbtuser:dbtuser . .

# Make sure virtual environment is in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set up dbt profile directory
RUN mkdir -p /home/dbtuser/.dbt && \
    chown -R dbtuser:dbtuser /home/dbtuser && \
    chown -R dbtuser:dbtuser /app

# Copy dbt profiles
COPY --chown=dbtuser:dbtuser dbt_project/profiles.yml.example /app/profiles.yml

# Switch to non-root user
USER dbtuser

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DBT_PROFILES_DIR=/app

# Install dbt dependencies on startup
RUN cd dbt_project && dbt deps

# Default command
CMD ["dbt", "run", "--project-dir", "dbt_project", "--profiles-dir", "/app"]