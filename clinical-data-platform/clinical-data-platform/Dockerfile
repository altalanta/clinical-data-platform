# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV POETRY_VERSION=1.8.3 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

COPY pyproject.toml .
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY src ./src
COPY configs ./configs
COPY sql ./sql
COPY dbt ./dbt

# Non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "clinical_platform.api.main"]

