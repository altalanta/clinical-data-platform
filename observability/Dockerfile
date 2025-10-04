# Multi-stage build for Clinical Data Platform Observability Stack
FROM alpine:3.18 as builder

# Build arguments
ARG VERSION=dev
ARG BUILD_DATE
ARG VCS_REF

# Install dependencies for setup script
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-requests \
    py3-yaml

# Runtime stage
FROM alpine:3.18 as runtime

# Labels for container metadata
LABEL org.opencontainers.image.title="Clinical Data Platform Observability"
LABEL org.opencontainers.image.description="Complete observability stack with Prometheus, Grafana, Loki, and Jaeger"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="Clinical Data Platform Team"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/altalanta/clinical-data-platform"

# Install runtime dependencies
RUN apk add --no-cache \
    docker-compose \
    python3 \
    py3-pip \
    py3-requests \
    py3-yaml \
    bash \
    curl

# Create non-root user for security
RUN addgroup -g 1000 appuser && adduser -u 1000 -G appuser -s /bin/bash -D appuser

# Set working directory
WORKDIR /app

# Copy observability configuration
COPY . .

# Create necessary directories
RUN mkdir -p data/prometheus data/grafana data/loki data/alertmanager \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose common observability ports
EXPOSE 3000 9090 9093 3100 16686

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python3 /app/scripts/health_check.py || exit 1

# Environment variables
ENV COMPOSE_PROJECT_NAME=clinical-observability
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["docker-compose", "-f", "docker-compose.observability.yml", "up"]