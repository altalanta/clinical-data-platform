# Deployment Guide

This guide covers deployment options for the Clinical Data Platform, from single-machine Docker deployments to production Kubernetes environments.

## Quick Start with Docker Compose

The fastest way to deploy the complete platform:

```bash
# Clone the repository
git clone https://github.com/altalanta/clinical-data-platform.git
cd clinical-data-platform

# Start all services
docker-compose up -d

# Include observability stack (optional)
docker-compose --profile observability up -d
```

### Services

After deployment, the following services will be available:

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI backend |
| Streamlit UI | http://localhost:8501 | Interactive dashboard |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| Prometheus | http://localhost:9090 | Metrics collection |
| Jaeger | http://localhost:16686 | Distributed tracing |

## Container Images

All components are available as multi-architecture Docker images supporting `linux/amd64` and `linux/arm64`:

### Core Platform Images

```bash
# Pull latest images
docker pull ghcr.io/altalanta/clinical-data-platform-api:latest
docker pull ghcr.io/altalanta/clinical-data-platform-ingest:latest
docker pull ghcr.io/altalanta/clinical-data-platform-analytics:latest
docker pull ghcr.io/altalanta/clinical-data-platform-streamlit:latest

# Or specific version
docker pull ghcr.io/altalanta/clinical-data-platform-api:v1.0.0
```

### Observability Stack

```bash
docker pull ghcr.io/altalanta/clinical-data-platform-observability:latest
```

## Individual Service Deployment

### API Service

```bash
docker run -d \
  --name clinical-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e DATABASE_URL=duckdb:///app/data/clinical.duckdb \
  ghcr.io/altalanta/clinical-data-platform-api:latest
```

### Streamlit Dashboard

```bash
docker run -d \
  --name clinical-ui \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -e API_BASE_URL=http://api:8000 \
  ghcr.io/altalanta/clinical-data-platform-streamlit:latest
```

### Data Ingestion

```bash
docker run -d \
  --name clinical-ingest \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/processing:/app/processing \
  ghcr.io/altalanta/clinical-data-platform-ingest:latest
```

## Production Deployment

### Environment Variables

Configure the following environment variables for production:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=duckdb:///app/data/clinical.duckdb
LOG_LEVEL=info
CORS_ORIGINS=["https://your-domain.com"]

# Security
JWT_SECRET_KEY=your-secure-secret-key
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100

# Observability
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
LOG_FORMAT=json

# HIPAA Compliance
PHI_READONLY_MODE=true
AUDIT_LOGGING_ENABLED=true
ENCRYPTION_AT_REST=true
```

### Persistent Volumes

Ensure data persistence by mounting volumes:

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  api:
    image: ghcr.io/altalanta/clinical-data-platform-api:latest
    volumes:
      - clinical-data:/app/data
      - clinical-logs:/app/logs
      - clinical-config:/app/config
    environment:
      - DATABASE_URL=duckdb:///app/data/clinical.duckdb
      - LOG_LEVEL=warning
      - PHI_READONLY_MODE=true

volumes:
  clinical-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/clinical-platform/data

  clinical-logs:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/clinical-platform/logs
```

### Reverse Proxy Configuration

#### Nginx

```nginx
upstream clinical_api {
    server 127.0.0.1:8000;
}

upstream clinical_ui {
    server 127.0.0.1:8501;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # API endpoints
    location /api/ {
        proxy_pass http://clinical_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Streamlit UI
    location / {
        proxy_pass http://clinical_ui/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### Traefik

```yaml
# traefik.yml
services:
  traefik:
    image: traefik:v3.0
    command:
      - --api.dashboard=true
      - --providers.docker=true
      - --entrypoints.web.address=:80
      - --entrypoints.websecure.address=:443
      - --certificatesresolvers.letsencrypt.acme.email=admin@your-domain.com
      - --certificatesresolvers.letsencrypt.acme.storage=acme.json
      - --certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./acme.json:/acme.json

  api:
    image: ghcr.io/altalanta/clinical-data-platform-api:latest
    labels:
      - traefik.http.routers.api.rule=Host(`your-domain.com`) && PathPrefix(`/api`)
      - traefik.http.routers.api.tls.certresolver=letsencrypt
      - traefik.http.services.api.loadbalancer.server.port=8000

  streamlit:
    image: ghcr.io/altalanta/clinical-data-platform-streamlit:latest
    labels:
      - traefik.http.routers.ui.rule=Host(`your-domain.com`)
      - traefik.http.routers.ui.tls.certresolver=letsencrypt
      - traefik.http.services.ui.loadbalancer.server.port=8501
```

## Kubernetes Deployment

### Namespace and Resources

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: clinical-platform
  labels:
    name: clinical-platform
    compliance: hipaa
```

### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: clinical-platform-config
  namespace: clinical-platform
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  LOG_LEVEL: "info"
  PHI_READONLY_MODE: "true"
  AUDIT_LOGGING_ENABLED: "true"
```

### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: clinical-platform-secrets
  namespace: clinical-platform
type: Opaque
stringData:
  JWT_SECRET_KEY: "your-secure-secret-key"
  DATABASE_URL: "duckdb:///app/data/clinical.duckdb"
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: clinical-api
  namespace: clinical-platform
  labels:
    app: clinical-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: clinical-api
  template:
    metadata:
      labels:
        app: clinical-api
    spec:
      containers:
      - name: api
        image: ghcr.io/altalanta/clinical-data-platform-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: clinical-platform-config
        - secretRef:
            name: clinical-platform-secrets
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: clinical-data-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: clinical-logs-pvc
```

### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: clinical-api-service
  namespace: clinical-platform
spec:
  selector:
    app: clinical-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: clinical-platform-ingress
  namespace: clinical-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: clinical-platform-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: clinical-api-service
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: clinical-ui-service
            port:
              number: 80
```

### Persistent Volumes

```yaml
# persistent-volumes.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: clinical-data-pvc
  namespace: clinical-platform
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: clinical-logs-pvc
  namespace: clinical-platform
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard
```

## Monitoring and Observability

### Helm Deployment

Deploy the observability stack using Helm:

```bash
# Add Prometheus community Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values prometheus-values.yaml

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --values grafana-values.yaml
```

### Prometheus Configuration

```yaml
# prometheus-values.yaml
prometheus:
  prometheusSpec:
    additionalScrapeConfigs:
    - job_name: clinical-api
      static_configs:
      - targets: ['clinical-api-service:80']
      metrics_path: /metrics
      scrape_interval: 30s

grafana:
  adminPassword: admin123
  persistence:
    enabled: true
    size: 10Gi
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
      - name: 'clinical-platform'
        orgId: 1
        folder: 'Clinical Platform'
        type: file
        disableDeletion: false
        options:
          path: /var/lib/grafana/dashboards/clinical-platform
```

## Security Considerations

### Network Security

1. **Use TLS/HTTPS**: Always encrypt traffic in production
2. **Network Segmentation**: Isolate database and internal services
3. **Firewall Rules**: Restrict access to necessary ports only
4. **VPN Access**: Require VPN for administrative access

### Authentication & Authorization

1. **JWT Tokens**: Configure secure JWT secret keys
2. **Role-Based Access**: Implement RBAC for different user types
3. **Rate Limiting**: Enable rate limiting to prevent abuse
4. **Session Management**: Configure secure session handling

### Data Protection

1. **Encryption at Rest**: Enable database encryption
2. **Encryption in Transit**: Use TLS for all communications
3. **Backup Encryption**: Encrypt all backup files
4. **PHI Handling**: Follow HIPAA compliance guidelines

### Container Security

```dockerfile
# Security-hardened Dockerfile example
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set security-focused environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install security updates
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup-script.sh

BACKUP_DIR="/opt/clinical-platform/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup DuckDB database
docker exec clinical-api \
  duckdb /app/data/clinical.duckdb \
  "EXPORT DATABASE '$BACKUP_DIR/clinical_$DATE'" \
  --csv

# Compress backup
tar -czf "$BACKUP_DIR/clinical_backup_$DATE.tar.gz" \
  -C "$BACKUP_DIR" "clinical_$DATE"

# Clean up old backups (keep 30 days)
find "$BACKUP_DIR" -name "clinical_backup_*.tar.gz" -mtime +30 -delete

echo "Backup completed: clinical_backup_$DATE.tar.gz"
```

### Automated Backup with Cron

```bash
# Add to crontab
# Backup daily at 2 AM
0 2 * * * /opt/clinical-platform/scripts/backup-script.sh

# Weekly full backup
0 3 * * 0 /opt/clinical-platform/scripts/full-backup-script.sh
```

## Troubleshooting

### Common Issues

1. **Container Won't Start**
   ```bash
   # Check logs
   docker logs clinical-api
   
   # Check container status
   docker ps -a
   
   # Inspect container
   docker inspect clinical-api
   ```

2. **Database Connection Issues**
   ```bash
   # Verify database file permissions
   ls -la data/clinical.duckdb
   
   # Test database connection
   docker exec clinical-api python -c "import duckdb; print(duckdb.connect('/app/data/clinical.duckdb').execute('SELECT 1').fetchone())"
   ```

3. **Performance Issues**
   ```bash
   # Check resource usage
   docker stats
   
   # Monitor logs
   docker logs -f clinical-api
   
   # Check Prometheus metrics
   curl http://localhost:8000/metrics
   ```

### Health Checks

```bash
# API health check
curl http://localhost:8000/health

# Database health check
curl http://localhost:8000/health/database

# Full system status
curl http://localhost:8000/status
```

## Scaling

### Horizontal Scaling

For high-availability deployments:

1. **API Layer**: Scale API containers behind a load balancer
2. **Database**: Consider distributed databases for large datasets
3. **Processing**: Use container orchestration for batch jobs
4. **Caching**: Implement Redis for session and query caching

### Vertical Scaling

Resource recommendations by deployment size:

| Deployment Size | CPU | Memory | Storage |
|----------------|-----|--------|---------|
| Development | 2 cores | 4 GB | 50 GB |
| Small Production | 4 cores | 8 GB | 200 GB |
| Medium Production | 8 cores | 16 GB | 500 GB |
| Large Production | 16+ cores | 32+ GB | 1+ TB |

For more deployment options and advanced configurations, see the [Infrastructure Guide](infrastructure.md).