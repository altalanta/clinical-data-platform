# Getting Started

## Requirements
- Python 3.10+
- Docker (optional, for containerized run)

## Quickstart
```bash
pip install -e .[dev]
pytest
```

For containers:

```bash
docker build -t ghcr.io/altalanta/clinical-data-platform:dev .
docker run --rm ghcr.io/altalanta/clinical-data-platform:dev
```
