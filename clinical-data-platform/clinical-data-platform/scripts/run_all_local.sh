#!/usr/bin/env bash
set -euo pipefail

make data
make minio
make ingest
make dbt
make analytics
make train

echo "Done. Start API: make api | Start UI: make ui"

