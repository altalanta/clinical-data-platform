# Runbook: Silver Pipeline SLO Breach

## Trigger
- Grafana alert **Silver Layer Freshness Breach** goes red
- `observability/freshness_sli.json` reports `status: breach`

## Immediate Checks
1. Open the latest SLI payload:
   ```bash
   cat observability/freshness_sli.json | jq
   ```
   - Confirm `freshness_minutes`, `slo_minutes`, and `silver_last_updated`.
2. Inspect observability stack health:
   ```bash
   docker compose -f observability/docker-compose.obsv.yml ps
   ```
3. Re-run the demo pipeline locally to reproduce:
   ```bash
   make obsv.demo
   ```

## Grafana Investigation
- URL: http://localhost:3000 (default credentials `admin`/`admin` — rotate for shared use)
- Dashboard: **Clinical Pipeline Observability → Observability folder**
  - **Silver Freshness (minutes):** Verify if metric stays above threshold.
  - **Ingestion Throughput / Stage Latency:** Look for drops or spikes correlated with the breach window.
  - **Pipeline Status timeline:** Identify which stage last completed successfully.
- Logs (Explore → Loki):
  - Filter by run: `{pipeline="dbt", run_id="<value>"}`
  - Error filtering: `{level="error"}`
  - Freshness context: `{pipeline="freshness"}` to review SLI events.
- Traces (Explore → Tempo):
  - Search by run ID: `pipeline.run_id=<value>`
  - Sort spans by `latency_ms` to pinpoint the slowest stage.

## Common Causes & Next Steps
- **Stale silver data** – `silver_last_updated` older than 2h. Re-run `python -m src.pipelines.dbt_demo` and confirm timestamp updates.
- **Upstream ingest failure** – Check Loki logs for `status="error"` or missing `pipeline_complete` events in ingress stages.
- **Collector/Grafana outage** – If dashboards missing new logs/traces, restart stack:
  ```bash
  make obsv.down
  make obsv.up
  ```
- **Clock drift** – Confirm system time; freshness computations depend on UTC timestamps from `_last_update.txt`.

## After Action
- Update `observability/freshness_sli.json` artifact in incident ticket.
- If thresholds need tuning, adjust `--slo-minutes` in `src/common/freshness.py` & dashboard thresholds, then document change.
- When ready, rotate Grafana admin credentials and configure email/webhook contact points in `observability/rules/freshness_slo.yaml`.
