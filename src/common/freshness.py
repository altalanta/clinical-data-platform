"""Freshness SLI/SLO utilities for the silver layer."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import logging_config
from .logging_config import event_context, get_logger
from .otel import pipeline_span

DEFAULT_SLO_MINUTES = 120
DEFAULT_OUTPUT = Path("observability/freshness_sli.json")


@dataclass(frozen=True)
class FreshnessResult:
    freshness_minutes: Optional[int]
    last_updated: Optional[datetime]
    status: str


def _parse_timestamp(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unsupported timestamp format in '{raw}'") from exc


def compute_freshness_minutes(path: str) -> FreshnessResult:
    """Return freshness minutes since last silver update and derived status."""
    last_updated: Optional[datetime] = None
    file_path = Path(path)
    now = datetime.now(timezone.utc)

    if file_path.exists():
        raw = file_path.read_text(encoding="utf-8").strip()
        if raw:
            last_updated = _parse_timestamp(raw)

    freshness_minutes: Optional[int] = None
    if last_updated:
        delta = now - last_updated
        freshness_minutes = max(int(delta.total_seconds() // 60), 0)

    return FreshnessResult(freshness_minutes=freshness_minutes, last_updated=last_updated, status="unknown")


def _status_for_freshness(freshness_minutes: int, slo_minutes: int) -> str:
    if freshness_minutes <= slo_minutes:
        return "ok"
    if freshness_minutes <= int(slo_minutes * 1.25):
        return "warning"
    return "breach"


def write_freshness_sli(path: str, slo_minutes: int, output: Path = DEFAULT_OUTPUT) -> FreshnessResult:
    logging_config.configure_logging()
    logger = get_logger("freshness")

    run_id = f"freshness-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    with pipeline_span(
        "silver_freshness_check",
        pipeline="freshness",
        stage="sli",
        run_id=run_id,
        dataset="silver",
        attributes={"slo_minutes": slo_minutes},
    ):
        interim = compute_freshness_minutes(path)
        freshness_value = interim.freshness_minutes if interim.freshness_minutes is not None else slo_minutes * 10
        status = _status_for_freshness(freshness_value, slo_minutes)
        final = FreshnessResult(
            freshness_minutes=freshness_value,
            last_updated=interim.last_updated,
            status=status,
        )
        output.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "silver_last_updated": final.last_updated.isoformat() if final.last_updated else None,
            "freshness_minutes": final.freshness_minutes,
            "slo_minutes": slo_minutes,
            "status": final.status,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        logger.info(
            "freshness_sli",
            extra=event_context(
                pipeline="freshness",
                stage="sli",
                dataset="silver",
                run_id=run_id,
                freshness_minutes=final.freshness_minutes,
                slo_minutes=slo_minutes,
                status=final.status,
            ),
        )

    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute silver layer freshness SLI")
    parser.add_argument("--path", required=True, help="Path to file containing last update timestamp")
    parser.add_argument(
        "--slo-minutes",
        type=int,
        default=DEFAULT_SLO_MINUTES,
        help="Freshness SLO threshold in minutes (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Where to write the SLI JSON payload",
    )
    args = parser.parse_args()

    write_freshness_sli(args.path, args.slo_minutes, Path(args.output))


if __name__ == "__main__":  # pragma: no cover
    main()
