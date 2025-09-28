"""OpenTelemetry helpers for pipeline instrumentation."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from opentelemetry import trace
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError as exc:  # pragma: no cover - surface clearer error at runtime
    raise RuntimeError(
        "opentelemetry-exporter-otlp-proto-http must be installed to emit traces"
    ) from exc

_TRACER_PROVIDER: Optional[TracerProvider] = None
_DEFAULT_ENDPOINT = "http://localhost:4318/v1/traces"


def init_tracer(
    service_name: str = "clinical-data-pipeline",
    endpoint: Optional[str] = None,
    resource_attributes: Optional[Dict[str, str]] = None,
) -> trace.Tracer:
    """Initialise a tracer provider with an OTLP HTTP exporter."""
    global _TRACER_PROVIDER

    if _TRACER_PROVIDER is None:
        resolved_endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", _DEFAULT_ENDPOINT)

        resource_attrs = {"service.name": service_name}
        if resource_attributes:
            resource_attrs.update(resource_attributes)

        exporter = OTLPSpanExporter(endpoint=resolved_endpoint, timeout=5)
        _TRACER_PROVIDER = TracerProvider(resource=Resource.create(resource_attrs))
        _TRACER_PROVIDER.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(_TRACER_PROVIDER)

        # Capture HTTP calls (e.g. dbt docs download) if they occur.
        RequestsInstrumentor().instrument()

    return trace.get_tracer(service_name)


@contextmanager
def pipeline_span(
    name: str,
    *,
    pipeline: str,
    stage: str,
    run_id: str,
    dataset: Optional[str] = None,
    attributes: Optional[Dict[str, str]] = None,
) -> Iterator[Span]:
    """Context manager for consistent span attributes and error recording."""
    tracer = init_tracer()
    span = tracer.start_span(name)
    span.set_attribute("pipeline.name", pipeline)
    span.set_attribute("pipeline.stage", stage)
    span.set_attribute("pipeline.run_id", run_id)
    if dataset:
        span.set_attribute("pipeline.dataset", dataset)
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)

    start = time.perf_counter()

    try:
        with trace.use_span(span, end_on_exit=True) as _span:
            yield _span
    except Exception as exc:  # pragma: no cover - runtime safeguard
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        span.set_attribute("latency_ms", round(elapsed_ms, 3))
