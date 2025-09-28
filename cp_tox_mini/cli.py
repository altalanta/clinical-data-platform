"""Typer CLI surfacing the CP tox mini pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from .diagnostics import LEAKAGE_JSON, run_diagnostics
from .dose_response import IC50_JSON, summarize_ic50_table
from .eval import METRICS_JSON, evaluate_model
from .features import build_features
from .fuse import fuse_modalities
from .io import MANIFEST_PATH, PROCESSED_DIR, RAW_DIR, download_inputs, validate_manifest
from .reporting import build_reports
from .train import MODEL_PATH, train_model

app = typer.Typer(help="Deterministic CP tox mini pipeline")


def _set_reproducible_env() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")


def _require_path(path: Path, message: str) -> None:
    if not path.exists():
        typer.echo(message)
        raise typer.Exit(code=1)


@app.command()
def download() -> None:
    """Copy deterministic demo inputs and emit a manifest."""
    _set_reproducible_env()
    entries = download_inputs()
    typer.echo(f"Downloaded {len(entries)} assets to {RAW_DIR}")
    validate_manifest()
    typer.echo(f"Manifest written to {MANIFEST_PATH}")


@app.command()
def features() -> None:
    """Compute cell painting + chemical features."""
    _set_reproducible_env()
    validate_manifest()
    output = build_features()
    typer.echo(f"Features saved to {output}")


@app.command()
def fuse() -> None:
    """Join modalities and materialise fused dataset."""
    _set_reproducible_env()
    validate_manifest()
    output = fuse_modalities()
    typer.echo(f"Fused dataset saved to {output}")


@app.command()
def train() -> None:
    """Train the baseline classifier."""
    _set_reproducible_env()
    validate_manifest()
    output = train_model()
    typer.echo(f"Model stored at {output}")


@app.command()
def eval() -> None:  # noqa: A003 - CLI verb
    """Evaluate the trained model and generate metrics/figures."""
    _set_reproducible_env()
    validate_manifest()
    payload = evaluate_model()
    typer.echo(f"Metrics written to {METRICS_JSON}: {payload}")


@app.command()
def diagnostics(iterations: int = typer.Option(100, help="Permutation iterations")) -> None:
    """Run leakage probes and permutation diagnostics."""
    _set_reproducible_env()
    validate_manifest()
    payload = run_diagnostics(iterations=iterations)
    typer.echo(f"Diagnostics written to {LEAKAGE_JSON}: {payload}")


@app.command()
def ic50(csv_path: Optional[Path] = typer.Option(None, help="Dose-response CSV")) -> None:
    """Estimate IC50 using an LL4 fit."""
    _set_reproducible_env()
    validate_manifest()
    default_path = RAW_DIR / "ic50_example.csv"
    path = csv_path or default_path
    if not path.exists():
        typer.echo(f"CSV not found at {path}")
        raise typer.Exit(code=1)
    result = summarize_ic50_table(path)
    typer.echo(f"IC50 summary saved to {IC50_JSON}: {result}")


@app.command()
def report() -> None:
    """Render Markdown + HTML reports and publish to docs/."""
    _set_reproducible_env()
    validate_manifest()
    _require_path(METRICS_JSON, "Run eval step first to generate metrics.")
    _require_path(LEAKAGE_JSON, "Run diagnostics step first to generate leakage summary.")
    build_reports()
    typer.echo("Reports generated in reports/ and docs/")


@app.command()
def all() -> None:
    """Run the full deterministic demo pipeline."""
    _set_reproducible_env()
    download_inputs()
    typer.echo("Inputs ready; validating manifest...")
    validate_manifest()
    build_features()
    fuse_modalities()
    train_model()
    metrics_payload = evaluate_model()
    typer.echo(f"Model metrics: {metrics_payload}")
    diag_payload = run_diagnostics(iterations=50)
    typer.echo(f"Diagnostics: {diag_payload}")
    summarize_ic50_table(RAW_DIR / "ic50_example.csv")
    build_reports()
    typer.echo("Pipeline complete. See reports/ and docs/ for outputs.")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
