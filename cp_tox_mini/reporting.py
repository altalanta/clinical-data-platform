"""Build Markdown and HTML reports for the CP tox mini demo."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

from markdown import markdown

from .diagnostics import LEAKAGE_JSON
from .eval import FIGURES_DIR, METRICS_JSON, REPORTS_DIR

REPORT_MD = REPORTS_DIR / "cp-tox-mini_report.md"
REPORT_HTML = REPORTS_DIR / "cp-tox-mini_report.html"
LANDING_HTML = REPORTS_DIR / "index.html"
DOCS_DIR = Path("docs")

_TEMPLATE_CARD = Path(__file__).resolve().parent / "model_card.md"


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _render_model_card(metrics_payload: Dict[str, object], leakage_payload: Dict[str, object]) -> Path:
    template = _TEMPLATE_CARD.read_text(encoding="utf-8")
    rendered = template.format(
        auroc=metrics_payload["auroc"],
        ap=metrics_payload["ap"],
        brier=metrics_payload["brier"],
        ece=metrics_payload["ece"],
        n_train=metrics_payload["n_train"],
        n_test=metrics_payload["n_test"],
        leakage_risk=leakage_payload["risk_of_leakage"],
        leakage_note=leakage_payload["notes"],
    )
    model_card_path = REPORTS_DIR / "model_card.md"
    model_card_path.write_text(rendered, encoding="utf-8")
    return model_card_path


def _metrics_table(metrics_payload: Dict[str, object]) -> str:
    return (
        "| Metric | Value |\n"
        "| --- | --- |\n"
        f"| AUROC | {metrics_payload['auroc']:.3f} |\n"
        f"| Average Precision | {metrics_payload['ap']:.3f} |\n"
        f"| Brier Score | {metrics_payload['brier']:.3f} |\n"
        f"| ECE (10 bins) | {metrics_payload['ece']:.3f} |\n"
        f"| Train Samples | {metrics_payload['n_train']} |\n"
        f"| Test Samples | {metrics_payload['n_test']} |\n"
    )


def _build_report_markdown(metrics_payload: Dict[str, object], leakage_payload: Dict[str, object]) -> str:
    md_parts = [
        "# CP-Tox Mini Pipeline Report",
        "",
        "## Metrics",
        _metrics_table(metrics_payload),
        "",
        "## Risk of Leakage",
        f"- Plate probe AUROC: {leakage_payload['plate_probe_auroc']:.3f} (AP {leakage_payload['plate_probe_ap']:.3f})",
        f"- Layout probe AUROC: {leakage_payload['layout_probe_auroc']:.3f} (AP {leakage_payload['layout_probe_ap']:.3f})",
        f"- Permutation p-value (within plate): {leakage_payload['perm_p_value']:.3f}",
        f"- Risk assessment: **{leakage_payload['risk_of_leakage'].upper()}**",
        f"- Notes: {leakage_payload['notes']}",
        "",
        "## Figures",
        "![ROC](figures/roc.png)",
        "![Precision-Recall](figures/pr.png)",
        "![Calibration](figures/calibration.png)",
        "![Plate Probe](figures/leakage_probe_plate.png)",
        "![Layout Probe](figures/leakage_probe_layout.png)",
    ]
    return "\n".join(md_parts) + "\n"


def _write_html(markdown_text: str, destination: Path) -> None:
    html_body = markdown(markdown_text, extensions=["tables"])
    wrapped = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>CP-Tox Mini Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; line-height: 1.6; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
    h1, h2 {{ color: #123A5F; }}
    img {{ max-width: 100%; height: auto; margin-bottom: 1rem; }}
    code {{ background: #f5f5f5; padding: 0.1rem 0.3rem; }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    destination.write_text(wrapped, encoding="utf-8")


def _build_landing_page(metrics_payload: Dict[str, object], leakage_payload: Dict[str, object], model_card_path: Path) -> None:
    metrics_table = _metrics_table(metrics_payload)
    landing_md = "\n".join(
        [
            "# CP-Tox Mini",
            "",
            "This is the deterministic clinical toxicity mini-pipeline demo.",
            "",
            "## Highlights",
            metrics_table,
            "",
            "### Leakage Summary",
            f"- Risk level: **{leakage_payload['risk_of_leakage'].upper()}**",
            f"- Permutation p-value: {leakage_payload['perm_p_value']:.3f}",
            "",
            "### Artifacts",
            "- [Full report](cp-tox-mini_report.html)",
            f"- [Model card]({model_card_path.name})",
            "- Figures:",
            "  - [ROC](figures/roc.png)",
            "  - [Precision-Recall](figures/pr.png)",
            "  - [Calibration](figures/calibration.png)",
            "  - [Plate probe](figures/leakage_probe_plate.png)",
            "  - [Layout probe](figures/leakage_probe_layout.png)",
        ]
    )
    _write_html(landing_md, LANDING_HTML)


def build_reports() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_payload = _load_json(METRICS_JSON)
    leakage_payload = _load_json(LEAKAGE_JSON)

    model_card_path = _render_model_card(metrics_payload, leakage_payload)

    markdown_text = _build_report_markdown(metrics_payload, leakage_payload)
    REPORT_MD.write_text(markdown_text, encoding="utf-8")
    _write_html(markdown_text, REPORT_HTML)

    _build_landing_page(metrics_payload, leakage_payload, model_card_path)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LANDING_HTML, DOCS_DIR / "index.html")

    docs_figures = DOCS_DIR / "figures"
    if docs_figures.exists():
        shutil.rmtree(docs_figures)
    if FIGURES_DIR.exists():
        shutil.copytree(FIGURES_DIR, docs_figures)

    shutil.copyfile(REPORT_HTML, DOCS_DIR / REPORT_HTML.name)
    shutil.copyfile(model_card_path, DOCS_DIR / model_card_path.name)


__all__ = ["build_reports", "REPORT_MD", "REPORT_HTML", "LANDING_HTML"]
