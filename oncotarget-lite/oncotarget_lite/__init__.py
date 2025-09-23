"""
Oncotarget-lite: A rigorous machine learning pipeline for oncology target discovery.

This package implements production-ready ML pipelines with real datasets,
proper preprocessing, robust evaluation with bootstrap confidence intervals,
and comprehensive ablation studies.
"""

__version__ = "1.0.0"
__author__ = "Oncotarget-lite Team"

from . import io, features, splits, models, metrics, calibration, plotting, registry

__all__ = [
    "io",
    "features", 
    "splits",
    "models",
    "metrics",
    "calibration",
    "plotting",
    "registry",
]