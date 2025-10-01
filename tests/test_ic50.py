from __future__ import annotations

import math

import numpy as np

from cp_tox_mini.dose_response import IC50Result, estimate_ic50


def _generate_curve(ic50: float) -> tuple[np.ndarray, np.ndarray]:
    doses = np.array([5, 10, 20, 40, 80, 160, 320, 640], dtype=float)
    bottom, top, hill = 0.1, 0.9, 1.3
    log_ic50 = math.log10(ic50)
    response = bottom + (top - bottom) / (1 + np.exp(hill * (np.log10(doses) - log_ic50)))
    return doses, response


def test_estimate_ic50_close_to_truth() -> None:
    expected_ic50 = 50.0
    doses, response = _generate_curve(expected_ic50)
    result: IC50Result = estimate_ic50(doses, response)
    assert abs(result.ic50 - expected_ic50) <= expected_ic50 * 0.1


def test_negative_dose_rejected() -> None:
    try:
        estimate_ic50([0, 1, 2], [0.1, 0.2, 0.3])
    except ValueError as exc:  # pragma: no branch
        assert "Doses must be positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for non-positive doses")
