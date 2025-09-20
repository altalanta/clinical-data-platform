from __future__ import annotations

from math import erf, sqrt
from typing import Tuple

import numpy as np


def welch_t_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Compute Welch's t statistic and approximate p-value via normal approx.

    Returns (t_stat, p_value). For small samples, prefer SciPy.
    """
    a = a.astype(float)
    b = b.astype(float)
    ma, mb = np.nanmean(a), np.nanmean(b)
    va, vb = np.nanvar(a, ddof=1), np.nanvar(b, ddof=1)
    na, nb = np.isfinite(a).sum(), np.isfinite(b).sum()
    se = sqrt(va / na + vb / nb)
    t = 0.0 if se == 0 else (ma - mb) / se
    # Normal approx two-sided p-value
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, p


def chi_square_test(obs: np.ndarray) -> Tuple[float, float, int]:
    """Pearson Chi-squared for contingency table; returns (chi2, p, dof).

    p-value approximated via normal tail of chi2 with dof; for demo only.
    """
    obs = obs.astype(float)
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    expected = row_sums @ col_sums / total
    chi2 = np.nansum((obs - expected) ** 2 / expected)
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    # Wilson-Hilferty approximation
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / sqrt(2 / (9 * dof)) if dof > 0 else 0
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return chi2, p, dof

