import numpy as np

from clinical_platform.analytics.stats_utils import chi_square_test, welch_t_test


def test_welch_t_test_runs():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 3, 4, 5, 6])
    t, p = welch_t_test(a, b)
    assert isinstance(t, float) and isinstance(p, float)


def test_chi_square_test_runs():
    obs = np.array([[10, 20], [20, 10]])
    chi2, p, dof = chi_square_test(obs)
    assert dof == 1
    assert chi2 > 0 and 0 <= p <= 1

