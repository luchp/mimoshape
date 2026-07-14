"""Tests for the nonparametric stationarity tests (Bendat & Piersol ch. 4)."""

import numpy as np
import pytest

from mimoshape import stationarity


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_segment_statistic_shapes_and_remainder(rng):
    y = rng.standard_normal((3, 1005))  # remainder of 5 dropped
    s = stationarity.segment_statistic(y, 10, "ms")
    assert s.shape == (3, 10)
    s1 = stationarity.segment_statistic(y[0], 10, "ms")  # 1-D promoted
    assert s1.shape == (1, 10)
    np.testing.assert_allclose(s1[0], s[0])


def test_segment_statistic_known_values():
    y = np.concatenate([np.zeros(50), 2.0 * np.ones(50)])
    np.testing.assert_allclose(stationarity.segment_statistic(y, 2, "mean")[0], [0.0, 2.0])
    np.testing.assert_allclose(stationarity.segment_statistic(y, 2, "ms")[0], [0.0, 4.0])


def test_segment_statistic_gaussian_kurtosis(rng):
    y = rng.standard_normal(2**16)
    k = stationarity.segment_statistic(y, 4, "kurtosis")
    np.testing.assert_allclose(k, 3.0, atol=0.2)
    sk = stationarity.segment_statistic(y, 4, "skewness")
    np.testing.assert_allclose(sk, 0.0, atol=0.1)


def test_segment_statistic_errors():
    with pytest.raises(ValueError, match="at least"):
        stationarity.segment_statistic(np.zeros(10), 8)
    with pytest.raises(ValueError, match="unknown statistic"):
        stationarity.segment_statistic(np.zeros(100), 4, "median")


def test_reverse_arrangements_exact_counts():
    dec = stationarity.reverse_arrangements_test([4.0, 3.0, 2.0, 1.0])
    assert dec.statistic[0] == 6  # every pair reversed
    inc = stationarity.reverse_arrangements_test([1.0, 2.0, 3.0, 4.0])
    assert inc.statistic[0] == 0
    assert inc.z[0] == -dec.z[0]  # symmetric null


def test_reverse_arrangements_detects_trend(rng):
    s = np.linspace(1.0, 2.0, 32) + 0.05 * rng.standard_normal(32)
    res = stationarity.reverse_arrangements_test(s)
    assert res.p[0] < 0.01


def test_runs_known_values():
    alt = stationarity.runs_test(np.tile([1.0, -1.0], 5))  # 10 runs, n1=n2=5
    assert alt.statistic[0] == 10
    assert alt.z[0] == pytest.approx((10 - 6.0) / np.sqrt(2000.0 / 900.0))
    grouped = stationarity.runs_test([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert grouped.statistic[0] == 2
    assert grouped.z[0] < 0  # too few runs


def test_runs_test_one_sided_raises():
    with pytest.raises(ValueError, match="one side"):
        stationarity.runs_test(np.ones(8))


def test_stationary_record_passes_both():
    y = np.random.default_rng(2).standard_normal((2, 2**14))
    report = stationarity.stationarity_report(y, num_segments=32)
    for stat in ("ms", "kurtosis"):
        assert np.all(report[stat]["reverse_arrangements"].p > 0.05)
        assert np.all(report[stat]["runs"].p > 0.05)


def test_variance_ramp_fails_ms(rng):
    n = 2**14
    y = rng.standard_normal(n) * (1.0 + 2.0 * np.arange(n) / n)
    report = stationarity.stationarity_report(y, num_segments=32, stats=("ms",))
    assert report["ms"]["reverse_arrangements"].p[0] < 1e-6
    assert report["ms"]["segments"].shape == (1, 32)
