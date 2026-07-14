"""Nonparametric stationarity tests on segment statistics.

The classical engineering check for weak-sense stationarity of a measured
record (Bendat & Piersol, *Random Data*, ch. 4): partition the record into
``num_segments`` equal segments, compute a per-segment statistic (mean
square for variance stationarity, sample kurtosis for moment stationarity,
...), and test the resulting sequence for structure with two
distribution-free tests:

* :func:`reverse_arrangements_test` -- sensitive to monotonic trends,
* :func:`runs_test` -- sensitive to slow wandering and clustering.

Both return z-scores against the exact null mean and variance under the
stationary (exchangeable-segments) hypothesis, and two-sided normal
p-values; ``p < alpha`` rejects stationarity. A record that fails on the
mean-square sequence needs one spectral model per section (multimodel
synthesis); failure on skewness/kurtosis alone calls for per-section moment
targets.
"""

from typing import NamedTuple

import numpy as np
from scipy.special import erfc


class TestResult(NamedTuple):
    """Per-channel test outcome: raw statistic, z-score, two-sided p-value."""

    statistic: np.ndarray
    z: np.ndarray
    p: np.ndarray


def segment_statistic(y, num_segments, stat="ms"):
    """Per-segment statistic of a (multi-channel) record.

    Splits ``y`` (channels x samples, 1-D allowed) into ``num_segments``
    equal segments (the remainder is dropped) and returns an array of shape
    ``(num_channels, num_segments)``. ``stat`` is one of ``mean`` (segment
    mean), ``ms`` (segment mean square, not centred), ``skewness`` or
    ``kurtosis`` (centred and normalised per segment).
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    nj, n = y.shape
    seg_len = n // num_segments
    if seg_len < 2:
        raise ValueError(f"{num_segments} segments need at least {2 * num_segments} samples, got {n}")
    seg = y[:, : num_segments * seg_len].reshape(nj, num_segments, seg_len)
    if stat == "mean":
        return np.mean(seg, axis=2)
    if stat == "ms":
        return np.mean(seg**2, axis=2)
    u = seg - np.mean(seg, axis=2, keepdims=True)
    s = np.sqrt(np.mean(u**2, axis=2))
    if stat == "skewness":
        return np.mean(u**3, axis=2) / s**3
    if stat == "kurtosis":
        return np.mean(u**4, axis=2) / s**4
    raise ValueError(f"unknown statistic {stat!r}; use mean, ms, skewness or kurtosis")


def reverse_arrangements_test(s):
    """Reverse-arrangements trend test (Bendat & Piersol sec. 4.5.2).

    For each channel of ``s`` (channels x segments, 1-D allowed) counts the
    reverse arrangements ``A`` -- pairs ``i < j`` with ``s_i > s_j``. Under
    stationarity ``A`` has mean ``N(N-1)/4`` and variance
    ``N(N-1)(2N+5)/72``; a monotonic trend drives ``A`` to an extreme.
    Returns a :class:`TestResult` of arrays over channels.
    """
    s = np.atleast_2d(np.asarray(s, dtype=float))
    n = s.shape[1]
    later = np.triu(np.ones((n, n), dtype=bool), k=1)
    a = np.sum((s[:, :, None] > s[:, None, :]) & later, axis=(1, 2)).astype(float)
    mean = n * (n - 1) / 4.0
    var = n * (n - 1) * (2 * n + 5) / 72.0
    z = (a - mean) / np.sqrt(var)
    return TestResult(a, z, erfc(np.abs(z) / np.sqrt(2.0)))


def runs_test(s):
    """Runs test about the median (Bendat & Piersol sec. 4.5.1).

    For each channel of ``s`` (channels x segments, 1-D allowed) classifies
    segments as above/below the channel median (ties dropped) and counts the
    number of runs ``R``. Under stationarity ``R`` has mean
    ``1 + 2 n1 n2 / n`` and variance ``2 n1 n2 (2 n1 n2 - n) / (n^2 (n-1))``;
    too few runs means clustering (slow wandering), too many means
    alternation. Returns a :class:`TestResult` of arrays over channels.
    """
    s = np.atleast_2d(np.asarray(s, dtype=float))
    runs = np.empty(s.shape[0])
    z = np.empty(s.shape[0])
    for k, row in enumerate(s):
        above = row[row != np.median(row)] > np.median(row)
        n1 = int(np.sum(above))
        n2 = int(above.size - n1)
        n = n1 + n2
        if n1 == 0 or n2 == 0:
            raise ValueError(f"channel {k}: all segment statistics on one side of the median")
        r = 1 + int(np.sum(above[1:] != above[:-1]))
        mean = 1.0 + 2.0 * n1 * n2 / n
        var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n**2 * (n - 1.0))
        runs[k] = r
        z[k] = (r - mean) / np.sqrt(var)
    return TestResult(runs, z, erfc(np.abs(z) / np.sqrt(2.0)))


def stationarity_report(y, num_segments=32, stats=("ms", "kurtosis")):
    """Run both tests on several segment statistics of a record.

    Returns ``{stat: {"segments": array, "reverse_arrangements": TestResult,
    "runs": TestResult}}``. The record is used as given (not detrended);
    remove a known static offset first if the ``mean`` statistic is included.
    """
    report = {}
    for stat in stats:
        s = segment_statistic(y, num_segments, stat)
        report[stat] = {
            "segments": s,
            "reverse_arrangements": reverse_arrangements_test(s),
            "runs": runs_test(s),
        }
    return report
