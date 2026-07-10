"""Target estimation from measured multi-channel records.

Pipeline (see the paper, section "Estimating targets from measured data"):

1. ``multitaper_csd`` -- Thomson multitaper CSD estimate ``G(f)``, Hermitian
   positive semi-definite by construction (average of rank-one outer
   products of DPSS eigencoefficients).
2. ``csd_to_frf`` -- per-bin Cholesky factor ``H(f)`` with ``H H^* = G``,
   zero DC/Nyquist bins, optionally row-scaled so the synthesised signal
   matches prescribed per-channel variances.
3. ``estimate_moment_targets`` -- sample joint moments of the record as
   ``MomentTarget`` values for the same channel tuples.

Records are arrays of shape ``(Nj, N)`` (channels by samples).
"""

import numpy as np
from scipy.signal import windows

from . import moments
from .shaper import MomentTarget


def multitaper_csd(y, nw=4.0, num_tapers=None, nfft=None):
    """Thomson multitaper CSD estimate of a multi-channel record.

    Long records are split into consecutive segments of ``nfft`` samples and
    the per-segment estimates are averaged.  Returns ``G`` of shape
    ``(Nj, Nj, nfft//2 + 1)`` with ``G[p, q, f] = mean_k Y_p Y_q*`` over the
    ``K`` tapers; Hermitian PSD at every bin by construction.

    ``nw`` is the time half-bandwidth product; ``num_tapers`` defaults to
    ``2*nw - 1`` (the effectively leakage-free set).  ``nfft`` defaults to
    the whole record length (single segment).
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    nj, n = y.shape
    if nfft is None:
        nfft = n
    if nfft < 8 or (nfft & (nfft - 1)) != 0:
        raise ValueError(f"nfft {nfft} must be a power of 2 (>= 8)")
    if n < nfft:
        raise ValueError(f"record length {n} shorter than nfft {nfft}")
    if num_tapers is None:
        num_tapers = int(2 * nw - 1)
    if num_tapers < 1:
        raise ValueError(f"need at least one taper, got {num_tapers}")

    tapers, ratios = windows.dpss(nfft, nw, Kmax=num_tapers, return_ratios=True)
    weights = ratios / np.sum(ratios)

    nf = nfft // 2 + 1
    G = np.zeros((nj, nj, nf), dtype=complex)
    num_segments = n // nfft
    for s in range(num_segments):
        segment = y[:, s * nfft : (s + 1) * nfft]
        # eigencoefficients: (K, Nj, Nf)
        Y = np.fft.rfft(tapers[:, None, :] * segment[None, :, :], axis=2)
        G += np.einsum("k,kpf,kqf->pqf", weights, Y, np.conj(Y))
    return G / num_segments


def csd_to_frf(G, variance=None):
    """Per-bin Cholesky factor ``H`` of a CSD estimate, ready for synthesis.

    DC and Nyquist bins are zeroed (required by ``SynthesisProblem``).  When
    ``variance`` (shape ``(Nj,)``) is given, the rows of ``H`` are scaled so
    that a synthesised block has exactly those expected per-channel
    variances; this preserves coherence and cross-phase.  Raises
    ``numpy.linalg.LinAlgError`` when a bin is not positive definite --
    regularise or band-limit the estimate rather than silently patching it.
    """
    G = np.asarray(G, dtype=complex)
    nf = G.shape[2]
    H = np.zeros_like(G)
    for f in range(1, nf - 1):
        H[:, :, f] = np.linalg.cholesky(G[:, :, f])
    if variance is not None:
        current = synthesis_variance(H)
        scale = np.sqrt(np.asarray(variance, dtype=float) / current)
        H *= scale[:, None, None]
    return H


def synthesis_variance(H):
    """Expected per-channel variance of a block synthesised from ``H`` with
    unit-modulus random phases: ``(2/Nt^2) sum_f sum_j |H_kjf|^2``."""
    nt = 2 * (H.shape[2] - 1)
    return (2.0 / nt**2) * np.sum(np.abs(H) ** 2, axis=(1, 2))


def estimate_moment_targets(y, index_tuples, weight=1.0):
    """Sample normalised joint moments of a record as ``MomentTarget`` list.

    Realisable by construction: the targets are the record's own statistics.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    y = y - np.mean(y, axis=1, keepdims=True)
    return [
        MomentTarget(tuple(idx), moments.normalized_moment(y, tuple(idx)), weight)
        for idx in index_tuples
    ]


def signal_stats(x):
    """Per-channel summary statistics of a (multi-channel) signal.

    Returns a dict of arrays keyed by ``mean, std, skewness, kurtosis, crest``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    mean = np.mean(x, axis=1)
    u = x - mean[:, None]
    std = np.sqrt(np.mean(u**2, axis=1))
    return {
        "mean": mean,
        "std": std,
        "skewness": np.mean(u**3, axis=1) / std**3,
        "kurtosis": np.mean(u**4, axis=1) / std**4,
        "crest": (np.max(u, axis=1) - np.min(u, axis=1)) / std,
    }
