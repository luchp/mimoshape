"""Estimation pipeline tests: multitaper CSD -> Cholesky H -> moment targets.

Includes a full measured-target round trip: estimate targets from a record,
synthesise a block, and verify CSD structure and moments are reproduced.
"""

import numpy as np
import pytest

from mimoshape import estimate, moments
from mimoshape.shaper import SynthesisProblem, MimoShaper


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def record(rng):
    """Two correlated non-Gaussian channels, 8 segments of 512 samples."""
    n = 8 * 512
    base = rng.standard_normal(n)
    y0 = base + 0.3 * rng.standard_normal(n)
    y0 = y0 + 0.4 * y0**2 - np.mean(0.4 * y0**2)  # skewed, heavy-tailed
    y1 = 0.7 * base + 0.5 * rng.standard_normal(n)
    return np.vstack([y0, y1])


def test_multitaper_csd_is_hermitian_psd(record):
    G = estimate.multitaper_csd(record, nw=4.0, nfft=512)
    assert G.shape == (2, 2, 257)
    for f in range(G.shape[2]):
        Gf = G[:, :, f]
        np.testing.assert_allclose(Gf, Gf.conj().T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(Gf) >= -1e-12)


def test_multitaper_csd_rejects_bad_nfft(record):
    with pytest.raises(ValueError):
        estimate.multitaper_csd(record, nfft=500)
    with pytest.raises(ValueError):
        estimate.multitaper_csd(record[:, :100], nfft=512)


def test_csd_to_frf_factorises(record):
    G = estimate.multitaper_csd(record, nfft=512)
    H = estimate.csd_to_frf(G)
    assert np.all(H[:, :, 0] == 0) and np.all(H[:, :, -1] == 0)
    for f in range(1, G.shape[2] - 1):
        np.testing.assert_allclose(
            H[:, :, f] @ H[:, :, f].conj().T, G[:, :, f], atol=1e-10
        )


def test_csd_to_frf_variance_scaling(record, rng):
    G = estimate.multitaper_csd(record, nfft=512)
    target_var = np.var(record, axis=1)
    H = estimate.csd_to_frf(G, variance=target_var)
    np.testing.assert_allclose(estimate.synthesis_variance(H), target_var, rtol=1e-12)
    # expected variance matches the ensemble average of random-phase blocks
    problem = SynthesisProblem(H)
    sampled = np.zeros(2)
    num_blocks = 50
    for _ in range(num_blocks):
        phase = rng.uniform(-np.pi, np.pi, (2, G.shape[2] - 2))
        sampled += np.mean(problem.signal(phase) ** 2, axis=1)
    np.testing.assert_allclose(sampled / num_blocks, target_var, rtol=0.1)


def test_estimate_moment_targets(record):
    tuples = [(0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 1, 1)]
    targets = estimate.estimate_moment_targets(record, tuples)
    assert [t.indices for t in targets] == tuples
    stats = estimate.signal_stats(record)
    assert targets[0].value == pytest.approx(stats["skewness"][0])
    assert targets[1].value == pytest.approx(stats["kurtosis"][0])
    assert targets[0].value > 0.5  # the record is built skewed
    assert targets[1].value > 3.5  # and heavy-tailed


def test_signal_stats_known_values():
    x = np.arange(8.0)
    s = estimate.signal_stats(x)
    assert s["mean"][0] == pytest.approx(3.5)
    assert s["std"][0] == pytest.approx(2.29128785)
    assert s["skewness"][0] == pytest.approx(0.0, abs=1e-12)
    assert s["kurtosis"][0] == pytest.approx(1.76190476)
    assert s["crest"][0] == pytest.approx(3.05505046)


def test_measured_target_round_trip(record, rng):
    """Estimate targets from the record, synthesise, verify reproduction."""
    G = estimate.multitaper_csd(record, nfft=512)
    H = estimate.csd_to_frf(G, variance=np.var(record, axis=1))
    tuples = [(0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 1, 1)]
    targets = estimate.estimate_moment_targets(record, tuples)

    problem = SynthesisProblem(H, targets=targets)
    shaper = MimoShaper(problem, max_time=30, rng=rng)
    x = shaper.make_block()

    achieved = [moments.normalized_moment(x, t.indices) for t in targets]
    wanted = [t.value for t in targets]
    np.testing.assert_allclose(achieved, wanted, rtol=0.05)
