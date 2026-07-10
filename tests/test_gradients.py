"""Analytic gradients must match central-difference numerical gradients.

These tests are the contract for every gradient formula in the paper: raw
joint moments (diagonal and cross), the variance shortcut, normalised
moments, endpoint value/slope, and the assembled loss.
"""

import numpy as np
import pytest

from mimoshape import moments
from mimoshape.shaper import MomentTarget, EndpointTarget, CrestTarget, SynthesisProblem

NT = 32
NF = NT // 2 + 1
NJ = 3
TOL = 1e-6


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def H(rng):
    """Random full MIMO frequency response with zero DC/Nyquist bins."""
    h = rng.standard_normal((NJ, NJ, NF)) + 1j * rng.standard_normal((NJ, NJ, NF))
    h[:, :, 0] = 0
    h[:, :, -1] = 0
    return h


@pytest.fixture
def phase(rng):
    return rng.uniform(-np.pi, np.pi, (NJ, NF - 2))


def check(analytic_full, func, phase):
    """Compare the free-phase slice of an analytic gradient to numerics."""
    numeric = moments.numerical_gradient(func, phase)
    np.testing.assert_allclose(analytic_full[:, 1:-1], numeric, atol=TOL)


@pytest.mark.parametrize(
    "indices",
    [
        (0, 0),          # variance
        (1, 1, 1),       # skewness
        (2, 2, 2, 2),    # kurtosis
        (0, 1, 2),       # co-skewness
        (0, 0, 1, 1),    # co-kurtosis
        (0, 1, 1, 2),    # mixed 4th order
    ],
)
def test_grad_raw_moment(H, phase, indices):
    u, v, x = moments.uvx(H, phase)
    analytic = moments.grad_raw_moment(H, u, x, indices)
    check(analytic, lambda p: moments.raw_moment(moments.uvx(H, p)[2], indices), phase)


def test_grad_variance_matches_raw(H, phase):
    u, v, x = moments.uvx(H, phase)
    for k in range(NJ):
        fast = moments.grad_variance(H, u, v, k)
        slow = moments.grad_raw_moment(H, u, x, (k, k))
        np.testing.assert_allclose(fast, slow, atol=1e-12)


@pytest.mark.parametrize(
    "indices",
    [
        (1, 1, 1),       # skewness
        (2, 2, 2, 2),    # kurtosis
        (0, 1, 2),       # co-skewness
        (0, 0, 1, 1),    # co-kurtosis
    ],
)
def test_grad_normalized_moment(H, phase, indices):
    u, v, x = moments.uvx(H, phase)
    _, analytic = moments.grad_normalized_moment(H, u, v, x, indices)
    check(
        analytic,
        lambda p: moments.normalized_moment(moments.uvx(H, p)[2], indices),
        phase,
    )


@pytest.mark.parametrize("k", range(NJ))
def test_grad_endpoint_value(H, phase, k):
    u, _, _ = moments.uvx(H, phase)
    analytic = moments.grad_endpoint_value(H, u, k)
    check(
        analytic,
        lambda p: moments.endpoint_value(H, moments.uvx(H, p)[0])[k],
        phase,
    )


@pytest.mark.parametrize("k", range(NJ))
def test_grad_endpoint_slope(H, phase, k):
    u, _, _ = moments.uvx(H, phase)
    analytic = moments.grad_endpoint_slope(H, u, k)
    check(
        analytic,
        lambda p: moments.endpoint_slope(H, moments.uvx(H, p)[0])[k],
        phase,
    )


def test_endpoint_value_matches_signal(H, phase):
    u, _, x = moments.uvx(H, phase)
    np.testing.assert_allclose(moments.endpoint_value(H, u), x[:, 0], atol=1e-12)


@pytest.mark.parametrize("k", range(NJ))
def test_grad_memoryless(H, phase, k):
    """General memoryless identity checked with g(x) = cos(x)."""
    u, _, x = moments.uvx(H, phase)
    analytic = moments.grad_memoryless(H, u, -np.sin(x[k]), k)
    check(
        analytic,
        lambda p: np.mean(np.cos(moments.uvx(H, p)[2][k])),
        phase,
    )


@pytest.mark.parametrize("k", range(NJ))
@pytest.mark.parametrize("beta", [2.0, 20.0])
def test_grad_crest_surrogate(H, phase, k, beta):
    u, v, x = moments.uvx(H, phase)
    val, analytic = moments.grad_crest_surrogate(H, u, v, x, k, beta)
    assert val == pytest.approx(moments.crest_surrogate(x, k, beta))
    check(
        analytic,
        lambda p: moments.crest_surrogate(moments.uvx(H, p)[2], k, beta),
        phase,
    )


def test_crest_surrogate_bounds(H, phase):
    """Surrogate approaches max|x|/std from below as beta grows."""
    _, _, x = moments.uvx(H, phase)
    crest = np.max(np.abs(x[0])) / np.sqrt(np.mean(x[0] ** 2))
    lo = moments.crest_surrogate(x, 0, 5.0)
    hi = moments.crest_surrogate(x, 0, 500.0)
    assert lo < hi < crest
    assert hi == pytest.approx(crest, abs=np.log(2 * x.shape[1]) / 500.0)


def test_grad_full_loss(H, phase):
    problem = SynthesisProblem(
        H,
        targets=[
            MomentTarget((0, 0, 0), 0.5),
            MomentTarget((1, 1, 1, 1), 4.0, weight=2.0),
            MomentTarget((0, 0, 1, 1), 1.5, weight=0.5),
            MomentTarget((0, 1, 2), 0.1),
        ],
        endpoints=[EndpointTarget(k) for k in range(NJ)],
        crests=[CrestTarget(0, beta=10.0, weight=0.7)],
    )
    grad = np.empty_like(phase)
    problem.loss(phase, grad)
    numeric = moments.numerical_gradient(lambda p: problem.loss(p), phase)
    np.testing.assert_allclose(grad, numeric, atol=TOL)


def test_problem_rejects_second_order_target():
    with pytest.raises(ValueError):
        MomentTarget((0, 0), 1.0)


def test_problem_rejects_nonzero_nyquist():
    h = np.ones((1, 1, NF), dtype=complex)
    with pytest.raises(ValueError):
        SynthesisProblem(h)
