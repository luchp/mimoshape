"""MIMO phase-shaping synthesiser.

``SynthesisProblem`` assembles the loss and analytic gradient from a target
set (pure numerics, testable without an optimiser).  ``MimoShaper`` wires the
problem into scipy's L-BFGS-B and produces signal blocks.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from . import moments


@dataclass(frozen=True)
class MomentTarget:
    """Target for a normalised joint moment ``M_i``.

    ``indices`` is the channel tuple (length >= 3), e.g. ``(k, k, k)`` for
    skewness of channel k, ``(i, i, j, j)`` for the pair co-kurtosis.
    """

    indices: tuple
    value: float
    weight: float = 1.0

    def __post_init__(self):
        if len(self.indices) < 3:
            raise ValueError(
                f"Moment tuple {self.indices} has order {len(self.indices)}; "
                "order >= 3 required (second order is fixed by the CSD)"
            )


@dataclass(frozen=True)
class FunctionTarget:
    """Memoryless functional ``Q = mean_t g(x_k[t])`` for one channel.

    ``g`` and ``gprime`` are vectorised pointwise callables (the setting of
    Proposition 1).  With ``value`` set the loss gains the moment-style term
    ``0.5 weight (Q - value)^2``; with ``value=None`` the loss gains
    ``weight * Q`` and the functional is minimised directly.
    """

    channel: int
    g: callable
    gprime: callable
    value: float = None
    weight: float = 1.0


@dataclass(frozen=True)
class ScaledFunctionTarget(FunctionTarget):
    """``FunctionTarget`` on the std-normalised signal ``z = x_k / std(x_k)``.

    Scale-invariant functionals live here: ``g(z) = z**4`` matches the
    normalised kurtosis, ``g(z) = |z|**p`` is the l_p crest objective of
    Guillaume et al.; the variance chain rule is handled analytically.
    """


@dataclass(frozen=True)
class CrestTarget:
    """Directly minimised smooth crest surrogate for one channel.

    Adds ``weight * (1/beta) log mean cosh(beta x_k / std)`` to the loss --
    a smooth stand-in for ``max|x_k|/std`` with bias ``~log(2 Nt)/beta``.
    For low crest factors run a beta continuation: optimise at a small
    ``beta``, then warm-start successively doubled betas via
    ``MimoShaper.make_block(start=...)``.
    """

    channel: int
    beta: float = 20.0
    weight: float = 1.0

    def __post_init__(self):
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")


@dataclass(frozen=True)
class EndpointTarget:
    """Head value and slope constraint for channel ``k`` (C1 block splicing)."""

    channel: int
    value: float = 0.0
    slope: float = 0.0
    value_weight: float = 1.0
    slope_weight: float = 1.0


@dataclass
class SynthesisProblem:
    """Loss and analytic gradient for a target set under a fixed ``H``.

    ``H`` has shape ``(Nj, Nj, Nf)`` with zero DC and Nyquist bins, typically
    a Cholesky factor of the target CSD per frequency bin.
    """

    H: np.ndarray
    targets: list = field(default_factory=list)
    endpoints: list = field(default_factory=list)
    crests: list = field(default_factory=list)
    functions: list = field(default_factory=list)

    def __post_init__(self):
        self.H = np.asarray(self.H, dtype=complex)
        if self.H.ndim != 3 or self.H.shape[0] != self.H.shape[1]:
            raise ValueError(f"H must be (Nj, Nj, Nf), got {self.H.shape}")
        if np.any(self.H[:, :, 0] != 0) or np.any(self.H[:, :, -1] != 0):
            raise ValueError("H must have zero DC and Nyquist bins")
        nt = 2 * (self.H.shape[2] - 1)
        if nt < 8:
            raise ValueError(f"Block length {nt} must be >= 8")
        nj = self.H.shape[0]
        for t in self.targets:
            if any(not 0 <= i < nj for i in t.indices):
                raise ValueError(f"Target {t.indices} out of range for {nj} channels")
        for e in self.endpoints:
            if not 0 <= e.channel < nj:
                raise ValueError(f"Endpoint channel {e.channel} out of range")
        for c in self.crests:
            if not 0 <= c.channel < nj:
                raise ValueError(f"Crest channel {c.channel} out of range")
        for f in self.functions:
            if not 0 <= f.channel < nj:
                raise ValueError(f"Function channel {f.channel} out of range")

    @property
    def num_channels(self):
        return self.H.shape[0]

    @property
    def num_free_phases(self):
        return self.H.shape[0] * (self.H.shape[2] - 2)

    def loss(self, phase, grad_out=None):
        """Weighted squared-error loss; fills ``grad_out`` (same shape as
        ``phase``) with the analytic gradient when provided."""
        u, v, x = moments.uvx(self.H, phase)
        want_grad = grad_out is not None
        total = 0.0
        grad = np.zeros_like(u, dtype=float) if want_grad else None

        for t in self.targets:
            if want_grad:
                m, dm = moments.grad_normalized_moment(self.H, u, v, x, t.indices)
                grad += (t.weight * (m - t.value)) * dm
            else:
                m = moments.normalized_moment(x, t.indices)
            total += 0.5 * t.weight * (m - t.value) ** 2

        for c in self.crests:
            if want_grad:
                val, dval = moments.grad_crest_surrogate(
                    self.H, u, v, x, c.channel, c.beta
                )
                grad += c.weight * dval
            else:
                val = moments.crest_surrogate(x, c.channel, c.beta)
            total += c.weight * val

        for f in self.functions:
            scaled = isinstance(f, ScaledFunctionTarget)
            arg = x[f.channel]
            if scaled:
                arg = arg / np.sqrt(np.mean(arg**2))
            q = np.mean(f.g(arg))
            if f.value is None:
                total += f.weight * q
                factor = f.weight
            else:
                total += 0.5 * f.weight * (q - f.value) ** 2
                factor = f.weight * (q - f.value)
            if want_grad:
                if scaled:
                    dq = moments.grad_scaled_memoryless(
                        self.H, u, v, x, f.channel, f.gprime
                    )
                else:
                    dq = moments.grad_memoryless(
                        self.H, u, f.gprime(arg), f.channel
                    )
                grad += factor * dq

        if self.endpoints:
            head = moments.endpoint_value(self.H, u)
            slope = moments.endpoint_slope(self.H, u)
            for e in self.endpoints:
                err0 = head[e.channel] - e.value
                err1 = slope[e.channel] - e.slope
                total += 0.5 * (e.value_weight * err0**2 + e.slope_weight * err1**2)
                if want_grad:
                    grad += (e.value_weight * err0) * moments.grad_endpoint_value(
                        self.H, u, e.channel
                    )
                    grad += (e.slope_weight * err1) * moments.grad_endpoint_slope(
                        self.H, u, e.channel
                    )

        if want_grad:
            grad_out[:] = grad[:, 1:-1]
        return total

    def signal(self, phase):
        """Time signal ``x`` for the given free phases, shape ``(Nj, Nt)``."""
        return moments.uvx(self.H, phase)[2]


class _StopEarly(Exception):
    """Raised from the objective to unwind ``scipy.optimize.minimize`` once
    the loss threshold, progress callback, or wall-clock budget fires."""


class MimoShaper:
    """Optimises the free phases of a ``SynthesisProblem`` with L-BFGS-B.

    ``progress`` is an optional callable ``progress(loss) -> bool``; returning
    True stops the optimisation.  Reporting stays out of the core numerics.
    """

    def __init__(
        self,
        problem,
        progress=None,
        max_time=60.0,
        stop_loss=1e-4,
        ftol_rel=1e-5,
        xtol_rel=1e-5,
        rng=None,
    ):
        self.problem = problem
        self.progress = progress
        self.max_time = max_time
        self.stop_loss = stop_loss
        self.ftol_rel = ftol_rel
        self.xtol_rel = xtol_rel
        self.rng = np.random.default_rng() if rng is None else rng
        self.last_result = None
        self.last_phase = None
        self._last_x = None
        self._deadline = None

    def _objective(self, flat_phase):
        self._last_x = flat_phase
        phase = flat_phase.reshape(self.problem.num_channels, -1)
        grad = np.empty_like(phase)
        loss = self.problem.loss(phase, grad)
        if self.progress is not None and self.progress(loss):
            raise _StopEarly()
        if loss < self.stop_loss:
            raise _StopEarly()
        if self._deadline is not None and time.perf_counter() > self._deadline:
            raise _StopEarly()
        return loss, grad.ravel()

    def make_block(self, start=None):
        """Optimise the free phases and return the time signal ``x`` of shape
        ``(Nj, Nt)``.

        ``start`` warm-starts from given free phases (flat or ``(Nj, Nf-2)``),
        e.g. ``shaper.last_phase`` from a previous stage of a beta
        continuation; by default a fresh random phase is drawn.
        """
        n = self.problem.num_free_phases
        if start is None:
            start = self.rng.uniform(-np.pi, np.pi, n)
        else:
            start = np.asarray(start, dtype=float).reshape(n)

        self._deadline = (
            None if self.max_time is None else time.perf_counter() + self.max_time
        )
        self._last_x = start
        # ftol_rel maps directly onto scipy's relative function-value
        # tolerance; xtol_rel has no exact scipy analogue for L-BFGS-B, so it
        # is mapped onto the gradient-norm tolerance gtol (xtol_rel<=0, as
        # used throughout this codebase to mean "disabled", falls back to
        # scipy's default gtol).
        options = {
            "ftol": self.ftol_rel,
            "gtol": self.xtol_rel if self.xtol_rel > 0 else 1e-10,
            "maxiter": 1_000_000,
            "maxfun": 2_000_000,
        }
        try:
            result = minimize(
                self._objective, start, jac=True, method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n, options=options,
            )
            flat = result.x
            self.last_result = result.status
        except _StopEarly:
            flat = self._last_x
            self.last_result = 0

        phase = flat.reshape(self.problem.num_channels, -1)
        self.last_phase = phase
        return self.problem.signal(phase)
