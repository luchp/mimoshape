"""Pure numerics for phase-domain moment shaping.

Notation follows the paper (paper/sss.tex):

* ``H`` -- complex frequency response, shape ``(Nj, Nj, Nf)`` with
  ``Nf = Nt//2 + 1`` rfft bins.  DC and Nyquist bins must be zero.
* ``phase`` -- free phases ``psi``, shape ``(Nj, Nf - 2)`` (bins 1..Nf-2).
* ``u`` -- unit-modulus source spectrum, shape ``(Nj, Nf)``.
* ``v`` -- shaped spectrum ``H @ u``, shape ``(Nj, Nf)``.
* ``x`` -- time signal ``irfft(v)``, shape ``(Nj, Nt)``.

Gradients are with respect to the *full* phase grid ``(Nj, Nf)``; callers
slice ``[:, 1:-1]`` to obtain the free-phase gradient.

All functions are pure: no I/O, no state, no optimiser dependencies.
"""

import numpy as np


def uvx(H, phase):
    """Source spectrum, shaped spectrum and time signal for the given phases.

    Returns ``(u, v, x)`` as defined in the module docstring.
    """
    nj, nfree = phase.shape
    u = np.zeros((nj, nfree + 2), dtype=complex)
    u[:, 1:-1] = np.exp(1j * phase)
    v = np.einsum("kjf,jf->kf", H, u)
    x = np.fft.irfft(v, axis=1)
    return u, v, x


def raw_moment(x, indices):
    """Raw joint moment ``P_i = mean_t prod_a x[i_a, t]`` for a channel tuple."""
    prod = np.ones(x.shape[1])
    for i in indices:
        prod = prod * x[i]
    return np.mean(prod)


def grad_raw_moment(H, u, x, indices):
    """Phase gradient of the raw joint moment ``P_i``.

    Implements the key observation of the paper:
    ``dP_i/dpsi_qg = (2/Nt^2) im( u* sum_a H[i_a]* F_g[prod_{b!=a} x_b] )``.
    Repeated indices share their FFT.  Returns shape ``(Nj, Nf)``.
    """
    nt = x.shape[1]
    total = np.zeros_like(u)
    for a, count in _index_counts(indices):
        remaining = list(indices)
        remaining.remove(a)  # one factor of channel a removed
        partial = np.ones(nt)
        for b in remaining:
            partial = partial * x[b]
        total += count * np.conj(H[a]) * np.fft.rfft(partial)
    return (2.0 / nt**2) * np.imag(np.conj(u) * total)


def grad_variance(H, u, v, k):
    """Phase gradient of the variance ``P_(k,k)``; free given ``v``.

    ``dP_(k,k)/dpsi_qg = (4/Nt^2) im( u* H[k]* v[k] )``.
    """
    nt = 2 * (u.shape[1] - 1)
    return (4.0 / nt**2) * np.imag(np.conj(u) * np.conj(H[k]) * v[k])


def normalized_moment(x, indices):
    """Normalised joint moment ``M_i = P_i / prod_a sqrt(P_(i_a,i_a))``."""
    p = raw_moment(x, indices)
    scale = 1.0
    for i in indices:
        scale *= np.sqrt(raw_moment(x, (i, i)))
    return p / scale


def grad_normalized_moment(H, u, v, x, indices):
    """Phase gradient of ``M_i`` via the chain rule on the normalisation.

    Returns ``(M_i, gradient)`` with gradient shape ``(Nj, Nf)``.
    """
    p = raw_moment(x, indices)
    dp = grad_raw_moment(H, u, x, indices)
    scale = 1.0
    correction = np.zeros_like(dp)
    for i in set(indices):
        m = indices.count(i)
        p2 = raw_moment(x, (i, i))
        scale *= p2 ** (0.5 * m)
        correction += (0.5 * m / p2) * grad_variance(H, u, v, i)
    return p / scale, (dp - p * correction) / scale


def endpoint_value(H, u):
    """Head value ``x_k(0)`` per channel, shape ``(Nj,)``."""
    nt = 2 * (u.shape[1] - 1)
    return (2.0 / nt) * np.einsum("kjf,jf->k", H.real, u.real) - (
        2.0 / nt
    ) * np.einsum("kjf,jf->k", H.imag, u.imag)


def grad_endpoint_value(H, u, k):
    """Phase gradient of ``x_k(0)``: ``-(2/Nt) im(H[k] u)``, shape ``(Nj, Nf)``."""
    nt = 2 * (u.shape[1] - 1)
    return -(2.0 / nt) * np.imag(H[k] * u)


def endpoint_slope(H, u):
    """Head slope ``dx_k/dt (0)`` per channel (unit sample time), shape ``(Nj,)``."""
    nt = 2 * (u.shape[1] - 1)
    omega = 2.0 * np.pi * np.arange(u.shape[1]) / nt
    return -(2.0 / nt) * np.einsum("f,kf->k", omega, np.imag(np.einsum("kjf,jf->kf", H, u)))


def grad_endpoint_slope(H, u, k):
    """Phase gradient of the head slope: ``-(2 w_g/Nt) re(H[k] u)``."""
    nt = 2 * (u.shape[1] - 1)
    omega = 2.0 * np.pi * np.arange(u.shape[1]) / nt
    return -(2.0 / nt) * omega * np.real(H[k] * u)


def _index_counts(indices):
    """Distinct channels of a tuple with multiplicities, preserving order."""
    seen = {}
    for i in indices:
        seen[i] = seen.get(i, 0) + 1
    return list(seen.items())


def numerical_gradient(func, phase, h=1e-6):
    """Central-difference gradient of a scalar ``func(phase)``.

    Slow and for testing only: verifies the analytic gradients above.
    """
    grad = np.zeros_like(phase)
    for idx in np.ndindex(phase.shape):
        p0 = phase[idx]
        phase[idx] = p0 - h
        y1 = func(phase)
        phase[idx] = p0 + h
        y2 = func(phase)
        phase[idx] = p0
        grad[idx] = (y2 - y1) / (2.0 * h)
    return grad
