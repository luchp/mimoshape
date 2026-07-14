"""Piecewise (multimodel) synthesis and block merging.

A record that fails the stationarity tests (:mod:`mimoshape.stationarity`)
cannot be summarised by one spectral model.  The multimodel approach cuts
the record into equal sections, estimates one model per section (multitaper
CSD -> Cholesky ``H`` plus sample moment targets, see
:mod:`mimoshape.estimate`), synthesises blocks per section, and joins the
blocks with one of three merge strategies:

* ``crossfade`` -- rotate each next block to best match the outgoing tail
  (a circular shift is a pure linear phase: statistically neutral), then
  equal-power cos/sin crossfade,
* ``c1`` -- constrain each block's head value and slope to continue the
  previous block (:class:`mimoshape.shaper.EndpointTarget`), concatenate,
* ``zero`` -- constrain all heads to zero value and slope, concatenate.

All functions are stateless; reporting stays with the caller.
"""

from dataclasses import dataclass

import numpy as np

from . import estimate, moments
from .shaper import EndpointTarget, SynthesisProblem, MimoShaper

MERGE_CHOICES = ("crossfade", "c1", "zero")


def moment_tuples(num_channels, skewness=True, kurtosis=True, coskewness=False, cokurtosis=True):
    """Standard moment index tuples for an ``num_channels``-channel problem.

    Per channel: skewness ``(k, k, k)`` and kurtosis ``(k, k, k, k)``.  Per
    channel pair: co-kurtosis ``(i, i, j, j)`` and the two co-skewnesses
    ``(i, i, j)``, ``(i, j, j)``.
    """
    tuples = []
    for k in range(num_channels):
        if skewness:
            tuples.append((k, k, k))
        if kurtosis:
            tuples.append((k, k, k, k))
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            if cokurtosis:
                tuples.append((i, i, j, j))
            if coskewness:
                tuples.append((i, i, j))
                tuples.append((i, j, j))
    return tuples


@dataclass(frozen=True)
class SectionModel:
    """Spectral model and moment targets of one record section."""

    H: np.ndarray
    targets: list
    variance: np.ndarray  # per-channel section variance (endpoint weighting)


def estimate_section_models(record, num_sections, index_tuples, nw=4.0, nfft=None):
    """One :class:`SectionModel` per equal section of ``record``.

    ``record`` is ``(Nj, N)``; the remainder of ``N // num_sections`` is
    dropped.  Each section is demeaned before estimation.  ``nfft`` defaults
    to the largest power of 2 that fits the section.  Raises
    ``numpy.linalg.LinAlgError`` when a section's CSD estimate is not
    positive definite (too few tapers/averages for the channel count).
    """
    record = np.atleast_2d(np.asarray(record, dtype=float))
    n = record.shape[1]
    sec_len = n // num_sections
    if nfft is None:
        nfft = 2 ** int(np.log2(sec_len))
    if sec_len < nfft:
        raise ValueError(f"section length {sec_len} shorter than nfft {nfft}")
    models = []
    for s in range(num_sections):
        section = record[:, s * sec_len : (s + 1) * sec_len]
        section = section - np.mean(section, axis=1, keepdims=True)
        variance = np.var(section, axis=1)
        if np.any(variance == 0):
            raise ValueError(f"section {s} has a constant channel")
        G = estimate.multitaper_csd(section, nw=nw, nfft=nfft)
        H = estimate.csd_to_frf(G, variance=variance)
        targets = estimate.estimate_moment_targets(section, index_tuples)
        models.append(SectionModel(H, targets, variance))
    return models


def head_state(x):
    """Head value and spectral head slope ``dx/dt(0)`` per channel (unit sample time).

    The slope is the derivative of the trigonometric interpolant at sample 0,
    matching the spectral-derivative definition enforced by
    :class:`mimoshape.shaper.EndpointTarget` (a central difference
    ``(x[1] - x[-1])/2`` underestimates the slope of broadband blocks).
    """
    nt = x.shape[1]
    omega = 2.0 * np.pi * np.fft.rfftfreq(nt)
    slope = np.fft.irfft(1j * omega * np.fft.rfft(x, axis=1), n=nt, axis=1)[:, 0]
    return x[:, 0], slope


def best_shift(tail, block):
    """Circular shift of ``block`` maximising correlation with ``tail``.

    Blocks are periodic, so a rotation is a pure linear phase: it changes
    nothing statistically but lets us splice where the waveforms agree
    (WSOLA-style alignment).  The same shift is applied to all channels,
    preserving cross-spectra and cross-moments.
    """
    nt = block.shape[1]
    padded = np.zeros_like(block)
    padded[:, : tail.shape[1]] = tail
    corr = np.fft.irfft(
        np.conj(np.fft.rfft(padded, axis=1)) * np.fft.rfft(block, axis=1), n=nt, axis=1
    )
    return int(np.argmax(np.sum(corr, axis=0)))


def merge_crossfade(blocks, fade):
    """Join blocks with aligned equal-power crossfades of ``fade`` samples.

    Each next block is rotated to best match the outgoing tail
    (:func:`best_shift`), then faded in with cos/sin weights
    (variance-preserving for uncorrelated signals).  Returns ``(Nj, total)``
    with ``total = sum(len) - (len(blocks) - 1) * fade``.
    """
    theta = np.pi / 2 * (np.arange(fade) + 0.5) / fade
    w_out, w_in = np.cos(theta), np.sin(theta)
    out = blocks[0]
    for block in blocks[1:]:
        shift = best_shift(out[:, -fade:], block)
        rolled = np.roll(block, -shift, axis=1)
        mix = out[:, -fade:] * w_out + rolled[:, :fade] * w_in
        out = np.concatenate([out[:, :-fade], mix, rolled[:, fade:]], axis=1)
    return out


@dataclass
class MultiModelResult:
    """Blocks and joined signal of a multimodel synthesis run."""

    blocks: list  # raw per-section blocks, section-major order
    merged: np.ndarray  # (Nj, ~total), blocks joined with the merge strategy
    models: list  # the SectionModel per section
    achieved: list  # per block: {indices: achieved normalised moment}


def synthesize_multimodel(
    models,
    blocks_per_section=1,
    merge="crossfade",
    fade=None,
    endpoint_weight=10.0,
    max_time=5.0,
    stop_loss=1e-10,
    rng=None,
    progress=None,
):
    """Synthesise and join blocks for a list of :class:`SectionModel`.

    ``merge`` is one of ``crossfade``, ``c1``, ``zero``.  For ``c1`` each
    block's head is constrained to continue the previous block; for ``zero``
    all heads are pinned to zero value and slope.  ``endpoint_weight`` is
    made dimensionless by scaling with 1/variance per channel.  ``fade``
    (crossfade length in samples) defaults to ``Nt // 16``.  ``progress``
    is an optional callable ``progress(done_blocks, total_blocks)``.
    Returns a :class:`MultiModelResult`.
    """
    if merge not in MERGE_CHOICES:
        raise ValueError(f"merge must be one of {MERGE_CHOICES}, got {merge!r}")
    rng = np.random.default_rng() if rng is None else rng
    total = len(models) * blocks_per_section

    blocks = []
    achieved = []
    prev = None
    for model in models:
        nj = model.H.shape[0]
        ep_w = endpoint_weight / np.maximum(model.variance, 1e-30)
        for _ in range(blocks_per_section):
            if merge == "zero":
                endpoints = [EndpointTarget(k, 0.0, 0.0, ep_w[k], ep_w[k]) for k in range(nj)]
            elif merge == "c1" and prev is not None:
                # the periodic continuation of the previous block ends at its
                # own head state: match it for a C1 joint
                head, slope = head_state(prev)
                endpoints = [
                    EndpointTarget(k, head[k], slope[k], ep_w[k], ep_w[k]) for k in range(nj)
                ]
            else:
                endpoints = []
            problem = SynthesisProblem(model.H, targets=model.targets, endpoints=endpoints)
            shaper = MimoShaper(problem, max_time=max_time, stop_loss=stop_loss, rng=rng)
            prev = shaper.make_block()
            blocks.append(prev)
            achieved.append(
                {t.indices: float(moments.normalized_moment(prev, t.indices)) for t in model.targets}
            )
            if progress is not None:
                progress(len(blocks), total)

    if merge == "crossfade" and len(blocks) > 1:
        merged = merge_crossfade(blocks, blocks[0].shape[1] // 16 if fade is None else fade)
    else:
        merged = np.hstack(blocks)
    return MultiModelResult(blocks, merged, list(models), achieved)
