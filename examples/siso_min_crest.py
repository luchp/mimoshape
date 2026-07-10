"""SISO demo: minimum crest factor via logcosh surrogate with beta continuation.

Optimises a flat-spectrum block for minimum peak/std by minimising the smooth
surrogate (1/beta) log mean cosh(beta x/std), doubling beta each stage and
warm-starting from the previous optimum.  The result is a constant-envelope,
noise-like waveform (no chirp structure appears in the spectrogram).

Run with:  uv run --extra examples python examples/siso_min_crest.py
"""

import numpy as np
import matplotlib.pyplot as plt

from mimoshape import CrestTarget, SynthesisProblem, MimoShaper


def crest(x):
    return np.max(np.abs(x)) / np.sqrt(np.mean(x**2))


def main():
    nt = 2**12
    nf = nt // 2 + 1
    H = np.zeros((1, 1, nf), dtype=complex)
    H[0, 0, 1:-1] = 1.0

    rng = np.random.default_rng(0)
    start = None
    for beta in (5, 10, 20, 40, 80, 160):
        problem = SynthesisProblem(H, crests=[CrestTarget(0, beta=beta)])
        shaper = MimoShaper(problem, max_time=20, ftol_rel=1e-7, rng=rng)
        x = shaper.make_block(start=start)
        start = shaper.last_phase
        print(f"beta {beta:>4d}: crest {crest(x[0]):.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
    axes[0].plot(x[0], linewidth=0.4)
    axes[0].set_title(f"minimum-crest block, crest = {crest(x[0]):.3f} (sine = 1.414)")
    axes[0].set_xlabel("sample")
    axes[0].grid(alpha=0.4)
    axes[1].specgram(x[0], NFFT=256, Fs=1.0, noverlap=192)
    axes[1].set_title("spectrogram: constant-envelope, noise-like (no chirp structure)")
    axes[1].set_xlabel("sample")
    axes[1].set_ylabel(r"$f/f_s$")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
