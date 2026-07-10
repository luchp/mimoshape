"""MIMO demo: estimate CSD + moment targets from a record, resynthesise.

Run with:  uv run --extra examples python examples/mimo_from_record.py
"""

import numpy as np
import matplotlib.pyplot as plt

from synthsig import estimate, moments
from synthsig.shaper import SynthesisProblem, MimoShaper


def make_record(rng, n):
    """Surrogate 'measured' record: two correlated, non-Gaussian channels."""
    base = rng.standard_normal(n)
    y0 = base + 0.3 * rng.standard_normal(n)
    y0 = y0 + 0.4 * y0**2 - np.mean(0.4 * y0**2)
    y1 = 0.7 * base + 0.5 * rng.standard_normal(n)
    return np.vstack([y0, y1])


def main():
    rng = np.random.default_rng(3)
    nfft = 1024
    record = make_record(rng, 16 * nfft)

    G = estimate.multitaper_csd(record, nw=4.0, nfft=nfft)
    H = estimate.csd_to_frf(G, variance=np.var(record, axis=1))
    tuples = [(0, 0, 0), (1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 1, 1)]
    targets = estimate.estimate_moment_targets(record, tuples)

    problem = SynthesisProblem(H, targets=targets)
    x = MimoShaper(problem, max_time=60, rng=rng).make_block()

    print(f"{'tuple':>14} {'target':>8} {'achieved':>9}")
    for t in targets:
        print(f"{str(t.indices):>14} {t.value:8.3f} {moments.normalized_moment(x, t.indices):9.3f}")

    fig, axes = plt.subplots(2, 1, sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(x[k])
        ax.set_ylabel(f"channel {k}")
        ax.grid()
    fig.suptitle("Synthesised MIMO block from measured targets")
    plt.show()


if __name__ == "__main__":
    main()
