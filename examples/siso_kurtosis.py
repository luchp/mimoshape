"""SISO demo: flat spectrum, shaped kurtosis, zero endpoints.

Run with:  uv run --extra examples python examples/siso_kurtosis.py
"""

import numpy as np
import matplotlib.pyplot as plt

from mimoshape import MomentTarget, EndpointTarget, SynthesisProblem, MimoShaper


def stats(x):
    std = np.sqrt(np.mean(x**2))
    return {
        "std": std,
        "skewness": np.mean(x**3) / std**3,
        "kurtosis": np.mean(x**4) / std**4,
        "crest": np.max(np.abs(x)) / std,
    }


def main():
    nt = 2**12
    nf = nt // 2 + 1
    H = np.zeros((1, 1, nf), dtype=complex)
    H[0, 0, 1:-1] = 1.0

    problem = SynthesisProblem(
        H,
        targets=[
            MomentTarget((0, 0, 0), 0.0),      # zero skewness
            MomentTarget((0, 0, 0, 0), 5.0),   # heavy-tailed kurtosis
        ],
        endpoints=[EndpointTarget(0)],          # start/end at rest
    )
    def report(loss):
        print(f"loss {loss:.3e}")
        return False

    shaper = MimoShaper(problem, progress=report)
    x = shaper.make_block()

    print({k: round(v, 3) for k, v in stats(x[0]).items()})
    plt.plot(x[0])
    plt.grid()
    plt.title("SISO block, kurtosis target 5.0")
    plt.show()


if __name__ == "__main__":
    main()
