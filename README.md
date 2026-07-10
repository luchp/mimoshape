# synthsig

Phase-domain synthesis of multi-channel (MIMO) signals that match a prescribed
cross-spectral density (CSD) together with user-selected higher-order diagonal
and joint moments (skewness, kurtosis, co-skewness, co-kurtosis).

The CSD is enforced structurally through a Cholesky factor `H`; the remaining
phase degrees of freedom are optimised with analytic gradients (CCSAQ from
NLopt). See `paper/sss.tex` for the full derivation.

## Layout

- `src/synthsig/moments.py` — pure numerics: signals, moments, analytic gradients
- `src/synthsig/shaper.py` — target set, loss assembly, NLopt wiring
- `src/synthsig/estimate.py` — targets from measured records: multitaper CSD → Cholesky `H`, sample moments
- `tests/` — analytic-vs-numerical gradient checks
- `examples/` — runnable demos
- `scripts/make_figures.py` — regenerates every figure and table in the paper
  (fixed seeds): `uv run --extra examples python scripts/make_figures.py`
- `paper/` — LaTeX source of the paper

## Quick start

```python
import numpy as np
from synthsig import MomentTarget, EndpointTarget, SynthesisProblem, MimoShaper

nt = 4096
nf = nt // 2 + 1
H = np.zeros((1, 1, nf), dtype=complex)
H[0, 0, 1:-1] = 1.0  # flat spectrum, zero DC and Nyquist

problem = SynthesisProblem(
    H,
    targets=[MomentTarget((0, 0, 0), 0.0), MomentTarget((0, 0, 0, 0), 4.0)],
    endpoints=[EndpointTarget(0)],
)
x = MimoShaper(problem).make_block()  # shape (1, nt)
```

## Development

```
uv sync --group dev
uv run pytest
```

## License

MIT (code). The paper is licensed CC BY 4.0.
