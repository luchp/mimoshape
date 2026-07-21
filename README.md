# mimoshape

Pure-NumPy optimization engine for the phase-domain synthesis of multi-channel (MIMO) signals
that match a prescribed Cross-Spectral Density (CSD) matrix while simultaneously optimizing 
an arbitrary ensemble of user-definable, smooth memoryless functionals.

ThThe CSD is enforced structurally through a Cholesky factor `H`; the remaining
phase degrees of freedom are optimised with analytic gradients (scipy's
L-BFGS-B). See `papers/26293/sss.tex` for the full derivation.

Specific targets are included for higher-order diagonal and joint moments (skewness, kurtosis, co-skewness, co-kurtosis), and 
directly minimised functionals such as a smooth crest-factor surrogate.

There are two live web applications where you try it yourself: [Crest minimizer](https://sensemagic.nl/app_mimoshape) and [MIMO synthesizer](https://sensemagic.nl/app_mimoshape_file)

## Layout

- `src/mimoshape/moments.py` — pure numerics: signals, moments, analytic gradients
- `src/mimoshape/shaper.py` — target set, loss assembly, scipy L-BFGS-B wiring
- `src/mimoshape/estimate.py` — targets from measured records: multitaper CSD → Cholesky `H`, sample moments
- `src/mimoshape/stationarity.py` — segment-statistic stationarity tests (reverse arrangements, runs)
- `src/mimoshape/multimodel.py` — piecewise synthesis for non-stationary records: per-section models, block merging (crossfade / C1 / zero)
- `tests/` — analytic-vs-numerical gradient checks
- `examples/` — runnable demos
- `scripts/figures` — regenerates every figure and table in the papers, pass in -p paperid, where paperid is the directory in papers/
  (fixed seeds): `figures -p 26293`
- `papers/` — LaTeX source of the papers

## Quick start

```python
import numpy as np
from mimoshape import MomentTarget, EndpointTarget, SynthesisProblem, MimoShaper

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

Minimum crest factor (peak/std) via the smooth logcosh surrogate with beta
continuation — see `examples/siso_min_crest.py`:

```python
from mimoshape import CrestTarget

start = None
for beta in (5, 10, 20, 40, 80, 160):
    shaper = MimoShaper(SynthesisProblem(H, crests=[CrestTarget(0, beta=beta)]))
    x = shaper.make_block(start=start)
    start = shaper.last_phase
```

## Development

```
uv sync --group dev
uv run pytest
```
## Generate figures and tables from paper with id 26293

```
scripts\figures -p 26293
```

## License

This reference implementation is released under the **MIT License**. The underlying paper is licensed under **CC BY 4.0**.

--- 

### Commercial C Implementation

A high-performance C implementation, with zero dependencies, zero dynamic memory allocations, and optimized for real-time/embedded deployment, is available under a commercial license. 

For licensing inquiries or evaluation, please contact luc@sensemagic.nl

## How to Cite

See CITATION.cff 


