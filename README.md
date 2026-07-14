# mimoshape

Pure-NumPy optimization engine for the phase-domain synthesis of multi-channel (MIMO) signals
that match a prescribed Cross-Spectral Density (CSD) matrix while simultaneously optimizing 
an arbitrary ensemble of user-definable, smooth memoryless functionals.

The CSD is enforced structurally through a Cholesky factor `H`; the remaining
phase degrees of freedom are optimised with analytic gradients (CCSAQ from
NLopt). See `paper/sss.tex` for the full derivation.

Specific targets are included for higher-order diagonal and joint moments (skewness, kurtosis, co-skewness, co-kurtosis), and 
directly minimised functionals such as a smooth crest-factor surrogate.

There are two live web applications where you try it yourself [Crest example](https://sensemagic.nl/app_mimoshape) [MIMO synthesis](https://sensemagic.nl/app_mimoshape_file)

## Layout

- `src/mimoshape/moments.py` — pure numerics: signals, moments, analytic gradients
- `src/mimoshape/shaper.py` — target set, loss assembly, NLopt wiring
- `src/mimoshape/estimate.py` — targets from measured records: multitaper CSD → Cholesky `H`, sample moments
- `src/mimoshape/stationarity.py` — segment-statistic stationarity tests (reverse arrangements, runs)
- `src/mimoshape/multimodel.py` — piecewise synthesis for non-stationary records: per-section models, block merging (crossfade / C1 / zero)
- `tests/` — analytic-vs-numerical gradient checks
- `examples/` — runnable demos
- `scripts/make_figures.py` — regenerates every figure and table in the paper
  (fixed seeds): `uv run --extra examples python scripts/make_figures.py`
- `paper/` — LaTeX source of the paper

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

## License

MIT (code). The paper is licensed CC BY 4.0.

## 📖 How to Cite

If you use this framework or the `mimoshape` optimization engine in your research, please cite the published journal article:

## 📖 How to Cite

If you use this framework, the `mimoshape` repository, or the mathematical derivations in your research or software, please cite the companion paper:

| Type | Reference |
| :--- | :--- |
| **APA** | Holtkamp, L. (2026). *Synthetic Signal Shaping: Analytic-gradient phase optimization for a target cross-spectral density and higher-order moments and cross-moments*. SenseMagic, Netherlands. |
| **BibTeX** | See the code block below for the canonical database entry. |

```bibtex
@techreport{holtkamp2026synthetic,
  author      = {Holtkamp, Luc},
  title       = {Synthetic Signal Shaping: Analytic-gradient Phase Optimization over Cross-Spectral Density Constraints and Arbitrary Memoryless Functionals},
  institution = {SenseMagic},
  address     = {Netherlands},
  year        = {2026},
  note        = {Source code available at \url{https://github.com/luchp/mimoshape}},
  url         = {https://github.com/luchp/mimoshape}
}
```

