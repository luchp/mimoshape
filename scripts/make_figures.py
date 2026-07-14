"""Regenerate all figures and tables for the paper.

Every number in the paper's examples section comes from this script, with
fixed seeds, so reviewers can reproduce them:

    uv run --extra examples python scripts/make_figures.py

Outputs land in paper/figures/*.pdf and paper/tables/*.tex.
"""

import pathlib
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nlopt
from scipy.optimize import minimize

from mimoshape import estimate, moments, multimodel, stationarity
from mimoshape.shaper import (
    MomentTarget,
    EndpointTarget,
    CrestTarget,
    ScaledFunctionTarget,
    SynthesisProblem,
    MimoShaper,
)

PAPER = pathlib.Path(__file__).resolve().parent.parent / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
ROAD_NPZ = pathlib.Path(__file__).resolve().parent.parent / "data" / "roadsection_220s_300hz.npz"

MIMO_TUPLES = [(0, 0, 0), (1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 1, 1)]
TUPLE_LABELS = {
    (0, 0, 0): r"skewness ch.\,0",
    (1, 1, 1): r"skewness ch.\,1",
    (0, 0, 0, 0): r"kurtosis ch.\,0",
    (1, 1, 1, 1): r"kurtosis ch.\,1",
    (0, 0, 1, 1): r"co-kurtosis $(0,0,1,1)$",
}


def flat_siso_problem(nt, kurtosis, endpoint=True):
    nf = nt // 2 + 1
    H = np.zeros((1, 1, nf), dtype=complex)
    H[0, 0, 1:-1] = 1.0
    return SynthesisProblem(
        H,
        targets=[MomentTarget((0, 0, 0), 0.0), MomentTarget((0, 0, 0, 0), kurtosis)],
        endpoints=[EndpointTarget(0)] if endpoint else [],
    )


def make_record(rng, n):
    """Surrogate measured record: two correlated, coloured, non-Gaussian channels.

    Channel 0 is low-frequency weighted; channel 1 adds a resonance, so the
    PSD, coherence and cross-phase plots all have visible structure.
    """
    base = rng.standard_normal(n)
    ff = np.fft.rfftfreq(n)
    lowpass = 1.0 / (1.0 + (ff / 0.08) ** 2)
    resonance = 1.0 / np.abs(1.0 + 2j * 0.05 * (ff / 0.2) - (ff / 0.2) ** 2)

    def colour(sig, mag):
        return np.fft.irfft(np.fft.rfft(sig) * mag, n)

    y0 = colour(base, lowpass) + 0.3 * colour(rng.standard_normal(n), lowpass)
    y0 = y0 + 4.0 * y0**2 - np.mean(4.0 * y0**2)  # skewed, heavy-tailed
    y1 = 0.7 * colour(base, resonance) + 0.5 * colour(
        rng.standard_normal(n), resonance
    )
    return np.vstack([y0, y1])


def mimo_problem(rng, nfft=1024):
    record = make_record(rng, 16 * nfft)
    G = estimate.multitaper_csd(record, nw=4.0, nfft=nfft)
    H = estimate.csd_to_frf(G, variance=np.var(record, axis=1))
    targets = estimate.estimate_moment_targets(record, MIMO_TUPLES)
    return SynthesisProblem(H, targets=targets), G


def fig_siso_block():
    """Random-phase (Gaussian) vs kurtosis-shaped block, same spectrum."""
    rng = np.random.default_rng(11)
    nt = 2**12
    problem = flat_siso_problem(nt, kurtosis=5.0)
    x_shaped = MimoShaper(problem, max_time=60, rng=rng).make_block()[0]
    phase = rng.uniform(-np.pi, np.pi, (1, nt // 2 - 1))
    x_random = problem.signal(phase)[0]

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(7, 4))
    for ax, x, name in [
        (axes[0], x_random, "random phase"),
        (axes[1], x_shaped, "shaped phase"),
    ]:
        kurt = moments.normalized_moment(x[None, :], (0, 0, 0, 0))
        ax.plot(x, linewidth=0.4)
        ax.set_ylabel(f"{name}\nkurtosis {kurt:.2f}")
        ax.grid(alpha=0.4)
    axes[1].set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(FIGURES / "siso_block.pdf")
    plt.close(fig)


def fig_convergence():
    """Loss vs objective evaluation: feasible vs infeasible kurtosis target.

    The infeasible target (kurtosis 1.0, below the achievable floor) shows
    L-BFGS-B descending smoothly onto the feasibility boundary; the residual
    loss measures the infeasibility gap.  The kurtosis actually reached by
    the infeasible run is the empirical floor for this spectrum and is
    exported as the ``\\kurtfloor`` macro used by the paper.
    """
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for kurtosis, style, label in [
        (5.0, "C0", "feasible target (kurtosis 5.0)"),
        (1.0, "C1", "infeasible target (kurtosis 1.0)"),
    ]:
        rng = np.random.default_rng(12)
        problem = flat_siso_problem(2**12, kurtosis=kurtosis)
        losses = []

        def record_loss(loss):
            losses.append(loss)
            return False

        shaper = MimoShaper(
            problem, progress=record_loss,
            max_time=15, stop_loss=1e-10, ftol_rel=1e-12, xtol_rel=1e-12, rng=rng,
        )
        shaper.make_block()
        ax.semilogy(losses, style, linewidth=0.8, label=label)
        if kurtosis == 5.0:
            (TABLES / "convergence_stats.tex").write_text(
                f"\\newcommand{{\\convergenceevals}}{{{len(losses)}}}\n"
            )
        if kurtosis == 1.0:
            x = problem.signal(shaper.last_phase)
            floor = moments.normalized_moment(x, (0, 0, 0, 0))
            (TABLES / "kurt_floor.tex").write_text(
                f"\\newcommand{{\\kurtfloor}}{{{floor:.2f}}}\n"
            )
    ax.set_xlabel("objective evaluation")
    ax.set_ylabel(r"loss $\Xi$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.pdf")
    plt.close(fig)


def fig_restarts():
    """Distribution of the final loss over many random restarts.

    Checks for local-minimum trapping of the phase optimisation: the SISO
    kurtosis problem is re-optimised from 64 independent random starts.
    """
    problem = flat_siso_problem(2**12, kurtosis=5.0)
    finals = []
    for seed in range(64):
        shaper = MimoShaper(
            problem, max_time=5, stop_loss=1e-10, ftol_rel=1e-9,
            rng=np.random.default_rng(100 + seed),
        )
        shaper.make_block()
        finals.append(problem.loss(shaper.last_phase))
    finals = np.array(finals)

    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(np.log10(finals), bins=24, color="C0", alpha=0.8)
    ax.set_xlabel(r"$\log_{10}$ final loss $\Xi$")
    ax.set_ylabel("restarts")
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "restarts.pdf")
    plt.close(fig)
    (TABLES / "restart_stats.tex").write_text(
        f"median $10^{{{np.median(np.log10(finals)):.1f}}}$, "
        f"worst $10^{{{np.max(np.log10(finals)):.1f}}}$\n"
    )


def gaussian_scaling_problem(nj, nt, rng):
    """``nj``-channel problem with a flat partially coherent CSD and
    Gaussian-consistent targets: skewness 0 and kurtosis 3 per channel plus
    every pair co-kurtosis at its jointly Gaussian value ``1 + 2 rho_ij^2``.

    Feasible by the central limit theorem, so the timing rows measure
    convergence to a realisable target set at growing channel count.
    """
    nf = nt // 2 + 1
    mix = np.eye(nj) + 0.5 * np.tril(rng.standard_normal((nj, nj)), -1)
    H = np.zeros((nj, nj, nf), dtype=complex)
    H[:, :, 1:-1] = mix[:, :, None]
    cov = mix @ mix.T
    rho = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    targets = []
    for k in range(nj):
        targets.append(MomentTarget((k, k, k), 0.0))
        targets.append(MomentTarget((k, k, k, k), 3.0))
    for i in range(nj):
        for j in range(i + 1, nj):
            targets.append(MomentTarget((i, i, j, j), 1.0 + 2.0 * rho[i, j] ** 2))
    return SynthesisProblem(H, targets=targets)


def table_timing():
    """Wall-clock cost per block for several sizes and objective mixes."""

    def run(name, problem, max_time=60.0, stop_loss=1e-10, **kwargs):
        evals = [0]

        def count(loss):
            evals[0] += 1
            return False

        shaper = MimoShaper(
            problem, progress=count, max_time=max_time, stop_loss=stop_loss,
            rng=np.random.default_rng(7), **kwargs,
        )
        t0 = time.perf_counter()
        shaper.make_block()
        dt = time.perf_counter() - t0
        nj, _, nf = problem.H.shape
        nt = 2 * (nf - 1)
        row = (
            f"{name} & {nj} & {nt} & {problem.num_free_phases} & "
            f"{evals[0]} & {dt:.2f} & {1e3 * dt / evals[0]:.2f} \\\\"
        )
        return row, evals[0], dt

    rows = [
        run(f"SISO skew+kurt+endpoint", flat_siso_problem(nt, kurtosis=5.0))[0]
        for nt in [2**10, 2**12, 2**14]
    ]
    nt = 2**12
    nf = nt // 2 + 1
    H = np.zeros((1, 1, nf), dtype=complex)
    H[0, 0, 1 : nf // 2] = 1.0
    rows.append(
        run(
            "SISO crest ($\\beta=80$)",
            SynthesisProblem(H, crests=[CrestTarget(0, beta=80)]),
            ftol_rel=1e-7,
        )[0]
    )
    problem, _ = mimo_problem(np.random.default_rng(13))
    rows.append(run("MIMO 2ch, CSD + 5 (cross-)moments", problem)[0])
    scaling = []
    for nj in [4, 8, 16, 32]:
        num = 2 * nj + nj * (nj - 1) // 2
        row, evals, dt = run(
            f"MIMO scaling, {num} moment targets",
            gaussian_scaling_problem(nj, 2**10, np.random.default_rng(21)),
            max_time=300.0,
        )
        rows.append(row)
        scaling.append((nj, 1e3 * dt / evals))

    # empirical per-evaluation scaling exponent vs the O(Nj^2) FFT-count claim
    slope = np.polyfit(
        np.log([nj for nj, _ in scaling]), np.log([ms for _, ms in scaling]), 1
    )[0]
    (TABLES / "timing_stats.tex").write_text(
        f"\\newcommand{{\\scalingslope}}{{{slope:.2f}}}\n"
    )

    lines = [
        r"\begin{tabular}{l r r r r r r}",
        r"problem & $N_j$ & $N_t$ & phases & evals & s/block & ms/eval \\",
        r"\hline",
        *rows,
        r"\end{tabular}",
    ]
    (TABLES / "timing.tex").write_text("\n".join(lines) + "\n")


class _StopEarly(Exception):
    """Raised once the tracked loss drops below the matched stop threshold,
    so scipy's optimisers halt on the same criterion as CCSAQ's ``stopval``."""


def _scipy_minimize(problem, method, x0, stop_loss, record=None):
    """Run ``scipy.optimize.minimize`` on ``problem`` with the identical
    analytic loss/gradient CCSAQ uses, same box bounds, matched early stop."""
    n = problem.num_free_phases

    def fun(flat):
        phase = flat.reshape(problem.num_channels, -1)
        grad = np.empty_like(phase)
        loss = problem.loss(phase, grad)
        if record is not None:
            record.append(loss)
        if loss < stop_loss:
            raise _StopEarly()
        return loss, grad.ravel()

    options = {"maxiter": 5000, "ftol": 1e-16}
    if method == "L-BFGS-B":
        options.update(gtol=1e-14, maxfun=20000)
    try:
        minimize(fun, x0, jac=True, method=method,
                 bounds=[(-np.pi, np.pi)] * n, options=options)
    except _StopEarly:
        pass


def _ccsaq_minimize(problem, x0, stop_loss, record=None):
    """Run NLopt's CCSAQ directly (``MimoShaper`` now runs L-BFGS-B, so the
    comparison talks to NLopt itself) with the same early-stop threshold and
    per-evaluation loss tracking as ``_scipy_minimize``."""
    n = problem.num_free_phases

    def objective(flat_phase, flat_grad):
        phase = flat_phase.reshape(problem.num_channels, -1)
        if flat_grad.size > 0:
            grad = np.empty_like(phase)
            loss = problem.loss(phase, grad)
            flat_grad[:] = grad.ravel()
        else:
            loss = problem.loss(phase)
        if record is not None:
            record.append(loss)
        return loss

    opt = nlopt.opt(nlopt.LD_CCSAQ, n)
    opt.set_lower_bounds(np.full(n, -np.pi))
    opt.set_upper_bounds(np.full(n, np.pi))
    opt.set_min_objective(objective)
    opt.set_maxtime(30)
    opt.set_stopval(stop_loss)
    opt.set_ftol_rel(1e-14)
    opt.set_xtol_rel(1e-14)
    opt.optimize(x0)


def fig_and_table_optimizer_comparison():
    """CCSAQ vs scipy's generic bound-constrained gradient optimisers.

    The loss and its analytic gradient make no NLopt-specific assumption, so
    the identical objective can be handed to any bound-constrained
    gradient-based optimiser.  Left: loss vs evaluation for the reference
    single-channel kurtosis-5 problem (one seed, matched early-stop
    threshold).  Right: wall-clock time per block vs free-phase count,
    log-log, averaged over independent seeds; the table gives the same
    numbers together with evaluation counts.
    """
    stop_loss = 1e-10
    methods = ["CCSAQ", "L-BFGS-B", "SLSQP"]
    colours = {"CCSAQ": "C0", "L-BFGS-B": "C1", "SLSQP": "C2"}

    def run(method, problem, x0, record):
        if method == "CCSAQ":
            _ccsaq_minimize(problem, x0, stop_loss, record=record)
        else:
            _scipy_minimize(problem, method, x0, stop_loss, record=record)

    fig, (ax_conv, ax_scale) = plt.subplots(1, 2, figsize=(9, 3.4))

    # --- left panel: convergence trace, one seed, reference problem ---
    nt_ref = 2**12
    x0_ref = np.random.default_rng(500).uniform(
        -np.pi, np.pi, flat_siso_problem(nt_ref, 5.0).num_free_phases
    )
    for method in methods:
        losses = []
        run(method, flat_siso_problem(nt_ref, 5.0), x0_ref, losses)
        ax_conv.semilogy(losses, colours[method], linewidth=0.9, label=method)
    ax_conv.set_xlabel("objective evaluation")
    ax_conv.set_ylabel(r"loss $\Xi$")
    ax_conv.legend(fontsize=8)
    ax_conv.grid(alpha=0.4, which="both")

    # --- right panel + table: wall-clock scaling with free-phase count ---
    nts = [2**8, 2**10, 2**12]
    n_seeds = 8
    stats = {}
    for method in methods:
        times_ms, mean_evals = [], []
        for nt in nts:
            times, evalcounts = [], []
            for seed in range(n_seeds):
                problem = flat_siso_problem(nt, 5.0)
                x0 = np.random.default_rng(600 + seed).uniform(
                    -np.pi, np.pi, problem.num_free_phases
                )
                losses = []
                t0 = time.perf_counter()
                run(method, problem, x0, losses)
                times.append(time.perf_counter() - t0)
                evalcounts.append(len(losses))
            times_ms.append(1e3 * np.mean(times))
            mean_evals.append(np.mean(evalcounts))
        stats[method] = (times_ms, mean_evals)
        free_phases = [nt // 2 - 1 for nt in nts]
        ax_scale.loglog(free_phases, times_ms, colours[method] + "o-", label=method)
    ax_scale.set_xlabel(r"free phases $N_t/2-1$")
    ax_scale.set_ylabel("wall-clock per block (ms)")
    ax_scale.tick_params(axis='x', which="both", labelrotation=30)
    ax_scale.legend(fontsize=8)
    ax_scale.grid(alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "optimizer_comparison.pdf")
    plt.close(fig)
   
    header2 = " & ".join(f"\\multicolumn{{2}}{{c}}{{$N_{{\\rm free}}={nt // 2 - 1}$}}" for nt in nts)
    lines = [
        r"\begin{tabular}{l r r r r r r}",
        " & " + header2 + r" \\",
        r"method & " + " & ".join(["ms/block & evals"] * len(nts)) + r" \\",
        r"\hline",
    ]
    for method in methods:
        times_ms, mean_evals = stats[method]
        cells = " & ".join(f"{t:.1f} & {e:.0f}" for t, e in zip(times_ms, mean_evals))
        lines.append(f"{method} & {cells} \\\\")
    lines.append(r"\end{tabular}")
    (TABLES / "optimizer_comparison.tex").write_text("\n".join(lines) + "\n")

    slsqp_ratio = stats["SLSQP"][0][-1] / stats["CCSAQ"][0][-1]
    lbfgsb_ratio = stats["CCSAQ"][1][-1] / stats["L-BFGS-B"][1][-1]
    (TABLES / "optimizer_stats.tex").write_text(
        f"\\newcommand{{\\slsqpratio}}{{{slsqp_ratio:.0f}}}\n"
        f"\\newcommand{{\\lbfgsbratio}}{{{lbfgsb_ratio:.1f}}}\n"
        f"\\newcommand{{\\optfreemax}}{{{nts[-1] // 2 - 1}}}\n"
    )


def mimo_blocks(problem, rng, num_blocks, **shaper_kwargs):
    """Optimised blocks and their spectra for the MIMO examples."""
    shaper = MimoShaper(problem, max_time=60, rng=rng, **shaper_kwargs)
    xs, vs = [], []
    for _ in range(num_blocks):
        x = shaper.make_block()
        xs.append(x)
        vs.append(np.fft.rfft(x, axis=1))
    return xs, vs


def fig_and_table_mimo():
    """MIMO measured-target example: traces, moment table, CSD match."""
    rng = np.random.default_rng(13)
    problem, G_target = mimo_problem(rng)
    xs, vs = mimo_blocks(problem, rng, num_blocks=32)

    # --- time traces of the first block
    x = xs[0]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 4))
    for k, ax in enumerate(axes):
        ax.plot(x[k], linewidth=0.4)
        ax.set_ylabel(f"channel {k}")
        ax.grid(alpha=0.4)
    axes[1].set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(FIGURES / "mimo_traces.pdf")
    plt.close(fig)

    # --- moment table (achieved mean and spread over the block ensemble)
    lines = [
        r"\begin{tabular}{l l r r}",
        r"target & tuple $\mathbf{i}$ & $\mu_\mathbf{i}$ & achieved (mean $\pm$ std) \\",
        r"\hline",
    ]
    for t in problem.targets:
        vals = [moments.normalized_moment(x, t.indices) for x in xs]
        idx = ",".join(str(i) for i in t.indices)
        lines.append(
            f"{TUPLE_LABELS[t.indices]} & $({idx})$ & {t.value:.3f} & "
            f"{np.mean(vals):.3f} $\\pm$ {np.std(vals):.3f} \\\\"
        )
    lines.append(r"\end{tabular}")
    (TABLES / "mimo_moments.tex").write_text("\n".join(lines) + "\n")

    # --- CSD reproduction: estimate the synthesised ensemble with the *same*
    # multitaper estimator used for the record, so both sides carry comparable
    # estimator variance (raw per-block cross-spectra are rank-one samples
    # with far fewer degrees of freedom and would scatter much more).
    nt = 2 * (G_target.shape[2] - 1)
    G_real = estimate.multitaper_csd(np.hstack(xs), nw=4.0, nfft=nt)
    G_real *= 2.0 / nt  # unit-norm-taper estimate -> (2/Nt^2) H H* convention
    # target CSD in the same per-block variance scaling as H
    H = problem.H
    G_scaled = np.einsum("pjf,qjf->pqf", H, np.conj(H)) * (2.0 / nt**2)

    def coherence(g):
        return np.abs(g[0, 1, 1:-1]) ** 2 / (
            np.abs(g[0, 0, 1:-1]) * np.abs(g[1, 1, 1:-1])
        )

    ff = np.arange(G_target.shape[2]) / nt  # normalised frequency
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    for k in range(2):
        axes[0].semilogy(
            ff[1:-1], np.abs(G_scaled[k, k, 1:-1]), f"C{k}", label=f"target ch.{k}"
        )
        axes[0].semilogy(
            ff[1:-1],
            np.abs(G_real[k, k, 1:-1]),
            f"C{k}--",
            alpha=0.7,
            label=f"synthesised ch.{k}",
        )
    axes[0].set_ylabel("PSD")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(ff[1:-1], coherence(G_scaled), "C0", label="target")
    axes[1].plot(ff[1:-1], coherence(G_real), "C1--", alpha=0.7, label="synthesised")
    axes[1].set_ylabel("coherence 0-1")
    axes[1].legend(fontsize=8)
    axes[2].plot(ff[1:-1], np.angle(G_scaled[0, 1, 1:-1]), "C0")
    axes[2].plot(ff[1:-1], np.angle(G_real[0, 1, 1:-1]), "C1--", alpha=0.7)
    axes[2].set_ylabel("cross phase 0-1")
    axes[2].set_xlabel(r"frequency $f/f_s$")
    for ax in axes:
        ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)  # review: keep tick labels legible
    fig.savefig(FIGURES / "csd_match.pdf")
    plt.close(fig)


def fig_crest():
    """Crest minimisation: beta continuation vs direct high-beta starts.

    Left: achieved crest factor vs surrogate stiffness beta, for a full flat
    spectrum, a half-band (zero tail) spectrum and a half band with a 10%
    raised-cosine edge taper, seed-averaged, including the physical crest of
    the continued optimum on an 8x oversampled reconstruction.  Right: the
    final tapered minimum-crest block (the full-band optimum hides its
    physical peaks between the samples).
    """
    nt = 2**12
    betas = [5, 10, 20, 40, 80, 160, 320]
    seeds = [0, 1, 2]

    def crest_of(x):
        return np.max(np.abs(x)) / np.sqrt(np.mean(x**2))

    def band_H(band, taper=0.0):
        """Flat band with optional raised-cosine roll-off (fraction of Nyquist)."""
        nf = nt // 2 + 1
        edge = int(round(band * (nf - 1)))
        H = np.zeros((1, 1, nf), dtype=complex)
        H[0, 0, 1:edge] = 1.0
        if taper > 0:
            w = int(round(taper * (nf - 1)))
            k = np.arange(w)
            H[0, 0, edge - w : edge] = 0.5 * (1 + np.cos(np.pi * k / w))
        return H

    def optimise(H, beta, start):
        problem = SynthesisProblem(H, crests=[CrestTarget(0, beta=beta)])
        shaper = MimoShaper(problem, max_time=30, ftol_rel=1e-7, xtol_rel=1e-9)
        x = shaper.make_block(start=start)
        return x, shaper.last_phase

    fig, (ax_beta, ax_block) = plt.subplots(
        1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    best_block = None
    results = {}
    cases = [
        (1.0, 0.0, "full spectrum", "C0"),
        (0.5, 0.0, "half band", "C1"),
        (0.5, 0.10, "half band, 10% taper", "C2"),
    ]
    for band, taper, label, style in cases:
        H = band_H(band, taper)
        n = nt // 2 - 1
        direct = np.empty((len(seeds), len(betas)))
        continued = np.empty_like(direct)
        physical = np.empty_like(direct)
        kurts = np.empty(len(seeds))
        for i, seed in enumerate(seeds):
            start0 = np.random.default_rng(seed).uniform(-np.pi, np.pi, n)
            phase = start0
            for j, beta in enumerate(betas):
                x, _ = optimise(H, beta, start0)
                direct[i, j] = crest_of(x[0])
                x, phase = optimise(H, beta, phase)
                continued[i, j] = crest_of(x[0])
                physical[i, j] = moments.oversampled_crest(x, 0)
            kurts[i] = moments.normalized_moment(x, (0, 0, 0, 0))
            if taper > 0 and i == 0:
                best_block = x[0]
        results[label] = (direct, continued, physical, kurts)
        if taper == 0:
            ax_beta.semilogx(
                betas, direct.mean(axis=0), style + "o--", label=f"{label}, direct"
            )
        ax_beta.semilogx(
            betas, continued.mean(axis=0), style + "s-", label=f"{label}, continued"
        )
        ax_beta.semilogx(
            betas,
            physical.mean(axis=0),
            style + ":",
            label=f"{label}, physical (8x)",
        )
    ax_beta.axhline(np.sqrt(2), color="k", linewidth=0.6, linestyle=":")
    ax_beta.annotate(r"sine $\sqrt{2}$", (betas[-2], np.sqrt(2)), fontsize=8,
                     textcoords="offset points", xytext=(2, 3))
    ax_beta.set_xlabel(r"surrogate stiffness $\beta$")
    ax_beta.set_ylabel(r"crest factor $\max|x|/\sigma$")
    ax_beta.legend(fontsize=7)
    ax_beta.grid(alpha=0.4, which="both")

    ax_block.plot(best_block, linewidth=0.4)
    ax_block.set_xlabel("sample")
    ax_block.set_ylabel(
        f"sampled crest {crest_of(best_block):.3f}\n"
        f"physical crest {moments.oversampled_crest(best_block[None, :], 0):.3f}"
    )
    ax_block.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "crest_beta.pdf")
    plt.close(fig)

    # seed-averaged headline numbers at the final beta, quoted in the paper
    full_d, full_c, full_p, full_k = results["full spectrum"]
    _, half_c, half_p, _ = results["half band"]
    _, _, taper_p, _ = results["half band, 10% taper"]
    (TABLES / "crest_stats.tex").write_text(
        f"\\newcommand{{\\crestfullsampled}}{{{full_c[:, -1].mean():.2f}}}\n"
        f"\\newcommand{{\\crestfulldirect}}{{{full_d[:, -1].mean():.2f}}}\n"
        f"\\newcommand{{\\crestfullphysical}}{{{full_p[:, -1].mean():.2f}}}\n"
        f"\\newcommand{{\\crestfullkurt}}{{{full_k.mean():.2f}}}\n"
        f"\\newcommand{{\\cresthalfsampled}}{{{half_c[:, -1].mean():.2f}}}\n"
        f"\\newcommand{{\\cresthalfphysical}}{{{half_p[:, -1].mean():.2f}}}\n"
        f"\\newcommand{{\\cresttaperphysical}}{{{taper_p[:, -1].mean():.2f}}}\n"
    )


# Retzler et al. 2022 (Automatica 146:110654) benchmark setups: block length
# and active harmonics (flat unless amplitudes given; scaling is CF-invariant).
RETZLER_SETUPS = {
    "A": (2048, np.arange(1, 32), None),
    "B": (1024, np.arange(1, 17), np.sin((2 * np.arange(1, 17) - 1) / 32 * np.pi)),
    "C": (2048, np.array([1, 2, 4, 8, 16, 32]), None),
    "D": (
        8192,
        np.array([10, 12, 15, 18, 22, 27, 33, 40, 48, 58, 70, 84, 100]),
        np.sin((2 * np.arange(1, 14) - 1) / 26 * np.pi),
    ),
    "E": (800, np.arange(1, 101), None),
}

# Published sampled-CF (min, avg) over 1000 random starts, Retzler et al. Table 2.
RETZLER_PUBLISHED = [
    (
        r"Van der Ouderaa \emph{et al.}~\cite{vanderouderaa1988peak} (clipping)",
        {"A": (1.4637, 1.5468), "B": (1.4579, 1.5372), "C": (2.096, 2.0968),
         "D": (2.076, 2.1944), "E": (1.4857, 1.5564)},
    ),
    (
        r"Guillaume \emph{et al.}~\cite{guillaume1991crest} ($\ell_p$ Chebyshev)",
        {"A": (1.3563, 1.4085), "B": (1.4042, 1.437), "C": (2.0139, 2.0142),
         "D": (1.9877, 2.0063), "E": (1.3565, 1.3727)},
    ),
    (
        r"Retzler \emph{et al.}~\cite{retzler2022crest} (nonlinear opt.)",
        {"A": (1.3513, 1.4041), "B": (1.4004, 1.4316), "C": (2.011, 2.0123),
         "D": (1.9815, 1.9961), "E": (1.3512, 1.3683)},
    ),
    (
        r"Janeiro \emph{et al.}~\cite{janeiro2020abc} (bee colony)",
        {"A": (1.5409, 1.6314), "B": (1.4181, 1.4978), "C": (2.011, 2.0124),
         "D": (1.9862, 2.0257), "E": (1.7858, 1.8947)},
    ),
]


def crest_of(x):
    return np.max(np.abs(x)) / np.sqrt(np.mean(x**2))


def multisine_H(nt, bins, amps):
    nf = nt // 2 + 1
    H = np.zeros((1, 1, nf), dtype=complex)
    H[0, 0, bins] = 1.0 if amps is None else amps
    return H


CREST_BETAS = [5.0 * 2**k for k in range(10)]  # 5 .. 2560, doubled each stage


def table_crest_benchmark():
    """Crest-factor benchmark on the five multisine setups of Retzler et al.

    Runs the beta-continued crest surrogate and -- demonstrating that the
    framework subsumes the classical l_p objective as a scaled-function
    target -- an l_p continuation with doubling p, both from independent
    random starts, and tabulates sampled crest min/avg against the published
    1000-start statistics of the dedicated crest optimisers.
    """
    num_seeds = 25
    opts = dict(max_time=30.0, ftol_rel=1e-11, xtol_rel=0.0)
    betas = CREST_BETAS
    ps = [2**k for k in range(2, 10)]  # 4 .. 512

    def lp_target(p):
        # |z|^p with exponent clipping so large p stays finite in float64
        def g(z):
            return np.exp(np.minimum(p * np.log(np.maximum(np.abs(z), 1e-12)), 500.0))

        def gprime(z):
            loga = np.log(np.maximum(np.abs(z), 1e-12))
            return p * np.exp(np.minimum((p - 1) * loga, 500.0)) * np.sign(z)

        return ScaledFunctionTarget(0, g, gprime)

    def continuation(H, seed, problems):
        rng = np.random.default_rng(seed)
        phase = rng.uniform(-np.pi, np.pi, H.shape[2] - 2)
        for problem in problems:
            shaper = MimoShaper(problem, **opts)
            x = shaper.make_block(start=phase)
            phase = shaper.last_phase
        return crest_of(x[0])

    ours = [
        (
            r"this work, crest surrogate ($\beta$-continued)",
            lambda H, s: continuation(
                H, s, [SynthesisProblem(H, crests=[CrestTarget(0, beta=b)]) for b in betas]
            ),
        ),
        (
            r"this work, $\ell_p$ scaled-function target",
            lambda H, s: continuation(
                H, s, [SynthesisProblem(H, functions=[lp_target(p)]) for p in ps]
            ),
        ),
    ]

    def fmt_pub(v):  # reproduce published decimals, no false precision
        return f"{v:.4f}".rstrip("0").rstrip(".")

    our_rows, times = [], []
    for label, run in ours:
        cells = []
        for name, (nt, bins, amps) in RETZLER_SETUPS.items():
            H = multisine_H(nt, bins, amps)
            t0 = time.perf_counter()
            cf = np.array([run(H, seed) for seed in range(num_seeds)])
            times.append((time.perf_counter() - t0) / num_seeds)
            cells.append(f"{cf.min():.4f}/{cf.mean():.4f}")
            print(f"  {label} {name}: min {cf.min():.4f} avg {cf.mean():.4f} "
                  f"({times[-1]:.1f}s/start)")
        our_rows.append(f"{label} & " + " & ".join(cells) + r" \\")

    pub_rows = [
        label + " & " + " & ".join(
            f"{fmt_pub(v[0])}/{fmt_pub(v[1])}" for v in (stats[s] for s in RETZLER_SETUPS)
        ) + r" \\"
        for label, stats in RETZLER_PUBLISHED
    ]
    lines = [
        r"\begin{tabular}{l c c c c c}",
        r"method & A & B & C & D & E \\",
        r"\hline",
        *pub_rows,
        r"\hline",
        *our_rows,
        r"\end{tabular}",
    ]
    (TABLES / "crest_benchmark.tex").write_text("\n".join(lines) + "\n")
    (TABLES / "crest_bench_stats.tex").write_text(
        f"\\newcommand{{\\crestbenchseeds}}{{{num_seeds}}}\n"
        f"\\newcommand{{\\crestbenchtimes}}{{{min(times):.1f}--{max(times):.0f}}}\n"
    )


def _continuation_lbfgsb(H, seed, betas):
    """Beta-continued crest surrogate via ``MimoShaper`` (L-BFGS-B), counting
    loss evaluations spent across the whole ladder."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, H.shape[2] - 2)
    total_evals = 0
    for beta in betas:
        problem = SynthesisProblem(H, crests=[CrestTarget(0, beta=beta)])
        count = [0]

        def prog(loss):
            count[0] += 1
            return False

        shaper = MimoShaper(problem, progress=prog, max_time=30.0,
                             ftol_rel=1e-11, xtol_rel=0.0)
        x = shaper.make_block(start=phase)
        phase = shaper.last_phase
        total_evals += count[0]
    return crest_of(x[0]), total_evals


def _continuation_ccsaq(H, seed, betas):
    """Same beta continuation, but driving NLopt's CCSAQ directly (bypassing
    ``MimoShaper``, which now runs L-BFGS-B), counting loss evaluations."""
    rng = np.random.default_rng(seed)
    n = H.shape[2] - 2
    phase = rng.uniform(-np.pi, np.pi, n)
    total_evals = 0
    for beta in betas:
        problem = SynthesisProblem(H, crests=[CrestTarget(0, beta=beta)])
        count = [0]

        def objective(flat_phase, flat_grad):
            p = flat_phase.reshape(1, -1)
            if flat_grad.size > 0:
                grad = np.empty_like(p)
                loss = problem.loss(p, grad)
                flat_grad[:] = grad.ravel()
            else:
                loss = problem.loss(p)
            count[0] += 1
            return loss

        opt = nlopt.opt(nlopt.LD_CCSAQ, n)
        opt.set_lower_bounds(np.full(n, -np.pi))
        opt.set_upper_bounds(np.full(n, np.pi))
        opt.set_min_objective(objective)
        opt.set_maxtime(30.0)
        opt.set_ftol_rel(1e-11)
        opt.set_xtol_rel(0.0)
        phase = opt.optimize(phase)
        total_evals += count[0]
    x = problem.signal(phase.reshape(1, -1))
    return crest_of(x[0]), total_evals


def table_optimizer_beta_continuation():
    """CCSAQ vs L-BFGS-B on the actual beta-continued crest surrogate.

    Runs the identical continuation used above (same five Retzler multisine
    setups, same beta ladder from 5 to 2560) with NLopt's CCSAQ swapped in
    for comparison, reporting achieved crest and loss evaluations spent --
    the evidence behind switching the package's default optimiser from
    CCSAQ to L-BFGS-B.
    """
    num_seeds = 10

    rows = []
    ccsaq_evals, lbfgsb_evals = [], []
    for name, (nt, bins, amps) in RETZLER_SETUPS.items():
        H = multisine_H(nt, bins, amps)
        cc_cf, cc_ev, lb_cf, lb_ev = [], [], [], []
        for seed in range(num_seeds):
            cf, ev = _continuation_ccsaq(H, seed, CREST_BETAS)
            cc_cf.append(cf)
            cc_ev.append(ev)
            cf, ev = _continuation_lbfgsb(H, seed, CREST_BETAS)
            lb_cf.append(cf)
            lb_ev.append(ev)
        ccsaq_evals.append(np.mean(cc_ev))
        lbfgsb_evals.append(np.mean(lb_ev))
        rows.append(
            f"{name} & {np.mean(cc_cf):.4f} & {np.mean(cc_ev):.0f} & "
            f"{np.mean(lb_cf):.4f} & {np.mean(lb_ev):.0f} \\\\"
        )
        print(f"  setup {name}: CCSAQ cf={np.mean(cc_cf):.4f} evals={np.mean(cc_ev):.0f}"
              f"  L-BFGS-B cf={np.mean(lb_cf):.4f} evals={np.mean(lb_ev):.0f}")

    lines = [
        r"\begin{tabular}{l c c c c}",
        r" & \multicolumn{2}{c}{CCSAQ} & \multicolumn{2}{c}{L-BFGS-B} \\",
        r"setup & crest & evals & crest & evals \\",
        r"\hline",
        *rows,
        r"\end{tabular}",
    ]
    (TABLES / "optimizer_beta_continuation.tex").write_text("\n".join(lines) + "\n")

    reduction = np.mean(np.array(ccsaq_evals) / np.array(lbfgsb_evals))
    (TABLES / "optimizer_beta_continuation_stats.tex").write_text(
        f"\\newcommand{{\\betacontevalreduction}}{{{reduction:.0f}}}\n"
    )


def load_road_record():
    """Measured road record from the shipped npz: (record, fs, names, units).

    12 wheel-hub force/moment channels (left/right measuring hubs) sampled at
    300 Hz on a test track, static offsets removed.  The npz stores the native
    int16 samples with per-channel scale; see data/roadsection_220s_300hz.npz.
    """
    z = np.load(ROAD_NPZ)
    y = z["data_int16"].astype(np.float64) * z["scale"][:, None]
    y -= np.mean(y, axis=1, keepdims=True)
    return y, float(z["fs"]), list(z["names"]), list(z["units"])


def road_tuples(num_channels):
    """Per-channel skewness+kurtosis and left/right co-kurtosis pairs.

    Channels come interleaved (left, right) per physical quantity, so the
    pair (2i, 2i+1) is the same force/moment on the two wheel hubs.
    """
    tuples = multimodel.moment_tuples(num_channels, cokurtosis=False)
    tuples += [(k, k, k + 1, k + 1) for k in range(0, num_channels, 2)]
    return tuples


def fig_and_table_road():
    """Measured road-record example: 12-channel targets, traces, CSD, stats."""
    rng = np.random.default_rng(17)
    record, fs, names, units = load_road_record()
    nj, n = record.shape
    nfft = 1024

    G_target = estimate.multitaper_csd(record, nw=4.0, nfft=nfft)
    H = estimate.csd_to_frf(G_target, variance=np.var(record, axis=1))
    targets = estimate.estimate_moment_targets(record, road_tuples(nj))
    problem = SynthesisProblem(H, targets=targets)

    t0 = time.time()
    # 30 coupled targets share one loss: tighten the stop criteria so no
    # single moment parks a visible offset inside the tolerance ball
    xs, _ = mimo_blocks(
        problem, rng, num_blocks=32, stop_loss=1e-10, ftol_rel=1e-9, xtol_rel=1e-8
    )
    block_seconds = (time.time() - t0) / 32

    # --- moment tables: diagonal per channel, left/right co-kurtosis pairs
    by_index = {t.indices: t for t in targets}

    def ach(indices):
        vals = [moments.normalized_moment(x, indices) for x in xs]
        return f"{np.mean(vals):.3f} $\\pm$ {np.std(vals):.4f}"

    lines = [
        r"\begin{tabular}{l r r r r}",
        r"channel & $\hat\mu_{(k,k,k)}$ & achieved & $\hat\mu_{(k,k,k,k)}$ & achieved \\",
        r"\hline",
    ]
    for k, name in enumerate(names):
        skew, kurt = by_index[(k, k, k)], by_index[(k, k, k, k)]
        lines.append(
            f"\\texttt{{{name}}} & {skew.value:.3f} & {ach(skew.indices)} & "
            f"{kurt.value:.3f} & {ach(kurt.indices)} \\\\"
        )
    lines.append(r"\end{tabular}")
    (TABLES / "road_moments.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{tabular}{l r r}",
        r"left/right pair & $\hat\mu_\mathbf{i}$ & achieved \\",
        r"\hline",
    ]
    for k in range(0, nj, 2):
        t = by_index[(k, k, k + 1, k + 1)]
        lines.append(
            f"\\texttt{{{names[k]}}}/\\texttt{{{names[k + 1]}}} & "
            f"{t.value:.3f} & {ach(t.indices)} \\\\"
        )
    lines.append(r"\end{tabular}")
    (TABLES / "road_cokurt.tex").write_text("\n".join(lines) + "\n")

    # --- traces: measured excerpt vs one synthesised block, extreme channels
    shown = [names.index("FZMRHL"), names.index("MZMRHL")]
    excerpt = slice(16 * nfft, 17 * nfft)
    tt = np.arange(nfft) / fs
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(7, 4))
    for row, k in enumerate(shown):
        axes[row, 0].plot(tt, record[k, excerpt], linewidth=0.4)
        axes[row, 1].plot(tt, xs[0][k], "C1", linewidth=0.4)
        for col in (0, 1):
            axes[row, col].grid(alpha=0.4)
        axes[row, 0].set_ylabel(f"{names[k]} [{units[k]}]")
    axes[0, 0].set_title("measured excerpt", fontsize=9)
    axes[0, 1].set_title("synthesised block", fontsize=9)
    for col in (0, 1):
        axes[1, col].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(FIGURES / "road_traces.pdf")
    plt.close(fig)

    # --- CSD reproduction for the most coherent physical pair (FZ left/right)
    p, q = names.index("FZMRHL"), names.index("FZMRHR")
    G_real = estimate.multitaper_csd(np.hstack(xs), nw=4.0, nfft=nfft)
    G_real *= 2.0 / nfft
    G_scaled = np.einsum("pjf,qjf->pqf", H, np.conj(H)) * (2.0 / nfft**2)

    def coherence(g):
        return np.abs(g[p, q, 1:-1]) ** 2 / (
            np.abs(g[p, p, 1:-1]) * np.abs(g[q, q, 1:-1])
        )

    ff = np.arange(G_target.shape[2])[1:-1] * fs / nfft
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    for k, style in ((p, "C0"), (q, "C2")):
        axes[0].semilogy(ff, np.abs(G_scaled[k, k, 1:-1]), style, label=f"target {names[k]}")
        axes[0].semilogy(ff, np.abs(G_real[k, k, 1:-1]), style + "--", alpha=0.7,
                         label=f"synthesised {names[k]}")
    axes[0].set_ylabel(r"PSD [$\mathrm{N^2/bin}$]")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(ff, coherence(G_scaled), "C0", label="target")
    axes[1].plot(ff, coherence(G_real), "C1--", alpha=0.7, label="synthesised")
    axes[1].set_ylabel("coherence L/R")
    axes[1].legend(fontsize=8)
    axes[2].plot(ff, np.angle(G_scaled[p, q, 1:-1]), "C0")
    axes[2].plot(ff, np.angle(G_real[p, q, 1:-1]), "C1--", alpha=0.7)
    axes[2].set_ylabel("cross phase L/R")
    axes[2].set_xlabel("frequency [Hz]")
    for ax in axes:
        ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    fig.savefig(FIGURES / "road_csd.pdf")
    plt.close(fig)

    # --- generated statistics quoted in the text (no manual transcription)
    nf = G_target.shape[2]
    cond = np.array([np.linalg.cond(G_target[:, :, k]) for k in range(1, nf - 1)])
    d = np.real(np.einsum("kkf->kf", G_target))
    coh = np.abs(G_target) ** 2 / (d[:, None, :] * d[None, :, :])
    iu = np.triu_indices(nj, 1)
    coh_max = coh[iu[0], iu[1], 1:-1].max()

    report = stationarity.stationarity_report(record, num_segments=32)
    rejected = np.zeros(nj, dtype=bool)
    for stat in report.values():
        for test in ("reverse_arrangements", "runs"):
            rejected |= stat[test].p < 0.01
    stats = estimate.signal_stats(record)
    exponent = int(np.floor(np.log10(cond.max())))
    mantissa = cond.max() / 10.0**exponent
    (TABLES / "road_stats.tex").write_text(
        f"\\newcommand{{\\roadcondmax}}{{{mantissa:.0f} \\times 10^{{{exponent}}}}}\n"
        f"\\newcommand{{\\roadcohmax}}{{{coh_max:.4f}}}\n"
        f"\\newcommand{{\\roadblockms}}{{{1000 * block_seconds:.0f}}}\n"
        f"\\newcommand{{\\roadstatrejected}}{{{int(rejected.sum())}}}\n"
        f"\\newcommand{{\\roadkurtmax}}{{{stats['kurtosis'].max():.2f}}}\n"
        f"\\newcommand{{\\roadcrestmax}}{{{stats['crest'].max():.2f}}}\n"
    )


def fig_and_table_multimodel():
    """Minimal multimodel validation on a two-regime composite record.

    A surrogate record whose second half is rougher (variance x6, heavier
    tails) rejects stationarity; a single compromise model reproduces the
    pooled statistics but no regime structure, while eight per-section
    models with crossfaded joints track the segment mean-square profile and
    the per-regime kurtosis.  Quotes reverse-arrangements z-scores on the
    32-segment mean-square sequence for record, single-model and multimodel
    syntheses.
    """
    rng = np.random.default_rng(29)
    n, num_sections, nfft = 2**16, 8, 2048
    ff = np.fft.rfftfreq(n)
    lowpass = 1.0 / (1.0 + (ff / 0.08) ** 2)
    resonance = 1.0 / np.abs(1.0 + 2j * 0.05 * (ff / 0.2) - (ff / 0.2) ** 2)

    def colour(sig, mag):
        return np.fft.irfft(np.fft.rfft(sig) * mag, n)

    base = rng.standard_normal(n)
    record = np.vstack(
        [
            colour(base, lowpass) + 0.3 * colour(rng.standard_normal(n), lowpass),
            0.7 * colour(base, resonance) + 0.5 * colour(rng.standard_normal(n), resonance),
        ]
    )
    # second half: rough regime, level x2.5 with moderately heavier tails
    rough = 2.5 * record[:, n // 2 :]
    z = rough / np.std(rough, axis=1, keepdims=True)
    record[:, n // 2 :] = rough * (1.0 + 0.06 * z**2)

    tuples = multimodel.moment_tuples(2)
    kwargs = dict(max_time=10.0, stop_loss=1e-10, rng=np.random.default_rng(31))
    models = multimodel.estimate_section_models(record, num_sections, tuples, nfft=nfft)
    multi = multimodel.synthesize_multimodel(models, blocks_per_section=4, **kwargs)
    pooled = multimodel.estimate_section_models(record, 1, tuples, nfft=nfft)
    single = multimodel.synthesize_multimodel(pooled, blocks_per_section=32, **kwargs)

    num_segments = 32

    def ms_profile(y):
        return stationarity.segment_statistic(y[0], num_segments, "ms")[0]

    def ra_z(y):  # worst-channel reverse-arrangements z on the mean-square sequence
        s = stationarity.segment_statistic(y, num_segments, "ms")
        return np.abs(stationarity.reverse_arrangements_test(s).z).max()

    def half_kurtosis(y):
        half = y.shape[1] // 2
        return (
            moments.normalized_moment(y[:1, :half], (0, 0, 0, 0)),
            moments.normalized_moment(y[:1, half:], (0, 0, 0, 0)),
        )

    fig, (ax_trace, ax_ms) = plt.subplots(
        1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    scale = np.std(record[0])
    for offset, (y, label) in enumerate(
        [(record, "measured"), (multi.merged, "multimodel"), (single.merged, "single model")]
    ):
        tt = np.linspace(0.0, 1.0, y.shape[1])
        ax_trace.plot(tt, y[0] / scale - 8.0 * offset, linewidth=0.3, label=label)
    ax_trace.set_xlabel("record time (normalised)")
    ax_trace.set_yticks([])
    ax_trace.legend(fontsize=8, loc="upper left")
    ax_trace.grid(alpha=0.4)

    for y, label, style in [
        (record, "measured", "C0o-"),
        (multi.merged, "multimodel", "C1s-"),
        (single.merged, "single model", "C2^--"),
    ]:
        ax_ms.semilogy(np.arange(num_segments), ms_profile(y), style,
                       markersize=3, linewidth=0.8, label=label)
    ax_ms.set_xlabel(f"segment (of {num_segments})")
    ax_ms.set_ylabel("segment mean square")
    ax_ms.legend(fontsize=8)
    ax_ms.grid(alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "multimodel.pdf")
    plt.close(fig)

    rec_q, rec_r = half_kurtosis(record)
    mm_q, mm_r = half_kurtosis(multi.merged)
    sg_q, sg_r = half_kurtosis(single.merged)
    (TABLES / "multimodel_stats.tex").write_text(
        f"\\newcommand{{\\mmzrecord}}{{{ra_z(record):.1f}}}\n"
        f"\\newcommand{{\\mmzmulti}}{{{ra_z(multi.merged):.1f}}}\n"
        f"\\newcommand{{\\mmzsingle}}{{{ra_z(single.merged):.1f}}}\n"
        f"\\newcommand{{\\mmkurtrecord}}{{{rec_q:.1f}/{rec_r:.1f}}}\n"
        f"\\newcommand{{\\mmkurtmulti}}{{{mm_q:.1f}/{mm_r:.1f}}}\n"
        f"\\newcommand{{\\mmkurtsingle}}{{{sg_q:.1f}/{sg_r:.1f}}}\n"
    )


def fig_title_art():
    """Original title-page artwork, generated by the synthesiser itself.

    x-y trajectories of kurtosis-shaped, partially coherent two-channel
    low-pass blocks: the heavy tails cluster the strokes into drip-like
    excursions.  Replaces the copyrighted Pollock reproduction.
    """
    nt = 2**13
    nf = nt // 2 + 1
    ff = np.fft.rfftfreq(nt)
    lowpass = 1.0 / (1.0 + (ff / 0.008) ** 4)
    lowpass[0] = lowpass[-1] = 0.0
    H = np.zeros((2, 2, nf), dtype=complex)
    H[0, 0] = lowpass
    H[1, 0] = 0.6 * lowpass
    H[1, 1] = 0.8 * lowpass
    problem = SynthesisProblem(
        H,
        targets=[
            MomentTarget((0, 0, 0, 0), 6.0),
            MomentTarget((1, 1, 1, 1), 6.0),
            MomentTarget((0, 0, 1, 1), 2.5),
        ],
    )
    palette = ["#1a1a1a", "#8c2d19", "#c8a028", "#3a5a78", "#5c5048"]
    widths = [1.6, 1.1, 0.9, 0.7, 0.5]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed, (colour, lw) in enumerate(zip(palette, widths)):
        shaper = MimoShaper(problem, max_time=10, rng=np.random.default_rng(40 + seed))
        x = shaper.make_block()
        ax.plot(x[0], x[1], color=colour, linewidth=lw, alpha=0.75,
                solid_capstyle="round")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(FIGURES / "title_art.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    jobs = [
        fig_siso_block,
        fig_convergence,
        fig_restarts,
        table_timing,
        fig_and_table_optimizer_comparison,
        table_optimizer_beta_continuation,
        fig_crest,
        table_crest_benchmark,
        fig_and_table_mimo,
        fig_and_table_road,
        fig_and_table_multimodel,
        fig_title_art,
    ]
    import sys

    if len(sys.argv) > 1:  # optionally run a subset: make_figures.py fig_crest ...
        jobs = [job for job in jobs if job.__name__ in sys.argv[1:]]
    for job in jobs:
        t0 = time.time()
        job()
        print(f"{job.__name__}: {time.time() - t0:.1f}s")
    print(f"assets written to {FIGURES} and {TABLES}")


if __name__ == "__main__":
    main()
