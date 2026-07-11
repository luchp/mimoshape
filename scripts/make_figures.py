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

from mimoshape import estimate, moments
from mimoshape.shaper import (
    MomentTarget,
    EndpointTarget,
    CrestTarget,
    SynthesisProblem,
    MimoShaper,
)

PAPER = pathlib.Path(__file__).resolve().parent.parent / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"

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
    """Loss vs objective evaluation for the SISO kurtosis problem."""
    rng = np.random.default_rng(12)
    problem = flat_siso_problem(2**12, kurtosis=5.0)
    losses = []

    def record_loss(loss):
        losses.append(loss)
        return False

    shaper = MimoShaper(
        problem, progress=record_loss,
        max_time=60, stop_loss=1e-10, ftol_rel=1e-12, xtol_rel=1e-12, rng=rng,
    )
    shaper.make_block()

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.semilogy(losses, linewidth=0.8)
    ax.set_xlabel("objective evaluation")
    ax.set_ylabel(r"loss $\Xi$")
    ax.grid(alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.pdf")
    plt.close(fig)


def mimo_blocks(problem, rng, num_blocks):
    """Optimised blocks and their spectra for the MIMO example."""
    shaper = MimoShaper(problem, max_time=60, rng=rng)
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

    # --- moment table (block-averaged achieved values)
    lines = [
        r"\begin{tabular}{l l r r}",
        r"target & tuple $\mathbf{i}$ & $\mu_\mathbf{i}$ & achieved \\",
        r"\hline",
    ]
    for t in problem.targets:
        achieved = np.mean([moments.normalized_moment(x, t.indices) for x in xs])
        idx = ",".join(str(i) for i in t.indices)
        lines.append(
            f"{TUPLE_LABELS[t.indices]} & $({idx})$ & {t.value:.3f} & {achieved:.3f} \\\\"
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
    fig.savefig(FIGURES / "csd_match.pdf")
    plt.close(fig)


def fig_crest():
    """Crest minimisation: beta continuation vs direct high-beta starts.

    Left: achieved crest factor vs surrogate stiffness beta, for a full flat
    spectrum and a half-band (zero tail) spectrum, seed-averaged, including
    the physical crest of the continued optimum on an 8x oversampled
    reconstruction.  Right: the final half-band minimum-crest block (the
    full-band optimum hides its physical peaks between the samples).
    """
    nt = 2**12
    betas = [5, 10, 20, 40, 80, 160, 320]
    seeds = [0, 1, 2]

    def crest_of(x):
        return np.max(np.abs(x)) / np.sqrt(np.mean(x**2))

    def band_H(band):
        nf = nt // 2 + 1
        H = np.zeros((1, 1, nf), dtype=complex)
        H[0, 0, 1 : int(round(band * (nf - 1)))] = 1.0
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
    for band, label, style in [(1.0, "full spectrum", "C0"), (0.5, "half band", "C1")]:
        H = band_H(band)
        n = nt // 2 - 1
        direct = np.empty((len(seeds), len(betas)))
        continued = np.empty_like(direct)
        physical = np.empty_like(direct)
        for i, seed in enumerate(seeds):
            start0 = np.random.default_rng(seed).uniform(-np.pi, np.pi, n)
            phase = start0
            for j, beta in enumerate(betas):
                x, _ = optimise(H, beta, start0)
                direct[i, j] = crest_of(x[0])
                x, phase = optimise(H, beta, phase)
                continued[i, j] = crest_of(x[0])
                physical[i, j] = moments.oversampled_crest(x, 0)
            if band == 0.5 and i == 0:
                best_block = x[0]
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
    ax_beta.annotate(r"sine $\sqrt{2}$", (betas[0], np.sqrt(2)), fontsize=8,
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


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    for job in [fig_siso_block, fig_convergence, fig_crest, fig_and_table_mimo]:
        t0 = time.time()
        job()
        print(f"{job.__name__}: {time.time() - t0:.1f}s")
    print(f"assets written to {FIGURES} and {TABLES}")


if __name__ == "__main__":
    main()
