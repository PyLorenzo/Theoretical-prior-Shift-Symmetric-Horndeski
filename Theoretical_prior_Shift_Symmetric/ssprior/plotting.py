"""Figures.

Reproduces the diagnostics of the paper that matter for judging whether the
prior is trustworthy:

``slice``        Fig. 1 and 3 -- what the acceptance slab does to (c01, c02, d02)
``ansatz``       Fig. 2 -- phidot0^2/Lambda2^4 and phiddot0/Lambda3^3 below 1,
                 the a posteriori justification of the truncation of Eq. (9)
``histories``    Fig. 4 -- w(z) and alpha_B(z) for a few models, with the fit
``fit_quality``  Fig. 7 -- CDF of the maximum relative error against the 1% and
                 0.3% targets
``corner``       Fig. 5 and 9 -- the four parameters, with the fitted Gaussian
``corner_X``     Fig. 8 -- the Gaussianised basis
``alphaB2``      how much of the prior crosses alpha_B = 2 (Appendix A)

Style follows the rest of the user's plotting scripts: LaTeX serif labels,
named colours, dpi 300.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

__all__ = ["setup_style", "make_all"]

C_EXACT = "darkgreen"
C_MODEL = "indigo"
C_ALT = "chocolate"
C_WARN = "darkred"


def setup_style(usetex: bool = True) -> bool:
    """LaTeX if a working installation is present, a clean fallback otherwise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shutil import which

    ok = bool(usetex and which("latex"))
    plt.rcParams.update({
        "text.usetex": ok,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.autolayout": False,
    })
    return ok


def _lbl(tex: str, plain: str, usetex: bool) -> str:
    return tex if usetex else plain


# --------------------------------------------------------------------------- #

def plot_slice(ds, path, usetex):
    """Fig. 1: the Lagrangian coefficients that survive the acceptance slab."""
    import matplotlib.pyplot as plt

    names = ("c01", "c02", "d02")
    labels = [_lbl(r"$c_{01}$", "c01", usetex), _lbl(r"$c_{02}$", "c02", usetex),
              _lbl(r"$d_{02}$", "d02", usetex)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, nm, lab in zip(axes, names, labels):
        ax.hist(ds.column(nm), bins=40, density=True, color=C_EXACT, alpha=0.75)
        ax.set_xlabel(lab)
        ax.set_yticks([])
    axes[0].set_ylabel(_lbl(r"normalised", "normalised", usetex))
    fig.suptitle(_lbl(r"After the $\sum\Omega_i = 1$ slice ",
                      "After the sum(Omega)=1 slice ", usetex))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_ansatz(ds, path, usetex):
    """Fig. 2: are the neglected higher-order terms really suppressed?"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(ds.column("psi0_sq_over_2"), bins=40, density=True,
                 color=C_EXACT, alpha=0.75)
    axes[0].set_xlabel(_lbl(r"$\dot\phi_0^2/\Lambda_2^4$", "phidot0^2/Lambda2^4", usetex))
    axes[1].hist(np.abs(ds.column("phiddot0")), bins=40, density=True,
                 color=C_MODEL, alpha=0.75)
    axes[1].set_xlabel(_lbl(r"$|\ddot\phi_0|/\Lambda_3^3$", "|phiddot0|/Lambda3^3", usetex))
    for ax in axes:
        ax.axvline(1.0, color=C_WARN, ls="--", lw=1.2)
        ax.set_yticks([])
    fig.suptitle(_lbl(r"Both below 1: the truncation of higher-order terms is self-consistent",
                      "Both below 1: the truncation of higher-order terms is self-consistent",
                      usetex))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_fit_quality(ds, cfg, path, usetex):
    """Fig. 7: distribution of the maximum relative error of the parametrisation."""
    import matplotlib.pyplot as plt

    e_low = ds.column("err_low_z")
    e_rec = ds.column("err_rec")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, e, target, name in (
            (axes[0], e_low, cfg.fit.accept_rel_err_low_z,
             _lbl(r"$\max_{z<10}|\Delta\mathcal{O}/\mathcal{O}|$",
                  "max |dO/O| at z<10", usetex)),
            (axes[1], e_rec, cfg.fit.accept_rel_err_rec,
             _lbl(r"$|\Delta D_A(z_{\rm rec})/D_A|$", "|dD_A(z_rec)/D_A|", usetex))):
        good = e[np.isfinite(e) & (e > 0)]
        if good.size:
            ax.hist(np.log10(good), bins=40, density=True, color=C_EXACT, alpha=0.75)
        ax.axvline(np.log10(target), color=C_WARN, ls="--", lw=1.5,
                   label=_lbl(rf"target ${100*target:g}\%$", f"target {100*target:g}%",
                              usetex))
        frac = float(np.mean(good < target)) if good.size else 0.0
        ax.set_xlabel(_lbl(r"$\log_{10}$ " + name, "log10 " + name, usetex))
        ax.set_yticks([])
        ax.legend(title=f"{100*frac:.1f}% inside", fontsize=10)
    fig.suptitle(_lbl(r"Accuracy of the four-parameter fit ",
                      "Accuracy of the four-parameter fit ", usetex))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _corner(samples, labels, path, usetex, overlay=None, legend=None, ranges=None):
    """Minimal corner plot: 1-D histograms on the diagonal, 2-D below.

    ``legend`` is a pair (label of ``samples``, label of ``overlay``); it is
    drawn in the free upper-right corner instead of a figure title.
    """
    import matplotlib.pyplot as plt

    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.6 * d, 2.6 * d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=40, density=True, color=C_EXACT, alpha=0.7)
                if overlay is not None:
                    ax.hist(overlay[:, i], bins=40, density=True, histtype="step",
                            color=C_MODEL, lw=1.6)
                ax.set_yticks([])
            else:
                ax.plot(samples[:, j], samples[:, i], ".", ms=1.4, color=C_EXACT,
                        alpha=0.35, rasterized=True)
                if overlay is not None:
                    ax.plot(overlay[:, j], overlay[:, i], ".", ms=1.0, color=C_MODEL,
                            alpha=0.25, rasterized=True)
            if i == d - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            else:
                if i != j:
                    ax.set_yticklabels([])
            if ranges and labels[j] in ranges:
                ax.set_xlim(*ranges[labels[j]])
    if legend:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=C_EXACT, alpha=0.7, label=legend[0])]
        if overlay is not None and len(legend) > 1:
            handles.append(Line2D([], [], color=C_MODEL, lw=1.6, label=legend[1]))
        axes[0, d - 1].legend(handles=handles, loc="center", frameon=False,
                              fontsize=16, handlelength=1.6,
                              borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_corner_params(ds, prior, path, usetex):
    import numpy as np

    labels = [_lbl(r"$w_0$", "w0", usetex), _lbl(r"$w_a$", "wa", usetex),
              _lbl(r"$\hat\alpha_{\rm B}$", "alphaB_hat", usetex),
              _lbl(r"$m$", "m", usetex)]
    overlay = None
    if prior is not None:
        from .gaussianise import sample_prior
        overlay = sample_prior(prior, len(ds), rng=np.random.default_rng(1))
    return _corner(ds.params, labels, path, usetex, overlay,
                   legend=(_lbl(r"Exact prior", "Exact prior", usetex),
                           _lbl(r"Gaussian fit", "Gaussian fit", usetex)))


def plot_corner_X(ds, prior, path, usetex):
    import numpy as np

    from .gaussianise import to_X
    if prior is None:
        return None
    p = prior.exponents
    X = to_X(ds.params, p["p2"], p["p3"], p["p4"])
    rng = np.random.default_rng(2)
    overlay = rng.multivariate_normal(prior.mu, prior.cov, size=len(ds))
    labels = [_lbl(rf"$X_{i}$", f"X{i}", usetex) for i in (1, 2, 3, 4)]
    return _corner(X, labels, path, usetex, overlay,
                   legend=(_lbl(r"Exact, Gaussianised basis",
                                "Exact, Gaussianised basis", usetex),
                           _lbl(r"Gaussian fit", "Gaussian fit", usetex)))


def plot_alphaB2(ds, path, usetex):
    """Appendix A: how much of the prior crosses alpha_B = 2?"""
    import matplotlib.pyplot as plt

    aB = ds.column("alphaB_hat")
    crosses = ds.column("crosses_alphaB2").astype(bool)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(aB[~crosses], bins=40, color=C_EXACT, alpha=0.75,
            label=_lbl(r"$\alpha_{\rm B} < 2$ throughout", "alpha_B < 2 throughout",
                       usetex))
    if crosses.any():
        ax.hist(aB[crosses], bins=40, color=C_WARN, alpha=0.75,
                label=_lbl(r"crosses $\alpha_{\rm B} = 2$", "crosses alpha_B = 2",
                           usetex))
    ax.axvline(2.0, color="k", ls=":", lw=1.2)
    ax.set_xlabel(_lbl(r"$\hat\alpha_{\rm B}$", "alphaB_hat", usetex))
    ax.set_ylabel(_lbl("models", "models", usetex))
    ax.legend(title=f"{100*crosses.mean():.1f}% cross", fontsize=10)
    ax.set_title(_lbl(r"Appendix A: evolutions through $\alpha_{\rm B}=2$",
                      "Appendix A: evolutions through alpha_B = 2", usetex))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_histories(ds, cfg, path, usetex, n_show: int = 12):
    """Fig. 4: w(z) and alpha_B(z), exact against the fitted parametrisation."""
    import matplotlib.pyplot as plt

    from .background import slice_candidates, solve_background
    from .lagrangian import Lagrangian
    from .parametrised import E_cpl, alpha_B_param

    good = np.flatnonzero(ds.good)
    if good.size == 0:
        return None
    pick = good[np.linspace(0, good.size - 1, min(n_show, good.size)).astype(int)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    z_plot = np.linspace(0.0, 3.0, 200)
    N_plot = -np.log1p(z_plot)
    n_done = 0
    for i in pick:
        c01, c02, d02, H0 = ds.theta[i]
        lag = Lagrangian(c01=c01, c02=c02, d02=d02, d01=cfg.model.d01)
        cands = slice_candidates(lag, H0, cfg)
        sol = None
        for psi0, cosmo in cands:
            t = solve_background(lag, cosmo, psi0, cfg)
            if t.ok:
                sol = t
                break
        if sol is None:
            continue
        from scipy.interpolate import CubicSpline
        axes[0].plot(z_plot, CubicSpline(sol.N, sol.w_phi)(N_plot),
                     color=C_EXACT, lw=1.0, alpha=0.8)
        axes[1].plot(z_plot, CubicSpline(sol.N, sol.alpha_B)(N_plot),
                     color=C_EXACT, lw=1.0, alpha=0.8)

        w0, wa, aB, m = ds.params[i]
        a_plot = np.exp(N_plot)
        axes[0].plot(z_plot, w0 + wa * (1.0 - a_plot), color=C_MODEL, ls="--", lw=1.0)
        E_p, _ = E_cpl(N_plot, sol.cosmo.Omega_m, sol.cosmo.Omega_r, w0, wa)
        axes[1].plot(z_plot, alpha_B_param(E_p, aB, 4.0 / m),
                     color=C_MODEL, ls="--", lw=1.0)
        n_done += 1

    if n_done == 0:
        plt.close(fig)
        return None
    axes[0].set_xlabel(_lbl("$z$", "z", usetex))
    axes[0].set_ylabel(_lbl("$w$", "w", usetex))
    axes[1].set_xlabel(_lbl("$z$", "z", usetex))
    axes[1].set_ylabel(_lbl(r"$\alpha_{\rm B}$", "alpha_B", usetex))
    axes[1].axhline(2.0, color="k", ls=":", lw=1.0)
    axes[0].plot([], [], color=C_EXACT, label=_lbl("exact", "exact", usetex))
    axes[0].plot([], [], color=C_MODEL, ls="--", label=_lbl("fit", "fit", usetex))
    axes[0].legend(fontsize=10)
    fig.suptitle(_lbl(r"Exact histories and their fits. The fit "
                      r"minimises the error on \emph{observables}, not on these curves.",
                      "Exact histories and their fits. The fit minimises "
                      "the error on observables, not on these curves.", usetex),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #

def make_all(ds, prior, cfg, outdir: str, dpi: int = 300,
             usetex: bool = True) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    tex = setup_style(usetex)
    pre = os.path.join(outdir, cfg.name)
    made: List[str] = []
    for fn, suffix in (
            (lambda p: plot_slice(ds, p, tex), "slice.png"),
            (lambda p: plot_ansatz(ds, p, tex), "ansatz.png"),
            (lambda p: plot_fit_quality(ds, cfg, p, tex), "fit_quality.png"),
            (lambda p: plot_alphaB2(ds, p, tex), "alphaB2.png"),
            (lambda p: plot_histories(ds, cfg, p, tex), "histories.png"),
            (lambda p: plot_corner_params(ds, prior, p, tex), "corner.png"),
            (lambda p: plot_corner_X(ds, prior, p, tex), "corner_X.png")):
        path = f"{pre}_{suffix}"
        try:
            r = fn(path)
        except Exception as exc:                     # noqa: BLE001
            print(f"  skipped {suffix}: {type(exc).__name__}: {exc}")
            continue
        if r:
            made.append(r)
    return made
