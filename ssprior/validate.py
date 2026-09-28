"""Does the fitted Gaussian reproduce the *observables*?

Fig. 10 of the paper.  A good fit to the parameter distribution is not the
point: the prior is only useful if a cosmology drawn from it looks like a
cosmology drawn from the underlying theory.  So we draw from the multivariate
normal, map back to (w0, wa, alphaB_hat, m), evaluate the parametrised model and
compare the distributions of H(z=1), D_A(z=1), f(z=1), z_rec and D_A(z_rec)
against the exact sample, with a two-sample Kolmogorov-Smirnov test.

The factorisation assumption is also measured, not assumed.  In the exact
sample Omega_m is *derived* from (c01, c02, d02): it is not an independent
parameter.  Using the prior in an MCMC that samples Omega_m freely treats the
two as independent, so the report includes the correlations between Omega_m and
each X_i.  Large values would mean the prior is throwing away information.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["validate_prior"]

_OBS_LABELS = ("H(z=1)", "D_A(z=1)", "f(z=1)", "z_rec", "D_A(z_rec)")


def _exact_observables(ds) -> np.ndarray:
    """(N, 5) table of the summary observables of the exact sample."""
    E, DA, f, DA_rec = ds.split_obs()
    z = ds.z
    i1 = int(np.argmin(np.abs(z - 1.0)))
    H0 = ds.column("H0")
    return np.column_stack([E[:, i1] * H0, DA[:, i1], f[:, i1],
                            ds.column("z_rec"), DA_rec])


def _draw_observables(params, Omega_m, Omega_r, H0, cfg) -> np.ndarray:
    from .growth import GradientInstability
    from .observables import build_grid
    from .parametrised import observables_param

    N_quad = np.linspace(np.log(cfg.model.a_ini), 0.0, cfg.background.n_grid)
    rows = []
    for th, om, orad, h0 in zip(params, Omega_m, Omega_r, H0):
        w0, wa, aB, m = th
        grid = build_grid(cfg, cfg.cosmology.ombh2_of(h0), om * (h0 / 100.0) ** 2, N_quad)
        try:
            o = observables_param((w0, wa, aB, 4.0 / m), om, orad, h0, grid, cfg)
        except (GradientInstability, ValueError, FloatingPointError):
            continue
        i1 = int(np.argmin(np.abs(grid.z - 1.0)))
        if not np.all(np.isfinite([o.E[i1], o.DA[i1], o.f[i1], o.DA_rec])):
            continue
        rows.append([o.E[i1] * h0, o.DA[i1], o.f[i1], grid.z_rec, o.DA_rec])
    return np.array(rows) if rows else np.zeros((0, 5))


def validate_prior(ds, prior, cfg, n_draw: int = 0, seed: int = 12345,
                   workers: int = 1, accurate_only: bool = False) -> Dict:
    """Compare exact and Gaussian-drawn observable distributions.

    The exact side is the same population the prior was fitted on: every
    fitted model by default, as in RUFIAN, or only the accurate fits.
    """
    from scipy.stats import ks_2samp

    from .gaussianise import sample_prior, to_X

    good = ds.good if accurate_only else np.ones(len(ds), dtype=bool)
    sub = ds.subset(good)
    n_draw = n_draw or len(sub)
    rng = np.random.default_rng(seed)

    # the factorisation assumption made explicit: each draw of the four
    # parameters is paired with an (Omega_m, H0) bootstrapped from the exact
    # sample, exactly as an MCMC would treat them as independent
    idx = rng.integers(0, len(sub), size=n_draw)
    Om = sub.column("Omega_m")[idx]
    Orad = sub.column("Omega_r")[idx]
    H0 = sub.column("H0")[idx]

    params = sample_prior(prior, n_draw, rng=rng)
    n_ok = params.shape[0]
    if n_ok < n_draw:
        Om, Orad, H0 = Om[:n_ok], Orad[:n_ok], H0[:n_ok]

    exact = _exact_observables(sub)
    drawn = _draw_observables(params, Om, Orad, H0, cfg)

    report: Dict = {
        "n_exact": int(exact.shape[0]),
        "n_drawn_requested": int(n_draw),
        "n_drawn_evaluated": int(drawn.shape[0]),
        "observables": {},
    }
    lines = [f"validation of '{prior.name}': {exact.shape[0]} exact vs "
             f"{drawn.shape[0]} drawn cosmologies", ""]
    lines.append(f"  {'observable':12s} {'KS':>8s} {'p':>10s}   "
                 f"{'median exact':>14s} {'median drawn':>14s}")
    for i, label in enumerate(_OBS_LABELS):
        if drawn.shape[0] < 10:
            continue
        ks = ks_2samp(exact[:, i], drawn[:, i])
        entry = {"ks": float(ks.statistic), "pvalue": float(ks.pvalue),
                 "median_exact": float(np.median(exact[:, i])),
                 "median_drawn": float(np.median(drawn[:, i]))}
        report["observables"][label] = entry
        lines.append(f"  {label:12s} {entry['ks']:8.4f} {entry['pvalue']:10.3g}   "
                     f"{entry['median_exact']:14.4f} {entry['median_drawn']:14.4f}")

    # how good is the factorisation?
    X = to_X(sub.params, prior.exponents["p2"], prior.exponents["p3"],
             prior.exponents["p4"])
    corr = {}
    for j, nm in enumerate(prior.x_names):
        corr[f"Omega_m-{nm}"] = float(np.corrcoef(sub.column("Omega_m"), X[:, j])[0, 1])
        corr[f"H0-{nm}"] = float(np.corrcoef(sub.column("H0"), X[:, j])[0, 1])
    report["factorisation_correlations"] = corr
    lines += ["", "  correlations with the derived background parameters",
              "  (the prior is used as if these factorised; large values mean "
              "information is being discarded)"]
    for k, v in corr.items():
        flag = "  <-- strong" if abs(v) > 0.5 else ""
        lines.append(f"    {k:14s} {v:+.3f}{flag}")

    worst = max((e["ks"] for e in report["observables"].values()), default=np.nan)
    report["max_ks"] = float(worst)
    lines += ["", f"  worst KS statistic: {worst:.4f}"
                  f"   {'PASS' if worst < 0.05 else 'check the transform'}"]
    report["text"] = "\n".join(lines)
    return report
