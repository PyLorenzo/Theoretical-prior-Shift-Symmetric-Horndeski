"""Command line interface: ``python -m ssprior <subcommand>``.

Heavy imports (scipy solvers, matplotlib, camb, cobaya) live inside the
``cmd_*`` functions, so ``--help`` is instant and a command that does not need
EFTCAMB never touches the Fortran build.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from typing import Optional

__all__ = ["main", "build_parser"]


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _load_cfg(path: str):
    from .config import PriorConfig
    return PriorConfig.from_yaml(path)


def _default_out(cfg, suffix: str, given: Optional[str]) -> str:
    if given:
        return given
    os.makedirs(cfg.outdir, exist_ok=True)
    return cfg.path(suffix)


def _cfg_from_dataset(ds, override: Optional[str]):
    """Reuse the configuration stored in the dataset unless one is given."""
    from .config import PriorConfig
    if override:
        return _load_cfg(override)
    if not ds.config:
        raise SystemExit("this dataset carries no configuration; pass --config")
    return PriorConfig.from_dict(ds.config)


# --------------------------------------------------------------------------- #
#  sample
# --------------------------------------------------------------------------- #

def cmd_sample(args) -> int:
    from .pipeline import run_sampling

    cfg = _load_cfg(args.config)
    n = args.n or cfg.run.n_models
    workers = args.workers or cfg.run.workers
    out = _default_out(cfg, "samples.npz", args.output)

    print(cfg.summary())
    print(f"\ntarget {n} models, {workers} workers -> {out}\n")

    ds = run_sampling(cfg, n_target=n, workers=workers, output=out,
                      resume=args.resume, max_trials=args.max_trials,
                      progress=not args.no_progress)
    print()
    print(ds.summary())
    return 0 if len(ds) else 1


# --------------------------------------------------------------------------- #
#  fit
# --------------------------------------------------------------------------- #

def cmd_fit(args) -> int:
    """Re-run only the chi^2 minimisation, on the stored exact observables."""
    from .pipeline import run_refit
    from .store import Dataset

    ds = Dataset.load(args.samples)
    cfg = _cfg_from_dataset(ds, args.config)
    workers = args.workers or cfg.run.workers
    print(f"refitting {len(ds)} models on {workers} workers "
          f"(sigma = {cfg.fit.sigma_low_z:g} / {cfg.fit.sigma_rec:g})")
    run_refit(ds, cfg, workers=workers, progress=not args.no_progress)

    out = args.output or args.samples
    ds.save(out)
    print()
    print(ds.summary())
    print(f"\nwritten to {out}")
    return 0


# --------------------------------------------------------------------------- #
#  gaussianise
# --------------------------------------------------------------------------- #

def cmd_gaussianise(args) -> int:
    import json

    import numpy as np

    from .gaussianise import build_prior, optimise_exponents
    from .store import Dataset

    ds = Dataset.load(args.samples)
    cfg = _cfg_from_dataset(ds, args.config)
    # RUFIAN / Traykova et al. build the prior from every model whose fit could
    # be computed: the 1% / 0.3% targets are a diagnostic (Fig. 7), not a cut.
    mask = ds.good if args.accurate_only else np.ones(len(ds), dtype=bool)
    if args.stable_only:
        from .store import load_json_sidecar
        meta = load_json_sidecar(args.samples)
        flags = meta.get("eftcamb_stable")
        if flags is None:
            raise SystemExit("no EFTCAMB stability flags: run `ssprior eftcamb-check "
                             "--kind stability` first")
        mask = mask & np.asarray(flags, dtype=bool)
    sub = ds.subset(mask)
    print(f"using {len(sub)}/{len(ds)} models")
    if len(sub) < 20:
        raise SystemExit("too few models to fit a 4-D Gaussian")

    if args.optimise_exponents:
        p = optimise_exponents(sub.params)
        print(f"skewness-optimised exponents: p2={p[0]:.4f} p3={p[1]:.4f} p4={p[2]:.4f}"
              f"   (paper: 1/6 = {1/6:.4f}, 1/4 = 0.2500, 2)")

    from dataclasses import replace
    if args.optimise_exponents:
        cfg = replace(cfg, gaussianise=replace(cfg.gaussianise, optimise_exponents=True))
    if args.trim is not None:
        cfg = replace(cfg, gaussianise=replace(cfg.gaussianise, trim_quantile=args.trim))

    prior = build_prior(sub.params, cfg, name=cfg.name)
    out = _default_out(cfg, "prior.npz", args.output)
    prior.save(out)

    ref = None
    if args.reference and os.path.exists(args.reference):
        with open(args.reference) as fh:
            ref = json.load(fh).get(args.reference_key)
    print()
    print(prior.summary(ref))
    d = prior.diagnostics
    print(f"\n  Mardia skewness {d.get('mardia_skewness', float('nan')):.4f}   "
          f"kurtosis {d.get('mardia_kurtosis', float('nan')):.2f} "
          f"(Gaussian: {d.get('mardia_kurtosis_expected', float('nan')):.0f})")
    ks = d.get("ks_mahalanobis", {})
    print(f"  KS of the Mahalanobis distance vs chi^2_4: "
          f"D = {ks.get('statistic', float('nan')):.4f}, p = {ks.get('pvalue', float('nan')):.3g}")
    if "warning" in d:
        print(f"  WARNING: {d['warning']}")
    print(f"\nwritten to {out}")
    return 0


# --------------------------------------------------------------------------- #
#  validate
# --------------------------------------------------------------------------- #

def cmd_validate(args) -> int:
    from .validate import validate_prior
    from .store import Dataset, GaussianPrior

    ds = Dataset.load(args.samples)
    cfg = _cfg_from_dataset(ds, args.config)
    prior = GaussianPrior.load(args.prior)
    report = validate_prior(ds, prior, cfg, n_draw=args.n, seed=args.seed,
                            workers=args.workers, accurate_only=args.accurate_only)
    out = _default_out(cfg, "validation.json", args.output)
    import json
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(report["text"])
    print(f"\nwritten to {out}")
    return 0


# --------------------------------------------------------------------------- #
#  eftcamb-check
# --------------------------------------------------------------------------- #

def cmd_eftcamb_check(args) -> int:
    from . import eftcamb_check as ec
    from .store import Dataset

    ds = Dataset.load(args.samples)
    cfg = _cfg_from_dataset(ds, args.config)
    workers = args.workers or cfg.eftcamb.workers

    if args.kind == "background":
        rc = ec.check_background(ds, cfg, n=args.n)
    elif args.kind == "growth":
        rc = ec.check_growth(ds, cfg, n=args.n, calibrate=args.calibrate,
                             workers=workers, progress=not args.no_progress)
    else:
        rc = ec.check_stability(ds, cfg, samples_path=args.samples, workers=workers,
                                progress=not args.no_progress)
    return rc


# --------------------------------------------------------------------------- #
#  plot / export / info
# --------------------------------------------------------------------------- #

def cmd_plot(args) -> int:
    from . import plotting
    from .store import Dataset, GaussianPrior

    ds = Dataset.load(args.samples)
    cfg = _cfg_from_dataset(ds, args.config)
    prior = GaussianPrior.load(args.prior) if args.prior else None
    outdir = args.outdir or os.path.join(cfg.outdir, "plots")
    made = plotting.make_all(ds, prior, cfg, outdir, dpi=args.dpi,
                             usetex=not args.no_latex)
    for p in made:
        print("  wrote", p)
    return 0


def cmd_export(args) -> int:
    from .export import export_prior
    from .store import GaussianPrior

    prior = GaussianPrior.load(args.prior)
    paths = export_prior(prior, args.format, args.output, args.plugin_path,
                         prior_file=os.path.abspath(args.prior))
    for p in paths:
        print("  wrote", p)
    return 0


def cmd_info(args) -> int:
    import numpy as np

    from .store import Dataset, GaussianPrior, load_json_sidecar

    meta = load_json_sidecar(args.path)
    if meta.get("kind") == "gaussian_prior":
        prior = GaussianPrior.load(args.path)
        print(prior.summary())
        print("\n  correlation matrix:")
        for row in prior.corr:
            print("      " + "  ".join(f"{v:+7.4f}" for v in row))
        print("\n  diagnostics:", prior.diagnostics)
        return 0

    ds = Dataset.load(args.path)
    print(ds.summary())
    if len(ds):
        print("\n  derived quantities (median [5-95%]):")
        for nm in ("Omega_m", "Omega_phi", "alphaB0_exact", "w_phi0",
                   "z_rec", "psi0_sq_over_2"):
            v = ds.column(nm)
            print(f"   {nm:16s} {np.median(v):+9.4f}   "
                  f"[{np.percentile(v, 5):+9.4f}, {np.percentile(v, 95):+9.4f}]")
        cr = ds.column("crosses_alphaB2")
        print(f"\n  models crossing alpha_B = 2: {int(cr.sum())}/{len(ds)} "
              f"({100 * cr.mean():.1f}%)  [Appendix A]")
    return 0


# --------------------------------------------------------------------------- #
#  parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ssprior",
        description="Theoretical priors for shift-symmetric Horndeski "
                    "(Traykova et al. 2021) with EFTCAMB.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="draw Lagrangians, solve, fit the parametrisation")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("-n", type=int, default=None, help="number of accepted models")
    s.add_argument("-w", "--workers", type=int, default=None)
    s.add_argument("-o", "--output", default=None)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--max-trials", type=int, default=None)
    s.add_argument("--no-progress", action="store_true")
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("fit", help="re-run the chi^2 fit on stored observables")
    s.add_argument("samples")
    s.add_argument("-c", "--config", default=None)
    s.add_argument("-o", "--output", default=None)
    s.add_argument("-w", "--workers", type=int, default=None)
    s.add_argument("--no-progress", action="store_true")
    s.set_defaults(func=cmd_fit)

    s = sub.add_parser("gaussianise", help="transform, fit the multivariate normal")
    s.add_argument("samples")
    s.add_argument("-c", "--config", default=None)
    s.add_argument("-o", "--output", default=None)
    s.add_argument("--accurate-only", action="store_true",
                   help="drop models that missed the 1%%/0.3%% accuracy targets "
                        "(not done by the paper, which keeps every fitted model)")
    s.add_argument("--all", action="store_true",
                   help="include every fitted model (the default; kept for compatibility)")
    s.add_argument("--stable-only", action="store_true",
                   help="keep only models EFTCAMB declares stable")
    s.add_argument("--trim", type=float, default=None, metavar="Q",
                   help="drop points beyond this chi^2_4 quantile, e.g. 0.999")
    s.add_argument("--optimise-exponents", action="store_true",
                   help="re-derive the exponents of Eq. (23) by minimising skewness")
    s.add_argument("--reference", default=None,
                   help="JSON with the paper's mu and Sigma, for comparison")
    s.add_argument("--reference-key", default="lambda_zero")
    s.set_defaults(func=cmd_gaussianise)

    s = sub.add_parser("validate", help="resample the prior and compare observables")
    s.add_argument("samples")
    s.add_argument("-p", "--prior", required=True)
    s.add_argument("-c", "--config", default=None)
    s.add_argument("-n", type=int, default=0, help="draws (default: sample size)")
    s.add_argument("-o", "--output", default=None)
    s.add_argument("-w", "--workers", type=int, default=1)
    s.add_argument("--seed", type=int, default=12345)
    s.add_argument("--accurate-only", action="store_true",
                   help="compare against the accurate fits only (use it if the prior "
                        "was built with gaussianise --accurate-only)")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("eftcamb-check", help="cross-check against EFTCAMB")
    s.add_argument("samples")
    s.add_argument("--kind", choices=("background", "growth", "stability"),
                   default="background")
    s.add_argument("-c", "--config", default=None)
    s.add_argument("-n", type=int, default=50)
    s.add_argument("-w", "--workers", type=int, default=None)
    s.add_argument("--calibrate", action="store_true",
                   help="fit the alpha_B convention factor between EFTCAMB and "
                        "the Bellini-Sawicki basis")
    s.add_argument("--no-progress", action="store_true")
    s.set_defaults(func=cmd_eftcamb_check)

    s = sub.add_parser("plot", help="reproduce the figures")
    s.add_argument("samples")
    s.add_argument("-p", "--prior", default=None)
    s.add_argument("-c", "--config", default=None)
    s.add_argument("--outdir", default=None)
    s.add_argument("--dpi", type=int, default=300)
    s.add_argument("--no-latex", action="store_true")
    s.set_defaults(func=cmd_plot)

    s = sub.add_parser("export", help="write the prior in a usable form")
    s.add_argument("prior")
    s.add_argument("--format", choices=("json", "cobaya", "covmat"), default="cobaya")
    s.add_argument("-o", "--output", default=None)
    s.add_argument("--plugin-path", default=None,
                   help="python_path written into the Cobaya snippet")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("info", help="summarise any artefact")
    s.add_argument("path")
    s.set_defaults(func=cmd_info)

    return p


def main(argv=None) -> int:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
