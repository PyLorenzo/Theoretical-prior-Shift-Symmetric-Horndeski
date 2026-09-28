"""Cross-checks against EFTCAMB.

EFTCAMB is used in three places, all *outside* the chi^2 loop:

``background``  do H(z), D_A(z) and z_* from this package agree with CAMB?  The
                chi^2 weights observables at sigma = 1e-3, so a systematic
                offset at that level -- a different radiation density, a
                different neutrino treatment, a different z_* convention --
                would invalidate the whole fit.  Requirement: < 1e-4.

``growth``      how good is the quasi-static approximation, and -- the critical
                one -- what is the relation between EFTCAMB's alpha_B and the
                Bellini-Sawicki alpha_B of the paper?  EFTCAMB stores the
                braiding in the RPH basis and feeds it to the perturbations
                through ``EFTGamma2V`` (``007p7_ShiftSym_alphaB.f90:1069``); the
                normalisation relative to the alpha-basis is not documented in
                the source.  ``--calibrate`` fits a single scalar factor ``s``
                in ``alpha_B -> s alpha_B`` over a grid of (alphaB_hat, m).  The
                falsifiable claim is that ``s`` comes out *constant* over the
                grid; if it does not, the prior must not be exported.

``stability``   run EFTCAMB's ghost and gradient checks on the fitted four
                parameters of every model in the sample.  This is the part that
                has no counterpart in the paper and is the point of using
                EFTCAMB at all: it says what fraction of the theoretical prior
                the Boltzmann code that will run the MCMC actually accepts.

Why not inside the loop: a full ``get_results`` costs 0.33 s here, and the
cascade needs of order 150-200 evaluations per model, so 30k models would be
about 550 core-hours -- against minutes for the Python solver.  Worse, the
designer model imposes H(a) from the parametrisation, so it cannot represent the
exact Lagrangian background at all: using it in the loop would be comparing the
parametrised model against itself.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np

__all__ = ["check_background", "check_growth", "check_stability", "import_camb"]


def import_camb(build_path: str):
    """Import the EFTCAMB build, and make sure it is *that* build."""
    if not build_path:
        raise RuntimeError(
            "EFTCAMB is not configured: set eftcamb.build_path in the model YAML "
            "or export EFTCAMB_PATH=/path/to/EFTCAMB (the directory containing camb/).")
    build_path = os.path.abspath(build_path)
    if build_path not in sys.path:
        sys.path.insert(0, build_path)
    import camb
    if not os.path.abspath(camb.__file__).startswith(build_path):
        raise RuntimeError(
            f"imported camb from {camb.__file__}, not from {build_path}. "
            "Another camb is shadowing the EFTCAMB build on sys.path.")
    camb.set_feedback_level(0)
    return camb


def eft_params(cfg, alphaB_hat: float, m: float, w0: float, wa: float) -> Dict:
    """The EFTCAMB dictionary for the designer shift-symmetric model."""
    return {
        "EFTflag": 3,
        "DesignerEFTmodel": 3,
        "EFTwDE": 2,                       # CPL
        "Shift_Symmetric_alphaB0": float(alphaB_hat),
        "Shift_Symmetric_m": float(m),
        "Shift_Symmetric_alphaK0": float(cfg.eftcamb.alphaK0),
        "EFTw0": float(w0),
        "EFTwa": float(wa),
        "EFT_ghost_math_stability": False,
        "EFT_mass_math_stability": False,
        "EFT_ghost_stability": True,
        "EFT_gradient_stability": True,
        "EFT_mass_stability": False,
        "EFT_additional_priors": False,
        "EFTCAMB_turn_on_time": 1.0e-8,
        "feedback_level": 0,
    }


def _cosmo_kwargs(cfg, H0: float, Omega_m: float) -> Dict:
    h2 = (H0 / 100.0) ** 2
    ombh2 = cfg.cosmology.ombh2_of(H0)
    return dict(H0=float(H0), ombh2=float(ombh2),
                omch2=float(Omega_m * h2 - ombh2),
                mnu=0.0, tau=float(cfg.eftcamb.tau),
                As=float(cfg.eftcamb.As), ns=float(cfg.eftcamb.ns),
                nnu=float(cfg.cosmology.N_eff), TCMB=float(cfg.cosmology.T_cmb))


# --------------------------------------------------------------------------- #
#  background
# --------------------------------------------------------------------------- #

def check_background(ds, cfg, n: int = 50) -> int:
    """Compare E(z), the comoving distance and z_* against CAMB.  Blocking test."""
    camb = import_camb(cfg.eftcamb.build_path)

    from .observables import build_grid, z_star_hu_sugiyama
    from .parametrised import E_cpl
    from .observables import _comoving_distance

    sub = ds.subset(ds.good)
    n = min(n, len(sub))
    idx = np.linspace(0, len(sub) - 1, n).astype(int)
    N_quad = np.linspace(np.log(cfg.model.a_ini), 0.0, cfg.background.n_grid)

    dE, dD, dz = [], [], []
    print(f"comparing {n} models against EFTCAMB "
          f"({os.path.basename(cfg.eftcamb.build_path)})\n")
    for i in idx:
        w0, wa, aB, m = sub.params[i]
        H0 = sub.column("H0")[i]
        Om = sub.column("Omega_m")[i]
        pars = camb.set_params(dark_energy_model="EFTCAMB",
                               **_cosmo_kwargs(cfg, H0, Om),
                               **eft_params(cfg, aB, m, w0, wa))
        try:
            bg = camb.get_background(pars, no_thermo=False)
        except Exception as exc:                     # noqa: BLE001
            print(f"  model {i}: EFTCAMB refused ({type(exc).__name__}: "
                  f"{str(exc).splitlines()[-1][:70]})")
            continue

        grid = build_grid(cfg, cfg.cosmology.ombh2_of(H0), Om * (H0 / 100.0) ** 2, N_quad)
        E_q, _ = E_cpl(N_quad, Om, sub.column("Omega_r")[i], w0, wa)
        from scipy.interpolate import CubicSpline
        E_mine = CubicSpline(N_quad, E_q)(grid.N_z)
        D_mine = _comoving_distance(N_quad, E_q, H0, grid.N_z)

        E_camb = bg.hubble_parameter(grid.z) / H0
        D_camb = bg.comoving_radial_distance(grid.z)
        dE.append(np.abs(E_mine / E_camb - 1.0).max())
        pos = grid.z > 0.0                      # chi(0) = 0 on both sides
        dD.append(np.abs(D_mine[pos] / D_camb[pos] - 1.0).max())
        try:
            z_camb = bg.get_derived_params()["zstar"]
            dz.append(abs(z_star_hu_sugiyama(cfg.cosmology.ombh2_of(H0),
                                             Om * (H0 / 100.0) ** 2) / z_camb - 1.0))
        except Exception:                            # noqa: BLE001
            pass

    if not dE:
        print("no model could be evaluated by EFTCAMB")
        return 1

    def line(name, v, target):
        v = np.asarray(v)
        ok = "PASS" if np.median(v) < target else "FAIL"
        print(f"  {name:22s} median {np.median(v):.3e}  max {v.max():.3e}   "
              f"(target {target:.0e})  {ok}")
        return np.median(v) < target

    print("relative deviation, Python vs EFTCAMB:")
    ok = line("E(z), z < 10", dE, 1e-4)
    ok &= line("comoving distance", dD, 1e-4)
    if dz:
        line("z_star (Hu-Sugiyama)", dz, 5e-3)
    print("\nE(z) and the distances must agree: both sides use the same CPL\n"
          "background, so any discrepancy is a mismatch in the radiation density,\n"
          "the neutrino treatment or the w_DE integral convention -- and it would\n"
          "sit right on top of the sigma = 1e-3 weighting of Eq. (21).")
    return 0 if ok else 2


# --------------------------------------------------------------------------- #
#  growth / alpha_B convention
# --------------------------------------------------------------------------- #

def _growth_setup(build_path, cfg_pickle):
    """Worker-side factory (picklable): returns the per-task callable."""
    import pickle

    cfg = pickle.loads(cfg_pickle)
    camb = import_camb(build_path)

    def evaluate(payload):
        aB, m, w0, wa, H0, Om, zlist = payload
        pars = camb.set_params(dark_energy_model="EFTCAMB", lmax=500,
                               WantTransfer=True,
                               **_cosmo_kwargs(cfg, H0, Om),
                               **eft_params(cfg, aB, m, w0, wa))
        pars.set_matter_power(redshifts=list(zlist)[::-1], kmax=2.0)
        pars.NonLinear = 0
        r = camb.get_results(pars)
        s8 = np.asarray(r.get_sigma8())
        fs8 = np.asarray(r.get_fsigma8())
        # CAMB returns these in order of increasing time (decreasing z)
        return (fs8 / s8)[::-1]

    return evaluate


def check_growth(ds, cfg, n: int = 100, calibrate: bool = False,
                 workers: int = 4, progress: bool = True) -> int:
    """Quasi-static f(z) against EFTCAMB, and the alpha_B convention factor."""
    import pickle
    from functools import partial

    from .executor import HangSafePool
    from .growth import GradientInstability
    from .observables import build_grid
    from .parametrised import observables_param

    sub = ds.subset(ds.good)
    n = min(n, len(sub))
    idx = np.linspace(0, len(sub) - 1, n).astype(int)
    z_probe = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    N_quad = np.linspace(np.log(cfg.model.a_ini), 0.0, cfg.background.n_grid)

    payloads = []
    meta = []
    for i in idx:
        w0, wa, aB, m = sub.params[i]
        H0, Om, Orad = (sub.column("H0")[i], sub.column("Omega_m")[i],
                        sub.column("Omega_r")[i])
        payloads.append((aB, m, w0, wa, H0, Om, z_probe))
        meta.append((w0, wa, aB, m, H0, Om, Orad))

    setup = partial(_growth_setup, cfg.eftcamb.build_path, pickle.dumps(cfg))
    print(f"running EFTCAMB on {n} models ({workers} workers)...")
    with HangSafePool(setup, workers=workers, timeout=cfg.eftcamb.timeout) as pool:
        results = pool.run(payloads, desc="eftcamb growth", progress=progress)

    def f_python(scale, w0, wa, aB, m, H0, Om, Orad):
        grid = build_grid(cfg, cfg.cosmology.ombh2_of(H0), Om * (H0 / 100.0) ** 2, N_quad)
        o = observables_param((w0, wa, scale * aB, 4.0 / m), Om, Orad, H0, grid, cfg)
        from scipy.interpolate import CubicSpline
        return CubicSpline(grid.N_z[::-1], o.f[::-1])(-np.log1p(z_probe))

    def residual_for(scale, rows=None, refs=None):
        """Paired (python, eftcamb) f(z) tables at a given alpha_B rescaling."""
        rows = meta if rows is None else rows
        refs = results if refs is None else refs
        num, ref = [], []
        for (val, err, _), mrow in zip(refs, rows):
            if val is None or not np.all(np.isfinite(val)):
                continue
            try:
                fp = f_python(scale, *mrow)
            except (GradientInstability, ValueError):
                continue
            if not np.all(np.isfinite(fp)):
                continue
            num.append(fp)
            ref.append(np.asarray(val, dtype=float))
        if not num:
            return None, None
        return np.array(num), np.array(ref)

    fp, fe = residual_for(1.0)
    n_ok = 0 if fp is None else fp.shape[0]
    print(f"\n{n_ok}/{n} models evaluated by both "
          f"(timeouts {pool.timeouts}, deaths {pool.deaths})")
    if n_ok < 5:
        print("too few successful evaluations to draw any conclusion")
        return 1

    rel = np.abs(fp / fe - 1.0)
    print("\nquasi-static f(z) vs EFTCAMB fsigma8/sigma8, scale factor 1.0:")
    print(f"  {'z':>6s} {'median |df/f|':>15s} {'95th pct':>12s}")
    for j, z in enumerate(z_probe):
        print(f"  {z:6.2f} {np.median(rel[:, j]):15.4%} {np.percentile(rel[:, j], 95):12.4%}")

    rc = 0
    if calibrate:
        from scipy.optimize import minimize_scalar
        print("\ncalibrating the alpha_B convention factor...")

        def cost_on(rows, refs):
            def cost(s):
                a, b = residual_for(s, rows, refs)
                if a is None:
                    return 1e6
                return float(np.mean((a / b - 1.0) ** 2))
            return cost

        def best_scale(rows, refs) -> float:
            """Coarse grid, then a local refinement.

            The cost is extremely sharply peaked -- when the two codes really do
            use the same alpha_B, rescaling it by a part in a thousand already
            shows up -- and a bracketing method started on the full [0.2, 5]
            range walks straight past the minimum.  So scan first.
            """
            fn = cost_on(rows, refs)
            grid = np.concatenate([np.linspace(0.2, 5.0, 97),
                                   np.linspace(0.9, 1.1, 41)])
            vals = np.array([fn(s) for s in grid])
            s0 = float(grid[np.argmin(vals)])
            half = 0.05 * max(s0, 1.0)
            r = minimize_scalar(fn, bounds=(max(0.01, s0 - half), s0 + half),
                                method="bounded", options={"xatol": 1e-6})
            return float(r.x) if fn(r.x) < vals.min() else s0

        cost = cost_on(meta, results)
        s_best = best_scale(meta, results)
        # the falsifiable part: a genuine convention factor is the same on any
        # subset, so refit it on each half and compare
        half = len(meta) // 2
        halves = [best_scale(meta[:half], results[:half]),
                  best_scale(meta[half:], results[half:])]
        spread = (max(halves) - min(halves)) / s_best if s_best else np.nan
        print(f"  best-fit s = {s_best:.4f}   "
              f"(halves: {', '.join(f'{h:.4f}' for h in halves)}, "
              f"spread {spread:.2%})")
        print(f"  residual at s = 1:      {np.sqrt(cost(1.0)):.4%} rms")
        print(f"  residual at s = s_best: {np.sqrt(cost(s_best)):.4%} rms")
        if abs(s_best - 1.0) < 0.05 and spread < 0.05:
            print("\n  s is 1 within the noise: EFTCAMB's Shift_Symmetric_alphaB0 is\n"
                  "  the same alpha_B as the paper's, and the prior can be exported\n"
                  "  in its own units.")
        elif spread < 0.05:
            print(f"\n  s is constant but not 1. Set\n"
                  f"      eftcamb.alphaB_convention_factor: {s_best:.6f}\n"
                  "  in the config before exporting the prior.")
            rc = 2
        else:
            print("\n  s is NOT constant across the sample. The mismatch is not a\n"
                  "  simple normalisation; do not export the prior until this is\n"
                  "  understood.")
            rc = 3

    print("\nThe quasi-static approximation is expected to agree at the percent\n"
          "level, not at 1e-3: f from EFTCAMB is fsigma8/sigma8, a scale-averaged\n"
          "quantity from the full perturbation equations. This is a consistency\n"
          "check, not a regression test.")
    return rc


# --------------------------------------------------------------------------- #
#  stability
# --------------------------------------------------------------------------- #

def _stability_setup(build_path, cfg_pickle):
    import pickle

    cfg = pickle.loads(cfg_pickle)
    camb = import_camb(build_path)
    from camb.baseconfig import CAMBError

    def evaluate(payload):
        aB, m, w0, wa, H0, Om = payload
        try:
            pars = camb.set_params(dark_energy_model="EFTCAMB",
                                   **_cosmo_kwargs(cfg, H0, Om),
                                   **eft_params(cfg, aB, m, w0, wa))
            camb.get_background(pars, no_thermo=True)
        except CAMBError as exc:
            msg = str(exc).lower()
            if "unstable" in msg:
                return "unstable"
            if "background solver" in msg:
                return "background_failed"
            return "camb_error"
        except ValueError:
            return "init_failed"
        return "stable"

    return evaluate


def check_stability(ds, cfg, samples_path: str, workers: int = 8,
                    progress: bool = True) -> int:
    """Label every model with EFTCAMB's ghost / gradient verdict."""
    import pickle
    from collections import Counter
    from functools import partial

    from .executor import HangSafePool
    from .store import load_json_sidecar, save_json_sidecar

    payloads = [(ds.params[i, 2], ds.params[i, 3], ds.params[i, 0], ds.params[i, 1],
                 ds.column("H0")[i], ds.column("Omega_m")[i]) for i in range(len(ds))]

    setup = partial(_stability_setup, cfg.eftcamb.build_path, pickle.dumps(cfg))
    print(f"EFTCAMB stability check on {len(payloads)} models, {workers} workers")
    with HangSafePool(setup, workers=workers, timeout=cfg.eftcamb.timeout) as pool:
        results = pool.run(payloads, desc="eftcamb stability", progress=progress)

    verdicts = []
    for value, err, _ in results:
        verdicts.append(value if value is not None else ("timeout" if err == "eval timeout"
                                                         else "error"))
    counts = Counter(verdicts)
    stable = np.array([v == "stable" for v in verdicts], dtype=bool)

    print("\nverdicts:")
    for k, v in counts.most_common():
        print(f"  {k:20s} {v:6d}  ({100 * v / len(verdicts):.1f}%)")
    print(f"\n{stable.sum()}/{len(stable)} ({100 * stable.mean():.1f}%) of the "
          "theoretical prior is accepted by EFTCAMB's ghost and gradient checks.")
    print("This has no counterpart in the paper: it is what the Boltzmann code\n"
          "that will run the MCMC actually allows. Use `ssprior gaussianise\n"
          "--stable-only` to fit the prior on this subset.")

    meta = load_json_sidecar(samples_path)
    meta["eftcamb_stable"] = stable.tolist()
    meta["eftcamb_verdicts"] = dict(counts)
    meta["eftcamb_timeouts"] = int(pool.timeouts)
    save_json_sidecar(samples_path, meta)
    print(f"\nflags written to {os.path.splitext(samples_path)[0]}.json")
    return 0
