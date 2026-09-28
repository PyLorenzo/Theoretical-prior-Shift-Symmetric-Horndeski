"""One model, end to end: draw -> slice -> background -> growth -> observables -> fit.

``evaluate_model`` is a module-level function so it can be pickled to worker
processes; the configuration reaches them once through ``init_worker`` rather
than being re-pickled with every task.

Rejections are counted by cause and reported, never swallowed.  The acceptance
rate of each filter is itself prior information: a box that is rejected 99% of
the time is telling you the box is wrong, and a silent ``except: pass`` would
hide it.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .background import BgStatus, slice_candidates, solve_background
from .fitting import FitStatus, fit_parametrisation, initial_guess
from .growth import GradientInstability
from .lagrangian import Lagrangian
from .observables import build_grid, observables_from_background
from .store import ModelRecord

__all__ = ["init_worker", "evaluate_model", "run_sampling", "REJECT_REASONS"]

REJECT_REASONS = (
    "no_slice",          # no root of the tracker cubic gives an acceptable Omega_cdm
    "bg_failed",         # continuation failed (fold, Newton, asymptote, ...)
    "ghost",             # alpha_K + 1.5 alpha_B^2 <= 0 somewhere
    "gradient",          # cs2N2 <= 0 somewhere
    "growth_failed",     # growth integration blew up
    "fit_failed",        # the parametrised model could not be evaluated
)

_CFG = None


def init_worker(cfg) -> None:
    """Runs once per worker process."""
    global _CFG
    _CFG = cfg


def evaluate_model(theta, cfg=None) -> ModelRecord:
    """Evaluate one Lagrangian.  ``theta = (c01, c02, d02, H0)``."""
    cfg = cfg if cfg is not None else _CFG
    t0 = time.time()
    theta = np.asarray(theta, dtype=float)
    c01, c02, d02, H0 = theta

    lag = Lagrangian(c01=float(c01), c02=float(c02), d02=float(d02),
                     d01=cfg.model.d01)

    candidates = slice_candidates(lag, float(H0), cfg)
    if not candidates:
        return ModelRecord(theta, False, "no_slice")

    sol = None
    for psi0, cosmo in candidates:
        trial = solve_background(lag, cosmo, psi0, cfg)
        if trial.ok:
            sol = trial
            break
    if sol is None:
        return ModelRecord(theta, False, "bg_failed")
    n_candidates = len(candidates)

    # --- stability priors of Sec. III, exact and analytic on this side -------
    if np.any(sol.kinetic_D <= 0.0) or np.any(sol.rho_phi <= 0.0):
        return ModelRecord(theta, False, "ghost")

    omh2 = sol.cosmo.Omega_m * sol.cosmo.h ** 2
    grid = build_grid(cfg, sol.cosmo.ombh2, omh2, sol.N)

    try:
        exact = observables_from_background(sol, grid, cfg)
    except GradientInstability:
        return ModelRecord(theta, False, "gradient")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return ModelRecord(theta, False, "growth_failed")
    if not np.all(np.isfinite(exact.as_vector())):
        return ModelRecord(theta, False, "growth_failed")

    guess = initial_guess(sol, cfg)
    fit = fit_parametrisation(exact, grid, sol.cosmo, cfg, guess)
    if fit.status is FitStatus.FAILED:
        return ModelRecord(theta, False, "fit_failed")

    # phi ddot / Lambda3^3 today, for the Fig. 2 consistency check of the ansatz
    phiddot0 = float(sol.E[-1] * sol.dpsi_dN[-1])

    derived = np.array([
        sol.cosmo.Omega_m, sol.cosmo.Omega_cdm, sol.cosmo.Omega_r,
        sol.cosmo.Omega_phi, sol.psi0, sol.alpha_B[-1], sol.alpha_K[-1],
        sol.w_phi[-1], grid.z_rec, sol.kinetic_D[-1],
        float(sol.crosses_alpha_B_2()), 0.5 * sol.psi0 ** 2, phiddot0,
        guess[0], guess[1], guess[2], guess[3],
    ], dtype=float)

    quality = np.array([
        fit.chi2, fit.errors.get("low_z", np.nan), fit.errors.get("rec", np.nan),
        float(fit.status), float(BgStatus.OK), fit.nfev, fit.n_starts,
        fit.cond_JtJ, n_candidates, time.time() - t0,
    ], dtype=float)

    return ModelRecord(theta=theta, accepted=True, reason="",
                       derived=derived, params=fit.as_array(),
                       quality=quality, obs=exact.as_vector())


# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #

def run_sampling(cfg, n_target: int, workers: int, output: str,
                 batch: int = 0, max_trials: Optional[int] = None,
                 resume: bool = False, progress: bool = True,
                 verbose: bool = True):
    """Draw and evaluate until ``n_target`` models are accepted.

    Works in batches so that the checkpoint on disk is never more than one batch
    behind, and so an unproductive box is caught early rather than after hours.
    """
    import os
    from collections import Counter

    from .executor import parallel_map
    from .sampling import DrawStream
    from .store import Dataset

    dataset = None
    n_drawn = 0
    counters = Counter()
    if resume and os.path.exists(output):
        dataset = Dataset.load(output)
        counters.update(dataset.counters)
        n_drawn = int(counters.get("drawn", 0))
        if verbose:
            print(f"resuming from {output}: {len(dataset)} models, {n_drawn} draws")

    stream = DrawStream(cfg, n_drawn=n_drawn)
    # a 5% acceptance rate is typical for the calibrated box; oversample so each
    # batch actually delivers something, but keep batches small enough to
    # checkpoint often
    if batch <= 0:
        batch = max(256, min(8192, 20 * cfg.run.checkpoint_every))
    if max_trials is None:
        max_trials = 200 * n_target

    have = len(dataset) if dataset is not None else 0
    z_grid = None
    t_start = time.time()

    while have < n_target and counters.get("drawn", 0) < max_trials:
        want = n_target - have
        n_draw = int(min(batch, max(256, 30 * want)))
        points = stream.draw(n_draw)
        counters["drawn"] += n_draw

        records = parallel_map(
            evaluate_model, points, workers=workers, chunksize=4,
            desc=f"sampling ({have}/{n_target})", progress=progress,
            initializer=init_worker, initargs=(cfg,))

        for r in records:
            if not r.accepted:
                counters[r.reason] += 1
        accepted = [r for r in records if r.accepted]
        counters["accepted"] += len(accepted)

        if z_grid is None:
            z_grid = _z_grid(cfg)
        new = Dataset.from_records(accepted, z_grid, cfg.to_dict(),
                                   dict(counters), cfg.name)
        dataset = new if dataset is None else dataset.concat(new)
        dataset.counters = dict(counters)
        # trim any overshoot so the sample size is exactly what was asked for
        if len(dataset) > n_target:
            dataset = dataset.subset(np.arange(len(dataset)) < n_target)
            dataset.counters = dict(counters)
        dataset.save(output)
        have = len(dataset)
        if verbose:
            rate = 100.0 * counters["accepted"] / max(counters["drawn"], 1)
            print(f"  {have}/{n_target} accepted   acceptance {rate:.2f}%   "
                  f"{time.time() - t_start:.0f}s elapsed")

    if verbose and have < n_target:
        print(f"WARNING: stopped at {have}/{n_target} after {counters['drawn']} draws "
              f"(max_trials reached). The prior box is probably too wide.")
    return dataset


def _z_grid(cfg) -> np.ndarray:
    return cfg.grid.z_nodes()


# --------------------------------------------------------------------------- #
#  refitting stored observables
# --------------------------------------------------------------------------- #

def refit_one(payload):
    """Re-run the chi^2 cascade for one stored model.  Picklable, worker-side.

    ``payload`` carries everything the fit needs -- the exact observable vectors,
    the derived cosmology and the stage-0 guess -- so no background has to be
    re-solved.
    """
    from .background import Cosmology
    from .fitting import fit_parametrisation
    from .observables import Observables, build_grid

    cfg = _CFG
    (E, DA, f, DA_rec, z_rec, Omega_m, Omega_r, H0, guess) = payload
    cosmo = Cosmology(Omega_m=Omega_m, Omega_r=Omega_r, Omega_L=0.0, H0=H0,
                      H_fid=cfg.cosmology.H_fid, ombh2=cfg.cosmology.ombh2_of(H0))
    N_quad = np.linspace(np.log(cfg.model.a_ini), 0.0, cfg.background.n_grid)
    grid = build_grid(cfg, cosmo.ombh2, cosmo.Omega_m * cosmo.h ** 2, N_quad)
    exact = Observables(E=E, DA=DA, f=f, DA_rec=float(DA_rec),
                        z_rec=float(z_rec), H0=H0)
    res = fit_parametrisation(exact, grid, cosmo, cfg, tuple(guess))
    return (res.as_array(), res.chi2, res.errors.get("low_z", np.nan),
            res.errors.get("rec", np.nan), float(res.status), res.nfev,
            res.n_starts, res.cond_JtJ)


def run_refit(ds, cfg, workers: int = 1, progress: bool = True):
    """Re-fit a whole dataset in parallel, in place."""
    from .executor import parallel_map
    from .store import QUALITY_NAMES

    # the stored observables live on the dataset's own z grid: refitting them
    # with a configuration that defines another grid would pair each value with
    # the wrong redshift, silently
    if ds.z.shape != _z_grid(cfg).shape or not np.allclose(ds.z, _z_grid(cfg)):
        raise ValueError("the configuration's redshift grid differs from the one the "
                         "observables were computed on; re-run `ssprior sample`")

    E, DA, f, DA_rec = ds.split_obs()
    payloads = [
        (E[i], DA[i], f[i], DA_rec[i], ds.column("z_rec")[i],
         ds.column("Omega_m")[i], ds.column("Omega_r")[i], ds.column("H0")[i],
         [ds.column(k)[i] for k in
          ("guess_w0", "guess_wa", "guess_alphaB", "guess_u")])
        for i in range(len(ds))
    ]
    out = parallel_map(refit_one, payloads, workers=workers, chunksize=4,
                       desc="refitting", progress=progress,
                       initializer=init_worker, initargs=(cfg,))

    params = np.array([o[0] for o in out], dtype=float)
    quality = ds.quality.copy()
    cols = ("chi2", "err_low_z", "err_rec", "fit_status", "nfev",
            "n_starts", "cond_JtJ")
    for j, name in enumerate(cols, start=1):
        quality[:, QUALITY_NAMES.index(name)] = [o[j] for o in out]
    ds.params, ds.quality = params, quality
    return ds
