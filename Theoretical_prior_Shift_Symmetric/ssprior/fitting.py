"""Minimisation of the chi^2 of Eq. (21): exact cosmology -> {w0, wa, alphaB_hat, m}.

Optimiser
---------
``scipy.optimize.least_squares(method='trf')`` on the residual *vector*, not
``minimize`` on the scalar chi^2.  Gauss-Newton exploits the sum-of-squares
structure, ``trf`` handles the bounds natively, and the returned Jacobian gives
``J^T J`` -- half the Hessian -- for free, from which the condition number
diagnoses degenerate directions without any extra work.

Initialisation cascade
----------------------
The parametrised expansion history depends *only* on (w0, wa): alpha_B and m
enter nowhere in H(a) or D_A(z).  All of the information on (alphaB_hat, m) sits
in f(z).  The fit therefore separates almost completely, and is done in stages:

  0. closed form, no function evaluations at all -- w0 and wa from w_phi and its
     derivative today, alphaB_hat from alpha_B today, and u = 4/m from a linear
     regression of ln alpha_B against ln E over z < 10 (Eq. (17) is a straight
     line in those variables);
  1. (w0, wa) against E, D_A and D_A(z_rec) only -- no ODE, analytic residuals.
     This stage is not merely an initialisation: it is exact;
  2. (alphaB_hat, u) against f(z), with (w0, wa) frozen;
  3. joint refinement on all four, starting from 1+2.

If the paper's acceptance criteria are still not met, stages 2+3 are repeated
from a spread of starting values of u and the best chi^2 is kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.optimize import least_squares

from .growth import GradientInstability
from .observables import Observables, rel_residual, relative_errors, residuals
from .parametrised import ParamModel, E_cpl, observables_param
from .observables import _comoving_distance

__all__ = ["FitStatus", "FitResult", "fit_parametrisation", "initial_guess"]


class FitStatus(IntEnum):
    OK = 0              # meets the paper's 1% / 0.3% targets
    POOR = 1            # converged but outside the targets
    AT_BOUND = 2        # a parameter is pinned to a bound
    NOT_CONVERGED = 3   # optimiser gave up
    FAILED = 4          # the parametrised model could not be evaluated at all


@dataclass
class FitResult:
    params: ParamModel
    chi2: float
    errors: dict
    status: FitStatus
    nfev: int
    n_starts: int
    cond_JtJ: float

    @property
    def ok(self) -> bool:
        return self.status is FitStatus.OK

    def as_array(self) -> np.ndarray:
        return self.params.as_array()


# --------------------------------------------------------------------------- #
#  stage residuals
# --------------------------------------------------------------------------- #

def _bg_residuals(x2, exact: Observables, Omega_m, Omega_r, H0, grid, sigma_low, sigma_rec):
    """Residuals of E, D_A and D_A(z_rec).  Independent of (alphaB_hat, m)."""
    w0, wa = x2
    E_q, _ = E_cpl(grid.N_quad, Omega_m, Omega_r, w0, wa)
    if not np.all(np.isfinite(E_q)) or np.any(E_q <= 0.0):
        return np.full(2 * exact.E.size + 1, 1e6)
    from scipy.interpolate import CubicSpline
    E_z = CubicSpline(grid.N_quad, E_q)(grid.N_z)
    DA_z = _comoving_distance(grid.N_quad, E_q, H0, grid.N_z)
    DA_rec = float(_comoving_distance(grid.N_quad, E_q, H0,
                                      np.array([np.log(grid.a_rec)]))[0])
    return np.concatenate([
        rel_residual(E_z, exact.E, sigma_low),
        rel_residual(DA_z, exact.DA, sigma_low),
        np.atleast_1d(rel_residual(DA_rec, exact.DA_rec, sigma_rec)),
    ])


def _f_residuals(x2, w0, wa, exact, Omega_m, Omega_r, H0, grid, cfg, sigma_low):
    """Residuals of f(z) at fixed (w0, wa)."""
    aB_hat, u = x2
    try:
        pm = observables_param((w0, wa, aB_hat, u), Omega_m, Omega_r, H0, grid, cfg)
    except (GradientInstability, FloatingPointError, ValueError):
        return np.full(exact.f.size, 1e6)
    if not np.all(np.isfinite(pm.f)):
        return np.full(exact.f.size, 1e6)
    return rel_residual(pm.f, exact.f, sigma_low)


def _full_residuals(x4, exact, Omega_m, Omega_r, H0, grid, cfg, sigma_low, sigma_rec):
    w0, wa, aB_hat, u = x4
    try:
        pm = observables_param((w0, wa, aB_hat, u), Omega_m, Omega_r, H0, grid, cfg)
    except (GradientInstability, FloatingPointError, ValueError):
        return np.full(3 * exact.E.size + 1, 1e6)
    r = residuals(exact, pm, sigma_low, sigma_rec)
    return np.where(np.isfinite(r), r, 1e6)


# --------------------------------------------------------------------------- #
#  stage 0
# --------------------------------------------------------------------------- #

def initial_guess(sol, cfg) -> tuple:
    """Closed-form starting point from the exact solution.  Costs nothing.

    ``w(a) = w0 + wa(1-a)`` implies ``wa = -dw/dN`` at a = 1; ``alpha_B`` today
    gives ``alphaB_hat`` directly; and ``ln alpha_B = ln alphaB_hat - u ln E``
    is linear, so ``u`` is the slope of a regression over the observable range.
    """
    w0 = float(sol.w_phi[-1])
    wa = float(-np.gradient(sol.w_phi, sol.N)[-1])
    aB_hat = float(sol.alpha_B[-1])

    E_over_H0 = sol.E / sol.cosmo.h_tilde
    mask = (sol.a > 1.0 / (1.0 + cfg.grid.z_max)) & (sol.alpha_B > 0.0) & (E_over_H0 > 0.0)
    if mask.sum() >= 5 and aB_hat > 0.0:
        slope = np.polyfit(np.log(E_over_H0[mask]), np.log(sol.alpha_B[mask]), 1)[0]
        u = float(-slope)
    else:
        u = 4.0  # m = 1, the cubic-galileon value
    return w0, wa, aB_hat, u


def _clip(x, lo, hi, pad=1e-9):
    return float(min(max(x, lo + pad), hi - pad))


# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #

def fit_parametrisation(exact: Observables, grid, cosmo, cfg, guess) -> FitResult:
    """Run the cascade.  ``guess`` is the closed-form stage-0 tuple.

    Takes the cosmology and the starting point explicitly rather than a solved
    background, so ``ssprior fit`` can be re-run on stored observables with
    different weights or bounds without recomputing any physics.
    """
    f = cfg.fit
    Om, Orad, H0 = cosmo.Omega_m, cosmo.Omega_r, cosmo.H0
    sl, sr = f.sigma_low_z, f.sigma_rec

    b_w0, b_wa = f.bound_pair("w0"), f.bound_pair("wa")
    b_aB, b_u = f.bound_pair("alphaB_hat"), f.bound_pair("u")

    w0_0, wa_0, aB_0, u_0 = guess
    w0_0, wa_0 = _clip(w0_0, *b_w0), _clip(wa_0, *b_wa)
    aB_0, u_0 = _clip(aB_0, *b_aB), _clip(u_0, *b_u)

    nfev = 0

    # ---- stage 1: (w0, wa) on the expansion history alone ------------------
    try:
        r1 = least_squares(
            _bg_residuals, x0=[w0_0, wa_0],
            bounds=([b_w0[0], b_wa[0]], [b_w0[1], b_wa[1]]),
            args=(exact, Om, Orad, H0, grid, sl, sr),
            method="trf", x_scale="jac", xtol=f.xtol, ftol=f.ftol,
            max_nfev=f.max_nfev)
        w0, wa = r1.x
        nfev += r1.nfev
    except Exception:
        w0, wa = w0_0, wa_0

    best = None
    starts = [u_0] + [_clip(u_0 * s, *b_u) for s in f.multistart_u]
    seen = []

    for n_start, u_start in enumerate(starts, start=1):
        if any(abs(u_start - s) < 1e-6 for s in seen):
            continue
        seen.append(u_start)

        # ---- stage 2: (alphaB_hat, u) on f(z) ------------------------------
        try:
            r2 = least_squares(
                _f_residuals, x0=[aB_0, u_start],
                bounds=([b_aB[0], b_u[0]], [b_aB[1], b_u[1]]),
                args=(w0, wa, exact, Om, Orad, H0, grid, cfg, sl),
                method="trf", x_scale="jac", xtol=f.xtol, ftol=f.ftol,
                max_nfev=f.max_nfev)
            aB, u = r2.x
            nfev += r2.nfev
        except Exception:
            aB, u = aB_0, u_start

        # ---- stage 3: joint refinement -------------------------------------
        try:
            r3 = least_squares(
                _full_residuals, x0=[w0, wa, aB, u],
                bounds=([b_w0[0], b_wa[0], b_aB[0], b_u[0]],
                        [b_w0[1], b_wa[1], b_aB[1], b_u[1]]),
                args=(exact, Om, Orad, H0, grid, cfg, sl, sr),
                method="trf", x_scale="jac", xtol=f.xtol, ftol=f.ftol,
                max_nfev=f.max_nfev)
            nfev += r3.nfev
        except Exception:
            continue

        chi2_val = float(2.0 * r3.cost)
        if best is None or chi2_val < best[0]:
            best = (chi2_val, r3, n_start)
        if best[0] < 1.0:      # already far inside the target; stop early
            break

    if best is None:
        return FitResult(ParamModel(w0_0, wa_0, aB_0, 4.0 / u_0), np.inf, {},
                         FitStatus.FAILED, nfev, 1, np.inf)

    chi2_val, res, n_starts = best
    w0, wa, aB, u = res.x
    pm = ParamModel.from_u(w0, wa, aB, u)

    try:
        model = observables_param((w0, wa, aB, u), Om, Orad, H0, grid, cfg)
        errors = relative_errors(exact, model)
    except (GradientInstability, ValueError):
        return FitResult(pm, chi2_val, {}, FitStatus.FAILED, nfev, n_starts, np.inf)

    # condition number of J^T J: large values flag a degenerate (alphaB, u) pair
    try:
        s = np.linalg.svd(res.jac, compute_uv=False)
        cond = float((s[0] / s[-1]) ** 2) if s[-1] > 0 else np.inf
    except np.linalg.LinAlgError:
        cond = np.inf

    lo = np.array([b_w0[0], b_wa[0], b_aB[0], b_u[0]])
    hi = np.array([b_w0[1], b_wa[1], b_aB[1], b_u[1]])
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    at_bound = bool(np.any((res.x - lo) / span < 1e-6) or np.any((hi - res.x) / span < 1e-6))

    if errors["low_z"] < f.accept_rel_err_low_z and errors["rec"] < f.accept_rel_err_rec:
        status = FitStatus.OK
    elif at_bound:
        status = FitStatus.AT_BOUND
    elif res.status <= 0 or nfev >= f.max_nfev * len(starts):
        status = FitStatus.NOT_CONVERGED
    else:
        status = FitStatus.POOR

    return FitResult(pm, chi2_val, errors, status, nfev, n_starts, cond)
