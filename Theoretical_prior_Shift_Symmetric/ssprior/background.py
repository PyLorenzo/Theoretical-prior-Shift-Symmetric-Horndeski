"""Exact background evolution of the shift-symmetric KGB model.

This is the piece EFTCAMB cannot provide: its shift-symmetric module
(``007p7_ShiftSym_alphaB.f90``) is a *designer* model that imposes H(a) from a
parametrised w_DE, and no full-mapping model in the build takes the Lagrangian
coefficients (c01, c02, d02).  It plays the role hi_class plays in the paper.

Method
------
At every scale factor the system

    j(psi, E)  = j_i a^-3                                   (Eq. 12-13)
    3 E^2      = 3 h~^2 (Om a^-3 + Or a^-4 + OL) + rho(psi, E)   (Eq. 10)

is *algebraic* in (psi, E): two equations, two unknowns.  There is no ODE to
integrate, only a root to follow.  We therefore:

1. solve the tracker cubic in closed form at a = 1, where E = h~ is known
   exactly by definition, and read off Omega_phi,0 (this is the paper's
   "slice", Sec. III -- no shooting, which would bias the prior, Fig. 3);
2. continue the root *backwards* in ln a with an Euler predictor built from the
   analytic derivatives, followed by a Newton corrector with the analytic
   Jacobian;
3. validate the branch against the known early-time asymptote
   ``psi -> c01 / (3 d01 E)``, and reject the model if the continuation folds
   (the 2x2 determinant changes sign) or drifts onto another root.

Backwards is the right direction: forwards from a_ini there is no exact initial
condition to start from, whereas at a = 1 there is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .lagrangian import Lagrangian

__all__ = ["BgStatus", "Cosmology", "BackgroundSolution", "solve_background",
           "slice_candidates"]


class BgStatus(IntEnum):
    """Why a model was kept or dropped, never collapsed into a bare bool."""

    OK = 0
    NO_REAL_ROOT = 1          # tracker cubic has no usable positive root at a=1
    NEG_RHO = 2               # rho_phi <= 0 (Eq. 14 forbids it for c01 >= 0)
    OMEGA_OUT_OF_RANGE = 3    # slice rejects it: Omega_cdm outside its prior
    FOLD = 4                  # 2x2 determinant changed sign along the branch
    ASYMPTOTE_MISMATCH = 5    # continuation landed on the wrong early-time root
    NEWTON_FAIL = 6           # corrector did not converge
    NEG_E2 = 7                # E^2 went negative


@dataclass(frozen=True)
class Cosmology:
    """Background content, with densities as fractions of today's critical density."""

    Omega_m: float
    Omega_r: float
    Omega_L: float
    H0: float
    H_fid: float
    ombh2: float

    @property
    def h_tilde(self) -> float:
        """``H0 / H_fid`` -- appears because Lambda2, Lambda3 use a *fixed* fiducial."""
        return self.H0 / self.H_fid

    @property
    def h(self) -> float:
        return self.H0 / 100.0

    @property
    def Omega_cdm(self) -> float:
        return self.Omega_m - self.ombh2 / self.h ** 2

    @property
    def Omega_phi(self) -> float:
        return 1.0 - self.Omega_m - self.Omega_r - self.Omega_L

    def rho_bg(self, a):
        """``(rho_m + rho_r + rho_Lambda) / (M_P^2 H_fid^2)``, i.e. 3 h~^2 sum(Omega_i)."""
        h2 = self.h_tilde ** 2
        return 3.0 * h2 * (self.Omega_m * a ** -3 + self.Omega_r * a ** -4 + self.Omega_L)

    def drho_bg_dN(self, a):
        h2 = self.h_tilde ** 2
        return -3.0 * h2 * (3.0 * self.Omega_m * a ** -3 + 4.0 * self.Omega_r * a ** -4)


@dataclass
class BackgroundSolution:
    """Sampled exact background, all arrays ordered by increasing ``N = ln a``."""

    N: np.ndarray
    a: np.ndarray
    E: np.ndarray
    psi: np.ndarray
    dpsi_dN: np.ndarray
    dE_dN: np.ndarray
    rho_phi: np.ndarray
    p_phi: np.ndarray
    w_phi: np.ndarray
    alpha_B: np.ndarray
    alpha_K: np.ndarray
    dalpha_B_dN: np.ndarray
    cosmo: Cosmology
    status: BgStatus
    psi0: float
    n_candidates: int = 1

    @property
    def ok(self) -> bool:
        return self.status is BgStatus.OK

    @property
    def kinetic_D(self) -> np.ndarray:
        return self.alpha_K + 1.5 * self.alpha_B ** 2

    def crosses_alpha_B_2(self) -> bool:
        """Appendix A of the paper: evolutions through alpha_B = 2 may be singular.

        We flag them rather than excluding them -- exactly what the paper does.
        """
        return bool(np.any(self.alpha_B > 2.0))


# --------------------------------------------------------------------------- #
#  the "slice": which Lagrangians give an acceptable cosmology at all
# --------------------------------------------------------------------------- #

def slice_candidates(lag: Lagrangian, H0: float, cfg, Omega_L: float = 0.0):
    """Candidate (psi0, Cosmology) pairs for one Lagrangian, per Sec. III.

    Rather than shooting on one of the c_0i -- which the paper shows biases the
    resulting prior (Fig. 3) -- we read Omega_phi,0 off the exact solution and
    let it *define* Omega_m, then keep the point only if Omega_cdm lands inside
    its prior.  That is the "deformed slab" cutting through (c01, c02, d02).
    """
    cosmo_cfg = cfg.cosmology
    h_tilde = H0 / cosmo_cfg.H_fid
    Omega_r = cosmo_cfg.omega_r(H0)
    h2 = h_tilde ** 2
    out = []
    for psi0 in lag.tracker_roots(h_tilde):
        rho0 = lag.rho(psi0, h_tilde)
        if not np.isfinite(rho0) or rho0 <= 0.0:
            continue
        Omega_phi = rho0 / (3.0 * h2)
        Omega_m = 1.0 - Omega_r - Omega_L - Omega_phi
        cosmo = Cosmology(Omega_m=Omega_m, Omega_r=Omega_r, Omega_L=Omega_L,
                          H0=H0, H_fid=cosmo_cfg.H_fid, ombh2=cosmo_cfg.ombh2_of(H0))
        if not cosmo_cfg.Omega_cdm.contains(cosmo.Omega_cdm):
            continue
        out.append((float(psi0), cosmo))
    return out


# --------------------------------------------------------------------------- #
#  continuation
# --------------------------------------------------------------------------- #

def _derivatives(lag: Lagrangian, cosmo: Cosmology, psi, E, a, j_val):
    """Solve the 2x2 linear system for (dpsi/dN, dE/dN); also return the determinant.

    Differentiating the two defining equations with respect to N = ln a:

        j_psi psi' + j_E E'            = -3 j
        -rho_psi psi' + (6E - rho_E) E' = drho_bg/dN
    """
    pa = lag.partials(psi, E)
    d = 6.0 * E - pa.rho_E
    det = pa.j_psi * d + pa.j_E * pa.rho_psi
    if det == 0.0 or not np.isfinite(det):
        return np.nan, np.nan, det
    p = -3.0 * j_val
    q = cosmo.drho_bg_dN(a)
    dpsi = (p * d - pa.j_E * q) / det
    dE = (pa.j_psi * q - p * (-pa.rho_psi)) / det
    return dpsi, dE, det


def _newton(lag: Lagrangian, cosmo: Cosmology, psi, E, a, j_target, tol, max_iter):
    """Newton corrector on (F1, F2) with the analytic Jacobian."""
    rho_bg = cosmo.rho_bg(a)
    for _ in range(max_iter):
        f1 = lag.j(psi, E) - j_target
        f2 = 3.0 * E * E - rho_bg - lag.rho(psi, E)
        scale = max(abs(j_target), abs(lag.j(psi, E)), 1.0), max(abs(rho_bg), 1.0)
        if abs(f1) < tol * scale[0] and abs(f2) < tol * scale[1]:
            return psi, E, True
        pa = lag.partials(psi, E)
        a11, a12 = pa.j_psi, pa.j_E
        a21, a22 = -pa.rho_psi, 6.0 * E - pa.rho_E
        det = a11 * a22 - a12 * a21
        if det == 0.0 or not np.isfinite(det):
            return psi, E, False
        dpsi = (-f1 * a22 + a12 * f2) / det
        dE = (-a11 * f2 + a21 * f1) / det
        psi, E = psi + dpsi, E + dE
        if not (np.isfinite(psi) and np.isfinite(E)) or E <= 0.0:
            return psi, E, False
    f1 = lag.j(psi, E) - j_target
    f2 = 3.0 * E * E - rho_bg - lag.rho(psi, E)
    return psi, E, (abs(f1) < 1e-8 * max(abs(j_target), 1.0)
                    and abs(f2) < 1e-8 * max(abs(rho_bg), 1.0))


def solve_background(lag: Lagrangian, cosmo: Cosmology, psi0: float, cfg,
                     j_i: float = 0.0) -> BackgroundSolution:
    """Continue the (psi, E) root backwards from a = 1 to ``cfg.model.a_ini``."""
    bgc = cfg.background
    n = int(bgc.n_grid)
    N = np.linspace(np.log(cfg.model.a_ini), 0.0, n)      # ascending
    a = np.exp(N)

    psi = np.empty(n)
    E = np.empty(n)
    dpsi = np.empty(n)
    dE = np.empty(n)

    # j(a) = j_i a^-3 ; the tracker is j_i = 0
    j_vals = j_i * a ** -3

    status = BgStatus.OK
    det_sign = 0.0

    # start at the last index (a = 1) and walk down
    psi[-1], E[-1] = psi0, cosmo.h_tilde
    for i in range(n - 1, -1, -1):
        if i < n - 1:
            h = N[i] - N[i + 1]                            # negative step
            psi_p = psi[i + 1] + h * dpsi[i + 1]
            E_p = E[i + 1] + h * dE[i + 1]
            if not (np.isfinite(psi_p) and np.isfinite(E_p)) or E_p <= 0.0:
                status = BgStatus.NEG_E2
                break
            psi[i], E[i], conv = _newton(lag, cosmo, psi_p, E_p, a[i], j_vals[i],
                                         bgc.newton_tol, bgc.newton_max_iter)
            if not conv:
                status = BgStatus.NEWTON_FAIL
                break
        dpsi[i], dE[i], det = _derivatives(lag, cosmo, psi[i], E[i], a[i], j_vals[i])
        if not np.isfinite(dpsi[i]) or not np.isfinite(dE[i]):
            status = BgStatus.NEWTON_FAIL
            break
        if det_sign == 0.0:
            det_sign = np.sign(det)
        elif bgc.reject_on_fold and np.sign(det) != det_sign:
            status = BgStatus.FOLD
            break

    if status is BgStatus.OK:
        # the branch test: at a_ini the tracker must sit on psi ~ c01/(3 d01 E)
        asym = lag.tracker_asymptote(E[0])
        if asym == 0.0 or abs(psi[0] / asym - 1.0) > bgc.asymptote_tol:
            status = BgStatus.ASYMPTOTE_MISMATCH

    if status is not BgStatus.OK:
        empty = np.zeros(0)
        return BackgroundSolution(empty, empty, empty, empty, empty, empty, empty,
                                  empty, empty, empty, empty, empty,
                                  cosmo, status, psi0)

    rho_phi = lag.rho(psi, E)
    p_phi = lag.pressure(psi, E, dpsi)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_phi = np.where(np.abs(rho_phi) > 0.0, p_phi / rho_phi, np.nan)
    aB = lag.alpha_B(psi, E)
    aK = lag.alpha_K(psi, E)
    daB = lag.dalpha_B_dN(psi, E, dpsi, dE)

    return BackgroundSolution(N=N, a=a, E=E, psi=psi, dpsi_dN=dpsi, dE_dN=dE,
                              rho_phi=rho_phi, p_phi=p_phi, w_phi=w_phi,
                              alpha_B=aB, alpha_K=aK, dalpha_B_dN=daB,
                              cosmo=cosmo, status=BgStatus.OK, psi0=psi0)
