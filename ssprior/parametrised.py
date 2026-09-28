"""The four-parameter phenomenological model, Eq. (2) and (17).

    w(a)       = w0 + wa (1 - a)                                  [CPL]
    alpha_B(a) = alphaB_hat (H0 / H)^(4/m)

This is *exactly* what EFTCAMB's designer shift-symmetric model implements.  In
``fortran/eftcamb/07f_designer_models/007p7_ShiftSym_alphaB.f90``:

    RPH_PM_V = 0                                      ! alpha_M = 0
    RPH_AT_V = 0                                      ! alpha_T = 0
    RPH_AK_V = SS_alphaK0 * a                         ! alpha_K = alpha_K0 a
    RPH_AB_V = SS_alphaB0*((a*h0_Mpc)/adotoa)**(4/SS_m)

and ``adotoa`` is the conformal Hubble rate aH, so ``a h0_Mpc / adotoa = H0/H``.
The expansion history comes from ``EFTwDE = 2`` (CPL), through
``grhov_t = grhov * wDE%integral(a)``.

Internal parametrisation
------------------------
Everything here works in ``u = 4/m`` rather than ``m``.  ``alpha_B`` is
exponential in ``u`` but ``m`` sits in the denominator of the exponent: in ``m``
the chi^2 surface has curvature that blows up as m -> 0 and flattens for large
m, while in ``u`` the relation ``ln alpha_B = ln alphaB_hat - u ln E`` is
*linear*.  Gauss-Newton needs the latter.  ``m = 4/u`` is restored on output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ParamModel", "rho_de_ratio", "E_cpl", "observables_param"]


@dataclass(frozen=True)
class ParamModel:
    w0: float
    wa: float
    alphaB_hat: float
    m: float

    @property
    def u(self) -> float:
        return 4.0 / self.m

    @classmethod
    def from_u(cls, w0, wa, alphaB_hat, u) -> "ParamModel":
        return cls(float(w0), float(wa), float(alphaB_hat), 4.0 / float(u))

    def as_array(self) -> np.ndarray:
        return np.array([self.w0, self.wa, self.alphaB_hat, self.m])


def w_cpl(a, w0, wa):
    return w0 + wa * (1.0 - a)


def rho_de_ratio(a, w0, wa):
    """``rho_DE(a)/rho_DE(1)`` for CPL -- the closed form of EFTCAMB's ``wDE%integral``."""
    return a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))


def E_cpl(N, Omega_m, Omega_r, w0, wa):
    """``H/H0`` and ``dE/dN`` for a flat CPL background, both analytic."""
    a = np.exp(N)
    Omega_de = 1.0 - Omega_m - Omega_r
    rm = Omega_m * a ** -3
    rr = Omega_r * a ** -4
    fde = rho_de_ratio(a, w0, wa)
    rde = Omega_de * fde
    E2 = rm + rr + rde
    E = np.sqrt(E2)
    # d(E^2)/dN = -3 rm - 4 rr - 3 (1 + w) rde
    dE2 = -3.0 * rm - 4.0 * rr - 3.0 * (1.0 + w_cpl(a, w0, wa)) * rde
    return E, 0.5 * dE2 / E


def alpha_B_param(E, alphaB_hat, u):
    """Eq. (17) in terms of ``u = 4/m``: ``alpha_B = alphaB_hat E^-u``."""
    return alphaB_hat * E ** (-u)


def dalpha_B_dN_param(alpha_B, u, dE_dN, E):
    return -u * alpha_B * dE_dN / E


def observables_param(pm_u, Omega_m, Omega_r, H0, grid, cfg):
    """Observables of the parametrised model.

    ``pm_u`` is the tuple ``(w0, wa, alphaB_hat, u)``; the caller works in ``u``.
    Uses the same quadrature nodes and the same fixed-step RK4 as the exact
    model, so their discretisation errors cancel in the residual.
    """
    from .growth import cs2_N2, mu_qs, integrate_growth, initial_f, GradientInstability
    from .observables import observables_from_E
    from scipy.interpolate import CubicSpline

    w0, wa, aB_hat, u = pm_u

    E_q, _ = E_cpl(grid.N_quad, Omega_m, Omega_r, w0, wa)

    Nf = grid.N_growth
    E_f, dE_f = E_cpl(Nf, Omega_m, Omega_r, w0, wa)
    dEoE = dE_f / E_f
    aB_f = alpha_B_param(E_f, aB_hat, u)
    daB_f = dalpha_B_dN_param(aB_f, u, dE_f, E_f)
    a_f = np.exp(Nf)
    Om_a = Omega_m * a_f ** -3 / E_f ** 2

    cs2 = cs2_N2(aB_f, daB_f, dEoE, Om_a)
    if np.any(cs2 <= 0.0):
        raise GradientInstability("cs2N2 <= 0 for the parametrised model")
    mu = mu_qs(aB_f, cs2)

    f_coarse = integrate_growth(Nf, dEoE, Om_a, mu,
                                initial_f(Om_a[0], cfg.growth.f_start))
    f_of_N = CubicSpline(Nf[::2], f_coarse)
    return observables_from_E(grid.N_quad, E_q, H0, f_of_N, grid)
