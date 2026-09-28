"""Observables of Eq. (18)-(20) and the chi^2 of Eq. (21).

The fit is performed on *observables*, not on w(a) and alpha_B(a) -- this is the
central methodological point of Sec. IV: w and alpha_B are not observable, and
allowing the parametrisation to deviate from their best fit buys accuracy where
it matters.

Three quantities on a grid of 100 redshifts below z = 10, plus the angular
diameter distance at recombination:

    E(z) = H(z)/H0        D_A(z) = int_0^z dz'/H(z')      f(z) = dln delta_m/dln a

``D_A`` follows the paper's Eq. (19) literally, i.e. it is the comoving distance;
``comoving_to_angular`` converts when comparing against CAMB.

Everything that both the exact and the parametrised model need -- the redshift
grid, the quadrature nodes, the growth nodes, z_rec -- lives in a single
``Grid`` object built once per run, so the two sides are evaluated with
identical discretisations and their truncation errors cancel in the residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ["Grid", "Observables", "z_star_hu_sugiyama", "build_grid",
           "observables_from_background", "residuals", "rel_residual", "chi2",
           "relative_errors", "C_KM_S"]

C_KM_S = 299792.458  # speed of light, km/s


def z_star_hu_sugiyama(ombh2: float, omh2: float) -> float:
    """Redshift of last scattering, Hu & Sugiyama (1996) fitting formula.

    Depends only on (omega_b, omega_m), so it is identical for the exact and the
    parametrised model of a given sample -- D_A(z_rec) is then a like-for-like
    comparison -- while still varying across the prior, as in Fig. 10.
    """
    g1 = 0.0783 * ombh2 ** -0.238 / (1.0 + 39.5 * ombh2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * ombh2 ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * ombh2 ** -0.738) * (1.0 + g1 * omh2 ** g2)


@dataclass(frozen=True)
class Grid:
    """Discretisation shared by every model evaluated in one run."""

    z: np.ndarray            # (n_z,) observable redshifts, z < z_max
    N_quad: np.ndarray       # (n_quad,) ascending ln a, from ln a_ini to 0, for distances
    N_growth: np.ndarray     # (2 n_steps + 1,) fine grid for the RK4
    z_rec: float
    a_rec: float

    @property
    def a_z(self) -> np.ndarray:
        return 1.0 / (1.0 + self.z)

    @property
    def N_z(self) -> np.ndarray:
        return -np.log1p(self.z)


def build_grid(cfg, ombh2: float, omh2: float, N_quad: np.ndarray) -> Grid:
    g = cfg.grid
    z = g.z_nodes()
    if g.z_rec.startswith("fixed:"):
        z_rec = float(g.z_rec.split(":", 1)[1])
    else:
        z_rec = z_star_hu_sugiyama(ombh2, omh2)
    from .growth import growth_grid
    return Grid(z=z, N_quad=N_quad, N_growth=growth_grid(cfg),
                z_rec=z_rec, a_rec=1.0 / (1.0 + z_rec))


@dataclass
class Observables:
    """``E``, ``D_A`` and ``f`` on ``grid.z``, plus ``D_A`` at recombination."""

    E: np.ndarray
    DA: np.ndarray
    f: np.ndarray
    DA_rec: float
    z_rec: float
    H0: float

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.E, self.DA, self.f, [self.DA_rec]])

    def angular_diameter(self, z) -> np.ndarray:
        """``D_A`` in CAMB's convention, ``chi/(1+z)``, for the cross-check only.

        Eq. (19) of the paper calls ``int dz/H`` the angular diameter distance;
        it is the comoving distance.  The fit is unaffected (both sides use the
        same definition), but the comparison against CAMB is not.
        """
        return self.DA / (1.0 + np.asarray(z))


# --------------------------------------------------------------------------- #
#  distances
# --------------------------------------------------------------------------- #

def _comoving_distance(N_quad, E_over_H0, H0, N_targets):
    """``chi(z) = (c/H0) int_N^0 dN' / (a E)`` evaluated at ``N_targets``.

    The integrand is splined once and integrated exactly through the spline's
    antiderivative -- fourth-order accurate on the node spacing, and smooth in
    the model parameters, which a piecewise quadrature rule would not be.

    The *same* function and the *same* nodes serve the exact and the
    parametrised model, so the quadrature error cancels at leading order in the
    residual of Eq. (21).
    """
    a = np.exp(N_quad)
    integrand = 1.0 / (a * E_over_H0)
    cum = CubicSpline(N_quad, integrand).antiderivative()(N_quad)
    chi_of_N = CubicSpline(N_quad, cum[-1] - cum)      # distance from N to today
    chi = (C_KM_S / H0) * chi_of_N(N_targets)
    # chi(z = 0) = 0 by definition; the spline returns ~1e-15 there, and a
    # round-off "zero" in the denominator of a relative residual is fatal
    return np.where(np.asarray(N_targets) == 0.0, 0.0, chi)


def observables_from_E(N_quad, E_over_H0, H0, f_of_N, grid: Grid) -> Observables:
    """Assemble the observable vector from ``E(N) = H/H0`` and the growth rate."""
    spline_E = CubicSpline(N_quad, E_over_H0)
    N_z = grid.N_z
    E_z = spline_E(N_z)
    DA_z = _comoving_distance(N_quad, E_over_H0, H0, N_z)
    DA_rec = float(_comoving_distance(N_quad, E_over_H0, H0,
                                      np.array([np.log(grid.a_rec)]))[0])
    return Observables(E=E_z, DA=DA_z, f=f_of_N(N_z), DA_rec=DA_rec,
                       z_rec=grid.z_rec, H0=H0)


def observables_from_background(sol, grid: Grid, cfg) -> Observables:
    """Exact observables from a solved KGB background."""
    from .growth import cs2_N2, mu_qs, integrate_growth, initial_f, GradientInstability

    cosmo = sol.cosmo
    h_t = cosmo.h_tilde
    E_over_H0 = sol.E / h_t                       # H/H0

    # tabulate the growth coefficients on the fine RK4 grid
    Nf = grid.N_growth
    sp_E = CubicSpline(sol.N, sol.E)
    sp_dE = CubicSpline(sol.N, sol.dE_dN)
    sp_aB = CubicSpline(sol.N, sol.alpha_B)
    sp_daB = CubicSpline(sol.N, sol.dalpha_B_dN)

    E_f = sp_E(Nf)
    dEoE = sp_dE(Nf) / E_f
    aB_f = sp_aB(Nf)
    daB_f = sp_daB(Nf)
    a_f = np.exp(Nf)
    Om_a = cosmo.Omega_m * h_t ** 2 * a_f ** -3 / E_f ** 2

    cs2 = cs2_N2(aB_f, daB_f, dEoE, Om_a)
    if cfg.growth.stop_on_gradient_instability and np.any(cs2 <= 0.0):
        raise GradientInstability("cs2N2 <= 0 on the growth range")
    mu = mu_qs(aB_f, cs2)

    f_coarse = integrate_growth(Nf, dEoE, Om_a, mu,
                                initial_f(Om_a[0], cfg.growth.f_start))
    f_of_N = CubicSpline(Nf[::2], f_coarse)
    # grid.N_quad is the background's own N grid, so no interpolation is needed
    # here and the parametrised model is evaluated on exactly the same nodes.
    return observables_from_E(grid.N_quad, E_over_H0, cosmo.H0, f_of_N, grid)


# --------------------------------------------------------------------------- #
#  chi^2 of Eq. (21)
# --------------------------------------------------------------------------- #

TINY = 1e-100   # RUFIAN's stand-in for an exact zero (common.py / binning.py)


def rel_residual(model, exact, sigma):
    """``(model - exact) / (exact * sigma)`` with RUFIAN's treatment of zeros.

    RUFIAN replaces every observable equal to 0 by 1e-100 on both sides before
    forming the residual.  The only zero here is D_A(z = 0) on the 'ln1pz'
    grid, where both sides vanish identically: the residual is then exactly 0,
    instead of 0/0.
    """
    m = np.where(np.asarray(model) == 0.0, TINY, model)
    e = np.where(np.asarray(exact) == 0.0, TINY, exact)
    return (m - e) / (e * sigma)


def residuals(exact: Observables, model: Observables,
              sigma_low: float, sigma_rec: float) -> np.ndarray:
    """Relative residuals, weighted as in Eq. (21).

    ``sigma`` are relative tolerances: 1e-3 for every observable at z < 10 and
    1e-4 for D_A at recombination.  Same form as RUFIAN:
    ``(O_th - O_fit) / (O_th sigma)``.
    """
    r_E = rel_residual(model.E, exact.E, sigma_low)
    r_D = rel_residual(model.DA, exact.DA, sigma_low)
    r_f = rel_residual(model.f, exact.f, sigma_low)
    r_rec = rel_residual(model.DA_rec, exact.DA_rec, sigma_rec)
    return np.concatenate([r_E, r_D, r_f, np.atleast_1d(r_rec)])


def chi2(exact, model, sigma_low, sigma_rec) -> float:
    r = residuals(exact, model, sigma_low, sigma_rec)
    return float(r @ r)


def relative_errors(exact: Observables, model: Observables) -> dict:
    """Max relative deviation at z < 10 (per observable) and at recombination.

    These, not the chi^2, are the paper's acceptance criteria: < 1% below z = 10
    and < 0.3% on D_A(z_rec), for 99% of the models (Fig. 7).
    """
    e_E = np.abs(rel_residual(model.E, exact.E, 1.0))
    e_D = np.abs(rel_residual(model.DA, exact.DA, 1.0))
    e_f = np.abs(rel_residual(model.f, exact.f, 1.0))
    return {
        "E": float(e_E.max()), "DA": float(e_D.max()), "f": float(e_f.max()),
        "low_z": float(max(e_E.max(), e_D.max(), e_f.max())),
        "rec": float(abs(model.DA_rec / exact.DA_rec - 1.0)),
    }
