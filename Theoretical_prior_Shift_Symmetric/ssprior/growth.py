"""Quasi-static growth of matter perturbations, Eq. (20) and (22) of the paper.

    f' + f^2 + (2 + Hdot/H^2) f - (3/2) Omega_m(a) [1 + alpha_B^2 / (2 cs2N2)] = 0

with ' = d/dN and

    cs2N2 = (alpha_B - 2)(Hdot/H^2 - alpha_B/2) + dalpha_B/dN - 3 Omega_m(a)

``cs2N2`` is the product ``c_s^2 * (alpha_K + 1.5 alpha_B^2)``, which is
*independent of alpha_K* -- the reason alpha_K drops out of every observable and
is unconstrained by data (Sec. III).  Two checks are built into the test suite:

* matter domination in GR (alpha_B = 0, Hdot/H^2 = -3/2, Omega_m = 1) gives
  cs2N2 = 0, as it must when alpha_K = 0;
* in the k-essence limit (alpha_B = 0, alpha_K != 0) the same expression reduces
  to (rho_phi + p_phi)/(M_P^2 H^2), the known value of c_s^2 alpha_K.

Integration
-----------
Fixed-step RK4 on a grid shared by the exact and the parametrised model, not
``solve_ivp``.  This is deliberate.  The chi^2 of Eq. (21) weights the
observables at sigma = 1e-3, and an adaptive solver would place its nodes
differently for the two models, leaving a truncation error of the same order as
the signal being fitted.  With the same nodes and the same scheme the leading
truncation error cancels in the residual, and the residual is a smooth function
of the parameters -- which Gauss-Newton needs in order to converge.
``solve_ivp(LSODA, rtol=1e-10)`` is kept in the tests as an accuracy reference.
"""

from __future__ import annotations

import numpy as np

__all__ = ["growth_grid", "cs2_N2", "mu_qs", "integrate_growth", "GradientInstability"]


class GradientInstability(Exception):
    """``cs2N2 <= 0`` somewhere on the integration range."""


def growth_grid(cfg) -> np.ndarray:
    """``2 n_steps + 1`` nodes from ``ln a_start`` to 0.

    RK4 advances from node ``2i`` to ``2i+2`` and needs the midpoint ``2i+1``,
    so every quantity is tabulated on the fine grid and the coarse solution is
    read off ``[::2]``.
    """
    n = int(cfg.growth.n_steps)
    return np.linspace(np.log(cfg.growth.a_start), 0.0, 2 * n + 1)


def cs2_N2(alpha_B, dalpha_B_dN, dE_dN_over_E, Omega_m_a):
    """Eq. (22) denominator: ``c_s^2 (alpha_K + 1.5 alpha_B^2)``."""
    return ((alpha_B - 2.0) * (dE_dN_over_E - 0.5 * alpha_B)
            + dalpha_B_dN - 3.0 * Omega_m_a)


def mu_qs(alpha_B, cs2n2):
    """Quasi-static modification of the growth source, ``1 + alpha_B^2/(2 cs2N2)``."""
    return 1.0 + alpha_B * alpha_B / (2.0 * cs2n2)


def integrate_growth(N_fine, dE_dN_over_E, Omega_m_a, mu, f_start):
    """RK4 for ``df/dN``; returns ``f`` on the coarse nodes ``N_fine[::2]``.

    All coefficient arrays must already be tabulated on ``N_fine``.
    """
    n_coarse = (N_fine.size - 1) // 2
    h = N_fine[2] - N_fine[0]
    f = np.empty(n_coarse + 1)
    f[0] = f_start

    def rhs(i, fi):
        return (-fi * fi - (2.0 + dE_dN_over_E[i]) * fi
                + 1.5 * Omega_m_a[i] * mu[i])

    y = f_start
    for k in range(n_coarse):
        i0, im, i1 = 2 * k, 2 * k + 1, 2 * k + 2
        k1 = rhs(i0, y)
        k2 = rhs(im, y + 0.5 * h * k1)
        k3 = rhs(im, y + 0.5 * h * k2)
        k4 = rhs(i1, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(y):
            raise GradientInstability("growth rate diverged")
        f[k + 1] = y
    return f


def initial_f(Omega_m_a0: float, how: str = "gamma") -> float:
    """Initial condition at ``a_start``.

    The decaying mode is strongly attractive, so the result at z < 10 is
    insensitive to this choice (verified to 1e-5 in ``tests/``).
    """
    if how == "unity":
        return 1.0
    return float(Omega_m_a0 ** 0.55)
