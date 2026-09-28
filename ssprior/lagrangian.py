"""Shift-symmetric Horndeski (Kinetic Gravity Braiding) Lagrangian.

Implements the background functions of Traykova et al. 2021 (arXiv:2103.11195)
for the truncation of their Eq. (9),

    G2 = c01 X + (c02 / L2^4) X^2
    G3 = -(1 / L3^3) ( d01 X + (d02 / L2^4) X^2 )
    G4 = M_P^2 / 2 ,   G5 = 0 ,   X = phidot^2 / 2

with the conventional normalisations L2^4 = M_P^2 H_fid^2 and L3^3 = M_P H_fid^2.
H_fid is a *fixed fiducial* Hubble rate (Sec. III of the paper: "we set this
normalisation H0 to a fiducial value"), while the sampled H0 varies; the ratio
h_tilde = H0 / H_fid therefore appears explicitly in the Friedmann equation.

Dimensionless variables used everywhere in this package:

    psi = phidot / (M_P H_fid)      E = H / H_fid      N = ln a
    tilde quantities are divided by M_P^2 H_fid^2

so that 2X / (M_P^2 H_fid^2) = psi^2 and X / L2^4 = psi^2 / 2.

Two exact identities follow from the definitions and are used both as
simplifications and as unit tests (see ``tests/test_ssprior.py``):

    d rho / dE = psi * dj/dE = 3 E alpha_B
    rho        = psi * j - (1/2) (c01 + c02 psi^2 / 2) psi^2

the second being Eq. (14) of the paper, rho_phi = phidot J - G2.  On the tracker
(j = 0) it gives rho_phi = -G2, so a positive scalar energy density requires
c01 + c02 psi^2 / 2 < 0 -- the paper's statement that self-acceleration needs a
"wrong sign" kinetic term.

Every partial derivative here is analytic; the production code never takes a
finite difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Lagrangian", "Partials"]


@dataclass(frozen=True)
class Partials:
    """Analytic first derivatives of ``j`` and ``rho`` at a point (psi, E)."""

    j_psi: np.ndarray
    j_E: np.ndarray
    rho_psi: np.ndarray
    rho_E: np.ndarray


@dataclass(frozen=True)
class Lagrangian:
    """Coefficients of Eq. (9).

    ``d01`` is fixed to -1 by the normalisation of the field (Sec. III); its sign
    carries no loss of generality because L3 contains only odd powers of phi.
    """

    c01: float
    c02: float
    d02: float
    d01: float = -1.0

    # ---- background functions -------------------------------------------------

    def j(self, psi, E):
        """Shift current, Eq. (13), in units of M_P H_fid.

        The background equation of motion is ``J dot + 3 H J = 0``, i.e.
        ``j(a) = j_i a^-3``; the tracker attractor is ``j = 0``.
        """
        p2 = psi * psi
        return psi * (self.c01 + self.c02 * p2) - 3.0 * E * p2 * (self.d01 + self.d02 * p2)

    def rho(self, psi, E):
        """Scalar energy density, Eq. (11), in units of M_P^2 H_fid^2."""
        p2 = psi * psi
        return 0.5 * p2 * (self.c01 + 1.5 * self.c02 * p2) \
            - 3.0 * E * psi * p2 * (self.d01 + self.d02 * p2)

    def pressure(self, psi, E, dpsi_dN):
        """Scalar pressure, Eq. (11).  Needs ``dpsi/dN`` because ``p`` contains phi ddot.

        phi ddot = M_P H_fid^2 E dpsi/dN, so the cubic term reads
        ``(d01 + d02 psi^2) E psi^2 dpsi/dN``.
        """
        p2 = psi * psi
        return 0.5 * p2 * (self.c01 + 0.5 * self.c02 * p2) \
            + (self.d01 + self.d02 * p2) * E * p2 * dpsi_dN

    def g2(self, psi):
        """``G2`` evaluated on the background, in units of M_P^2 H_fid^2."""
        p2 = psi * psi
        return 0.5 * p2 * (self.c01 + 0.5 * self.c02 * p2)

    # ---- alpha functions, Eq. (15) -------------------------------------------

    def alpha_B(self, psi, E):
        """Braiding.  ``H^2 M_P^2 alpha_B = 2 X phidot H G_3X``.

        In the cubic-galileon limit (c02 = d02 = 0) the tracker gives
        psi = c01 / (3 d01 E), hence alpha_B ~ E^-4 -- the motivation for the
        parametrisation of Eq. (17).
        """
        p2 = psi * psi
        return -(self.d01 + self.d02 * p2) * psi * p2 / E

    def alpha_K(self, psi, E):
        """Kineticity.  ``H^2 M_P^2 alpha_K = 2X[G_2X + 2X G_2XX + 6 phidot H (G_3X + X G_3XX)]``.

        Never enters the observables: ``cs2_N2`` below is the product
        ``c_s^2 * (alpha_K + 1.5 alpha_B^2)``, which is alpha_K-independent.  That
        is precisely why alpha_K is unconstrained by data (Sec. III).
        """
        p2 = psi * psi
        return (p2 * (self.c01 + 3.0 * self.c02 * p2)
                - 6.0 * E * psi * p2 * (self.d01 + 2.0 * self.d02 * p2)) / (E * E)

    def kinetic_D(self, psi, E):
        """``D = alpha_K + 1.5 alpha_B^2``.  No-ghost requires ``D > 0``."""
        aB = self.alpha_B(psi, E)
        return self.alpha_K(psi, E) + 1.5 * aB * aB

    # ---- derivatives ----------------------------------------------------------

    def partials(self, psi, E) -> Partials:
        p2 = psi * psi
        return Partials(
            j_psi=self.c01 + 3.0 * self.c02 * p2
            - 6.0 * E * psi * (self.d01 + 2.0 * self.d02 * p2),
            j_E=-3.0 * p2 * (self.d01 + self.d02 * p2),
            rho_psi=psi * (self.c01 + 3.0 * self.c02 * p2)
            - 3.0 * E * p2 * (3.0 * self.d01 + 5.0 * self.d02 * p2),
            rho_E=-3.0 * psi * p2 * (self.d01 + self.d02 * p2),
        )

    def dalpha_B_dN(self, psi, E, dpsi_dN, dE_dN):
        """``d alpha_B / d ln a`` from the chain rule, analytic."""
        p2 = psi * psi
        aB = self.alpha_B(psi, E)
        dpsi_term = -(3.0 * self.d01 + 5.0 * self.d02 * p2) * p2 * dpsi_dN / E
        return dpsi_term - aB * dE_dN / E

    # ---- tracker solution -----------------------------------------------------

    def tracker_roots(self, E):
        """Real positive roots of ``j(psi, E) = 0`` at fixed ``E``, excluding psi = 0.

        Dividing the quartic ``j`` by psi leaves the cubic

            3 d02 E psi^3 - c02 psi^2 + 3 d01 E psi - c01 = 0

        which is solved in closed form.  Returns them sorted by magnitude.
        """
        coeffs = np.array([3.0 * self.d02 * E, -self.c02, 3.0 * self.d01 * E, -self.c01])
        # np.roots needs a non-degenerate leading coefficient
        nz = np.flatnonzero(np.abs(coeffs) > 1e-300)
        if nz.size == 0:
            return np.empty(0)
        coeffs = coeffs[nz[0]:]
        if coeffs.size < 2:
            return np.empty(0)
        roots = np.roots(coeffs)
        real = roots[np.abs(roots.imag) < 1e-9 * (1.0 + np.abs(roots.real))].real
        real = real[real > 1e-12]
        return np.sort(real)

    def tracker_asymptote(self, E):
        """Large-``E`` branch, ``psi -> c01 / (3 d01 E)``.

        This is the *physical* branch selector: at early times E is huge and the
        tracker drives psi to zero along this root.
        """
        return self.c01 / (3.0 * self.d01 * E)
