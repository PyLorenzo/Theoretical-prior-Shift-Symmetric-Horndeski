"""Cobaya likelihood: the shift-symmetric theoretical prior as a Gaussian.

Implements Eq. (24)-(25) of Traykova et al. 2021 as an external likelihood, so
it multiplies the data likelihood exactly as the paper's MontePython module did.

Two bases, both correct, with different trade-offs:

``basis: X``         sample the Gaussianised variables {X1..X4} directly and let
                     Cobaya derive EFTCAMB's parameters from them with
                     ``drop``/``value``.  This is what the paper does (Sec. VI):
                     the chain moves in the basis where the distribution really
                     is a normal, so the proposal matrix is meaningful and the
                     sampler does not have to learn a curved degeneracy.
                     ``ssprior export --format cobaya`` writes the matching
                     ``params`` block.

``basis: physical``  sample ``Shift_Symmetric_alphaB0``, ``Shift_Symmetric_m``,
                     ``EFTw0``, ``EFTwa`` as usual and transform internally.
                     Simpler to bolt onto an existing YAML, but the posterior is
                     curved in these variables.

``include_jacobian`` controls whether ``ln|dX/dtheta|`` is added.  It only has
an effect when ``basis: physical``; in the ``X`` basis ``logp`` never reaches
that branch and the flag is ignored.

The prior is the fitted normal in the X basis with a flat measure there.  The
Jacobian is what preserves that distribution under a change of coordinates, so
with ``basis: physical`` you want ``include_jacobian: True`` to sample the same
distribution the ``X`` basis gives you.  Leaving it ``False`` there samples a
genuinely different distribution (on the prior in ``results/``: mean
``alphaB0`` 1.515 instead of 1.295).  The default is ``False`` only so that the ``X``-basis YAML
reads honestly; it is the wrong choice in the physical basis.

No Jacobian is needed to read physical posteriors off an ``X``-basis chain --
the derived columns already carry the right measure, because transforming
samples is not the same operation as transforming a density.

Usage::

    likelihood:
      ss_theory_prior.SSTheoryPrior:
        python_path: /path/to/Theoretical_prior_Shift_Symmetric/cobaya_plugins
        prior_file:  /path/to/Theoretical_prior_Shift_Symmetric/results/shift_symmetric_lambda0_prior.npz
        basis: X
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from cobaya.likelihood import Likelihood

PHYSICAL_NAMES = ("Shift_Symmetric_alphaB0", "Shift_Symmetric_m", "EFTw0", "EFTwa")
X_DEFAULT = ("X1", "X2", "X3", "X4")


class SSTheoryPrior(Likelihood):
    """Gaussian theoretical prior on the shift-symmetric parametrisation."""

    prior_file: str = ""
    basis: str = "X"                      # 'X' | 'physical'
    include_jacobian: bool = False
    max_mahalanobis: Optional[float] = None   # hard cut, e.g. 30 ~ chi^2_4(0.99999)
    x_names: Optional[list] = None

    def initialize(self):
        if not self.prior_file or not os.path.exists(self.prior_file):
            raise ValueError(f"SSTheoryPrior: prior_file {self.prior_file!r} not found")
        if self.basis not in ("X", "physical"):
            raise ValueError(f"SSTheoryPrior: unknown basis {self.basis!r}")

        if self.prior_file.endswith(".json"):
            with open(self.prior_file) as fh:
                payload = json.load(fh)
            mu = np.asarray(payload["mu"], dtype=float)
            cov = np.asarray(payload["cov"], dtype=float)
            exponents = payload["exponents"]
            names = payload.get("x_names", list(X_DEFAULT))
            n_samples = payload.get("n_samples", 0)
            factor = payload.get("alphaB_convention_factor")
        else:
            with np.load(self.prior_file) as f:
                mu, cov = np.asarray(f["mu"]), np.asarray(f["cov"])
            meta_path = os.path.splitext(self.prior_file)[0] + ".json"
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as fh:
                    meta = json.load(fh)
            exponents = meta.get("exponents", {"p2": 1.0 / 6.0, "p3": 0.25, "p4": 2.0})
            names = meta.get("x_names", list(X_DEFAULT))
            n_samples = meta.get("n_samples", 0)
            factor = meta.get("alphaB_convention_factor")

        self._mu = mu
        self._inv_cov = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("SSTheoryPrior: covariance is not positive definite")
        self._lognorm = -0.5 * (len(mu) * np.log(2.0 * np.pi) + logdet)
        self._p2 = float(exponents["p2"])
        self._p3 = float(exponents["p3"])
        self._p4 = float(exponents["p4"])
        self._names = list(self.x_names or names)
        self._factor = factor

        self.log.info("loaded a %d-D Gaussian prior from %d models (%s basis)",
                      len(mu), n_samples, self.basis)
        if self.basis == "physical" and not self.include_jacobian:
            self.log.warning(
                "basis='physical' with include_jacobian=False samples a different "
                "distribution from the X basis (the prior is the fitted normal in X "
                "with a flat measure there). Set include_jacobian: true unless you "
                "specifically mean N(X(theta)) with a flat measure in theta.")
        if factor is None:
            self.log.warning(
                "alphaB_convention_factor is not calibrated in this prior file; "
                "the prior assumes EFTCAMB's Shift_Symmetric_alphaB0 is the same "
                "alpha_B as the paper's. Run `ssprior eftcamb-check --kind growth "
                "--calibrate` to confirm.")
        elif abs(factor - 1.0) > 1e-6:
            self.log.info("applying alpha_B convention factor %.6f", factor)

    def get_requirements(self):
        if self.basis == "X":
            return {n: None for n in self._names}
        return {n: None for n in PHYSICAL_NAMES}

    # ---- the transform of Eq. (23) ----------------------------------------

    def _to_X(self, aB, m, w0, wa) -> np.ndarray:
        return np.array([aB, m * aB ** self._p2, w0 * m ** self._p3,
                         wa * m ** self._p4])

    def logp(self, **params_values):
        if self.basis == "X":
            X = np.array([float(params_values[n]) for n in self._names])
            extra = 0.0
        else:
            aB = float(params_values["Shift_Symmetric_alphaB0"])
            m = float(params_values["Shift_Symmetric_m"])
            if aB <= 0.0 or m <= 0.0:
                return -np.inf
            if self._factor:
                aB = aB / self._factor          # back to the paper's normalisation
            X = self._to_X(aB, m, float(params_values["EFTw0"]),
                           float(params_values["EFTwa"]))
            extra = (self._p2 * np.log(aB) + (self._p3 + self._p4) * np.log(m)
                     if self.include_jacobian else 0.0)

        if not np.all(np.isfinite(X)):
            return -np.inf
        d = X - self._mu
        d2 = float(d @ self._inv_cov @ d)
        if self.max_mahalanobis is not None and d2 > self.max_mahalanobis:
            return -np.inf
        return self._lognorm - 0.5 * d2 + extra
