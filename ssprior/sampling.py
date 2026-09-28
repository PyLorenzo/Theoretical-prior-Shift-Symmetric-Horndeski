"""Drawing Lagrangian coefficients from the prior box.

Uniform, uncorrelated priors on (c01, c02, d02) and H0, as in Sec. III.  The
physical constraints are *not* imposed by construction but by rejection -- this
is the paper's "slice" (Fig. 3): solving for one of the c_0i by shooting gives a
prior that depends on which coefficient was solved for, because the shooting
introduces a non-linear correction to the measure.  Sampling everything and
projecting down the constraint slab avoids that bias, at the cost of an
acceptance rate of a few percent.

Sobol is the default: the slab is a thin, curved region and a low-discrepancy
sequence fills it far more evenly than pseudo-random draws at the same cost.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DrawStream"]


class DrawStream:
    """Batched draws of ``(c01, c02, d02, H0)``, resumable.

    ``n_drawn`` is the number of points already produced; restoring it and
    re-drawing reproduces the same stream, which is what makes ``--resume``
    exact rather than approximate.
    """

    def __init__(self, cfg, n_drawn: int = 0):
        self.cfg = cfg
        self.ranges = [cfg.prior.c01, cfg.prior.c02, cfg.prior.d02, cfg.prior.H0]
        self.n_drawn = int(n_drawn)
        self._buffer = np.zeros((0, len(self.ranges)))
        self._sobol = None
        if cfg.prior.sampler == "sobol":
            from scipy.stats import qmc
            self._sobol = qmc.Sobol(d=len(self.ranges), scramble=True,
                                    seed=cfg.prior.seed)
            if self.n_drawn:
                self._sobol.fast_forward(self.n_drawn)
        else:
            self._rng = np.random.default_rng(cfg.prior.seed)
            if self.n_drawn:
                self._rng.random((self.n_drawn, len(self.ranges)))

    def _raw(self, n: int) -> np.ndarray:
        """``n`` uniform deviates in the unit hypercube.

        scipy warns when a single Sobol draw is not a power of two, because an
        isolated block of that size is not balanced.  Here successive calls
        continue one sequence, so it is the *union* of the batches -- a genuine
        Sobol prefix -- that has to be balanced, not each batch on its own.  The
        warning is therefore silenced deliberately rather than worked around by
        distorting the batch sizes.
        """
        if self._sobol is None:
            return self._rng.random((n, len(self.ranges)))
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*power of 2.*",
                                    category=UserWarning)
            return self._sobol.random(n)

    def draw(self, n: int) -> np.ndarray:
        """Return an ``(n, 4)`` array of ``(c01, c02, d02, H0)``."""
        u = self._raw(int(n))
        self.n_drawn += int(n)
        return np.column_stack([r.draw(u[:, i]) for i, r in enumerate(self.ranges)])
