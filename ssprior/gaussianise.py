"""The Gaussianising transform of Eq. (23) and the multivariate normal fit.

    X1 = alphaB_hat
    X2 = m  alphaB_hat^(1/6)
    X3 = w0 m^(1/4)
    X4 = wa m^2

The transform is triangular, so its inverse is explicit and its Jacobian is a
product of diagonal entries:

    |dX/dtheta| = alphaB_hat^(1/6) m^(9/4)

The Jacobian is *not* needed to reproduce the paper, which samples with a flat
measure in X and treats the fitted normal as the prior in that basis.  It is
needed if one wants the density in the physical basis instead; both are
supported by the Cobaya plugin and the distinction is spelled out in the README,
because it is the easiest thing to get wrong.

"Gaussianised" is a claim, not a definition, so ``diagnostics`` measures it:
Mardia's multivariate skewness and kurtosis, a KS test of the Mahalanobis
distance against chi^2 with 4 degrees of freedom, and 1-D KS tests on the
whitened components.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .store import GaussianPrior

__all__ = ["to_X", "from_X", "log_jacobian", "fit_gaussian", "diagnostics",
           "mardia_skewness", "sample_prior", "optimise_exponents"]

X_NAMES = ("X1", "X2", "X3", "X4")


def to_X(params: np.ndarray, p2: float, p3: float, p4: float) -> np.ndarray:
    """``(..., 4)`` array of (w0, wa, alphaB_hat, m) -> the Gaussianised basis."""
    params = np.atleast_2d(np.asarray(params, dtype=float))
    w0, wa, aB, m = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    return np.column_stack([aB, m * aB ** p2, w0 * m ** p3, wa * m ** p4])


def from_X(X: np.ndarray, p2: float, p3: float, p4: float) -> np.ndarray:
    """Exact inverse: X1 -> alphaB_hat -> m -> w0, wa."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    aB = X[:, 0]
    m = X[:, 1] / aB ** p2
    w0 = X[:, 2] / m ** p3
    wa = X[:, 3] / m ** p4
    return np.column_stack([w0, wa, aB, m])


def log_jacobian(params: np.ndarray, p2: float, p3: float, p4: float) -> np.ndarray:
    """``ln |dX/dtheta|`` with theta = (w0, wa, alphaB_hat, m).

    The transform is triangular in the order (alphaB_hat, m, w0, wa), so the
    determinant is the product of the diagonal: 1 * aB^p2 * m^p3 * m^p4.
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    aB, m = params[:, 2], params[:, 3]
    return p2 * np.log(aB) + (p3 + p4) * np.log(m)


# --------------------------------------------------------------------------- #
#  fitting
# --------------------------------------------------------------------------- #

def fit_gaussian(X: np.ndarray, trim_quantile: Optional[float] = None):
    """Maximum-likelihood mean and covariance, optionally after trimming.

    Returns ``(mu, cov, mask)``.  Trimming removes points beyond the given
    quantile of the chi^2_4 distribution of the Mahalanobis distance; it is off
    by default because the paper does not do it, but X4 = wa m^2 amplifies tails
    and it is worth checking the prior is not driven by a handful of outliers.
    """
    X = np.asarray(X, dtype=float)
    mask = np.ones(X.shape[0], dtype=bool)
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False, ddof=1)
    if trim_quantile is not None and X.shape[0] > 50:
        from scipy.stats import chi2 as chi2_dist
        d2 = mahalanobis2(X, mu, cov)
        mask = d2 <= chi2_dist.ppf(trim_quantile, df=X.shape[1])
        if mask.sum() > 20:
            mu = X[mask].mean(axis=0)
            cov = np.cov(X[mask], rowvar=False, ddof=1)
    return mu, cov, mask


def mahalanobis2(X, mu, cov) -> np.ndarray:
    d = np.asarray(X, dtype=float) - np.asarray(mu, dtype=float)
    return np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)


def mardia_skewness(W: np.ndarray) -> float:
    """Mardia's multivariate skewness of whitened data ``W`` (shape ``(n, p)``).

    The definition is

        b1 = (1/n^2) sum_{i,j} (w_i . w_j)^3 ,

    which, evaluated literally, needs the ``n x n`` Gram matrix: 6.6 GB at
    n = 30 000.  Expanding the cube of the inner product,

        (w_i . w_j)^3 = sum_{a,b,c} w_ia w_ib w_ic  w_ja w_jb w_jc ,

    and exchanging the sums over (i, j) and (a, b, c),

        b1 = (1/n^2) sum_{a,b,c} M_abc^2 ,   M_abc = sum_i w_ia w_ib w_ic .

    ``M`` is the ``p x p x p`` third-moment tensor of the whitened sample, so the
    cost is O(n p^3) time and O(p^3) extra memory (64 numbers for p = 4)
    instead of O(n^2).  Same quantity, not an approximation.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    M = np.einsum("ia,ib,ic->abc", W, W, W)
    return float((M * M).sum() / n ** 2)


def diagnostics(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> Dict:
    """Quantify how Gaussian the transformed sample actually is."""
    from scipy.stats import kstest, norm, chi2 as chi2_dist

    X = np.asarray(X, dtype=float)
    n, p = X.shape
    out: Dict = {"n": int(n), "dim": int(p)}

    try:
        L = np.linalg.cholesky(cov)
        W = np.linalg.solve(L, (X - mu).T).T           # whitened
    except np.linalg.LinAlgError:
        out["error"] = "covariance not positive definite"
        return out

    # Mardia's multivariate skewness and kurtosis (O(n) memory, see above)
    out["mardia_skewness"] = mardia_skewness(W)
    out["mardia_kurtosis"] = float((np.einsum("ij,ij->i", W, W) ** 2).mean())
    out["mardia_kurtosis_expected"] = float(p * (p + 2))

    d2 = np.einsum("ij,ij->i", W, W)
    ks = kstest(d2, lambda x: chi2_dist.cdf(x, df=p))
    out["ks_mahalanobis"] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}
    out["ks_1d"] = {
        X_NAMES[i]: float(kstest(W[:, i], norm.cdf).statistic) for i in range(p)
    }
    out["cond_cov"] = float(np.linalg.cond(cov))
    if n < 2000:
        out["warning"] = (f"only {n} models: the covariance is noisy "
                          "(the paper used 30000)")
    return out


def build_prior(params: np.ndarray, cfg, name: str = "run",
                weights: Optional[np.ndarray] = None) -> GaussianPrior:
    """Full pipeline: transform, fit, diagnose, package."""
    g = cfg.gaussianise
    p2, p3, p4 = g.p2, g.p3, g.p4
    if g.optimise_exponents:
        p2, p3, p4 = optimise_exponents(params, (p2, p3, p4))
    X = to_X(params, p2, p3, p4)
    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    mu, cov, mask = fit_gaussian(X, g.trim_quantile)
    diag = diagnostics(X[mask], mu, cov)
    diag["n_dropped_nonfinite"] = int((~finite).sum())
    diag["n_dropped_trim"] = int((~mask).sum())
    return GaussianPrior(mu=mu, cov=cov,
                         exponents={"p2": float(p2), "p3": float(p3), "p4": float(p4)},
                         x_names=list(X_NAMES), x_ranges=dict(g.x_ranges),
                         n_samples=int(mask.sum()), diagnostics=diag,
                         alphaB_convention_factor=cfg.eftcamb.alphaB_convention_factor,
                         name=name)


def optimise_exponents(params: np.ndarray, start=(1.0 / 6.0, 0.25, 2.0)):
    """Find the exponents that minimise Mardia's skewness.

    An independent check on Eq. (23): if the optimum lands near (1/6, 1/4, 2),
    the paper's transform is confirmed rather than merely assumed.
    """
    from scipy.optimize import minimize

    def objective(p):
        X = to_X(params, *p)
        if not np.all(np.isfinite(X)):
            return 1e6
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False, ddof=1)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return 1e6
        W = np.linalg.solve(L, (X - mu).T).T
        return mardia_skewness(W)

    res = minimize(objective, np.asarray(start, dtype=float), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
    return tuple(float(v) for v in res.x)


def sample_prior(prior: GaussianPrior, n: int, rng=None,
                 respect_ranges: bool = True) -> np.ndarray:
    """Draw ``n`` points in X and map them back to (w0, wa, alphaB_hat, m).

    Draws outside the MCMC box of Eq. (27)-(28), or giving a non-positive
    alphaB_hat or m, are resampled: the normal has unbounded support but the
    physical parametrisation does not.
    """
    rng = np.random.default_rng() if rng is None else rng
    p = prior.exponents
    out = np.empty((0, 4))
    guard = 0
    while out.shape[0] < n and guard < 200:
        guard += 1
        X = rng.multivariate_normal(prior.mu, prior.cov, size=2 * n)
        keep = np.ones(X.shape[0], dtype=bool)
        if respect_ranges and prior.x_ranges:
            for i, nm in enumerate(prior.x_names):
                if nm in prior.x_ranges:
                    lo, hi = prior.x_ranges[nm]
                    keep &= (X[:, i] >= lo) & (X[:, i] <= hi)
        X = X[keep]
        if X.size == 0:
            continue
        th = from_X(X, p["p2"], p["p3"], p["p4"])
        good = np.all(np.isfinite(th), axis=1) & (th[:, 2] > 0) & (th[:, 3] > 0)
        out = np.vstack([out, th[good]])
    return out[:n]
