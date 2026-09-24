"""On-disk formats.

Three artefacts, all self-describing and all written atomically (``os.replace``
onto a temporary file) so an interrupted run never leaves a half-written file:

``<name>_samples.npz``  the accepted models: Lagrangian parameters, derived
                        cosmology, the *full* exact observable vectors, and the
                        fitted four parameters with their quality metrics.
``<name>_prior.npz``    mu and Sigma of the Gaussianised distribution.
``*.json``              a sidecar with the configuration, counters and
                        timestamp, next to every ``.npz``.

The exact observable vectors are stored rather than discarded (300 float64 per
model, ~72 MB for 30k -- negligible).  That makes ``ssprior fit`` re-runnable
with different weights, grids or bounds without recomputing any physics.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["ModelRecord", "Dataset", "GaussianPrior", "save_json_sidecar"]

SCHEMA = "ssprior/1"

#: column order of ``Dataset.theta``
THETA_NAMES = ("c01", "c02", "d02", "H0")
#: column order of ``Dataset.derived``
DERIVED_NAMES = ("Omega_m", "Omega_cdm", "Omega_r", "Omega_phi", "psi0",
                 "alphaB0_exact", "alphaK0_exact", "w_phi0", "z_rec",
                 "kinetic_D0", "crosses_alphaB2", "psi0_sq_over_2", "phiddot0",
                 # stage-0 closed-form guess, stored so `ssprior fit` can be
                 # re-run on the saved observables without re-solving the
                 # background
                 "guess_w0", "guess_wa", "guess_alphaB", "guess_u")
#: column order of ``Dataset.params`` -- the four parameters of the prior
PARAM_NAMES = ("w0", "wa", "alphaB_hat", "m")
#: column order of ``Dataset.quality``
QUALITY_NAMES = ("chi2", "err_low_z", "err_rec", "fit_status", "bg_status",
                 "nfev", "n_starts", "cond_JtJ", "n_candidates", "elapsed")


@dataclass
class ModelRecord:
    """Everything one sampled Lagrangian produced.  ``accepted`` gates the rest."""

    theta: np.ndarray
    accepted: bool
    reason: str = ""
    derived: Optional[np.ndarray] = None
    params: Optional[np.ndarray] = None
    quality: Optional[np.ndarray] = None
    obs: Optional[np.ndarray] = None


@dataclass
class Dataset:
    theta: np.ndarray                      # (N, 4)
    derived: np.ndarray                    # (N, len(DERIVED_NAMES))
    params: np.ndarray                     # (N, 4)
    quality: np.ndarray                    # (N, len(QUALITY_NAMES))
    obs: np.ndarray                        # (N, 3 n_z + 1)
    z: np.ndarray                          # (n_z,)
    config: Dict[str, Any] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    name: str = "run"

    # ---- views -------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.theta.shape[0])

    def column(self, name: str) -> np.ndarray:
        for names, block in ((THETA_NAMES, self.theta), (DERIVED_NAMES, self.derived),
                             (PARAM_NAMES, self.params), (QUALITY_NAMES, self.quality)):
            if name in names:
                return block[:, names.index(name)]
        raise KeyError(f"unknown column {name!r}")

    @property
    def n_z(self) -> int:
        return int(self.z.size)

    def split_obs(self):
        """Return ``(E, DA, f, DA_rec)`` as arrays over the sample."""
        n = self.n_z
        return (self.obs[:, :n], self.obs[:, n:2 * n],
                self.obs[:, 2 * n:3 * n], self.obs[:, 3 * n])

    @property
    def good(self) -> np.ndarray:
        """Boolean mask of models that met the paper's accuracy targets."""
        from .fitting import FitStatus
        return self.column("fit_status") == float(FitStatus.OK)

    def subset(self, mask: np.ndarray) -> "Dataset":
        mask = np.asarray(mask, dtype=bool)
        return Dataset(self.theta[mask], self.derived[mask], self.params[mask],
                       self.quality[mask], self.obs[mask], self.z,
                       dict(self.config), dict(self.counters), self.name)

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_records(cls, records: List[ModelRecord], z: np.ndarray,
                     config: Dict[str, Any], counters: Dict[str, int],
                     name: str) -> "Dataset":
        keep = [r for r in records if r.accepted and r.params is not None]
        if not keep:
            nd, nq = len(DERIVED_NAMES), len(QUALITY_NAMES)
            return cls(np.zeros((0, 4)), np.zeros((0, nd)), np.zeros((0, 4)),
                       np.zeros((0, nq)), np.zeros((0, 3 * z.size + 1)), z,
                       config, counters, name)
        return cls(
            theta=np.array([r.theta for r in keep], dtype=float),
            derived=np.array([r.derived for r in keep], dtype=float),
            params=np.array([r.params for r in keep], dtype=float),
            quality=np.array([r.quality for r in keep], dtype=float),
            obs=np.array([r.obs for r in keep], dtype=float),
            z=np.asarray(z, dtype=float), config=config, counters=counters, name=name)

    def concat(self, other: "Dataset") -> "Dataset":
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other
        merged = {k: self.counters.get(k, 0) + other.counters.get(k, 0)
                  for k in set(self.counters) | set(other.counters)}
        return Dataset(
            np.vstack([self.theta, other.theta]),
            np.vstack([self.derived, other.derived]),
            np.vstack([self.params, other.params]),
            np.vstack([self.quality, other.quality]),
            np.vstack([self.obs, other.obs]),
            self.z, self.config, merged, self.name)

    # ---- I/O ---------------------------------------------------------------

    def save(self, path: str) -> None:
        _atomic_npz(path, theta=self.theta, derived=self.derived,
                    params=self.params, quality=self.quality, obs=self.obs, z=self.z)
        save_json_sidecar(path, {
            "schema": SCHEMA, "name": self.name, "n_models": len(self),
            "theta_names": list(THETA_NAMES), "derived_names": list(DERIVED_NAMES),
            "param_names": list(PARAM_NAMES), "quality_names": list(QUALITY_NAMES),
            "counters": self.counters, "config": self.config,
        })

    @classmethod
    def load(cls, path: str) -> "Dataset":
        with np.load(path) as f:
            arrays = {k: f[k] for k in f.files}
        meta = load_json_sidecar(path)
        return cls(arrays["theta"], arrays["derived"], arrays["params"],
                   arrays["quality"], arrays["obs"], arrays["z"],
                   meta.get("config", {}), meta.get("counters", {}),
                   meta.get("name", "run"))

    def summary(self) -> str:
        n = len(self)
        if n == 0:
            return "empty dataset"
        good = self.good
        lines = [f"{self.name}: {n} accepted models, {good.sum()} ({100*good.mean():.1f}%) "
                 f"within the paper's targets"]
        for nm in PARAM_NAMES:
            v = self.column(nm)[good] if good.any() else self.column(nm)
            lines.append(f"   {nm:11s} median {np.median(v):+8.4f}   "
                         f"[{np.percentile(v, 5):+8.4f}, {np.percentile(v, 95):+8.4f}] (5-95%)")
        if self.counters:
            lines.append("   rejections: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.counters.items()) if v))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  the prior itself
# --------------------------------------------------------------------------- #

@dataclass
class GaussianPrior:
    """Multivariate normal of Eq. (24) in the Gaussianised basis {X1..X4}."""

    mu: np.ndarray
    cov: np.ndarray
    exponents: Dict[str, float]            # p2, p3, p4 of Eq. (23)
    x_names: List[str] = field(default_factory=lambda: ["X1", "X2", "X3", "X4"])
    x_ranges: Dict[str, Any] = field(default_factory=dict)
    n_samples: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    alphaB_convention_factor: Optional[float] = None
    name: str = "run"

    @property
    def inv_cov(self) -> np.ndarray:
        return np.linalg.inv(self.cov)

    @property
    def corr(self) -> np.ndarray:
        s = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(s, s)

    def save(self, path: str) -> None:
        _atomic_npz(path, mu=self.mu, cov=self.cov)
        save_json_sidecar(path, {
            "schema": SCHEMA, "name": self.name, "kind": "gaussian_prior",
            "x_names": self.x_names, "x_ranges": self.x_ranges,
            "exponents": self.exponents, "n_samples": self.n_samples,
            "diagnostics": self.diagnostics,
            "alphaB_convention_factor": self.alphaB_convention_factor,
        })

    @classmethod
    def load(cls, path: str) -> "GaussianPrior":
        with np.load(path) as f:
            mu, cov = f["mu"], f["cov"]
        meta = load_json_sidecar(path)
        return cls(mu=mu, cov=cov, exponents=meta.get("exponents", {}),
                   x_names=meta.get("x_names", ["X1", "X2", "X3", "X4"]),
                   x_ranges=meta.get("x_ranges", {}),
                   n_samples=int(meta.get("n_samples", 0)),
                   diagnostics=meta.get("diagnostics", {}),
                   alphaB_convention_factor=meta.get("alphaB_convention_factor"),
                   name=meta.get("name", "run"))

    def summary(self, reference: Optional[Dict[str, Any]] = None) -> str:
        lines = [f"Gaussian prior '{self.name}' from {self.n_samples} models",
                 "  mu = (" + ", ".join(f"{v:.4f}" for v in self.mu) + ")"]
        err = np.sqrt(np.diag(self.cov) / max(self.n_samples, 1))
        lines.append("  +- " + "  ".join(f"{v:.4f}" for v in err) + "  (Monte Carlo error on mu)")
        lines.append("  Sigma =")
        for row in self.cov:
            lines.append("      " + "  ".join(f"{v:+9.4f}" for v in row))
        if reference:
            ref_mu = np.asarray(reference["mu"], dtype=float)
            dev = (self.mu - ref_mu) / np.where(err > 0, err, np.nan)
            lines.append(f"  paper  = (" + ", ".join(f"{v:.4f}" for v in ref_mu) + ")")
            lines.append("  (this - paper)/MC error = "
                         + "  ".join(f"{v:+.1f}" for v in dev))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _atomic_npz(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:          # a file handle: np.savez would append '.npz'
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def _sidecar_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def save_json_sidecar(path: str, meta: Dict[str, Any]) -> None:
    meta = dict(meta)
    meta.setdefault("written", time.strftime("%Y-%m-%d %H:%M:%S"))
    meta.setdefault("host", platform.node())
    p = _sidecar_path(path)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    os.replace(tmp, p)


def load_json_sidecar(path: str) -> Dict[str, Any]:
    p = _sidecar_path(path)
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        return json.load(fh)
