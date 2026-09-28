"""Configuration: one YAML file describes a whole run.

Follows the conventions of ``stabmap/config.py``: a frozen dataclass tree built
by ``PriorConfig.from_yaml``, every relative path resolved against the config
file (not the working directory), and ``to_dict()`` round-tripping the whole
thing so it can be stored next to the results as provenance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

__all__ = ["PriorConfig", "Range", "ConfigError", "OMEGA_R_COEFF"]

#: rho_gamma / rho_crit at T_cmb = 1 K, h = 1.  Omega_gamma h^2 = COEFF * T^4.
#: Value pinned to CAMB's constants so the Python background matches to <1e-4
#: (verified by ``ssprior eftcamb-check --kind background``).
OMEGA_R_COEFF = 4.48130e-7  # per K^4


class ConfigError(Exception):
    """Raised for a malformed or inconsistent configuration."""


@dataclass(frozen=True)
class Range:
    min: float
    max: float

    def __post_init__(self):
        if not np.isfinite([self.min, self.max]).all() or self.max <= self.min:
            raise ConfigError(f"empty or non-finite range [{self.min}, {self.max}]")

    @property
    def width(self) -> float:
        return self.max - self.min

    def draw(self, u):
        """Map uniform deviates in [0, 1) onto the range."""
        return self.min + u * self.width

    def contains(self, x) -> bool:
        return bool(np.all((x >= self.min) & (x <= self.max)))


def _range(d: Dict[str, Any], key: str) -> Range:
    try:
        sub = d[key]
    except KeyError:
        raise ConfigError(f"missing range '{key}'") from None
    if not isinstance(sub, dict) or "min" not in sub or "max" not in sub:
        raise ConfigError(f"'{key}' must be a mapping with 'min' and 'max'")
    return Range(float(sub["min"]), float(sub["max"]))


# --------------------------------------------------------------------------- #
#  sections
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelSection:
    d01: float = -1.0
    lambda_mode: str = "zero"      # 'zero' (Omega_DE = Omega_phi) | 'free'
    tracker: bool = True           # j == 0 exactly; False integrates j = j_i a^-3
    a_ini: float = 1.0e-8

    def __post_init__(self):
        if self.lambda_mode not in ("zero", "free"):
            raise ConfigError(f"model.lambda_mode must be 'zero' or 'free', got {self.lambda_mode!r}")
        if self.d01 == 0.0:
            raise ConfigError("model.d01 must be non-zero (it sets the field normalisation)")


@dataclass(frozen=True)
class PriorSection:
    c01: Range
    c02: Range
    d02: Range
    H0: Range
    Omega_Lambda: Range | None = None
    sampler: str = "sobol"         # 'sobol' | 'uniform'
    seed: int = 42

    def __post_init__(self):
        if self.sampler not in ("sobol", "uniform"):
            raise ConfigError(f"prior.sampler must be 'sobol' or 'uniform', got {self.sampler!r}")


# Default baryon fraction of CLASS 2.x (input.c: Omega0_b = 0.022032 / 0.67556^2).
# RUFIAN (the code of Traykova et al.) never sets Omega_b or omega_b, so hi_class
# keeps this *fraction* fixed while h is sampled; omega_b = Omega_b h^2 varies.
CLASS_DEFAULT_OMEGA_B = 0.022032 / 0.67556 ** 2


@dataclass(frozen=True)
class CosmologySection:
    H_fid: float = 70.0
    ombh2: float = 0.0224
    # If set, the baryon *fraction* is held fixed and ombh2 = Omega_b h^2 varies
    # with the sampled H0 (RUFIAN / CLASS-default convention); ``ombh2`` is then
    # ignored.  If None, ``ombh2`` is held fixed instead.
    Omega_b: Optional[float] = None
    T_cmb: float = 2.7255
    N_eff: float = 3.046
    Omega_cdm: Range = field(default_factory=lambda: Range(0.15, 0.35))

    def __post_init__(self):
        if self.Omega_b is not None and not (0.0 < self.Omega_b < 1.0):
            raise ConfigError(f"cosmology.Omega_b must be in (0, 1), got {self.Omega_b}")

    def ombh2_of(self, H0: float) -> float:
        """Physical baryon density for a model with Hubble rate ``H0``."""
        if self.Omega_b is not None:
            return self.Omega_b * (H0 / 100.0) ** 2
        return self.ombh2

    def omega_r(self, H0: float) -> float:
        """Photons + massless neutrinos, as a fraction of today's critical density.

        Massive neutrinos are deliberately excluded (the paper does not include
        them, and they would break the <1e-4 agreement with CAMB that the chi^2
        target of 1e-3 relies on).
        """
        h2 = (H0 / 100.0) ** 2
        omega_gamma_h2 = OMEGA_R_COEFF * self.T_cmb ** 4
        return omega_gamma_h2 * (1.0 + 0.2271 * self.N_eff) / h2


@dataclass(frozen=True)
class BackgroundSection:
    n_grid: int = 600
    newton_tol: float = 1.0e-12
    newton_max_iter: int = 12
    reject_on_fold: bool = True
    asymptote_tol: float = 0.10


@dataclass(frozen=True)
class GrowthSection:
    a_start: float = 1.0e-3
    f_start: str = "gamma"         # 'gamma' -> Omega_m(a)^0.55 | 'unity'
    n_steps: int = 400
    stop_on_gradient_instability: bool = True

    def __post_init__(self):
        if self.f_start not in ("gamma", "unity"):
            raise ConfigError(f"growth.f_start must be 'gamma' or 'unity', got {self.f_start!r}")


@dataclass(frozen=True)
class GridSection:
    n_z: int = 100
    z_min: float = 0.01            # may be 0 for 'ln1pz'/'linear' (D_A(0) = 0 is handled)
    z_max: float = 10.0
    spacing: str = "log"           # 'log' | 'linear' | 'ln1pz' (uniform in ln(1+z), RUFIAN)
    z_rec: str = "hu_sugiyama"     # 'hu_sugiyama' | 'fixed:1090'
    n_quad: int = 2000

    def __post_init__(self):
        if self.spacing not in ("log", "linear", "ln1pz"):
            raise ConfigError("grid.spacing must be 'log', 'linear' or 'ln1pz', "
                              f"got {self.spacing!r}")
        if self.z_min < 0.0 or (self.spacing == "log" and self.z_min <= 0.0):
            raise ConfigError("grid.z_min must be >= 0, and > 0 for log spacing")

    def z_nodes(self) -> np.ndarray:
        """The observable redshifts, ascending.

        'ln1pz' reproduces RUFIAN: ``Binning._get_index_to_compare`` takes
        ``points_to_fit`` equally spaced *rows* of the CLASS background table
        between z_low_redshift_fit and z = 0, and CLASS 2.x writes that table at
        a constant step in ln a -- so the points are uniform in ln(1+z), and the
        last one is z = 0.
        """
        if self.spacing == "log":
            return np.geomspace(self.z_min, self.z_max, self.n_z)
        if self.spacing == "ln1pz":
            return np.expm1(np.linspace(np.log1p(self.z_min), np.log1p(self.z_max), self.n_z))
        return np.linspace(self.z_min, self.z_max, self.n_z)


@dataclass(frozen=True)
class FitSection:
    sigma_low_z: float = 1.0e-3
    sigma_rec: float = 1.0e-4
    # internal optimisation uses u = 4/m, which is far better conditioned than m
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "w0": (-3.0, 0.0),
        "wa": (-10.0, 10.0),
        "alphaB_hat": (1.0e-6, 50.0),
        "u": (0.05, 40.0),
    })
    multistart_u: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    accept_rel_err_low_z: float = 0.01
    accept_rel_err_rec: float = 0.003
    xtol: float = 1.0e-12
    ftol: float = 1.0e-12
    max_nfev: int = 400

    def bound_pair(self, name: str) -> Tuple[float, float]:
        lo, hi = self.bounds[name]
        return float(lo), float(hi)


@dataclass(frozen=True)
class GaussianiseSection:
    #: exponents of Eq. (23): X1 = aB, X2 = m aB^p2, X3 = w0 m^p3, X4 = wa m^p4
    p2: float = 1.0 / 6.0
    p3: float = 0.25
    p4: float = 2.0
    optimise_exponents: bool = False
    trim_quantile: float | None = None
    x_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "X1": (0.0, 10.0), "X2": (0.0, 15.0),
        "X3": (-10.0, 0.0), "X4": (-15.0, 30.0),
    })


def _resolve_build_path(value, abspath) -> str:
    """EFTCAMB location: the config value (with ``~`` and ``$VARS`` expanded),
    else the ``EFTCAMB_PATH`` environment variable, else "" (not configured)."""
    for candidate in (value, os.environ.get("EFTCAMB_PATH")):
        if not candidate:
            continue
        p = os.path.expanduser(os.path.expandvars(str(candidate)))
        if p and "$" not in p:
            return abspath(p)
    return ""


@dataclass(frozen=True)
class EftcambSection:
    #: directory of the EFTCAMB Python build (the one containing ``camb/``).
    #: ``~`` and ``$VARS`` are expanded; if empty or unresolved, the
    #: ``EFTCAMB_PATH`` environment variable is used.  Only ``eftcamb-check``
    #: needs it.
    build_path: str = ""
    alphaK0: float = 10.0
    #: multiplicative factor between EFTCAMB's RPH alpha_B and the Bellini-Sawicki
    #: alpha_B used by the paper.  Calibrated by ``eftcamb-check --calibrate``;
    #: None means "not yet calibrated" and ``export`` will refuse to apply it.
    alphaB_convention_factor: float | None = None
    timeout: float = 45.0
    workers: int = 8
    # fiducial cosmology for the EFTCAMB cross-checks
    tau: float = 0.055
    As: float = 2.1e-9
    ns: float = 0.967


@dataclass(frozen=True)
class RunSection:
    n_models: int = 30000
    workers: int = 8
    checkpoint_every: int = 200


# --------------------------------------------------------------------------- #
#  top level
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PriorConfig:
    name: str
    outdir: str
    model: ModelSection
    prior: PriorSection
    cosmology: CosmologySection
    background: BackgroundSection
    growth: GrowthSection
    grid: GridSection
    fit: FitSection
    gaussianise: GaussianiseSection
    eftcamb: EftcambSection
    run: RunSection
    source: str | None = None

    # ---- construction ---------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "PriorConfig":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw, source=os.path.abspath(path))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], source: str | None = None) -> "PriorConfig":
        base_dir = os.path.dirname(source) if source else os.getcwd()

        def abspath(p: str) -> str:
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))

        name = str(raw.get("name", "run"))
        outdir = abspath(str(raw.get("outdir", "../results")))

        model = ModelSection(**(raw.get("model") or {}))

        pr = raw.get("prior") or {}
        # `is not None` rather than `in`: a config round-tripped through
        # to_dict() carries every optional key with a null value
        lam = _range(pr, "Omega_Lambda") if pr.get("Omega_Lambda") is not None else None
        if model.lambda_mode == "free" and lam is None:
            raise ConfigError("model.lambda_mode='free' requires prior.Omega_Lambda")
        prior = PriorSection(
            c01=_range(pr, "c01"), c02=_range(pr, "c02"), d02=_range(pr, "d02"),
            H0=_range(pr, "H0"), Omega_Lambda=lam,
            sampler=str(pr.get("sampler", "sobol")), seed=int(pr.get("seed", 42)),
        )

        co = dict(raw.get("cosmology") or {})
        ocdm = _range(co, "Omega_cdm") if "Omega_cdm" in co else Range(0.15, 0.35)
        co.pop("Omega_cdm", None)
        cosmology = CosmologySection(Omega_cdm=ocdm, **co)

        fit_raw = dict(raw.get("fit") or {})
        bounds = fit_raw.pop("bounds", None)
        ms = fit_raw.pop("multistart_u", None)
        fit = FitSection(**fit_raw)
        if bounds:
            merged = dict(fit.bounds)
            merged.update({k: (float(v[0]), float(v[1])) for k, v in bounds.items()})
            fit = FitSection(**{**fit_raw, "bounds": merged,
                                "multistart_u": tuple(ms) if ms else fit.multistart_u})
        elif ms:
            fit = FitSection(**{**fit_raw, "multistart_u": tuple(ms)})

        ga_raw = dict(raw.get("gaussianise") or {})
        xr = ga_raw.pop("x_ranges", None)
        gauss = GaussianiseSection(**ga_raw)
        if xr:
            merged = dict(gauss.x_ranges)
            merged.update({k: (float(v[0]), float(v[1])) for k, v in xr.items()})
            gauss = GaussianiseSection(**{**ga_raw, "x_ranges": merged})

        eft_raw = dict(raw.get("eftcamb") or {})
        eft_raw["build_path"] = _resolve_build_path(eft_raw.get("build_path"), abspath)

        cfg = cls(
            name=name, outdir=outdir, model=model, prior=prior, cosmology=cosmology,
            background=BackgroundSection(**(raw.get("background") or {})),
            growth=GrowthSection(**(raw.get("growth") or {})),
            grid=GridSection(**(raw.get("grid") or {})),
            fit=fit, gaussianise=gauss,
            eftcamb=EftcambSection(**eft_raw),
            run=RunSection(**(raw.get("run") or {})),
            source=source,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.prior.c01.min >= 0.0:
            raise ConfigError(
                "prior.c01 must allow negative values: on the tracker "
                "rho_phi = -(c01 + c02 psi^2/2) psi^2 / 2, so a positive scalar "
                "energy density is impossible for c01 >= 0 with small c02."
            )
        if self.grid.z_max <= self.grid.z_min:
            raise ConfigError("grid.z_max must exceed grid.z_min")
        if self.model.a_ini >= self.growth.a_start:
            pass  # allowed: background starts earlier than the growth integration

    # ---- derived paths --------------------------------------------------------

    def path(self, suffix: str) -> str:
        return os.path.join(self.outdir, f"{self.name}_{suffix}")

    # ---- provenance -----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclasses.asdict turns Range into {'min':..,'max':..}; tuples into lists
        return d

    def summary(self) -> str:
        p = self.prior
        return (
            f"{self.name}  [lambda_mode={self.model.lambda_mode}, "
            f"tracker={self.model.tracker}]\n"
            f"  c01 in [{p.c01.min:g}, {p.c01.max:g}]   "
            f"c02 in [{p.c02.min:g}, {p.c02.max:g}]   "
            f"d02 in [{p.d02.min:g}, {p.d02.max:g}]\n"
            f"  H0  in [{p.H0.min:g}, {p.H0.max:g}] (H_fid = {self.cosmology.H_fid:g})   "
            f"Omega_cdm in [{self.cosmology.Omega_cdm.min:g}, {self.cosmology.Omega_cdm.max:g}]\n"
            f"  grid: {self.grid.n_z} z-points in [{self.grid.z_min:g}, {self.grid.z_max:g}] "
            f"({self.grid.spacing}), sigma = {self.fit.sigma_low_z:g} / "
            f"{self.fit.sigma_rec:g} at recombination"
        )
