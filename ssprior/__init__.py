"""Theoretical priors for shift-symmetric Horndeski cosmologies, with EFTCAMB.

Reproduces the construction of Traykova et al. 2021 (arXiv:2103.11195): sample
the Kinetic Gravity Braiding Lagrangian, solve the exact background, compress
each cosmology into {w0, wa, alphaB_hat, m} by minimising the error on
*observables*, and fit the resulting distribution with a multivariate normal in
the Gaussianised basis of Eq. (23).

Command line:  ``python -m ssprior --help``
"""

from .background import BackgroundSolution, BgStatus, Cosmology, solve_background
from .config import PriorConfig
from .fitting import FitResult, FitStatus, fit_parametrisation
from .gaussianise import build_prior, from_X, to_X
from .lagrangian import Lagrangian
from .observables import Observables, build_grid, observables_from_background
from .parametrised import ParamModel, observables_param
from .store import Dataset, GaussianPrior, ModelRecord

__version__ = "1.0.0"

__all__ = [
    "BackgroundSolution", "BgStatus", "Cosmology", "solve_background",
    "PriorConfig", "FitResult", "FitStatus", "fit_parametrisation",
    "build_prior", "from_X", "to_X", "Lagrangian", "Observables", "build_grid",
    "observables_from_background", "ParamModel", "observables_param",
    "Dataset", "GaussianPrior", "ModelRecord", "__version__",
]
