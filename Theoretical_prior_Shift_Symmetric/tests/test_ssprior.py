"""Test suite.  No Boltzmann code needed; runs in well under a minute.

    cd Theoretical_priors && python -m pytest tests -q

The tests that matter are the ones that check *identities*, not outputs: if the
continuity equation holds to machine precision along the solution, the algebraic
system, the analytic derivatives and the pressure are all simultaneously right.
The EFTCAMB comparisons live in ``ssprior eftcamb-check``, not here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssprior.background import (BgStatus, slice_candidates,  # noqa: E402
                                solve_background)
from ssprior.config import PriorConfig  # noqa: E402
from ssprior.gaussianise import (fit_gaussian, from_X, log_jacobian,  # noqa: E402
                                 mardia_skewness, to_X)
from ssprior.growth import cs2_N2, growth_grid, integrate_growth  # noqa: E402
from ssprior.lagrangian import Lagrangian  # noqa: E402
from ssprior.observables import build_grid, observables_from_background  # noqa: E402
from ssprior.parametrised import E_cpl, observables_param  # noqa: E402
from ssprior.fitting import fit_parametrisation, initial_guess  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "models", "shift_symmetric_lambda0.yaml")


@pytest.fixture(scope="module")
def cfg():
    return PriorConfig.from_yaml(CONFIG)


@pytest.fixture(scope="module")
def solved(cfg):
    """A representative accepted model, solved once for the whole module."""
    lag = Lagrangian(c01=-28.83, c02=27.12, d02=-116.80, d01=-1.0)
    cands = slice_candidates(lag, 73.6, cfg)
    assert cands, "the reference model must pass the slice"
    psi0, cosmo = cands[0]
    sol = solve_background(lag, cosmo, psi0, cfg)
    assert sol.status is BgStatus.OK
    return lag, sol


# --------------------------------------------------------------------------- #
#  T1: the algebraic identities of the Lagrangian
# --------------------------------------------------------------------------- #

def test_identity_rho_E_equals_3E_alphaB():
    """``drho/dE = psi dj/dE = 3 E alpha_B`` -- ties alpha_B to the Jacobian."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        lag = Lagrangian(*(rng.normal(size=3) * 5), d01=float(rng.normal()))
        psi, E = abs(rng.normal()) + 0.05, abs(rng.normal()) + 0.5
        pa = lag.partials(psi, E)
        assert pa.rho_E == pytest.approx(3.0 * E * lag.alpha_B(psi, E), rel=1e-12)
        assert pa.rho_E == pytest.approx(psi * pa.j_E, rel=1e-12)


def test_identity_rho_equals_psi_j_minus_g2():
    """Eq. (14): ``rho_phi = phidot J - G2``.  On the tracker, rho = -G2."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        lag = Lagrangian(*(rng.normal(size=3) * 5), d01=float(rng.normal()))
        psi, E = abs(rng.normal()) + 0.05, abs(rng.normal()) + 0.5
        assert lag.rho(psi, E) == pytest.approx(
            psi * lag.j(psi, E) - lag.g2(psi), rel=1e-12, abs=1e-14)


def test_partials_match_finite_differences():
    rng = np.random.default_rng(2)
    lag = Lagrangian(-12.0, 30.0, -40.0)
    psi, E, h = 0.37, 1.31, 1e-6
    pa = lag.partials(psi, E)
    assert pa.j_psi == pytest.approx((lag.j(psi + h, E) - lag.j(psi - h, E)) / (2 * h), rel=1e-6)
    assert pa.j_E == pytest.approx((lag.j(psi, E + h) - lag.j(psi, E - h)) / (2 * h), rel=1e-6)
    assert pa.rho_psi == pytest.approx((lag.rho(psi + h, E) - lag.rho(psi - h, E)) / (2 * h), rel=1e-6)


def test_tracker_roots_solve_the_current():
    lag = Lagrangian(-12.0, 40.0, -30.0)
    roots = lag.tracker_roots(1.0)
    assert roots.size >= 1
    for r in roots:
        assert lag.j(r, 1.0) == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------------------------------- #
#  T2-T3: the solved background
# --------------------------------------------------------------------------- #

def test_continuity_holds_to_machine_precision(solved):
    """``drho/dN + 3(rho + p) = 0`` with drho/dN from the analytic chain rule.

    This single check validates the algebraic system, the 2x2 derivative solve
    and the expression for the pressure all at once.
    """
    lag, sol = solved
    pa = lag.partials(sol.psi, sol.E)
    drho_dN = pa.rho_psi * sol.dpsi_dN + pa.rho_E * sol.dE_dN
    residual = drho_dN + 3.0 * (sol.rho_phi + sol.p_phi)
    assert np.abs(residual).max() / np.abs(sol.rho_phi).max() < 1e-12


def test_friedmann_closes_on_every_node(solved):
    lag, sol = solved
    F = 3.0 * sol.E ** 2 - sol.cosmo.rho_bg(sol.a) - sol.rho_phi
    assert np.abs(F).max() / (3.0 * sol.E ** 2).max() < 1e-12


def test_branch_lands_on_the_early_time_asymptote(solved):
    lag, sol = solved
    asym = lag.tracker_asymptote(sol.E[0])
    assert sol.psi[0] / asym == pytest.approx(1.0, rel=1e-6)


def test_equation_of_state_is_phantom(solved):
    """Footnote 5: on the tracker w_phi < -1, approaching -1 from below."""
    _, sol = solved
    late = sol.a > 0.5
    assert np.all(sol.w_phi[late] < -1.0 + 1e-6)
    assert sol.w_phi[-1] > sol.w_phi[late][0] - 1e-9   # rising towards -1


def test_no_ghost(solved):
    _, sol = solved
    assert np.all(sol.kinetic_D > 0.0)
    assert np.all(sol.rho_phi > 0.0)


def test_cubic_galileon_limit_is_exact(cfg):
    """c02 = d02 = 0: psi E is constant, alpha_B ~ E^-4, so m = 1 exactly."""
    lag = Lagrangian(c01=-3.30, c02=0.0, d02=0.0)
    cands = slice_candidates(lag, 70.0, cfg)
    assert cands
    psi0, cosmo = cands[0]
    sol = solve_background(lag, cosmo, psi0, cfg)
    assert sol.ok
    assert np.ptp(sol.psi * sol.E) / np.mean(sol.psi * sol.E) < 1e-9
    assert np.ptp(sol.alpha_B * sol.E ** 4) / np.mean(sol.alpha_B * sol.E ** 4) < 1e-9
    # the stage-0 guess must already read off m = 1
    _, _, _, u = initial_guess(sol, cfg)
    assert 4.0 / u == pytest.approx(1.0, rel=1e-3)


def test_background_grid_convergence(cfg, solved):
    """Doubling the grid must not move the observables.

    The tolerances are set relative to what they have to beat: Eq. (21) weights
    the observables at sigma = 1e-3 (1e-4 at recombination), so a discretisation
    error of 1e-8 to 1e-7 is four to five orders of magnitude inside the signal.
    Tightening these further would only be testing the spline, not the physics.
    """
    from dataclasses import replace
    lag, sol = solved
    grid = build_grid(cfg, sol.cosmo.ombh2, sol.cosmo.Omega_m * sol.cosmo.h ** 2, sol.N)
    o1 = observables_from_background(sol, grid, cfg)

    cfg2 = replace(cfg, background=replace(cfg.background,
                                           n_grid=2 * cfg.background.n_grid - 1))
    sol2 = solve_background(lag, sol.cosmo, sol.psi0, cfg2)
    grid2 = build_grid(cfg2, sol.cosmo.ombh2,
                       sol.cosmo.Omega_m * sol.cosmo.h ** 2, sol2.N)
    o2 = observables_from_background(sol2, grid2, cfg2)
    assert np.abs(o1.E / o2.E - 1.0).max() < 1e-6      # vs sigma = 1e-3
    from ssprior.observables import rel_residual
    assert np.abs(rel_residual(o1.DA, o2.DA, 1.0)).max() < 1e-6    # vs sigma = 1e-3
    assert abs(o1.DA_rec / o2.DA_rec - 1.0) < 1e-6     # vs sigma = 1e-4


# --------------------------------------------------------------------------- #
#  growth
# --------------------------------------------------------------------------- #

def test_cs2N2_vanishes_in_gr_matter_domination():
    """alpha_B = 0, Hdot/H^2 = -3/2, Omega_m = 1  =>  c_s^2 alpha = 0 (alpha_K = 0)."""
    assert cs2_N2(0.0, 0.0, -1.5, 1.0) == pytest.approx(0.0, abs=1e-14)


def test_growth_recovers_lcdm(cfg):
    """With alpha_B = 0 and w = -1 the quasi-static equation is the LCDM one.

    Compared against an independent adaptive integration of the same ODE, which
    also checks that the fixed-step RK4 is accurate and not merely consistent.
    """
    from scipy.integrate import solve_ivp

    Om, Orad, H0 = 0.31, 0.0, 70.0
    Nf = growth_grid(cfg)
    E_f, dE_f = E_cpl(Nf, Om, Orad, -1.0, 0.0)
    dEoE = dE_f / E_f
    Om_a = Om * np.exp(Nf) ** -3 / E_f ** 2
    mu = np.ones_like(Nf)
    f0 = Om_a[0] ** 0.55
    f_rk4 = integrate_growth(Nf, dEoE, Om_a, mu, f0)

    from scipy.interpolate import CubicSpline
    sp_dEoE, sp_Om = CubicSpline(Nf, dEoE), CubicSpline(Nf, Om_a)

    def rhs(N, y):
        return [-y[0] ** 2 - (2.0 + sp_dEoE(N)) * y[0] + 1.5 * sp_Om(N)]

    ref = solve_ivp(rhs, (Nf[0], 0.0), [f0], rtol=1e-11, atol=1e-13,
                    dense_output=True)
    assert abs(f_rk4[-1] / ref.y[0, -1] - 1.0) < 1e-7
    # and the familiar today value
    assert f_rk4[-1] == pytest.approx(Om ** 0.55, rel=0.02)


def test_growth_initial_condition_is_irrelevant(cfg):
    """The decaying mode is attractive: f(z<10) must not remember f(a_start)."""
    Om, Orad = 0.31, 0.0
    Nf = growth_grid(cfg)
    E_f, dE_f = E_cpl(Nf, Om, Orad, -1.0, 0.0)
    dEoE, Om_a = dE_f / E_f, Om * np.exp(Nf) ** -3 / E_f ** 2
    mu = np.ones_like(Nf)
    a = integrate_growth(Nf, dEoE, Om_a, mu, Om_a[0] ** 0.55)
    b = integrate_growth(Nf, dEoE, Om_a, mu, 1.0)
    assert abs(a[-1] / b[-1] - 1.0) < 1e-5


# --------------------------------------------------------------------------- #
#  fitting
# --------------------------------------------------------------------------- #

def test_fit_recovers_synthetic_parameters(cfg, solved):
    """Feed the fitter observables generated by the parametrisation itself.

    Isolates the optimiser from the physics: if this fails the cascade is broken,
    not the model.
    """
    _, sol = solved
    cosmo = sol.cosmo
    grid = build_grid(cfg, cosmo.ombh2, cosmo.Omega_m * cosmo.h ** 2, sol.N)
    truth = (-1.05, -0.30, 1.20, 1.60)          # w0, wa, alphaB_hat, m
    synthetic = observables_param((truth[0], truth[1], truth[2], 4.0 / truth[3]),
                                  cosmo.Omega_m, cosmo.Omega_r, cosmo.H0, grid, cfg)
    res = fit_parametrisation(synthetic, grid, cosmo, cfg,
                              guess=(-1.0, 0.0, 1.0, 2.0))
    assert res.ok
    for got, want in zip(res.params.as_array(), truth):
        assert got == pytest.approx(want, rel=2e-3)
    assert res.errors["low_z"] < 1e-5


def test_fit_meets_the_paper_targets(cfg, solved):
    _, sol = solved
    grid = build_grid(cfg, sol.cosmo.ombh2,
                      sol.cosmo.Omega_m * sol.cosmo.h ** 2, sol.N)
    exact = observables_from_background(sol, grid, cfg)
    res = fit_parametrisation(exact, grid, sol.cosmo, cfg, initial_guess(sol, cfg))
    assert res.ok
    assert res.errors["low_z"] < cfg.fit.accept_rel_err_low_z
    assert res.errors["rec"] < cfg.fit.accept_rel_err_rec


# --------------------------------------------------------------------------- #
#  Gaussianisation
# --------------------------------------------------------------------------- #

def test_X_round_trip():
    rng = np.random.default_rng(3)
    params = np.column_stack([rng.uniform(-1.3, -0.9, 200),
                              rng.uniform(-2.0, 0.0, 200),
                              rng.uniform(0.2, 3.0, 200),
                              rng.uniform(0.5, 4.0, 200)])
    X = to_X(params, 1 / 6, 0.25, 2.0)
    back = from_X(X, 1 / 6, 0.25, 2.0)
    assert np.allclose(back, params, rtol=1e-12, atol=1e-12)


def test_log_jacobian_matches_numerical_determinant():
    p2, p3, p4 = 1 / 6, 0.25, 2.0
    theta = np.array([[-1.05, -0.4, 1.3, 1.7]])
    h = 1e-6
    J = np.empty((4, 4))
    for j in range(4):
        d = np.zeros(4)
        d[j] = h
        J[:, j] = (to_X(theta + d, p2, p3, p4)[0]
                   - to_X(theta - d, p2, p3, p4)[0]) / (2 * h)
    num = np.log(abs(np.linalg.det(J)))
    assert log_jacobian(theta, p2, p3, p4)[0] == pytest.approx(num, rel=1e-5)


def test_fit_gaussian_recovers_a_known_normal():
    rng = np.random.default_rng(4)
    mu = np.array([1.5, 1.4, -1.16, -0.88])
    cov = np.array([[0.1475, -0.0916, 0.0160, -0.0469],
                    [-0.0916, 0.0776, -0.0087, 0.0326],
                    [0.0160, -0.0087, 0.0041, -0.0079],
                    [-0.0469, 0.0326, -0.0079, 0.0516]])
    X = rng.multivariate_normal(mu, cov, size=40000)
    mu_hat, cov_hat, _ = fit_gaussian(X)
    err = np.sqrt(np.diag(cov) / X.shape[0])
    assert np.all(np.abs(mu_hat - mu) < 4.0 * err)
    assert np.allclose(cov_hat, cov, atol=0.02)


# --------------------------------------------------------------------------- #
#  storage
# --------------------------------------------------------------------------- #

def test_dataset_round_trip(tmp_path, cfg):
    from ssprior.pipeline import evaluate_model
    from ssprior.store import Dataset

    rec = evaluate_model(np.array([-28.83, 27.12, -116.80, 73.6]), cfg=cfg)
    assert rec.accepted, rec.reason
    z = cfg.grid.z_nodes()
    ds = Dataset.from_records([rec], z, cfg.to_dict(), {"drawn": 1}, cfg.name)
    path = str(tmp_path / "ds.npz")
    ds.save(path)
    back = Dataset.load(path)
    assert len(back) == 1
    assert np.allclose(back.params, ds.params)
    assert np.allclose(back.obs, ds.obs)
    # and the configuration survives the round trip
    PriorConfig.from_dict(back.config)


def test_config_round_trip(cfg):
    again = PriorConfig.from_dict(cfg.to_dict())
    assert again.name == cfg.name
    assert again.prior.c01.min == cfg.prior.c01.min
    assert again.fit.bounds["u"] == cfg.fit.bounds["u"]
    assert again.gaussianise.p2 == cfg.gaussianise.p2


def test_mardia_skewness_matches_the_gram_definition():
    """The O(n p^3) tensor form equals the literal O(n^2) definition."""
    rng = np.random.default_rng(3)
    W = rng.standard_normal((700, 4)) + 0.4 * rng.standard_exponential((700, 4))
    naive = float(((W @ W.T) ** 3).sum() / W.shape[0] ** 2)
    assert mardia_skewness(W) == pytest.approx(naive, rel=1e-12)


def test_mardia_skewness_scales_to_the_paper_sample_size():
    """30 000 points must not build a 30 000 x 30 000 matrix."""
    W = np.random.default_rng(4).standard_normal((30_000, 4))
    b1 = mardia_skewness(W)
    # for a Gaussian, n b1 / 6 ~ chi^2 with p(p+1)(p+2)/6 = 20 dof -> b1 ~ 4e-3
    assert 0.0 < b1 < 0.02


def test_ln1pz_grid_matches_rufian(cfg):
    """100 points, uniform in ln(1+z), from z = 0 to z = 10 (Binning._get_index_to_compare)."""
    z = cfg.grid.z_nodes()
    assert cfg.grid.spacing == "ln1pz" and z.size == 100
    assert z[0] == 0.0 and z[-1] == pytest.approx(10.0, rel=1e-12)
    assert np.allclose(np.diff(np.log1p(z)), np.log(11.0) / 99)


def test_residual_is_zero_where_both_sides_vanish():
    """D_A(z = 0) = 0 exactly on both sides: residual 0, as with RUFIAN's 1e-100."""
    from ssprior.observables import rel_residual
    r = rel_residual(np.array([0.0, 1.001]), np.array([0.0, 1.0]), 1e-3)
    assert r[0] == 0.0 and r[1] == pytest.approx(1.0, rel=1e-9)


def test_baryon_fraction_convention(cfg):
    """Omega_b fixed (CLASS default) -> ombh2 scales as h^2."""
    c = cfg.cosmology
    assert c.Omega_b == pytest.approx(0.022032 / 0.67556 ** 2, rel=1e-4)
    assert c.ombh2_of(80.0) / c.ombh2_of(60.0) == pytest.approx((80 / 60) ** 2)
    lag = Lagrangian(c01=-25.0, c02=40.0, d02=-60.0)
    for psi0, cos in slice_candidates(lag, 73.6, cfg):
        assert cos.ombh2 == pytest.approx(c.Omega_b * 0.736 ** 2)
        assert cos.Omega_cdm == pytest.approx(cos.Omega_m - c.Omega_b)


def test_refit_refuses_a_different_grid(cfg):
    """Stored observables cannot be refitted on another redshift grid."""
    from dataclasses import replace
    from ssprior.pipeline import run_refit
    from ssprior.store import Dataset
    ds = Dataset(np.zeros((1, 4)), np.zeros((1, 17)), np.zeros((1, 4)),
                 np.zeros((1, 10)), np.zeros((1, 301)), np.geomspace(0.01, 10, 100),
                 cfg.to_dict(), {}, "x")
    with pytest.raises(ValueError):
        run_refit(ds, cfg, workers=1, progress=False)


def test_eftcamb_path_is_never_hard_coded(monkeypatch):
    """build_path comes from the YAML (with $VARS/~ expanded) or EFTCAMB_PATH, else ''."""
    from ssprior.config import PriorConfig
    base = PriorConfig.from_yaml(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "shift_symmetric_lambda0.yaml")).to_dict()
    monkeypatch.delenv("EFTCAMB_PATH", raising=False)
    base["eftcamb"]["build_path"] = "${EFTCAMB_PATH}"
    assert PriorConfig.from_dict(base).eftcamb.build_path == ""
    monkeypatch.setenv("EFTCAMB_PATH", "/opt/eftcamb")
    assert PriorConfig.from_dict(base).eftcamb.build_path == "/opt/eftcamb"
    base["eftcamb"]["build_path"] = "/somewhere/else"
    assert PriorConfig.from_dict(base).eftcamb.build_path == "/somewhere/else"
