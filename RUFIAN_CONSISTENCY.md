# Consistency with RUFIAN, the code of Traykova et al. (2021)

> Line-by-line comparison of `ssprior` with `horndeski-priors` (RUFIAN, commit `0634b015`): where the two agree, how `ssprior` was aligned, how much each choice matters, and what cannot be compared from that archive.

![language](https://img.shields.io/badge/language-Python%203.12-3776ab)
![part of](https://img.shields.io/badge/part%20of-Theoretical__priors-blue)

Traykova et al. write that their fits were made with *"a modified version of RUFIAN"* (`gitlab.com/dinatraykova/horndeski-priors`). The archive of that repository's `master` branch (commit `0634b015a5d7`, files dated 4 May 2020) contains RUFIAN as used for the thawing-quintessence priors of García-García et al.: the models in `computation.py` are `quintessence_monomial`, `_modulus`, `_axion`, `_eft` and `_tracker`, and **no shift-symmetric model**. The strings `c01`, `d02`, `Horndeski` that `grep` finds in the notebooks sit inside base64-encoded PNG outputs, not in code. The shift-symmetric modifications (the Lagrangian, the $\alpha_B$ parametrisation, the Gaussianisation of Eq. 23) are therefore not in the archive. What *is* in it is the machinery the modified version is built on: the cosmological priors, the redshift grid, the $\chi^2$ and the treatment of failed and inaccurate fits. Those are compared here.

## Table of Contents
- [Summary](#summary)
- [Point-by-point comparison](#point-by-point-comparison)
- [Differences that remain, and why](#differences-that-remain-and-why)
- [How much each choice matters](#how-much-each-choice-matters)
- [References](#references)

## Summary

| Item | RUFIAN | `ssprior` | Status |
|---|---|---|---|
| $h$ prior | $\mathcal U(0.6,0.8)$ | $H_0\in\mathcal U(60,80)$ | same |
| $\Omega_{\rm cdm}$ | $\mathcal U(0.15,0.35)$, sampled | derived, window $[0.15,0.35]$ | same measure (see below) |
| baryons | not set → CLASS default $\Omega_b=0.048275$ fixed | $\Omega_b=0.048275$ fixed | aligned |
| radiation | CLASS default: $T_{\rm cmb}=2.7255$, $N_{\rm ur}=3.046$, no massive $\nu$ | same | same |
| redshift points | 100 rows of the CLASS table, $0\le z\le10$ → uniform in $\ln(1+z)$, includes $z=0$ | uniform in $\ln(1+z)$, $0\le z\le10$, 100 points | aligned |
| residual | $(O_{\rm th}-O_{\rm fit})/(O_{\rm th}\sigma)$, zeros → $10^{-100}$ | same form, zeros handled the same way | aligned |
| weights | $\sigma=10^{-3}$, $\sigma_{\rm rec}=10^{-4}$ | same | same |
| models entering the prior | every model whose fit ran; accuracy only reported | every fitted model (`--accurate-only` adds a cut) | aligned |
| failed models | skipped, a new one drawn | same | same |
| $f(z)$ | CLASS (`work_in_class.cosmo_extra.growthrate_at_z`) | quasi-static ODE, Eq. (20) | not comparable, see below |
| $z_{\rm rec}$, $D_A(z_{\rm rec})$ | CLASS thermodynamics | Hu–Sugiyama $z_*$ | not comparable, see below |
| optimiser | `wicm.fit` from a fit to the $w$ curve | cascade + multistart `least_squares` | same objective |

## Point-by-point comparison

### Cosmological priors — same

`computation.py:31-34` draws, for every model, `h ~ U(0.6, 0.8)` and `Omega_cdm ~ U(0.15, 0.35)`; nothing else. `ssprior` uses $H_0\in[60,80]$ and accepts a model when the derived $\Omega_{\rm cdm}$ lies in $[0.15,0.35]$. These are the same measure: sampling $\Omega_{\rm cdm}$ uniformly on $R$ and keeping $|\sum_i\Omega_i-1|<\epsilon$ gives

```math
p(\theta\mid\text{keep})\propto\int_R d\Omega_{\rm cdm}\,\mathbb 1\big[|\Omega_{\rm cdm}-\Omega^*_{\rm cdm}(\theta)|<\epsilon\big]=2\epsilon\,\mathbb 1_R\big(\Omega^*_{\rm cdm}(\theta)\big),
```

because $\sum_i\Omega_i-1$ is linear in $\Omega_{\rm cdm}$ with unit slope ([`SLICE_AND_PRIOR_BOX.md`](SLICE_AND_PRIOR_BOX.md), Step 6). The archive's RUFIAN still lets hi_class tune $V_0$ (`Omega_smg: -1`, `computation.py:25-28`); the paper's footnote 7 says the shift-symmetric runs switched that off (`Omega_smg_debug`), which is the slice.

### Baryons — aligned

RUFIAN passes only `h` and `Omega_cdm` to hi_class (`binning.py:169-186`, `computation.py:31-34`). CLASS 2.x sets its defaults in `input.c` *before* reading the user's `h`, with `Omega0_b = 0.022032/0.67556²`, and only overwrites it if `Omega_b` or `omega_b` is given. Hence in RUFIAN the baryon **fraction** is fixed:

```math
\Omega_b=\frac{0.022032}{0.67556^2}=0.048275,\qquad \omega_b=\Omega_b\,h^2\in[0.0174,\,0.0309].
```

`ssprior` does the same through `cosmology.Omega_b` ([`config.py:116-120`](ssprior/config.py#L116-L120), `ombh2_of`). The window on $\Omega_\phi$ is then

```math
\Omega_\phi\in\big[\,1-\Omega_r-\Omega_b-0.35,\ 1-\Omega_r-\Omega_b-0.15\,\big]=[0.6016,\,0.8016]\ \text{to}\ [0.6017,\,0.8017]\quad(H_0=60\to80),
```

almost independent of $H_0$. With $\omega_b$ held fixed instead, the window would drift from $[0.588,0.788]$ to $[0.615,0.815]$ across the same $H_0$ range.

> [!NOTE]
> The value is an inference from CLASS's defaults, not something written in RUFIAN. CLASS ≥ 2.9 uses `0.02238280/0.67810²` = 0.048678 instead (+0.8%); set `cosmology.Omega_b` accordingly if the hi_class build of the paper is known to be newer.

### Redshift points — aligned

`binning.py:638-645`:

```python
index_to_compare = (np.linspace(len(z) - sum(z <= self.z_low_redshift_fit),
                                len(z)-1, self.points_to_fit) + 0.5).astype(int)
```

takes `points_to_fit` equally spaced **rows** of the CLASS background table between the first row with $z\le z_{\rm low}$ and the last row ($z=0$). CLASS 2.x fills that table with a step $\Delta\tau=\epsilon/(aH)$, i.e. $\Delta\ln a=aH\,\Delta\tau=\epsilon$ constant (`back_integration_stepsize` = $7\times10^{-3}$, about 340 rows below $z=10$). Equally spaced rows are therefore equally spaced in $\ln a=-\ln(1+z)$:

```math
z_k=\exp\!\Big(\frac{k}{99}\ln 11\Big)-1,\qquad k=0,\dots,99 .
```

The paper's "100 points at $z<10$" is `points_to_fit = 100`, `z_low_redshift_fit = 10`. `ssprior` builds the same nodes with `grid.spacing: ln1pz` ([`config.py:171`](ssprior/config.py#L171), `z_nodes`). The grid sets the weights of Eq. (21), so its shape matters, not only its resolution:

| grid | points below $z=1$ | points below $z=0.1$ | median node |
|---|---|---|---|
| RUFIAN / `ssprior`, uniform in $\ln(1+z)$ | 29 | 4 | $z=2.32$ |
| log-spaced in $z$ from 0.01 (for comparison) | 66 | 33 | $z=0.32$ |

A log-spaced grid would give two thirds of the $\chi^2$ to $z<1$; RUFIAN's spreads it evenly in e-folds.

### Residuals and zeros — aligned

`binning.py:292-298, 375, 385` build the observable vector `[H, D_A, f, D_A(z_rec)]`, set every exact zero to `1e-100`, and minimise

```math
\chi^2=\sum_k\left(\frac{O^{\rm th}_k-O^{\rm fit}_k}{O^{\rm th}_k\,\sigma_k}\right)^2,\qquad \sigma_k=10^{-3}\ (z\le10),\quad \sigma_{\rm rec}=10^{-4}.
```

`ssprior` uses the same residual. It works with $E=H/H_0$ instead of $H$ and with the comoving distance instead of $D_A$, but both ratios cancel in a relative residual, because the exact and the fitted model share $H_0$ and $z$. The grid contains $z=0$, where $D_A=0$ on both sides: `rel_residual` ([`observables.py:182`](ssprior/observables.py#L182)) applies RUFIAN's replacement, so the residual is exactly 0 there, and `_comoving_distance` ([`observables.py:107`](ssprior/observables.py#L107)) returns an exact 0 at $z=0$ instead of the spline's $\sim10^{-15}$ round-off, which would otherwise sit in the denominator.

### Which models enter the prior — aligned

RUFIAN skips a model only when hi_class or the fit raises, and draws another in its place (`binning.py:833-862`). The maximum relative deviations are written to the `reldev` files and summarised (Fig. 7 of the paper); `analyze_coeffs.py` never removes a model on accuracy grounds (its `selection`, lines 554-573, selects PCA dimensions, not models). The paper says the same: the 30 000 fitted parameter sets *"can be used to build our theoretical priors"*. `ssprior gaussianise` and `validate` use every fitted model by default; `--accurate-only` restricts them to the fits that meet the 1% / 0.3% targets.

## Differences that remain, and why

- **$f(z)$.** RUFIAN takes $f$ from CLASS through `work_in_class.cosmo_extra.growthrate_at_z` (`common.py:323`), a package not included in the archive, with `output: mPk` and `z_max_pk` set "for relative errors in f" (`binning.py:62-66`). How the modified version computes $f$ for the shift-symmetric model is not visible; the paper writes the quasi-static Eq. (20), which is what `ssprior` integrates. The quasi-static $f$ agrees with EFTCAMB's $f\sigma_8/\sigma_8$ to 0.03%.
- **$z_{\rm rec}$.** RUFIAN reads `z_rec` and `da_rec` from CLASS's recombination (`common.py:273-275`). `ssprior` has no recombination code and uses the Hu–Sugiyama $z_*$, which differs from CLASS by $\sim2\times10^{-3}$. The exact and the fitted model are compared at the same $z_{\rm rec}$, so the effect on the fitted parameters is second order.
- **Where the parametrised model is evaluated.** RUFIAN's default (`fit_with_class=True`, `binning.py:305-322`) runs the fitted model through CLASS; `ssprior` evaluates it in Python on the same nodes as the exact model. The objective is the same.
- **Optimiser.** RUFIAN starts from a least-squares fit to the $w$ curve up to $z_{\rm rec}+200$ and runs one local fit (`binning.py:363-376`). `ssprior` runs a staged fit with several starts in $u=4/m$ and keeps the lowest $\chi^2$. Both minimise the same $\chi^2$; they can differ only for models with several local minima, where `ssprior` returns the lower one.
- **Fraction of accurate fits.** In the run, 96.5% of the fits meet the 1% / 0.3% targets (28 948 of 30 000), against the 99% quoted by the paper. In all 1052 fits that miss them the largest error is in $f$, at exactly $z=0$ in 1048 cases and below $z=0.1$ in the other 4; none misses the 0.3% target on $D_A(z_{\rm rec})$. They are a well-defined population: 99.9% of them have $\hat\alpha_B>2$ (median 2.50, against 1.21 for the accurate fits) and small exponents (median $m=0.87$, against 1.64). These are genuine minima of the $\chi^2$: the grid puts only 4 of its 100 points below $z=0.1$, where $f$ departs fastest from the parametrisation. The difference from 99% may come from how the modified RUFIAN computes $f$, which the archive does not show.

## How much each choice matters

**Grid and baryons.** Three runs of 800 models each on the same Sobol stream: A with a log-spaced grid from $z=0.01$ and $\omega_b=0.0224$ fixed, B with RUFIAN's grid and $\omega_b$ fixed, C with RUFIAN's grid and $\Omega_b$ fixed (the `ssprior` configuration). The grid does not enter the slice, so A and B accept exactly the same models and A → B is a paired comparison with very small errors.

| Change | $X_1$ | $X_2$ | $X_3$ | $X_4$ |
|---|---|---|---|---|
| log grid → RUFIAN grid, paired (A → B) | +0.0153 ± 0.0019 (+1.2%) | −0.0096 ± 0.0006 (−0.6%) | +0.0076 ± 0.0003 (+0.6%) | **−0.0418 ± 0.0005 (−5.7%)** |
| $\omega_b$ → $\Omega_b$ fixed, same models (B → C) | 0 | 0 | 0 | 0 |
| $\omega_b$ → $\Omega_b$ fixed, whole samples (B → C) | −0.002 ± 0.024 | −0.002 ± 0.014 | −0.000 ± 0.002 | −0.001 ± 0.007 |

- **The grid is the choice that matters.** It moves $X_4=w_am^2$ by −5.7% and $X_1$ by +1.2%. Every model moves, not only a few (median per-model shift of $X_4$: 0.042).
- **The baryon convention does not change any fit.** For a given $(c_{01},c_{02},d_{02},H_0)$, $\Omega_\phi$ comes from the Lagrangian and $\Omega_m=1-\Omega_r-\Omega_\phi$ is fixed by closure; only its split into $\Omega_b+\Omega_{\rm cdm}$ changes, and $E(z)$, $D_A$, $f$ depend on $\Omega_m$ alone. What changes is *which* models pass the $\Omega_{\rm cdm}$ window: 30 of 800 differ, and $\langle\Omega_\phi\rangle$ goes from 0.7030 to 0.7008. The effect on $\mu$ is below the noise of 800 models; through $\alpha_{B,0}\approx2\Omega_\phi$ it is expected at the −0.3% level.

**Which models enter the prior.** On the 30 000-model run, restricting the fit of the normal to the 28 948 accurate models would give

| | $\mu_{X_1}$ | $\mu_{X_2}$ | $\mu_{X_3}$ | $\mu_{X_4}$ | $\Sigma_{11}$ |
|---|---|---|---|---|---|
| every fitted model (RUFIAN, default) | 1.2921 | 1.6322 | −1.1763 | −0.7750 | 0.214 |
| accurate fits only (`--accurate-only`) | 1.2447 | 1.6561 | −1.1808 | −0.7766 | 0.153 |

so the choice moves $\mu_{X_1}$ by 3.8% and $\Sigma_{11}$ by 40%. The 3.5% of fits that miss the targets almost all have $\hat\alpha_B>2$ (median 2.50), in the tail of $X_1=\hat\alpha_B$, which is why they widen $\Sigma_{11}$ so much.

## References

- `horndeski-priors-master.zip`, commit `0634b015a5d7`: `computation.py`, `binning.py`, `common.py`, `analyze_coeffs.py`, `Glamdring-binning.sh`.
- D. Traykova et al., arXiv:2103.11195v3, Sec. IV–V and footnote 7.
- C. García-García, E. Bellini, P. G. Ferreira, D. Traykova, M. Zumalacárregui, arXiv:1911.02868 (RUFIAN's first use).
- D. Blas, J. Lesgourgues, T. Tram, *CLASS II*, arXiv:1104.2933 (background integration and default parameters).
