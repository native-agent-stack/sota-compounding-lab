# ADR-0029: Advanced Sim-to-Merit Convergence (`Hansen SPA`/`MCS`, `Conformal PID` Stops, `Fixed-Share Hedge` + Hierarchical EB Pooling & `H109..H114` Blueprints)

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Stage-2 Pre-Wall Statistical Gate, Stage-3 Conformal Risk Calibration & Stage-4 Merit Farm Allocation (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`)
- **Evidence Grade:** E4 (`src/sota_compounding/advanced_merit_sim_convergence.py`, `tests/test_adr0029_advanced_merit_sim_convergence.py` incl. `80` Hypothesis property proofs, cross-referenced with `sota-upstream-shelf-lock ADR-0020`, `ruff`, `ty`, `pytest`, `org_visibility_doctor.py`, `gitleaks 0`)
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/native-agent-stack/sota-upstream-shelf-lock, https://github.com/bashtage/arch/commit/d3e858fc1c99932bb6936cbaf0d1b45b136266e5, https://github.com/scikit-learn-contrib/MAPIE/commit/e4c4f5f7daeaa7c8b09323852ad1997e506891a2
- **Revisit Trigger:** Pre-registration of `Trial #105` (`t*(105) = 4.2084`) on `H109..H113` pre-wall (`<= 2026-10-15`) or `2026-12-06` `21:00 ET` `23/5` Consolidated SIP cutover.

## Context and Problem Statement

While `ADR-0003`..`ADR-0028` established `HLZ t*(N)`, `skfolio` `CPCV`, `ml4t-diagnostic` `DSR`/`PBO`/`MinTRL`/`RAS`, and scalar Empirical-Bayes + stationary `UCB1` weighting, four quantitative gaps remained in the `Sim -> Pre-Wall Backtest -> Merit -> Paper Performance Manifest` loop:

1. **Missing `Hansen SPA` & `Model Confidence Set (MCS)`:** White's Reality Check loses power when dominated strategies are included, and single-benchmark tests do not return the benchmark-free surviving roster `M*` (`Hansen 2005 JBES`; `Hansen, Lunde & Nason 2011 Econometrica`).
2. **Parametric / Static Stop & Drawdown Thresholds:** Fixed quantile/ATR stops break under non-exchangeable volatility jumps (`Angelopoulos, Candès & Tibshirani 2023 NeurIPS`; `Gibbs & Candès 2021/2024`).
3. **Stationary `UCB1` Regret & Single-Strategy Outlier Over-Allocation:** Stationary `UCB1` lags regime flips and ignores full-information shadow-marked daily paper returns (`Freund & Schapire 1997`), while scalar `kappa` shrinkage ignores economic cluster structure (`Jensen, Kelly & Pedersen 2023 JF`).
4. **Next-Wave Primary-Literature Strategy Coverage:** The pre-wall factory needed pre-registration-ready blueprints spanning 2021–2026 institutional research (`Overnight/BOATS Bifurcation`, `Residual Directed Lead-Lag + EOD LETF Flow`, `Zero-New-Feed EDGAR Form 4 / XML 13D / Lazy Prices / ATM Overhang`, `Triple-Barrier Meta-Labeling`, and future `Databento DBN` L2/L3 Integrated OFI + NOII).

## Decision Outcome

1. **Ship `src/sota_compounding/advanced_merit_sim_convergence.py`:**
   - `evaluate_hansen_spa_and_mcs()`: Politis–Romano (1994) stationary block bootstrap with Bartlett HAC variance, Hansen (2005) consistent/lower/upper SPA `p`-values, and Hansen–Lunde–Nason (2011) `MCS` surviving set `M*`.
   - `ConformalPidStopCalibrator`: Online Conformal PID / ACI intrabar MAE quantile controller strictly clamped inside `ADR-0022 <= 2.00%` (`TP OFF`).
   - `update_fixed_share_hedge_weights()` & `compute_hierarchical_eb_merit_pool()`: Full-information Fixed-Share Hedge/MWU non-stationary strategy allocator + Jensen–Kelly–Pedersen (2023) Hierarchical Empirical-Bayes cluster shrinkage + `compute_asymmetric_vol_crash_brake()` (`<= 1.0x` ceiling honoring `Cederburg et al. 2020 JFE`).
   - `NEXT_WAVE_STRATEGY_BLUEPRINTS` (`H109..H114`) & `build_advanced_sim_merit_convergence_manifest()`: SHA-256 sealed governance manifest strictly locked to `<= 2026-10-15`, `PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`, `hlz_trial_consumed = False` (`Trial #105` `t*(105) = 4.2084` preserved).

## Alternatives Considered

- **Add heavy runtime dependencies (`arch`, `MAPIE`, `numpyro`, `JAX`) directly into `sota-compounding-lab`'s core `pyproject.toml` immediately:** Rejected — keeps core zero-bloat while implementing exact deterministic math in stdlib Python and recording the 18 verified optional research libraries in `sota-upstream-shelf-lock ADR-0020`.
- **Keep stationary `UCB1` and allow `> 1.0x` vol-timing leverage:** Rejected — `Cederburg et al. (2020 JFE)` show OOS leverage scaling degrades Sharpe under structural breaks, and full-information `Fixed-Share Hedge` strictly dominates partial-feedback `UCB1` when all strategies are shadow-marked daily.

## Verification Witness

- `uv run pytest -q` (full suite incl. `tests/test_adr0029_advanced_merit_sim_convergence.py` with `80` Hypothesis property proofs `PASS`).
- `uv run ruff check . && uv run ruff format --check .` (`PASS`), `scripts/architecture_catalog.py --check` (`29/29` ADRs, `0` drift), `scripts/org_visibility_doctor.py --offline` (`PASS`), and `gitleaks protect --staged --no-banner` (`0` leaks).
