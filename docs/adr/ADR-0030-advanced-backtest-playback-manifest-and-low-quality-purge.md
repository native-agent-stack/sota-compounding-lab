# ADR-0030: Advanced Backtest Playback Performance Manifest (`sota.backtest-playback-performance-manifest.v1`), Suitable Multi-Regime Time Periods & 14-Point Low-Quality Practice Purge

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Stage-1/2 Pre-Wall Playback Governance, Multi-Regime Horizon Calibration & Low-Quality Anti-Pattern Firewall (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`)
- **Evidence Grade:** E4 (`src/sota_compounding/backtest_playback_manifest.py`, `src/sota_compounding/converged_strategy_factory.py`, `tests/test_adr0030_backtest_playback_manifest.py` incl. `80` Hypothesis property proofs, cross-linked with `sota-upstream-shelf-lock ADR-0021` and `sota-mover-ranker ADR-0069`, `ruff`, `ty`, `pytest`, `org_visibility_doctor.py`, `gitleaks 0`)
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/native-agent-stack/sota-upstream-shelf-lock, https://github.com/native-agent-stack/sota-mover-ranker, https://github.com/bashtage/arch/commit/d3e858fc1c99932bb6936cbaf0d1b45b136266e5, https://github.com/amazon-science/chronos-forecasting/commit/1224fafa91e1ecd6674f5204066b159e40b5e4a1
- **Revisit Trigger:** Execution of `Trial #105` (`t*(105) = 4.2084`) on `W1_PRIMARY_PREWALL_LAKE` (`2024-01-02 .. 2026-10-15`, `679` sessions) or `2026-12-06` `21:00 ET` `23/5` SIP cutover.

## Context and Problem Statement

A comprehensive ecosystem audit (`2026-11-20`) identified legacy low-quality backtest practices that risk silent false discoveries: toy `30d` (`~21`-session) intraday windows far below `ADIA Lab #19` `MinTRL`, ex-post live-mover survivorship universes, flat `4 bps` RT slippage without `2x` + square-root impact stress, hard Take-Profit caps truncating right tails, legacy `3.5%–4.0%` daily stops violating `ADR-0022 <= 2.00%`, trade-count Sharpe scaling, stationary `UCB1` on full-information manifests, `> 1.0x` vol-timing leverage (`Cederburg et al. 2020 JFE`), non-commercial `timesfm v3.0` weight references, and banned copyleft/scraper packages (`yfinance`, `backtrader`, `backtesting.py`, `cvxportfolio`, `ArcticDB`, `pybroker`, `mlfinlab`).

## Decision Outcome

1. **Ship `src/sota_compounding/backtest_playback_manifest.py` (`sota.backtest-playback-performance-manifest.v1`):**
   - **`SUITABLE_PLAYBACK_HORIZONS` (`W1..W5`) + `compute_min_trl_sessions()`:** Locks intraday/event playback to `W1_PRIMARY_PREWALL_LAKE` (`2024-01-02 .. 2026-10-15`, `679` sessions, `C(6,2)=15` CPCV `purge=20`/`embargo=20`), swing/daily stress to `W2_MULTI_REGIME_SWING_STRESS` (`2020-01-02 .. 2026-10-15`, `1,707` sessions spanning COVID, 2021 euphoria, 2022 bear, 2023–2026 high-rate dispersion), `88 FR 76896` XML 13D / BOATS to `W3_MODERN_XML_13D_BOATS` (`2024-02-05 .. 2026-10-15`, `655` sessions), post-cutover shadow to `W4_POST_2026_12_06_23X5_SIP_SHADOW`, and live paper parity to `W5_LIVE_PAPER_PARITY_REPLAY`.
   - **14-Point Fail-Closed `LOW_QUALITY_PRACTICE_PURGE_REGISTRY` (`audit_low_quality_practices()`):** Mechanically rejects every single low-quality backtest anti-pattern and banned non-permissive/scraper library.
   - **`build_advanced_backtest_playback_manifest()` & `render_playback_manifest_markdown()`:** Unifies calendar-day AR(1)-adjusted Sharpe, `1x`/`2x` net bps, `Hansen SPA`, `HLN MCS M*`, `Conformal PID` stops (`<= 2.00%`), `Fixed-Share Hedge`, `Hierarchical EB` cluster pooling, and `Cederburg-safe` (`<= 1.0x`) vol crash brake into a SHA-256 sealed dimensionless artifact (`hlz_trial_consumed = False`, `Trial #105` `t*(105) = 4.2084` preserved).
2. **Upgrade `converged_strategy_factory.py`:** Expand `RETIRED_TRAP_DENYLIST` (`LEGACY_SP500_ADDITION_POP_CHURN`, `VOL_TIMING_LEVERAGE_ABOVE_1X`, `EX_POST_LIVE_MOVER_SURVIVORSHIP_BACKTEST`, `TOY_HORIZON_BELOW_MIN_TRL`, `FIXED_TAKE_PROFIT_CAP_CUTTING_TAIL`), flag `timesfm` v3.0 non-commercial weight restriction, and register clean `Apache-2.0` `chronos2` + frontier adapters (`arch`, `linearmodels`, `databento_dbn`, `delta_rs`, `ducklake`, `mapie`, `river`, `numpyro`, `cvxpy_clarabel`).

## Alternatives Considered

- **Allow short 30-day diagnostic runs to pass merit promotion gates with a soft warning:** Rejected — violates `ADIA Lab #19` `MinTRL` power requirements and re-opens single-regime overfitting.

## Verification Witness

- `scripts/architecture_catalog.py --check` (`30/30` ADRs, `0` drift), `pytest -q` (full suite incl. `tests/test_adr0030_backtest_playback_manifest.py` with `80` Hypothesis proofs `PASS`), `ruff check .`, `ty check`, `scripts/org_visibility_doctor.py --offline` (`PASS`), and `gitleaks protect --staged --no-banner` (`0` leaks).
