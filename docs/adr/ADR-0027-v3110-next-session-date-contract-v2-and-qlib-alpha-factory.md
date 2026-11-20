# ADR-0027: v3.11.0 Calendar-Aligned `next_session_date` Pre-Wall Lake, `sota.strategy-contract.v2`, `ADR-0059` TCA Cost Calibration & Trial `#104` Qlib `Alpha158` Benchmark

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Pre-Wall Institutional Alpha Factory (`S1`/`S2`, `PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`)
- **Evidence Grade:** E4
- **GitHub Evidence:** https://github.com/microsoft/qlib/commit/da920b7f954f48ab1bb64117c976710de198373e, https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/native-agent-stack/sota-upstream-shelf-lock
- **Revisit Trigger:** Any future pre-registered `Trial #105+` clearing `HLZ t*(N)`, `DSR > 0.95`, `PBO < 0.25`, `CPCV >= 70%`, and positive `2x` square-root impact net expectancy on `<= 2026-10-15`.

## Context and Problem Statement

Prior `materialize_prewall_df10_df6.py` derived forward targets via per-symbol `LEAD(split_open, 1) OVER (PARTITION BY symbol ORDER BY trade_date)` without an explicit `XNYS` calendar-aligned `next_session_date` join. Whenever a symbol missed 1 or more sessions (halt/illiquid gap), `LEAD(..., 1)` jumped across multiple trading dates, inducing multi-day span overlap across `CombinatorialPurgedCV` fold boundaries. In addition, Paper `P1-P4` fill attribution needed one unified `sota.strategy-contract.v2` ("Write Once, Run Everywhere") contract with `sota-<strategy_slug>-<lane>-<seq>` broker client IDs feeding `ADR-0059` TCA back into `S1`/`S2` cost calibration, plus an honest `microsoft/qlib` `Alpha158` + `LightGBM` pre-wall (`<= 2026-10-15`) benchmark.

## Decision Outcome

1. **Calendar-Aligned `next_session_date` Pre-Join (`materialize_prewall_df10_df6.py` & `run_v3110_alpha_factory_s1_s2.py`):**
   - Pre-join official `XNYS` calendar `next_session_date = LEAD(trade_date, 1)` (strictly `<= 2026-10-15`) and join `next1_split_*` + `next1_split_volume` strictly on `(symbol, next_session_date)` directly from canonical deduplicated `universe/{2024,2025,2026}/daily_split.csv.gz` (fail closed if any yearly split CSV is missing — zero raw-OHLC fallback; unknown non-XNYS dates fail closed).
   - Materialized `derived/df10_daily_20d_baselines_next_session.parquet` on `Noesis-Carbon` ext4 (`7,039,270` rows, `7,024,987` aligned target rows, `680` distinct sessions, `679` verified 1-step pairs, `0` mismatches, `0` post-wall breaches, SHA-256 `793ecc1e9bab14f8f704e25552f6e4f9e02974d1d2deb0849878ac2cf9f56cc2`).
2. **Standardize `sota.strategy-contract.v2` & `ADR-0059` TCA Cost Feedback (`src/sota_compounding/strategy_contract_v2.py`):**
   - Unified `build_client_order_id_v2` / `parse_client_order_id_v2` (`sota-<slug>-<lane>-<seq>`, `<= 48` chars, integer `seq <= 999_999_999_999`, full cross-repo slug parity incl. `boats_obb_stage`), `QUARANTINED_PAPER_STRATEGIES`, `validate_next_session_date_alignment`, and Empirical-Bayes (`kappa=20`) conservative `calibrate_sim_cost_from_paper_tca` (ratchets sim cost assumptions upward from realized Paper slippage/adverse selection; falls back transparently to `PRIOR_CONSERVATIVE_FLOOR` `4.0 bps` when `0` empirical fills supplied).
3. **Honest E4 Pre-Registered Trial `#104` (`H108_QLIB_ALPHA158_LGBM_NEXT_SESSION_PREWALL_CPCV`, spec SHA-256 `e0b6d6b3091b30eb2d0b38b221a26a0978fc69edf00f47be1e1e0e41fc115505`):**
   - `ADIA Lab #19` power gate approved (`680` dates, `t*(104)=4.2063`, `SR_min=2.5606 <= 5.0`, `MinTRL=496` days, `MDE=13.711 bps`) and wired fail-closed directly into `evaluate_strict_adr0027_gate` (strict `DSR > 0.95`, `PBO < 0.25`, fail-closed parsed non-empty SEC 90d dilution join).
   - `S1` **18-feature `Qlib Alpha158`-inspired OHLCV+Liquidity `LightGBM` subset** (`microsoft/qlib` pinned for full 158-expression expansion) evaluated as a retrospective `C(6,2)=15` `skfolio` `CombinatorialPurgedCV(purged_size=20, embargo_size=20)` distribution diagnostic (matching the 20-session DF10 rolling lookback) with strictly frozen `T-1` universe (`1,318,701` liquid 90d dilution-shielded rows, `263,146` OOS events / `679` dates, `145` missing/halted/0-volume next-session entries explicitly scored `0.0` unfilled, exact `2.00%` stop-touch boundary): `1x = -19.24 bps` (`t=-12.02`), `2x = -44.52 bps` (`t=-27.78`), `DSR=0.0000`, `CPCV_pos=0.00`, true **Bailey et al. (2014) symmetric `C(6,3)=20` CSCV `PBO=0.60`** over the 5-config OOS matrix -> **`FAILED`** (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`, zero Paper capital allocation).
   - `S2` real **`hftbacktest 2.4.2` `RiskAdverseQueueModel`** passive-GTX replay executed over date- and timestamp-verified pre-wall (`<= 2026-10-15`, `700` date dirs, `0` post-wall, `35,950` raw SIP Parquet files; `QQQ` `2026-10-15`, `951` bars, `3,804` events, `40` steps executed, `2/11` passive queue fills -> `18.18%` fill rate confirming passive fills are non-trivial and cannot be assumed free) + calibrated `4.0 bps` 1x / `8.0 bps` 2x floor propagated into `S1` (`receipt_sha256: 5e073f988d9c545090b8a133416b99ae7c43b34d5284b0048eb20e892e9da5a0`).

## Alternatives Considered

- **Keep unaligned `LEAD(split_open, 1)` without `next_session_date`:** Rejected; silently bridges missing sessions and violates CPCV temporal isolation.
- **Promote Qlib `Alpha158` benchmark based on gross returns without `2x` impact + borrow:** Rejected; fails HLZ `t*(104)` and `DSR > 0.95` decisively under realistic costs.

## Verification Witness

- Executed live on `Noesis-Carbon` WSL2 ext4 pre-wall lake (`<= 2026-10-15`); verified via `tests/test_adr0027_strategy_contract_v2_and_next_session.py`, full `pytest`, `ruff`, `scripts/architecture_catalog.py --check`, `scripts/org_visibility_doctor.py --offline`, and `gitleaks 0`.

## Rollback

Revert this commit and remove the derived `df10_daily_20d_baselines_next_session.parquet` artifact; existing `df10_daily_20d_baselines.parquet` and `ADR-0026` remain intact.
