# ADR-0038: Tier-0 Full-Resolution, Always-Full-Resolution, Zero-Residual, Zero-Fallback Backbone Rule

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Full-Resolution 9-Engine Quantitative Backbone, `:8794` Strict Full-Resolution Sidecar, & Fail-Closed Merge Gate (`src/sota_compounding/upstream_lib_strategy_e2e.py`, `src/sota_compounding/gpu_scoring_sidecar.py`, `scripts/run_upstream_lib_e2e_and_live_routing.py`, `scripts/merge_gate.py`, `docs/SOTA_UPSTREAM_LIB_E2E_REGIME_PERSYMBOL_MANIFEST.{json,md}`, `tests/test_adr0038_full_resolution_zero_fallback_gate.py`), compounding upon `ADR-0035`, `ADR-0036`, and `ADR-0037` (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`, `hlz_trial_consumed = False`)
- **Evidence Grade:** E4 (Authentic 9/9 full-resolution pre-wall `<= 2026-10-15` execution sealed at SHA-256 `fcd06e224d68de521fcf6e3fc4f593a53a76d0bce311ec892b120d47aa31ccc8`: `nautilus_trader v2.0.0rc9` Rust `BacktestEngine` real 15-Equity `Strategy` replay `10,500` bars / `10,500` iterations / `654` orders / `590` fills / `295` positions replacing prior `iterations=0` boot stub; `hftbacktest v2.4.2` Rust `HashMapMarketDepthBacktest` + `RiskAdverseQueueModel` over `12` real pre-wall L1 quote+trade Parquet slices / `1,138,548` events / `11/12` fills / `-48.37 bps` honest adverse-selection markout replacing pure-Python `hftbacktest_queue_parity_ref`; `RTX 4090` `cuda:0` `10,000`-rep block bootstrap in `332.31 ms` with `require_gpu=True`; `polars 2.1.0` + `duckdb 2.0.0` + `pandera 0.35.0` + `exchange_calendars 4.13.2` `7,836,606` 1m bars / `700` XNYS sessions / `0` post-wall bytes; `vectorbt 1.2.0` + `statsmodels 0.15.0` + `skfolio 1.12.7` + `ml4t-diagnostic 0.2.0` + `arch 9.0.0`; `pytest` + Hypothesis red-team suite, `ruff`, `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py`, `gitleaks 0`)
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/nautechsystems/nautilus_trader/commit/da6b57aefdd62b44f675e434ffcfefca7f4025dd, https://github.com/nkaz001/hftbacktest/commit/5f3ec40b2afb764e0fea112f941ed85523ef4e88, https://github.com/polakowo/vectorbt/commit/a52a984b631624bb9daf41cb19f532fe559cf0ef, https://github.com/NVIDIA/cudf/commit/ece1ff92c8e5a61ac87f6cc43ef5cf763659377d
- **Revisit Trigger:** Addition of a newly pinned full-resolution engine tier (e.g., real-data `Qlib` cross-sectional factor pipeline or `Databento` MBO L3 replay) to `REQUIRED_FULL_RESOLUTION_ENGINES`.

## Context and Problem Statement

A forensic audit on `2026-11-20` revealed three residual shortcuts inside the `ADR-0035` E2E runner and manifest that violated the owner's Tier-0 Full-Resolution mandate (**full resolution and always full resolution — zero residual stubs, zero silent fallbacks**):
1. `nautilus_trader` booted an empty `BacktestEngine` with `Venue("XNAS")`, zero instruments, zero bars, zero strategies, and logged `"executed": true` with `iterations=0`.
2. `hftbacktest` was represented by a pure-Python OHLC loop (`hftbacktest_queue_parity_ref`) instead of invoking the pinned Rust `hftbacktest v2.4.2` `HashMapMarketDepthBacktest` + `RiskAdverseQueueModel` over real pre-wall L1 quote+trade tick slices.
3. RTX 4090 GPU scoring/bootstrap endpoints lacked a fail-closed `require_gpu=True` mode and strict manifest endpoint (`/api/full_resolution/manifest`) rejecting degraded or incomplete manifests.

## Decision Outcome

1. **Permanent Tier-0 Full-Resolution Gate (`enforce_full_resolution_zero_fallback_gate` & `audit_python_source_for_full_resolution`):**
   - Permanently bans `FORBIDDEN_RESIDUAL_OR_FALLBACK_RUN_KEYS` (`hftbacktest_queue_parity_ref`, `cpu_fallback`, `python_bar_loop_fallback`, `synthetic_smoke`, `nautilus_boot_only`), `iterations <= 0`, `fallback_used=True`, `synthetic=True`, `degraded=True`, non-Rust `hftbacktest`, non-CUDA `rtx4090_gpu_cuda`, and any `post_wall_n > 0`.
   - Enforces AST-level detection of empty `BacktestEngine.run()` stubs (including inline subprocess scripts) and sealed-manifest verification directly inside `scripts/merge_gate.py`.
2. **Real 9/9 Full-Resolution Execution (`scripts/run_upstream_lib_e2e_and_live_routing.py`):**
   - Runs a real `NautilusTrader v2.0.0rc9` `SotaBreakoutParityStrategy` across `15` pre-wall `Equity` instruments (`10,500` bars, `10,500` iterations, `654` orders, `590` fills, `295` positions, `ADR-0022 <= 2.00%` stops).
   - Runs real Rust `hftbacktest v2.4.2` `HashMapMarketDepthBacktest` + `RiskAdverseQueueModel` in `Noesis-Carbon` over `12` real pre-wall paired L1 quote+trade Parquet slices (`1,138,548` events).
   - Executes real RTX 4090 `cuda:0` `10,000`-rep circular block bootstrap via `:8794` with `require_gpu=True` and seals with `strict_full_resolution=True`.
3. **Strict Fail-Closed `:8794` Sidecar Endpoints (`src/sota_compounding/gpu_scoring_sidecar.py`):**
   - Adds `GET /api/full_resolution/status` and `GET /api/full_resolution/manifest` (returning HTTP 503 if SHA-256 or `ADR-0038` gate fails) plus `require_gpu=True` / `SOTA_REQUIRE_FULL_RESOLUTION_GPU=1` HTTP 503 refusal of silent CPU fallbacks.

## Alternatives Considered

1. **Keep Soft Warning Flags or Optional Fallbacks:** Rejected unconditionally by owner Tier-0 mandate ("full resolution and always full resolution, no residual or fallbacks").

## Consequences

- **Positive:** Eliminates `iterations=0` theater, pure-Python residual bar loops masquerading as `hftbacktest`, and silent CPU degradation at both runtime and `merge_gate.py`.
- **Negative / Safeguards:** Regenerating the E2E manifest requires all 9 real engines (including `Noesis-Carbon` Rust `hftbacktest v2.4.2`, `meridian` `NautilusTrader v2.0.0rc9`, and live `:8794` RTX 4090 CUDA) — failing loudly if any component is missing.

## Verification Witness

- **Real 9/9 Sealed Pre-Wall Receipt:** `docs/SOTA_UPSTREAM_LIB_E2E_REGIME_PERSYMBOL_MANIFEST.{json,md}` (`full_resolution_verified=True`, `residual_fallbacks_count=0`, `engines_verified=9/9`, SHA-256 `fcd06e224d68de521fcf6e3fc4f593a53a76d0bce311ec892b120d47aa31ccc8`).
- **Deterministic Gates:** `tests/test_adr0038_full_resolution_zero_fallback_gate.py` (including 80-example Hypothesis red-team fuzzing), full `pytest`, `ruff check`, `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py`, `scripts/org_visibility_doctor.py --offline`, and `gitleaks 0`.
