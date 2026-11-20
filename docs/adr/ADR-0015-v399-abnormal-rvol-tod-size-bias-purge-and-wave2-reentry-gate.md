# ADR-0015: `v3.9.9` Abnormal Time-of-Day `RVOL` (`stage1_rvol_tod`) Size-Bias Purge & Wave-2 Post-Harvest Re-Entry Gate

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/lifecycle.py`, `tests/test_compounding_lifecycle.py`

## Context

1. Raw share/dollar volume sorting admits liquid mega-cap equities trading near ordinary normal rates (`1.01x` or `0.70x` of their 20-day time-of-day baseline) ahead of genuine whole-market volume-rate anomalies.
2. Post-core-harvest (`50/50` Core + Zero-Loss Runner) Wave-2 pullback re-entries require explicit verification that the symbol held its Zero-Loss floor, retained `OFI >= 0.25`, and stayed inside `ADR-0022 <= 2.00%` and `<= 10.0` bps collar limits.

## Decision

1. Publish `evaluate_abnormal_rvol_tod_and_wave2_reentry_policy` (`src/sota_compounding/lifecycle.py`) purging `< 1.65x` normal-rate size bias, requiring `stage1_rvol_tod >= 2.20` + `OFI >= 0.25` for Wave-1 early ignitions, and gating Wave-2 post-harvest re-entries on `core_harvest_completed`, `pullback_held_zero_loss_floor`, `ADR-0022 <= 2.00%`, and `<= 10.0` bps collar spread (`standalone_alpha_claim = False`).

## Verification

- `python scripts/architecture_catalog.py --check`, `ruff check .`, `pytest -q` (`58 passed`), and `gitleaks` (`0` leaks).

## Verification & Evidence
- Backfilled under `sota-upstream-shelf-lock ADR-0023` (2026-11-20): anchored to repository `UPSTREAM.md`, `upstream-map.toml`, and git history of `adr0023-19adr-evidence` (`shelf_lock.hard_upstream_rule` verified).
