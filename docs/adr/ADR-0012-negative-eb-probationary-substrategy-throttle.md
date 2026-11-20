# ADR-0012: `PROBATION_NEGATIVE_EB_THROTTLE` (`0.15x`) for Persistent Negative-EB Sub-Strategy Bleeders

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/lifecycle.py`, `tests/test_compounding_lifecycle.py`

## Context

Live `1,018`-order RTH attribution showed `POST_VIX_REVERSAL_MAJOR_GROWTH_THRUST` (`12` closed round-trips, `8.3%` win rate, `-0.68R` realized, `-0.47R` EB) retaining `0.65x` sizing (`UCB = 0.89x`) under `TIGHTENED_NOISE_FILTER`, dragging portfolio Profit Factor to `0.46` despite positive sibling alpha leaders.

## Decision

1. Extend `tune_empirical_bayes_ucb` with `alloc_multiplier` and `PROBATION_NEGATIVE_EB_THROTTLE`: whenever `closed_roundtrips >= 10`, `eb_expectancy_r <= -0.25`, and `closed_mean_r < -0.25`, cap `ucb_priority_weight <= 0.45`, require `min_ofi_pct >= +25.0%`, and throttle `alloc_multiplier = 0.15` (`85%` capital bleed reduction while preserving never-gated micro-exploration).

## Verification

- `python scripts/architecture_catalog.py --check`, `ruff check .`, `pytest -q`, and `gitleaks` (`0` leaks).

## Verification Witness

- **Cataloged Implementation & Test Artifacts:** `src/sota_compounding/lifecycle.py`, `tests/test_compounding_lifecycle.py`
- **Deterministic Gates:** Verified by `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py` (`UPSTREAM.md` + `upstream-map.toml` + `18` shelf pins), and `pytest -q`.
