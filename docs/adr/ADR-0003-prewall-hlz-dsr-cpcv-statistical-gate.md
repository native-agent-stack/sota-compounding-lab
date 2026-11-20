# ADR-0003: Pre-Wall (`<= 2026-10-15`), Harvey–Liu–Zhu `t*(N)` & Native-Frequency `DSR` Gate

- **Status:** Accepted
- **Date:** 2026-11-17
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `hlz_dynamic_t_hurdle`, `evaluate_full_lifecycle`

## Context

Across `80+` cumulative pre-registered trials (`HYPOTHESIS_LEDGER.md`), conventional `t >= 1.96` thresholds and annualized-Sharpe unit mismatches produce false discoveries.

## Decision

1. Reject any backtest sample with `sample_max_date > 2026-10-15` (`Study-Wall Violation`).
2. Scale the required date-clustered `t`-statistic monotonically with cumulative org trials `N` via Harvey, Liu & Zhu (2016) (`t*(1) = 3.000`, `t*(80) = 4.1466`).
3. Require `dsr_probability >= 0.95` computed at native return frequency alongside `>= 5` recombined `CombinatorialPurgedCV` paths (`cpcv_pbo <= 0.30`) and positive net expectancy under `2x` transaction + Almgren–Chriss square-root impact costs.

## Verification Witness

- **Cataloged Implementation & Test Artifacts:** `hlz_dynamic_t_hurdle`, `evaluate_full_lifecycle`
- **Deterministic Gates:** Verified by `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py` (`UPSTREAM.md` + `upstream-map.toml` + `18` shelf pins), and `pytest -q`.
