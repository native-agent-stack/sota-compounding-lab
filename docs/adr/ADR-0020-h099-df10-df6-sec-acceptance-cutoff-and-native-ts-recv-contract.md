# ADR-0020: Pre-Wall `H099` `DF10` 20-Session Same-Clock Baseline, `DF6` Dated RegSHO Threshold List, SEC `accepted_at` Wall Filter & Native `ts_recv_ns` Readiness Contract

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/prewall_h099_enrichment.py`, `tests/test_prewall_h099_enrichment.py`

## Context

`T3` (`ctx_ff6ccc3463b0`, `Codex gpt-6.2-sol`, peer receipt `#377` / Lesson `94c3e0cf6bbf9fde`) empirically audited all `29,499` non-empty partitions (`53,063,777` rows) of `Z:/sota-data/prewall-lake` under `ADR-0018` `E2_BOUNDED_LATENCY_500MS` (`<= 2026-10-15 10:45:00.495858 ET`) and proved three exact PIT prerequisites missing from `published-orb` event slices before `H099` (`SHA 42cc7cfe...`) can compute CPCV / `1x`/`2x` Almgren-Chriss / DSR:
1. **`DF10` Same-Clock Intraday Baselines:** `published-orb` (`orb_1m_bars` alias) stores `<= 5`-minute event slices only (`0 / 10,669` symbol-days have all 20 prior XNYS sessions; max `5 / 20`).
2. **`DF6` Dated FINRA RegSHO Threshold List:** Short-volume and short-interest files do not provide dated Rule 203(c)(6) threshold membership snapshots.
3. **SEC Acceptance-Timestamp Leakage:** `11` SEC `EFFECT` records had acceptance timestamps past the `2026-10-15` ET study wall.

## Decision

1. Ship `src/sota_compounding/prewall_h099_enrichment.py` enforcing fail-closed, zero-fabrication contracts:
   - `compute_df10_same_clock_baseline()` requires `>= 20` strictly prior XNYS sessions (`d < target_date <= 2026-10-15`) at the same ET clock bucket and rejects any look-ahead date leakage.
   - `validate_df6_regsho_threshold_snapshot()` requires dated daily RegSHO threshold snapshots with `published_ts_utc <= decision_ts_utc` and `trade_date <= 2026-10-15`.
   - `filter_sec_filings_pit_prewall()` purges and counts all SEC filings missing `accepted_at_utc` or with `America/New_York` date `> 2026-10-15` (closing the `11` post-wall `EFFECT` records).
   - `audit_h099_prewall_lake_readiness()` composes `ADR-0018` `ts_recv_ns` with `DF10`, `DF6`, and SEC cutoff checks into three deterministic SHA-256 tiers (`BLOCKED_MISSING_PIT_PREREQUISITES`, `E2_BOUNDED_500MS_H099_EXPLORATORY_READY`, `E4_NATIVE_H099_LIVE_PROMOTION_ELIGIBLE`).
2. Add deterministic & Hypothesis property tests in `tests/test_prewall_h099_enrichment.py`.

## Verification

- `python scripts/architecture_catalog.py --write && python scripts/architecture_catalog.py --check`, `ruff check .`, `ty check src/sota_compounding`, `pytest -q`, `python scripts/org_visibility_doctor.py`, and `gitleaks git --no-banner` (`0` leaks).
