# ADR-0019: `v3.10.6` `America/New_York` Overnight `BOATS` `decision_ns` Parity, `5 bps` Stop Floor Constant & Canonical `DF10..DF12` Feed Normalization

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/symbol_ranking_v310.py`, `tests/test_symbol_ranking_v310.py`

## Context

`ADR-0017` added UTC `decision_ns // 1_000_000_000` study-wall enforcement in `evaluate_df12_bitemporal_symbol_ranking_handoff()`, but US equities `BOATS` (`20:00–23:59 ET`) decisions on pre-wall `2026-10-15` cross into `2026-10-16` UTC (`00:00–03:59Z`), causing false `PREWALL_DECISION_NS_VIOLATION` rejections during overnight studies. Additionally, `DF10..DF12` suffixes used uppercase strings while sibling repos (`sota-multi-model-workflows`, `meridian`) used lowercase `DF10_own_history_...` conventions.

## Decision

1. Evaluate `decision_ns // 1_000_000_000` in `America/New_York` via cross-OS `_to_ny_datetime()` (IANA `ZoneInfo("America/New_York")` with exact US DST pure-stdlib fallback for Windows hosts without `tzdata`), matching `sota-multi-model-workflows` `ADR-0031`.
2. Export `MIN_HARD_STOP_PCT = 0.05` (`5.0 bps`) and `CANONICAL_12_RANKING_FEEDS_NORMALIZED` (preserving `CANONICAL_12_RANKING_FEEDS` backward compatibility).
3. Add `America/New_York` overnight rollover, `10**400` overflow, and `Hypothesis` property-based stop/IEEE-754 invariant tests in `tests/test_symbol_ranking_v310.py`.

## Verification

- `python scripts/architecture_catalog.py --write && python scripts/architecture_catalog.py --check`, `ruff check .`, `pytest -q`, and `gitleaks git --no-banner` (`0` leaks).

## Verification Witness

- **Cataloged Implementation & Test Artifacts:** `src/sota_compounding/symbol_ranking_v310.py`, `tests/test_symbol_ranking_v310.py`
- **Deterministic Gates:** Verified by `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py` (`UPSTREAM.md` + `upstream-map.toml` + `18` shelf pins), and `pytest -q`.
