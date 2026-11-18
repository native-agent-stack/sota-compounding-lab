# ADR-0017: `v3.10.3` IEEE-754 Finite Guards & UTC `decision_ns` Study-Wall Enforcement in `DF1..DF12` Handoff

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/symbol_ranking_v310.py`, `tests/test_symbol_ranking_v310.py`

## Context

An adversarial cross-repo E2E audit (`cc-lead-e2e-review` `#291`/`#292`) surfaced two fail-open edge cases in `evaluate_df12_bitemporal_symbol_ranking_handoff()` (`ADR-0016`):
1. **IEEE-754 `NaN` / `±Inf` bypass:** Direct `<` / `>` threshold checks silently evaluated to `False` for `float("nan")`, allowing non-finite feature values (`rvol_tod`, `vol_z_tod`, `ofi`, `dry_powder_sec`, `spread_collar_bps`, `hard_stop_pct`, `cooldown_elapsed_sec`) to slip through without violation.
2. **Unchecked `decision_ns` wall drift:** Epoch nanosecond timestamps (`>= 2000-01-01T00:00:00Z`) were checked for monotonicity (`ts_event_ns <= ts_recv_ns <= decision_ns`) but not verified against `STUDY_WALL_DATE` (`2026-10-15`) or `study_date`, allowing a post-wall `2027` nanosecond timestamp paired with a pre-wall `study_date` string to pass.

## Decision

1. Add `_safe_finite()` rejecting booleans, non-reals, `OverflowError` giant ints, `NaN`, and `±Inf`, and rewrite all threshold gates as negated valid ranges (`not (v >= min)` / `not (lo <= v <= hi)`).
2. Convert epoch `decision_ns` (`>= 946_684_800_000_000_000`) via integer floor division (`decision_ns // 1_000_000_000`) in UTC and reject any timestamp exceeding `STUDY_WALL_DATE` (`2026-10-15`) or disagreeing with `study_date`, as well as overflowed timestamps.
3. Add regression tests covering `NaN`, `±Inf`, post-wall `2027` `decision_ns`, and `study_date` mismatch in `tests/test_symbol_ranking_v310.py`.

## Verification

- `python scripts/architecture_catalog.py --write && python scripts/architecture_catalog.py --check`, `ruff check .`, `pytest -q`, and `gitleaks git --no-banner` (`0` leaks).
