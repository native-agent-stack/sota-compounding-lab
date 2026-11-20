# ADR-0014: `v3.9.8` `Trial #96` (`H096`) Pure Dry-Powder `15s` Bid-Survival & Informed-Flow Joint Necessity (`N = 96 -> t*(97) = 4.1905`)

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `templates/sota_converged_strategy.template.json`, `src/sota_compounding/lifecycle.py`, `tests/test_compounding_lifecycle.py`

## Context

1. Pre-registered Trial `#96` (`H096_DRY_POWDER_15S_BID_SURVIVAL_10BPS_COLLAR_ORB_CPCV`, `spec_sha256 = 588d27755badb38da2ebe7a762c8106b210a0333842b0a58d0724531583fbb5f`, `<= 2026-10-15`, `700` `XNYS` sessions) tested pure dry-powder `15s` bid-survival (`0` resting passive bids in `(0, 15s]`, `min_lit_roundlot_px >= bp_entry`, `bp_15s >= bp_entry`) + `<= 10.0` bps collar entry.
2. Relaxing informed-flow selection to bare `OFI > 0` (`n = 124` / `107` dates) yielded `-12.729` bps net `1x` (`t = -0.7781 < 4.1881`, `FAILED` on `ALPHA_STRATEGY_MINOR`, advancing `N = 96 -> t*(97) = 4.1905`), whereas the strict diagnostic cohort (`OFI >= 0.25 & (ISO_F >= 0.15 | BJZZ >= 0.12)` + `15s` dry-powder bid-survival + `<= 10.0` bps collar, `n = 52` / `47` dates) realized **`+16.835` bps net `1x` / `+11.366` bps net `2x`**, `+2.947` bps positive selection (`+100.221` bps reversal vs `H094`), `11/15` (`73.33%`) disjoint CPCV `> 0`, and `3/3` years positive at `2x` (`+29.564` bps joint necessity over `OFI > 0`).

## Decision

1. Advance `templates/sota_converged_strategy.template.json` default `cumulative_trials_n` to `96` (`hlz_t_hurdle` synced to `hlz_dynamic_t_hurdle(96)`).
2. Publish `evaluate_dry_powder_15s_collar_policy` (`src/sota_compounding/lifecycle.py`) enforcing `resting_passive_bid_0_15s == False`, `bid_survived_15s == True`, `OFI >= 0.25 & (ISO_F >= 0.15 | BJZZ >= 0.12)`, and `<= 10.0` bps collar/spread under `PROTECTIVE_FILTER_OR_GOVERNANCE` (`standalone_alpha_claim = False`).

## Verification

- `python scripts/architecture_catalog.py --check`, `ruff check .`, `pytest -q` (`57 passed`), and `gitleaks` (`0` leaks).

## Verification & Evidence
- Backfilled under `sota-upstream-shelf-lock ADR-0023` (2026-11-20): anchored to repository `UPSTREAM.md`, `upstream-map.toml`, and git history of `adr0023-19adr-evidence` (`shelf_lock.hard_upstream_rule` verified).
