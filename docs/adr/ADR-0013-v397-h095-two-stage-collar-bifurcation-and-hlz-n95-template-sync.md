# ADR-0013: `v3.9.7` `H095` Two-Stage Collar Bifurcation, `N = 95` (`t*(95) = 4.1858`) Template Sync & `PROTECTIVE_FILTER_OR_GOVERNANCE` Discipline

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `templates/sota_converged_strategy.template.json`, `src/sota_compounding/strategy_template.py`, `tests/test_strategy_template.py`, `tests/test_compounding_lifecycle.py`

## Context

1. `templates/sota_converged_strategy.template.json` retained static defaults `cumulative_trials_n = 84` (`hlz_t_hurdle = 4.1604`) while the shared `HYPOTHESIS_LEDGER.md` advanced through pre-registered Trial `#95` (`H095_TWO_STAGE_LIT_QUEUE_15S_TO_10BPS_MARKETABLE_COLLAR_ORB_CPCV`, `spec_sha256 = 5abe51a4caacf4bf6955a331b495310fae1791a8d74a76c111bc9e6a35b5f4fa`, `t*(95) = 4.1858 -> t*(96) = 4.1882`).
2. Across `319` shielded ORB ignitions (`<= 2026-10-15`), `H095` improved pure-passive adverse selection by `+47.804 bps` (`-97.274 -> -49.470 bps`, `8/15` disjoint CPCV `> 0`, median `+0.0219`) and revealed that Stage-2 `+15s` bid-survived `<= 10.0 bps` marketable collars (`n = 55`) earn `+20.779 bps` gross / `+15.305 bps` net `1x` / `+9.832 bps` net `2x`, whereas Stage-1 `[0..15s]` immediate passive bid fills (`n = 134`, `80/134` `trade_through`) suffer `-8.736 bps` net `1x` seller-sweep toxicity (`combined = -1.740 bps` net `1x`, `t = -0.1089 < 4.1858`, logged `FAILED` on `ALPHA_STRATEGY_MINOR`).

## Decision

1. Advance `templates/sota_converged_strategy.template.json` default `cumulative_trials_n` to `95` and `hlz_t_hurdle` to `4.1858` (verified against `hlz_dynamic_t_hurdle(95)` in `validate_converged_strategy_template`).
2. Enforce strict Two-Track SemVer promotion (`v3.9.7` `PROTECTIVE_FILTER_OR_GOVERNANCE`): never post-hoc cherry-pick the `+15.305 bps` Stage-2 subset without an independent pre-registered `Trial #96` (`t*(96) = 4.1882`) and `REAL_BROKER_ONLY` `n_closed >= 30` Stage-3 paper roundtrips.

## Verification

- `python scripts/architecture_catalog.py --check`, `ruff check .`, `pytest -q` (`56 passed`), and `gitleaks git --staged --no-banner` (`0` leaks).

## Verification & Evidence
- Backfilled under `sota-upstream-shelf-lock ADR-0023` (2026-11-20): anchored to repository `UPSTREAM.md`, `upstream-map.toml`, and git history of `adr0023-19adr-evidence` (`shelf_lock.hard_upstream_rule` verified).
