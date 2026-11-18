# ADR-0006: Research-Converged Strategy Templates, Non-Repainting Pine v6 & 7-Feed PIT Contract

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/strategy_template.py`, `templates/sota_converged_strategy.template.json`, `templates/sota_pine_v6_nonrepainting.template.pine`, `tests/test_strategy_template.py`

## Context

New quantitative strategy hypotheses across the 4 SOTA archetypes (`PRE_TRAVEL_EARLY_IGNITION`, `FIRST_HAND_SEC_EPISODIC_PIVOT`, `MEGA_CAP_NEAR_LOSSLESS_ORB`, `CLIMAX_VWAP_MEAN_REVERSION`) risked schema drift, unpinned upstream citations, split-adjustment lookahead, intrabar repainting, or silent dilution exposure when scaffolded ad hoc across repositories.

## Decision

1. **3-Tier Research-Convergence Provenance (`ADR-0004` Alignment):** Every strategy manifest must pass `validate_converged_strategy_template` with a 64-hex `input_package_sha256`, `>= 1` 40-hex `pinned_upstream_commits` from the read-only upstream shelf, a non-empty `reproducible_command`, an `E0..E4` `evidence_grade`, and `other_repos_authority == "PHILOSOPHY_ONLY"`.
2. **7-Feed Point-in-Time (`PIT`) DataFeed Contract:** Enforce `rth_extended_feed == "sip"` (`04:00–20:00 ET`), `overnight_feed == "boats"` (`20:00–04:00 ET`), explicit `dec6_2026_23x5_cutover_enabled` boolean, raw universe filters (`>= $2.00`, `>= $2,000,000` dollar volume) paired with `return_adjustment == "split_dividend_all"` and `split_exclusion_window_sessions == [-1, 3]`, SEC EDGAR `acceptanceDateTime` with `900s` (`15m`) exponential freshness half-life, `S-3`/`424B5` dilution blocks, and `finra_short_vol_lag_sessions >= 1`.
3. **Pre-Wall Statistical, Queue & Paper-Only Gates:** Require `study_wall_date <= "2026-10-15"`, 64-hex `spec_sha256`, dynamic Harvey–Liu–Zhu `hlz_t_hurdle >= 3.0`, `cpcv_paths >= 5` (`purge >= 3`, `embargo >= 2`, `max_pbo <= 0.30`), `min_dsr >= 0.95`, `intrabar_tie_rule == "ADVERSE_STOP_FIRST"`, `cost_stress_multipliers == [1.0, 2.0]`, `live_execution_enabled is False`, `max_initial_stop_pct <= 2.0`, `alpaca_paper_max_rpm <= 200`, and `ibkr_paper_port == 4002` (`<= 3,000 msg/min`).
4. **Closed-Bar Non-Repainting Pine v6 Relay Template:** Ship `templates/sota_pine_v6_nonrepainting.template.pine` (`//@version=6`, `calc_on_every_tick=false`, `barstate.isconfirmed`, `alert.freq_once_per_bar_close`) emitting closed-bar JSON alerts aligned to `:8799/api/tv-webhook` with inward-quantized `ADR-0022 <= 2.00%` stops.
