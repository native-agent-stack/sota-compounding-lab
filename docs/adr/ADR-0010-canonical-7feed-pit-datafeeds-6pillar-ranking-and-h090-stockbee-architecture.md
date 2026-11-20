# ADR-0010: Canonical 7-Feed PIT DataFeeds, 6-Pillar SOTA Mover Ranking, Stockbee EP/Burst & `H090` Execution-Tier Architecture (`sota-ranking-and-datafeeds.v2`)

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/ranking_and_datafeeds.py`, `tests/test_ranking_and_datafeeds.py`, `scripts/architecture_catalog.py`

## Context

To ensure every quantitative research study, pre-wall backtest (`<= 2026-10-15`), and `24/5` Paper execution lane shares one mathematically verified DataFeed and Ranking System contract across all SOTA research skills (`microstructure-and-sec-catalyst-lab`, `stockbee-episodic-pivot-analyzer`, `stockbee-momentum-burst-screener`, `sota-quant-research-protocol`, `sota-dimensional-and-property-invariants`, and `sota-mover-ranker`), we needed an open-core validator that:
1. Enforces fail-closed Point-in-Time (`PIT`) lag invariants across all 7 canonical `sota-research-feeds.v2` DataFeeds;
2. Eliminates a live false-positive classification bug where `1h–8h` old intraday headlines (`1,800s < age <= 28,800s`, e.g., `88m`-old analyst notes) fell through into `TAPE_PRE_NEWS_IGNITION` (`incentive_score = 92.0`, `is_first_hand_early = True`);
3. Integrates `Trial #90` (`H090`: `<= 10 bps` spread + `>= 25%` `OFI` passive-bid execution tier achieving `+20.84 bps` net 1x and `5/5` `CPCV` positive OOS paths) alongside Stockbee Day-1 Episodic Pivot and `4%` Momentum Burst setup gates.

## Decision

1. **7-Feed PIT DataFeed Validator (`validate_pit_datafeed_observation`):**
   - Require all 7 canonical feeds (`alpaca_sip`, `alpaca_boats`, `alpaca_raw_vs_all_adjustment`, `sec_edgar_acceptance_datetime`, `finra_regsho_cnmsshvol_t1`, `corporate_actions_split_guard`, `ibkr_paper_4002_l1`) with zero lookahead: `sec_acceptance_delta_sec >= 0`, `finra_short_vol_lag_days >= 1` (`T-1`), `finra_short_interest_lag_bdays >= 8` (FINRA Rule 4560 `T+8` NYSE business days), `boats_quote_age_sec <= 900s`, `bar_age_sessions == 0` completed-session freshness (`stockbee-momentum-burst-screener`), `bs * 100.0` CTA/UTP round-lot share conversion, and `BJZZ (2021)` round/midpoint institutional vs sub-penny retail `x == 'D'` classification.
2. **True Pre-News Silence (`>= 8h`) vs Intraday Post-News Drift (`evaluate_first_hand_incentive`):**
   - `TAPE_PRE_NEWS_IGNITION` (`Gervais, Kaniel & Mingelgrin 2001`) requires `catalyst_age_seconds is None` or `(catalyst_age_seconds > 28,800s and news_count_36h <= 1)` plus `ofi_pct >= 15.0`; intraday `1,800s < age <= 28,800s` headlines classify deterministically as `TIER2_INTRADAY_POST_NEWS_DRIFT` (`ACTIVE_FOLLOW_THROUGH`, `is_first_hand_early = False`) under `15m` (`900s`) Vega/Tetlock exponential decay.
3. **6-Pillar Session-Regime Ranking + `H090` Execution Tiers (`compute_sota_mover_ranking`):**
   - Combine the 6 `[0..100]` percentile pillars (`move`, `unusual_vol` `RVOL_tod`, `inst_dark`, `iso_urgency`, `overnight`, `catalyst`) with `H090` execution-quality tiers (`TIER1_TIGHT_SPREAD_HIGH_OFI_PASSIVE` `<= 10 bps` & `OFI >= 25%` -> `+4.0` early-rank lift; `TIER2_TWO_STAGE_HYBRID_QUEUE` `<= 35 bps`; `TIER3_WIDE_SPREAD_OR_ADVERSE_OFI_BLOCKED`), Stockbee `4%`/dollar Momentum Burst qualification (`<= 2.00%` `ADR-0022` stop), and normalized `[0..100] * ucb_priority_weight` strategy arbitration.

## Verification Witness

- **Cataloged Implementation & Test Artifacts:** `src/sota_compounding/ranking_and_datafeeds.py`, `tests/test_ranking_and_datafeeds.py`
- **Deterministic Gates:** Verified by `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py` (`UPSTREAM.md` + `upstream-map.toml` + `18` shelf pins), and `pytest -q`.
