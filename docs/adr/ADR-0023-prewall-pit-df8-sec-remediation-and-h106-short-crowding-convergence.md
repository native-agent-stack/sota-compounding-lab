# ADR-0023: Pre-Wall PIT Remediation (`DF8`/`SEC`/`Universe`/`H099`) & Trial #102 (`H106`) Short-Crowding Cross-Sectional Convergence

- **Status:** Accepted
- **Date:** 2026-11-19
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/prewall_h099_enrichment.py`, `scripts/materialize_prewall_df10_df6.py`, `scripts/run_h106_df8_short_crowding_xsec.py`, `scripts/run_h105_sec_8k_episodic_pivot.py`, `scripts/run_optb_prewall_factory.py`, `scripts/sync_prewall_lake_fast_tier.py`, `tests/test_prewall_h099_enrichment.py`, `tests/test_prewall_optb_factory.py`

## Context

Read-only Lane 04/05 PIT audits (`sota-bus` `#299` / `#300`) and adversarial falsification identified five point-in-time data-hygiene and runner gaps across the pre-wall (`<= 2026-10-15`) lake and Option B scripts:
1. **`DF8` Post-Wall Leakage:** `public-tables/short_interest.parquet` (`1,480,016` rows) contained `45,096` post-wall records (`known_from > 2026-10-15`, `settlement` up to `2026-10-30`) despite `PREWALL_LAKE_MANIFEST.json` reporting `excluded_post_wall: 0`.
2. **Non-Reproducible Clean SEC Build & Unwired `ADR-0020` Helpers:** `derived/sec_filings_clean_prewall.parquet` lacked a reproducible builder in `scripts/`, `filter_sec_filings_pit_prewall()` and `audit_h099_prewall_lake_readiness()` had no script callers, and SEC dilution form filters diverged across `H103`/`H104`/`H105` (omitting `S-1MEF`, `424B2`, `424B7`, `424B8`, or `424B3`).
3. **`H105` Day-0 Close & Calendar Mapping Bugs:** `run_h105_sec_8k_episodic_pivot.py` filtered on Day-0 `d.raw_close >= 5.0` (6.5h after `09:30` entry), dropped Friday post-close 8-Ks (`+ INTERVAL 1 DAY`), mapped pre-2024 8-Ks onto `2024-01-31`, and passed `effective_trials=105.0` instead of `101.0`.
4. **Universe `2026` Manifest Delta:** `universe/2026/manifest.json` records `2,497,925` pre-trim full-year bars whereas the pre-wall trimmed gzip files hold `2,248,997` bars (`-248,928` trimmed post-`2026-10-15`).
5. **Untested Multi-Feed `DF8 + sec_float + DF5 + DF6 + SEC` Hypothesis (`H106`, Trial `#102`):** Required strictly causal, non-overlapping 1-day intraday MTM evaluation at `HLZ t*(102) = 4.2019` under dimensionless XBRL `EntityPublicFloat` (`USD`) parity.

## Decision

1. **Canonical SEC Form Sets & `DF8` Quarantine Filter (`src/sota_compounding/prewall_h099_enrichment.py`):**
   - Export `CANONICAL_SEC_IPO_FORMS`, `CANONICAL_SEC_SHELF_DILUTION_FORMS`, and `CANONICAL_SEC_ALL_DILUTION_FORMS`, extend `filter_sec_filings_pit_prewall()` to read raw `accepted`, tighten `_parse_iso_date()` to exact `YYYY-MM-DD`, and add `ShortInterestPrewallFilterResult` / `filter_short_interest_pit_prewall()` (`known_from <= 2026-10-15`, `settlement <= 2026-10-15`, `settlement < known_from`).
2. **Reproducible Materialization & `H099` Readiness Sealing (`scripts/materialize_prewall_df10_df6.py`, `scripts/sync_prewall_lake_fast_tier.py`):**
   - Add `materialize_sec_clean_prewall()` (`612,671` clean rows, `12` post-wall purged, `0` leaks), `materialize_df8_short_interest_clean_prewall()` (`1,434,920` clean rows, `45,096` post-wall purged, `max_known_from=2026-10-12`, `max_settlement=2026-09-30`), `audit_universe_and_h099_readiness()`, and `--pit-remediation-only` (sealing `prewall_pit_remediation_receipt.json` `sha256=16f65e4c...`, tier `E2_BOUNDED_500MS_H099_EXPLORATORY_READY`). Verify `df8_short_interest_clean_prewall.parquet` SHA-256 across `native_ext4` and `/dev/shm`.
3. **Remediate `H105` & Unify Option B Runners (`scripts/run_h105_sec_8k_episodic_pivot.py`, `scripts/run_optb_prewall_factory.py`):**
   - Wire `CANONICAL_SEC_*_FORMS` and `audit_h099_prewall_lake_readiness()` across runners. Fix `H105` to `prev_raw_close >= 5.0`, XNYS `trade_sessions` next-session lookup bounded to `0..5` calendar days, and `effective_trials=101.0`; re-run `H105` receipt (`events=6298`, `dates=635`, honest `FAILED`, `sha256=a84b7fa2...`). *(Historical `H103`/`H104` factory receipts remain exploratory reference `FAILED` records from pre-remediation code.)*
4. **Pre-Registered Trial `#102` (`H106`) on Noesis RTX 4090 (`scripts/run_h106_df8_short_crowding_xsec.py`):**
   - Pre-registered (`spec_sha256=88292e639cfbd9f436e3466b07f75faa5acdce6eb0b65c5cd051cc6fd0cfd947`) over non-overlapping `next1_split_open -> next1_split_close` (`<= 2026-10-15`), `ASOF` `DF8 known_from < td`, dimensionless dollar short-ratio `(short_shares * raw_close) / float_usd` (`sec_float` USD units), `ADR-0022 <= 2.00%` adverse-stop-first legs, `1x`/`2x` Almgren–Chriss + `5`/`10 bps` short borrow, `C(6,2)=15` CPCV (`ensemble_3m` primary).
   - **Empirical Outcome (Honest `E4 FAILED` — Never Promote):** `980,057` events across `679` dates (`2,196` symbols): `ensemble_3m` `1x = -13.2555 bps`, `2x_hac5 = -30.2436 bps`, `conservative_gate_t_2x = -18.2717 < 4.2019`, `DSR = 0.0`, `CPCV+ = 0.0`, `PBO = 1.0` (`sha256=fa88d7bc...`).

## Verification

- Adversarial `sota-falsifier` audit + delta pass, `ruff check --no-cache src scripts tests`, `pytest -q` (Win32 + WSL `.venv`), `python scripts/architecture_catalog.py --check` (`23/23` MADR ADRs), and `gitleaks` (`0` leaks).
