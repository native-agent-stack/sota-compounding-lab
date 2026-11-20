# ADR-0024: Trial #103 (`H107`) SIC Sector-Neutral Liquid Multi-Feed Residual Factor on RTX 4090 CPCV

- **Status:** Accepted
- **Date:** 2026-11-19
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `scripts/run_h107_sector_neutral_liquid_multifeed_rtx4090.py`, `tests/test_h107_sector_neutral_rtx4090.py`

## Context

Following `H106` (`Trial #102`, broad universe `-13.26 bps` `1x` / `-30.24 bps` `2x`), we pre-registered `Trial #103` (`H107_SECTOR_NEUTRAL_LIQUID_MULTIFEED_RESIDUAL_RTX4090_CPCV`, `spec_sha256=a0a41e11b2aa3618427e0e1141246175d8a98c88f79483d3efc2364e8ebc0fe6`, ADIA Lab #19 power gate `approved=true` for `D=679` vs `MinTRL=496.0`) to isolate whether stripping market/sector beta via 2-digit SIC (`sic2`) sector-neutralization and restricting to liquid common US equities (`prev_raw_close >= 10.0`, `mean_20d_dvol >= $25M`, `>= 6` symbols per `(trade_date, sic2)`) could clear `HLZ t*(103) = 4.2041` and `DSR >= 0.95`.

## Decision

1. **Deterministic Pre-Wall Execution (`scripts/run_h107_sector_neutral_liquid_multifeed_rtx4090.py`):**
   - Strictly enforces `trade_date <= 2026-10-15`, `next1_trade_date <= 2026-10-15`, `ASOF` PIT causality (`DF8 known_from < trade_date`, `sec_float filed < trade_date`), `90`-day SEC dilution exclusion on long legs, retired-trap firewall compliance, within-`(trade_date, sic2)` feature ranking (`[-1, +1]`), inverse-volatility weighted dollar-neutral top/bottom quintile legs, `ADR-0022 <= 2.00%` adverse-stop-first 1-day intraday MTM (`next1_split_open -> next1_split_close`), `1x`/`2x` Almgren-Chriss sqrt impact + `5`/`10 bps` short borrow, and `C(6,2)=15` disjoint `skfolio` `CombinatorialPurgedCV(purge=5d, embargo=5d)` on RTX 4090 (`cuda`, `native_ext4` tier).
2. **Honest Empirical Outcome (`E4 FAILED` — Never Promote, Protective Design Lesson Recorded):**
   - Executed on **NVIDIA GeForce RTX 4090** (`native_ext4`, `receipt_sha256=3f0d3b90f7a5b240ba6b6007d38fae92d0cc23b80d96b9c826061475c5fe5217`):
   - `871,676` events across `679` distinct pre-wall dates (`2,036` operating US equities, `48` `sic2` sectors).
   - Primary equal-weight 3-model ensemble (`PyTorch RTX 4090 CUDA MLP + XGBoost CUDA + Ridge`): `1x = -14.3105 bps` (`SE = 0.9751`), `2x_hac5 = -29.2959 bps` (`SE = 0.9655`), `conservative_gate_t_2x = -30.3419 < 4.2041`, `DSR = 0.0`, `CPCV+ = 0.0`, `PBO = 1.0`.
   - Diagnostic decomposition proves why: even on `$25M+ ADV` sector-neutral stocks, `long_stopped_day1_rate = 28.03%` and `short_stopped_day1_rate = 27.45%` under a `<= 2.00%` intraday adverse stop, combined with `13.0 bps` `1x` (`26.0 bps` `2x`) Almgren-Chriss impact + `5/10 bps` short borrow (`long_leg_2x = -21.67 bps`, `short_leg_2x = -36.92 bps`). Future Trial `#104+` designs must target **multi-day (`3–10` session) post-catalyst horizons** or **genuine sub-second microstructure fills**, never 1-day open-to-close daily-bar factor swaps.

## Verification

- `pytest -q tests/test_h107_sector_neutral_rtx4090.py` (`3 passed` on `Noesis-Carbon` `.venv`), `ruff check` & `ruff format --check src scripts tests`, `scripts/architecture_catalog.py --check` (`24/24`), `scripts/org_visibility_doctor.py --offline`, and `gitleaks protect --staged --no-banner` (`0` leaks).

## Verification Witness

- **Cataloged Implementation & Test Artifacts:** `scripts/run_h107_sector_neutral_liquid_multifeed_rtx4090.py`, `tests/test_h107_sector_neutral_rtx4090.py`
- **Deterministic Gates:** Verified by `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py` (`UPSTREAM.md` + `upstream-map.toml` + `18` shelf pins), and `pytest -q`.
