# ADR-0032: Pre-Registered `Trial #105` (`H109`) Real Pre-Wall (`512`-Session `BOATS 1m` + `SIP 1m`) Empirical Outcome & Governance Disposition

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Pre-Wall Empirical Alpha Falsification (`<= 2026-10-15`), `BOATS` Microstructure Governance & `ADIA Lab #19` / `HLZ` Ledger Integrity (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`)
- **Evidence Grade:** E4 (`scripts/run_h109_trial105_boats_bifurcation.py`, `docs/TRIAL105_H109_PREWALL_RECEIPT.json` SHA `79d8e91c452b...`, `tests/test_adr0032_trial105_h109_boats_bifurcation.py`, `sota-bus` Trial `#105` pre-registration SHA `db9a6f8ac768...`, `pytest`, `ruff`, `ty`, `org_visibility_doctor.py`, `gitleaks 0`)
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/native-agent-stack/sota-upstream-shelf-lock, https://github.com/native-agent-stack/sota-mover-ranker
- **Revisit Trigger:** Post-`2026-12-06` `23/5` SIP cutover (`W4` OOS shadow) or L2/L3 Databento `DBN` queue microstructure enrichment (`H114`).

## Context and Problem Statement

`ADR-0029` / `ADR-0030` prioritized `H109` (Overnight/`BOATS` Retail-Crowding Bifurcation + RTH `09:30–09:45` Opening VWAP Fade, citing Lou–Polk–Skouras 2019 JFE, Eaton–Shkilko–Werner 2025/26, and Lim 2026) as the first Wave-1 `Trial #105` test on real pre-wall (`<= 2026-10-15`) `BOATS 1m` + `SIP 1m` data.

## Decision Outcome

1. **Pre-Registration & Statistical Power Gate (`sota-quant-research-protocol`):**
   - `ADIA Lab #19 MinTRL/MDE` Power Gate passed BEFORE running (`512` pre-wall overlapping `BOATS 1m` + `SIP 1m` sessions `2024-10-01 .. 2026-10-15` >= `min_trl_days_at_sr3 = 497`, `HLZ t*(105) = 4.2084`, `SR_min_ann = 2.9524`).
   - Immutably pre-registered on `sota-bus` (`Trial #105`, `spec_sha256 = db9a6f8ac768bd1482c94b1d1c8559876aab0102b85464a5b443357a8342ec61`) with strictly causal `T-1` liquid US operating common equities (`raw close >= $5.00`, `20d ADV >= $25M`, ex-splits, strictly PIT SEC `accepted` timestamps, `ADR-0022 <= 2.00%` stop + uncapped 3-tier lossless Chandelier, flat `15:50 ET`, `14/14` low-quality purge audit = `PASS`).
2. **Honest Empirical Verdict (`FAILED_HLZ_OR_STAT_GATE`, Receipt `79d8e91c452b...`):**
   - **Primary `LEG_A_BOATS_FADE_SHORT` (`3,182` trades / `439` dates):** **DECISIVELY FALSIFIED** — `-1.51 bps/day` gross, **`-11.96 bps/day` `1x`** (`t = -2.6779`), **`-22.41 bps/day` `2x`** (`t = -5.0188`, `SR_2x = -3.803`, `0/5` CPCV paths positive, `DSR = 0.0`, `Hansen SPA p = 1.0`, negative across all 4 subperiods `2024Q4..2026`). **Governance Action:** Permanently refuse blind `09:35 ET` fading of `BOATS` gap-ups in Paper `P1–P4` and Live `:8798`.
   - **Diagnostic `LEG_B_8K_VWAP_HOLD_LONG` (`2,025` trades / `410` dates):** Positive gross (`+19.94 bps/day`) and **`1x` net (`+11.19 bps/day`, `t = +2.502`, `SR_1x = +1.961`, `5/5` CPCV positive at `2x` `+2.43 bps/day`, `Conformal PID` stop `0.9643%` at `10.37%` stop-out rate)**, but fails `2x` HLZ (`t = +0.5436 < 4.2084`, `DSR = 0.0`, `SPA p = 0.2967`, `2026` pre-wall crowding decay `-5.19 bps`). **Governance Action:** Retain strictly under `PROTECTIVE_FILTER_OR_GOVERNANCE` (never `ALPHA_STRATEGY_MINOR`).
   - Logged honestly to `sota-bus` as `FAILED`, advancing the ledger to `Trial #106` (`H111`/`H112` Form 4 / XML 13D / `Lazy Prices` next).

## Alternatives Considered

- **Retune Thresholds Post-Hoc to Force a Pass:** Hard-refused — violates immutable SHA-256 pre-registration and induces multiple-testing overfitting.

## Verification Witness

- `scripts/run_h109_trial105_boats_bifurcation.py` (`docs/TRIAL105_H109_PREWALL_RECEIPT.json`, `hygiene_14pt_passed = true`, `0` violations).
- `scripts/architecture_catalog.py --write && --check` (`32/32` ADRs, `0` drift).
- `pytest -q`, `ruff check .`, `ruff format --check .`, `ty check`, `scripts/org_visibility_doctor.py`, and `gitleaks protect --staged --no-banner` (`0` leaks).

## Rollback

Revert `ADR-0032`; the immutable `sota-bus` `Trial #105` outcome remains permanently recorded in `HYPOTHESIS_LEDGER.md`.
