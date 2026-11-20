# ADR-0033: Pre-Registered `Trial #106` (`H112`) Real `700`-Session Pre-Wall SEC `424B5` Dilution Short Empirical Outcome & Zero-Truncation Falsification Law

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Pre-Wall Empirical Alpha Falsification (`<= 2026-10-15`), `DF5` SEC Dilution Shield Validation & Backtest Anti-Truncation Integrity (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`)
- **Evidence Grade:** E4 (`scripts/run_h112_trial106_424b5_dilution_short.py`, `docs/TRIAL106_H112_PREWALL_RECEIPT.json` SHA `caea38a47158...`, `tests/test_adr0033_trial106_h112_424b5_dilution_short.py`, `sota-bus` Trial `#106` pre-registration SHA `71c7df7ca89b...`, `pytest`, `ruff`, `ty`, `org_visibility_doctor.py`, `gitleaks 0`)
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/native-agent-stack/sota-mover-ranker, https://github.com/native-agent-stack/sota-upstream-shelf-lock
- **Revisit Trigger:** Addition of historical `Form 4` insider clusters & `5-day XML SC 13D` tables (`H111`), or `H110`/`H113` (`Trial #107+`).

## Context and Problem Statement

After `Trial #105` (`H109`, `ADR-0032`) falsified blind `BOATS` overnight gap-up fading, `Trial #106` (`H112`) tested Billett, Floros & Garfinkel (2019 JFQA) / Gustafson & Iliev (2017 JFE) `424B5` equity offering dilution supply-shock shorts across `700` real pre-wall `SEC EDGAR` + `SIP 1m` sessions (`2024-01-02 .. 2026-10-15`).

## Decision Outcome

1. **Pre-Registration & Statistical Power Gate:**
   - `ADIA Lab #19` Power Gate passed (`679 >= 498` `MinTRL`, `HLZ t*(106) = 4.2106`, `SR_min_ann = 2.5651`), immutably pre-registered on `sota-bus` (`spec_sha256 = 71c7df7ca89b09cae3b3e5784381c3c79f2c3de5c5c7ce767f9c5fc1b4d34dc9`).
2. **Honest Empirical Verdict (`FAILED_HLZ_OR_STAT_GATE`, Receipt `caea38a47158...`):**
   - **Primary `H112_424B5_PRIMARY_RTH_SHORT` (`2,078` events / `596` dates, `14/14` hygiene PASS):** Gross `+2.88 bps/day`, Net `1x` **`-7.44 bps/day`** (`t = -1.98`), Net `2x` **`-17.75 bps/day`** (`t = -4.725`, `0/5` CPCV positive, `DSR = 0.0`, `SPA p = 1.0`). Unclamped Day-0 short drift was **`+24.94 bps` (`+45.78 bps` over 3 days, `raw_t = +3.11`)**, but `424B5` issuers average `396 bps` daily volatility, causing `0.50-sigma` `2.00%` hard stops to whipsaw out `22.06 bps` of gross drift before realization.
   - **Unclamped 3-Day MTM Diagnostic (`6,226` daily MTM legs / `671` dates):** **`+10.21 bps/day` gross**, `+5.29 bps/day` `1x` (`t = +0.57`), **`+0.37 bps/day` `2x`** (`t = +0.0403`, `SPA p = 0.4765`, `max_drawdown = 39.74%`, `ex_top3 = -4.65 bps`).
   - **Idle `S-3`/`EFFECT` Shelf Short (`1,201` events / `509` dates, `14/14` hygiene PASS):** **`-27.80 bps/day` `2x`** (`t = -5.92`, negative across `2024`/`2025`/`2026`).
3. **Two Permanent Governance & Research Laws:**
   - **Quantitative Validation of `DF5` Long Dilution Shield:** Confirms avoiding Long entries on `424B5` issuers saves **`+45.78 bps` per avoided offering** (`+24.94 bps` on Day 0), while refusing standalone `424B5`/`S-3` short bots under `ADR-0022 <= 2.00%` stops.
   - **Zero Phantom Daily-Loss Truncation Law:** Adversarial self-falsification during `Trial #106` caught that clamping daily close-to-close returns with `max(-200 bps, ret)` without true intrabar stop-out fabricated a phantom Sharpe (`5.18` vs true unclamped `0.025`). All multi-day MTM studies MUST remain 100% unclamped unless simulating full bar-by-bar stop-out + gap-through slippage.

## Alternatives Considered

- **Relax `ADR-0022 <= 2.00%` Stop Ceiling to Capture `424B5` 3-Day Drift:** Hard-refused — even unclamped 3-day MTM fails `HLZ` (`t_2x = +0.04`, `39.74%` drawdown, `excess_kurtosis = 11.28`) and violates our non-negotiable `ADR-0022 <= 2.00%` risk firewall.

## Verification Witness

- `docs/TRIAL106_H112_PREWALL_RECEIPT.json` (`SHA caea38a47158...`, `0` post-wall bytes read).
- `scripts/architecture_catalog.py --check` (`33/33` ADRs, `0` drift).
- `pytest -q`, `ruff check .`, `ruff format --check .`, `ty check`, `scripts/org_visibility_doctor.py`, and `gitleaks protect --staged --no-banner` (`0` leaks).

## Rollback

Revert `ADR-0033`; `Trial #106` remains immutably logged as `FAILED` in `sota-bus` `HYPOTHESIS_LEDGER.md`.
