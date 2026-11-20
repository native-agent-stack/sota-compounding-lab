# ADR-0031: Quality-and-SOTA-Only Experimental Repo Evaluation Policy (Unblock Technically SOTA Libraries; Retain Technical-Only Low-Quality Purge)

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Experimental R&D Source Governance & Playback Hygiene (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`)
- **Evidence Grade:** E4 (`src/sota_compounding/backtest_playback_manifest.py`, `src/sota_compounding/converged_strategy_factory.py`, `tests/test_adr0031_quality_first_sota_policy.py`, `sota-upstream-shelf-lock ADR-0022`, `ruff`, `ty`, `pytest`, `org_visibility_doctor.py`, `gitleaks 0`)
- **GitHub Evidence:** https://github.com/cvxgrp/cvxportfolio, https://github.com/man-group/ArcticDB, https://github.com/google-research/timesfm, https://github.com/xtdb/xtdb, https://github.com/ml4t/backtest, https://github.com/Nixtla/conformalforecast, https://github.com/NVIDIA/cudf
- **Revisit Trigger:** Only if any repository is ever redistributed externally to third parties (in which case OSI-permissive redistribution audits re-engage).

## Context and Problem Statement

By explicit owner direction (`2026-11-20`), the `native-agent-stack` environment operates strictly as a personal, effectively-private experimental research workstation (zero external stars/watchers, zero external redistribution). Excluding technically world-class libraries solely on copyleft/BSL/non-commercial license tags (`cvxgrp/cvxportfolio`, `man-group/ArcticDB`, `google-research/timesfm`, `xtdb/xtdb`, `edtechre/pybroker`, `freqtrade/freqtrade`) unnecessarily deprives our pre-wall R&D loop of frontier engineering.

## Decision Outcome

1. **Adopt `EXPERIMENTAL_EVALUATION_POLICY = "QUALITY_AND_SOTA_ONLY_NO_LICENSE_EXCLUSION"`:**
   - Unblock technically SOTA libraries inside `backtest_playback_manifest.py` and `converged_strategy_factory.py`: `cvxgrp/cvxportfolio`, `man-group/ArcticDB`, `google-research/timesfm` (alongside `Chronos-2`, `uni2ts` Moirai, `granite-tsfm` TTM), `xtdb/xtdb`, `edtechre/pybroker`, `freqtrade/freqtrade`, `ml4t/*` (2026 modular suite), `Nixtla/conformalforecast`, `alphanome-ai/sec-parser`, and `NVIDIA/cudf` + `NVIDIA/cuml`.
2. **Retain 100% Strict Technical-Only Low-Quality Purge:**
   - Continue hard-rejecting repositories that fail on **technical quality, staleness, or data corruption** (`ranaroussi/yfinance`, `mementum/backtrader`, `kernc/backtesting.py`, abandoned `hudson-and-thames/mlfinlab`, stale `cuemacro/tcapy` / `vollib/vollib`, archived `abides-jpmc-public`, unofficial TradingView cookie/WS scrapers) and keep all 14 statistical/execution hygiene checks intact (`ADR-0022 <= 2.00%` stop, `TP OFF`, `8/16 bps` `1x`/`2x`, `W1 679d`/`W2 1707d` `MinTRL`, `SPA`/`MCS`/`DSR`/`PBO`, `<= 1.0x` vol brake, `acceptanceDateTime` PIT).

## Alternatives Considered

- **Retain third-party redistribution license walls inside a personal zero-redistribution lab:** Rejected per owner directive; technical quality and SOTA rigor govern experimental selection.

## Verification Witness

- `scripts/architecture_catalog.py --check` (`31/31` ADRs, `0` drift), `pytest -q` (incl. `tests/test_adr0031_quality_first_sota_policy.py`), `ruff check .`, `ty check`, `scripts/org_visibility_doctor.py --offline` (`PASS`), and `gitleaks protect --staged --no-banner` (`0` leaks).
