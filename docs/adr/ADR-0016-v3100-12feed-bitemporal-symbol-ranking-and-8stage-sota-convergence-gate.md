# ADR-0016: `v3.10.0` Finalized 12-Feed (`DF1..DF12`) Bitemporal Symbol Ranking & 8-Stage SOTA Convergence Gate

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/symbol_ranking_v310.py`, `tests/test_symbol_ranking_v310.py`

## Context

`sota-compounding-lab` supplies strategy scaffolds and pre-wall lifecycle verification across the 6 autonomous paper-trading routes powered by `sota-multi-model-workflows` (`ADR-0024`). Under `v3.10.0`, the finalized `12`-Feed Symbol Ranking System DataFeeds (`DF1..DF12`, `sota-symbol-ranking-datafeeds.v3.10`) unify `DF10` own-history ToD normalization, `DF11` bitemporal PIT ordering (`ts_event_ns <= ts_recv_ns <= decision_ns`) + `H096` `15s` dry-powder collar, and `DF12` `microsoft/qlib` `RankIC` + `kappa=40` post-wall Empirical-Bayes feedback (`REAL_BROKER_ONLY`) grounded in `413` owner stars plus `microsoft/RD-Agent`, `databento/databento-python`, `unit8co/darts`, `Nixtla/statsforecast`, `stanfordnlp/dspy`, and `ADIA-Lab/MinTRL-MDE-PowerGate`.

## Decision

1. Ship `src/sota_compounding/symbol_ranking_v310.py` exposing `CANONICAL_12_RANKING_FEEDS` (`12` feeds) and `evaluate_df12_bitemporal_symbol_ranking_handoff()`.
2. Enforce `<= 2026-10-15` pre-wall study date, strict `ts_event_ns <= ts_recv_ns <= decision_ns`, `RVOL_tod >= 2.20`, `vol_z_tod >= 2.30`, `OFI >= 0.25`, `15.0s` dry-powder `<= 10.0 bps` collar, `ADR-0022 <= 2.00%` stop, and `900.0s` Wave-2 cooldown.

## Verification

- `python scripts/architecture_catalog.py --check`, `ruff check .`, `pytest -q`, and `gitleaks git --no-banner` (`0` leaks).
