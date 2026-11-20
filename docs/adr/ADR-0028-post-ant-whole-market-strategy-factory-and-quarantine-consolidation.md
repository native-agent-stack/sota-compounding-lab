# ADR-0028: Post-ANT Whole-Market Strategy Factory, `ADR-0064` Quarantine & Lossless Merit Contract (`v3.11.1`)

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Stage-4 Compounding Strategy Factory & Pre-Wall Governance (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`)
- **Evidence Grade:** E4
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/native-agent-stack/sota-mover-ranker, https://github.com/native-agent-stack/sota-multi-model-workflows, https://github.com/native-agent-stack/astrolabe
- **Revisit Trigger:** Pre-registration of `Trial #105` (`t*(105) = 4.2084`) or `2026-12-06` `21:00 ET` `23/5` Consolidated SIP cutover.

## Context and Problem Statement

Following `Trial #104` (`ADR-0027`, honest `FAILED` -> `PROTECTIVE_FILTER_OR_GOVERNANCE`, `PBO = 0.60`) and the transition into the Post-ANT whole-market era, `src/sota_compounding/converged_strategy_factory.py` needed explicit alignment with the `ADR-0064` Paper quarantine, the `6` proven SOTA strategy families, and symbol-agnostic `DF1..DF12` merit admission under shared `LosslessProtectionKernel` bounds (`ADR-0022 <= 2.00%`, `TP OFF`).

## Decision Outcome

1. **Extend `RETIRED_TRAP_DENYLIST` & Quarantine Registry:** Block `ADR0064_QUARANTINED_NOISY_PAPER_STRATEGIES` and `SINGLE_TICKER_IPO_DAY1_OVERFIT` across automated compounding plans and reflections.
2. **Ship `evaluate_post_ant_symbol_merit_contract`:** Validate any whole-market merited US symbol (`df_pillars_manifested >= 3`, `spread_bps <= 35.0`, `sec_dilution_clean=True`, `ADR-0022 <= 2.00%`, `TP OFF`, pre-wall `<= 2026-10-15`) strictly under `PROTECTIVE_FILTER_OR_GOVERNANCE` (`standalone_alpha_claim = False`; does not consume `Trial #105`).

## Alternatives Considered

- **Retaining Pre-ANT Single-Ticker IPO Promotion Defaults:** Rejected because Post-ANT whole-market targets must earn capital strictly on manifested `DF1..DF12` data and citation-backed lossless strategy eligibility.

## Verification Witness

- `ruff check . && ruff format --check .` (`PASS`), `scripts/architecture_catalog.py --check` (`28/28` ADRs, `0` drift), `uv run pytest -q` (incl. `tests/test_v3111_post_ant_factory_consolidation.py` `PASS`), `gitleaks --staged` (`0` leaks).
