# ADR-0037: Sim Lanes (`S1`/`S2`) Tier-0 SOTA-Source-Only Enforcement & Anti-Self-Invention Gate

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Pre-Wall Simulation & Research Factory Gate (`src/sota_compounding/upstream_lib_strategy_e2e.py`, `tests/test_adr0037_tier0_sim_sota_source_only.py`), aligned with `sota-mover-ranker` `ADR-0092` and `sota-dynamic-trail-c` `ADR-0033` (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`)
- **Evidence Grade:** E4 (`enforce_tier0_sota_source_only_sim_gate()` fail-closed enforcement rejecting `cpython-stdlib`/self-built/unpinned upstreams, missing primary literature, and all 15 purged self-invented anti-patterns; `pytest`, `ruff`, `scripts/architecture_catalog.py --check`, `scripts/org_visibility_doctor.py`, `gitleaks 0`)
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/nautechsystems/nautilus_trader/commit/da6b57aefdd62b44f675e434ffcfefca7f4025dd, https://github.com/nkaz001/hftbacktest/commit/5f3ec40b2afb764e0fea112f941ed85523ef4e88, https://github.com/polakowo/vectorbt/commit/a52a984b631624bb9daf41cb19f532fe559cf0ef
- **Revisit Trigger:** Addition of a newly pinned upstream repository to `SOTA_UPSTREAM_BACKTEST_PLATFORMS` and `~/Documents/upstream/PINS.md`.

## Context and Problem Statement

To eliminate self-invented heuristics at the earliest point in the compounding lifecycle (`Sim -> Paper -> Live`), Sim lanes (`S1`/`S2`) and all future research trials must enforce the exact same Tier-0 SOTA-Source-Only rule enforced in Paper `P1..P4` and Stage-5 Live: zero self-built, zero `cpython-stdlib`-only, and zero uncited strategies.

## Decision Outcome

Add `enforce_tier0_sota_source_only_sim_gate()` and `TIER0_PURGED_SELF_INVENTED_SIM_STRATEGIES` to `src/sota_compounding/upstream_lib_strategy_e2e.py`, requiring every Sim strategy to reference a pinned repository in `SOTA_UPSTREAM_BACKTEST_PLATFORMS`, a non-empty commit SHA/tag, and a peer-reviewed primary literature citation while rejecting all 15 purged self-invented anti-patterns.

## Alternatives Considered

Allowing ad-hoc prototypes into Sim before checking SOTA provenance: rejected because uncited prototypes leak into Paper manifests and burn multiple-testing (`HLZ`) trials.

## Consequences

- **Positive:** 100% end-to-end Tier-0 provenance parity across Sim (`ADR-0037`), Paper `P1` (`ADR-0033`), Paper `P2..P4` + Stage-5 Live + `merge_gate.py` (`ADR-0092`).
- **Negative / Safeguards:** Every future trial must cite its pinned SOTA upstream repo SHA + primary paper up front.
