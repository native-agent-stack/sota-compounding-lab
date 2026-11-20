# ADR-0021: Converged Multi-Strategy Foundation Factory, Retired-Trap Firewall & Compounding Learning Reflection Engine

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/converged_strategy_factory.py`, `tests/test_converged_strategy_factory.py`

## Context

After consolidating all 8 trading-engine repositories, 18 pinned upstream libraries, 17 quant skills, starred research frameworks (`Qlib`, `Kronos`, `TimesFM`, `NautilusTrader`, `hftbacktest`, `skfolio`, `ml4t-diagnostic`, `VectorBT`), and 101 pre-wall empirical trials (`H001`–`H102`), the ecosystem needed a single pure-Python open-core factory contract that:
1. Registers all 6 converged strategy archetypes (adding `IPO_DAY1_MICROSTRUCTURE_LOSSLESS` and `SLOW_COIL_VDU_ACCUMULATION` alongside the canonical four),
2. Programmatically blocks retired negative-expectancy traps (`H101` already-`POST` chases, `H092`/`H093` naive overnight carry, `H094` unshielded lit passive queue, `H096` relaxed `OFI`-only, `09:30–09:35 ET` opening taker chases),
3. Closes the **Compounding Learning Reflection** feedback loop (dimensionless capture efficiency, premature-stopout rate, Glosten–Milgrom slippage attribution, Empirical-Bayes UCB tuning, and bounded `ADR-0022 <= 2.00%` parameter updates with SHA-256 provenance), and
4. Generates study-wall-locked (`<= 2026-10-15`) batch factory plans for Stationary Workstation `Noesis` (64c / RTX 4090).

## Decision

1. Ship `src/sota_compounding/converged_strategy_factory.py` implementing:
   - `CONVERGED_FOUNDATION_CATALOG` (`6` archetypes) and `STARRED_FRAMEWORK_ADAPTERS` (`8` research/production adapters),
   - `check_retired_trap_firewall()` failing closed on all 5 proven negative-EV traps,
   - `reflect_and_finetune_batch()` emitting PII-free, SHA-256-sealed `CompoundingReflectionDigest` objects strictly respecting `ADR-0022 <= 2.00%`,
   - `build_noesis_prewall_factory_plan()` enforcing `study_wall_cutoff <= 2026-10-15` and dynamic `HLZ` `t*(N+1)` hurdles.
2. Add deterministic & Hypothesis property tests in `tests/test_converged_strategy_factory.py` and re-export via `src/sota_compounding/__init__.py`.

## Verification

- `python scripts/architecture_catalog.py --check`, `ruff check .`, `ty check src/sota_compounding`, `pytest -q`, `python scripts/org_visibility_doctor.py`, and `gitleaks git --no-banner` (`0` leaks).

## Verification Witness

- **Cataloged Implementation & Test Artifacts:** `src/sota_compounding/converged_strategy_factory.py`, `tests/test_converged_strategy_factory.py`
- **Deterministic Gates:** Verified by `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py` (`UPSTREAM.md` + `upstream-map.toml` + `18` shelf pins), and `pytest -q`.
