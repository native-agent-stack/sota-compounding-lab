# ADR-0011: v3.9.0 9-Feed `sota-ranking-datafeeds.v3.8` (`DF1–DF9`), Orca Lane Hubs v2 & `UPSTREAM.md` Lineage

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `UPSTREAM.md`, `src/sota_compounding/ranking_and_datafeeds.py`, `scripts/architecture_catalog.py`

## Context

Following `ADR-0010`, finalizing `v3.9.0` across the `native-agent-stack` ecosystem (`sota-multi-model-workflows` `ADR-0016`, `sota-harness-foundation` `ADR-0031`) required binding `sota-compounding-lab` (`T3_prewall_cpcv_gpu_and_strategy_factory` / `T4_T5_COMPOUNDING_GUARDIAN_VAULT`) to the **`9`-feed `sota-ranking-datafeeds.v3.8` (`DF1..DF9`)** contract, the empirical `64`-hex `prewallLakeSha256` (`a253421e873441740d5d4b410d993cfa7345fe34b88ec4ab7f4eecd537b3f234`) on `Z:/sota-data/prewall-lake`, `sota-orca-lane-hub.v2`, and root `UPSTREAM.md` SOTA framework & 18-pin provenance.

## Decision

1. Add root `UPSTREAM.md` mapping all `8` SOTA multi-model orchestration frameworks (`17` binaries, `11` workers, `7` vendors) and `18` `40`-hex quantitative upstream pins to `DF1..DF9`.
2. Route `56`-thread `CPCV` / `DSR` pre-wall (`<= 2026-10-15`) sweeps via `sota-orca-lane-hub.v2` (`folder:c4f93f9e-a710-4599-ac95-892082b03ec8`) onto `Noesis` / `* Noesis-Carbon` (`ext4`) under two-track promotion (`ALPHA_STRATEGY_MINOR` vs. `PROTECTIVE_FILTER_OR_GOVERNANCE`, `HLZ N >= 90`).

## Verification

- `uv run python scripts/architecture_catalog.py`, `uv run ruff check .`, `uv run pytest -q`, and `gitleaks detect --no-git --no-banner` (`0` leaks).
