# ADR-0001: Public Open-Core vs. Private Execution & Vault Organization Firewall

- **Status:** Accepted
- **Date:** 2026-11-17
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `scripts/org_visibility_doctor.py`, `scripts/architecture_catalog.py`

## Context

Within `github.com/native-agent-stack`, public repositories receive unlimited GitHub Actions CI minutes for deterministic linting, type-checking, property-based fuzzing, and ADR drift gates, while proprietary trading signals, broker execution adapters, and empirical strategy ledgers must remain strictly private.

## Decision

1. **Public Open-Core Tier (`visibility == PUBLIC`):** `.github`, `native-agent-stack`, `sota-compounding-lab`, `sota-multi-model-workflows`, `sota-harness-foundation`, `dsh-sota-harness`, and `sota-upstream-shelf-lock`. Public repositories contain zero credentials, zero broker account IDs, zero concrete host home paths, and zero proprietary live alpha weights.
2. **Private Execution & Vault Tier (`visibility == PRIVATE`, fail-closed):** `sota-mover-ranker`, `sota-dynamic-trail-c`, `cc-sota-harness`, `sota-private-vault`, `ecosystem-records`, and `trading-sota`. `org_visibility_doctor.py` hard-fails if any repository in `MUST_BE_PRIVATE` reports `isPrivate == false`.

## Verification Witness

- **Cataloged Implementation & Test Artifacts:** `scripts/org_visibility_doctor.py`, `scripts/architecture_catalog.py`
- **Deterministic Gates:** Verified by `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py` (`UPSTREAM.md` + `upstream-map.toml` + `18` shelf pins), and `pytest -q`.
