# ADR-0005: SHA-Pinned GitHub Actions CI & Isolated Git Worktree Governance

- **Status:** Accepted
- **Date:** 2026-11-17
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `.github/workflows/ci.yml`, `scripts/architecture_catalog.py`

## Context

Concurrent multi-model peer agents (`AGY`, `Claude Code`, `Codex`, `DSH`) iterate simultaneously across shared repositories. Unpinned third-party GitHub Actions or shared-working-tree edits create supply-chain and merge-collision risk.

## Decision

1. Pin all external GitHub Actions in `.github/workflows/ci.yml` to immutable 40-character commit SHAs (`actions/checkout@11d5960a326750d5838078e36cf38b85af677262`, `actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065`) with `permissions: contents: read`.
2. Author and verify multi-file changes in isolated `git worktree` branches coordinated via atomic SQLite WAL leases (`peer_bus.db`) before fast-forwarding `main`.

## Verification Witness

- **Cataloged Implementation & Test Artifacts:** `.github/workflows/ci.yml`
- **Deterministic Gates:** Verified by `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py` (`UPSTREAM.md` + `upstream-map.toml` + `18` shelf pins), and `pytest -q`.
