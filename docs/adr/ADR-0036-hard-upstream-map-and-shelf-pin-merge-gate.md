# ADR-0036: Fail-Closed `upstream-map.toml`, 18-Pin Shelf Parity & Zero-Placeholder Merge Gate for `sota-compounding-lab`

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Stage-2 Pre-Wall Lab Source Provenance & Hard Upstream-First Firewall (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`)
- **Evidence Grade:** E4 (`upstream-map.toml`, `scripts/merge_gate.py`, `tests/test_upstream_hard_gate.py`, `pytest`, `ruff`, `architecture_catalog.py --check`, `merge_gate.py`, `gitleaks 0`)
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/native-agent-stack/sota-upstream-shelf-lock
- **Revisit Trigger:** Addition of any new module under `src/sota_compounding/` or `scripts/`.

## Context and Problem Statement

An ecosystem-wide provenance audit (`2026-11-20`) identified that `sota-compounding-lab` maintained `UPSTREAM.md` without a mechanical `upstream-map.toml` or `scripts/merge_gate.py` verifier, unlike `meridian`, `astrolabe`, and `sota-mover-ranker`. Enacting the owner's hard upstream-source mandate required closing that mechanical gap across every tracked implementation file under `src/sota_compounding/` and `scripts/`.

## Decision Outcome

1. **Add `upstream-map.toml`** mapping 100% of tracked implementation files in `src/sota_compounding/` and `scripts/` (`install`, `integrate`, or `build-on`) to maintained SOTA libraries registered in `UPSTREAM.md`, together with `[shelf_pins]` pinning all 18 read-only shelf clones in `~/Documents/upstream/`.
2. **Ship fail-closed `scripts/merge_gate.py` & `tests/test_upstream_hard_gate.py`:**
   - Reject any unmapped file, unused pattern, banned low-quality library (`yfinance`, `backtrader`, `backtesting.py`, `mlfinlab`, `tcapy`, `vollib`, `tradingview_ta`), or non-library placeholder (`cpython-stdlib`, `academic-momentum-literature`).
   - Verify exact 40-char SHA parity and clean tree on all 18 read-only shelf clones when present on disk, zero account-ID/secret/home-path leaks, and zero-drift `docs/adr/CATALOG.md`.

## Alternatives Considered

- **Rely solely on `UPSTREAM.md` prose documentation:** Rejected — prose registers drift as new trial runners and strategy modules land.

## Verification Witness

- `pytest -q tests/test_upstream_hard_gate.py` (`PASS`).
- `python scripts/merge_gate.py` -> `[MERGE-GATE PASS]`.
- `python scripts/architecture_catalog.py --check` (`36/36` ADRs, `0` drift).
- `ruff check scripts/merge_gate.py tests/test_upstream_hard_gate.py` (`0` errors) & `gitleaks protect --staged --no-banner` (`0` leaks).
