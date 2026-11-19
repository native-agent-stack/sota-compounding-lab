# ADR-0018: Deterministic Bitemporal `ts_recv_ns` Availability Sidecar Contract (`E4` Strict Native vs `E2` Bounded-500ms Exploratory Projection)

- **Status:** Accepted
- **Date:** 2026-11-18
- **Layer:** Pre-Wall Statistical Simulation (`<= 2026-10-15`), Bitemporal Point-in-Time Provenance & `H099` Replay Governance (`L8` × `T3`)
- **Evidence Grade:** E4 (Deterministic & Hypothesis TDD Suite + `T3` `#363` Audit Finding Resolution + `gitleaks 0`)
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/native-agent-stack/native-agent-stack, https://github.com/databento/databento-python
- **Revisit Trigger:** Native `ts_recv_ns` Databento/SIP backfill completion or `v4.0.0` promotion

## Context and Problem Statement

During live `Noesis` pre-wall verification (`#363`, `noesis-codex:orca-task-7bad736791cd`), `T3` confirmed that all `29,500` Parquet partitions (`14,750` trades + `14,750` quotes) in `Z:/sota-data/prewall-lake/published-orb` satisfy the strict `<= 2026-10-15` ET cutoff, but carry exchange event time `t_ns` without a native per-record feed receipt timestamp `ts_recv_ns`. `T3` correctly failed closed rather than fabricating native receipt timestamps or running unverified look-ahead replays.

## Decision Outcome

- Introduce `src/sota_compounding/bitemporal_sidecar.py` enforcing a two-tier, zero-fabrication bitemporal contract:
  1. **`E4_NATIVE_BITEMPORAL_RECEIPT` (Default Strict Mode):** Requires native `ts_recv_ns` in the partition schema; required for `standalone_live_promotion_eligible = True`. Fails closed whenever `ts_recv_ns` is missing.
  2. **`E2_BOUNDED_LATENCY_PROJECTION` (Explicit Exploratory Opt-In):** Derives a conservative worst-case SIP propagation ceiling (`ts_recv_bound_ns = ts_event_ns + 500ms`, clamped to `[50ms, 5000ms]`) with SHA-256 sidecar provenance and `standalone_live_promotion_eligible = False`, unblocking pre-registered exploratory `H099` CPCV / `1x`/`2x` Almgren-Chriss studies without ever misrepresenting `t_ns` as native `ts_recv_ns`.
- Fail closed unconditionally on any `max_event_ts_ns > 2026-10-15` ET or non-finite/out-of-bounds latency bound.

## Alternatives Considered

- **Silently aliasing `ts_recv_ns = t_ns`:** Rejected because exchange timestamp equals zero-latency look-ahead and violates `DF11` bitemporal integrity.

## Verification Witness

- `uv run python scripts/architecture_catalog.py --write && --check` (`18/18` MADR 4.0 ADRs, `0` drift).
- `uv run pytest -q && ruff check . && ty check && python3 scripts/merge_gate.py && gitleaks git --no-banner` (`PASS`, `0` leaks).
