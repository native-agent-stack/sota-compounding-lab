# ADR-0007: Stationary 56-Thread CPCV Offload, `dirhash` Snapshot Locks & `learnharness` Postmortem Compounding

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/lifecycle.py`, `src/sota_compounding/strategy_template.py`, `scripts/architecture_catalog.py`, `scripts/org_visibility_doctor.py`

## Context

As the pre-registered hypothesis ledger scaled past `N >= 86` cumulative trials (`t*(87) = 4.1657`) across the `424`-entry research-converged catalog (`404` owner-starred + `20` evaluated repositories in `Meridian` `docs/adr/data/ecosystem-manifest.md`), multi-path `CombinatorialPurgedCV` (`CPCV`) sweeps, nanosecond `RiskAdverseQueueModel` replay, and post-trade failure distillation needed deterministic snapshot hashing and seamless dual-workstation Orca routing without clobbering active `24x5` paper leases.

## Decision

1. **`golang/mod` `dirhash`-Style Input Snapshot Locking (`ADR-0004` Shelf Parity):** Every Stage-2 pre-wall (`<= 2026-10-15`) evaluation binds a deterministic 64-hex `input_package_sha256` (`golang/mod` `dirhash` sorted-tree SHA-256 discipline) alongside `40`-hex pinned upstream commits (`skfolio/skfolio`, `ml4t/diagnostic`, `polars`, `duckdb`, `nautechsystems/nautilus_trader`, `nkaz001/hftbacktest`, `dgunning/edgartools`), keeping external research frameworks (`microsoft/qlib`, `QuantConnect/Lean`, `polakowo/vectorbt`, `openbb-finance/OpenBB`) strictly `PHILOSOPHY_ONLY` unless pinned on the local upstream shelf.
2. **CTA/UTP Round-Lot Queue Scaling & Dynamic HLZ Hurdle:** Passive-limit pre-wall simulations must scale `SIP` round-lot quote sizes (`bs * 100.0`, `as * 100.0`) inside `hftbacktest::RiskAdverseQueueModel` (`Canon 0169..0176`, `Trial #85`) and clear both `1.0x`/`2.0x` cost stress and the live `sota-bus` Harvey–Liu–Zhu `t*(N)` hurdle (`DSR >= 0.95`, `PBO <= 0.30`).
3. **Dual-Workstation Orca Co-op Offloading (`stablyai/orca` + `ComposioHQ/agent-orchestrator`):** Combinatorial `CPCV` and GPU-heavy feature sweeps route to the stationary `64`-thread / `128 GB` / RTX 4090 `Noesis` Orca runtime (`orca serve --no-pairing --refuse-public` + clean `Ubuntu 26.04.1` `Noesis-Carbon` WSL2 guest) with automatic `14`-thread Apple M5 Max (`64 GB`) offline failover and remote `sota-bus` lease-safe sync (`singleActiveBrokerHostOnly: true`).
4. **Structured `learnharness` / `LoopForge` Postmortem Distillation:** Rejected pre-wall trials and live Paper `MAE`/`MFE` outliers are distilled into failure-clustered root-cause rules (`amosblomqvist/learnharness`, `heohlee/learnharness-git`, `Tencent/LoopForge` pattern) in the private vault ledger (`native-agent-stack/sota-private-vault`) while keeping public `sota-compounding-lab` 100% free of proprietary weights, account IDs, or host paths.
