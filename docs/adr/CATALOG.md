# Architecture Decision Record Catalog (`sota-compounding-lab`)

Canonical MADR 4.0 decision index for `https://github.com/native-agent-stack/sota-compounding-lab`. Verified fail-closed in CI by `python scripts/architecture_catalog.py --check`.

| ADR | Title | Status | Date | Evidence / Enforced By |
| :--- | :--- | :--- | :--- | :--- |
| [ADR-0001](ADR-0001-public-open-core-vs-private-vault-org-firewall.md) | Public open-core vs. private execution & vault organization firewall | Accepted | 2026-11-17 | `scripts/org_visibility_doctor.py`, `scripts/architecture_catalog.py` |
| [ADR-0002](ADR-0002-five-stage-sim-backtest-paper-finetune-broker-pipeline.md) | Five-stage Sim -> Pre-Wall Backtest -> Paper -> Empirical-Bayes -> Broker pipeline | Accepted | 2026-11-17 | `src/sota_compounding/lifecycle.py`, `tests/test_compounding_lifecycle.py` |
| [ADR-0003](ADR-0003-prewall-hlz-dsr-cpcv-statistical-gate.md) | Pre-wall (`<= 2026-10-15`), Harvey–Liu–Zhu `t*(N)` & native-frequency `DSR` gate | Accepted | 2026-11-17 | `hlz_dynamic_t_hurdle`, `evaluate_full_lifecycle` |
| [ADR-0004](ADR-0004-dual-paper-venue-routing-and-adr0022-stops.md) | Dual paper venue routing (`200/min` Alpaca + `3,000/min` IBKR `:4002`) & `ADR-0022` stops | Accepted | 2026-11-17 | `quantize_nms_stop`, `export_broker_strategy_manifest` |
| [ADR-0005](ADR-0005-sha-pinned-ci-and-isolated-worktree-governance.md) | SHA-pinned GitHub Actions CI & isolated git worktree governance | Accepted | 2026-11-17 | `.github/workflows/ci.yml` |
| [ADR-0006](ADR-0006-research-converged-strategy-templates-and-7feed-pit-contract.md) | Research-converged strategy templates, non-repainting Pine v6 & 7-feed PIT contract | Accepted | 2026-11-18 | `src/sota_compounding/strategy_template.py`, `templates/`, `tests/test_strategy_template.py` |
