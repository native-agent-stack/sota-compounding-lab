# ADR-0039: Tier-0 Official Upstream Tearsheet Export & Hosted FINOS/OpenBB Workspace Integration (Zero Self-Built UI)

- **Status:** Accepted
- **Date:** 2026-11-21
- **Layer:** Backtest Visualization & Hosted Institutional UI (`L2` / `L6` — `scripts/export_upstream_tearsheets.py`, `upstream-map.toml`, `tests/test_adr0039_upstream_tearsheets.py`, Disk `Z:/sota-data/tearsheets/`, hosted `openbq-org/finos-workspace` `:1420` + `openbq-org/finos-workspace-plugins` + `openbq-org/backends-for-openbb` `:6900`), compounding upon `ADR-0036`, `ADR-0037`, `ADR-0038`, and `native-agent-stack/ADR-0047` / `ADR-0049` (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`, `hlz_trial_consumed = False`)
- **Evidence Grade:** E4 (Authentic full-resolution pre-wall `<= 2026-10-15` execution emitting 3 unmodified official upstream HTML tearsheets on Disk `Z:/sota-data/tearsheets/`: `nautilus_v2_prewall_tearsheet.html` from `nautilus_trader v2.0.0rc9` `BacktestEngine` `4,074` iterations / `148` orders / `148` fills / `74` positions vs `SPY (Pre-Wall)`, `vectorbt_prewall_portfolio.html` from `vectorbt v1.2.0` `2x` cost stress Sharpe `1.3796`, and `quantstats_prewall_tearsheet.html` from pinned `ranaroussi/quantstats v0.0.86` with fail-closed banned-feed blocker `status=OK`; official `openbq-org/finos-workspace@a0047d5` 5-container Docker Compose stack live on `:1420`/`:8000` with verified `plotly-charts@2.5.0` and `ag-grid@2.3.0` plugins connected to `:6900`; `ruff`, `pytest`, `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py`, `gitleaks 0`)
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/nautechsystems/nautilus_trader/commit/da6b57aefdd62b44f675e434ffcfefca7f4025dd, https://github.com/polakowo/vectorbt/commit/a52a984b631624bb9daf41cb19f532fe559cf0ef, https://github.com/ranaroussi/quantstats/commit/d1a214fb906e76f974f45cdccf923b601c67a756, https://github.com/openbq-org/finos-workspace/commit/a0047d5, https://github.com/openbq-org/finos-workspace-plugins/commit/d9b57e9014110e21d2ccef7c211f479558ca0c20, https://github.com/openbq-org/backends-for-openbb/commit/a6293707576e16edda8305adda95b07b6a4b968b
- **Revisit Trigger:** Adoption of an additional official upstream visualization plugin in `openbq-org/finos-workspace-plugins` or `nautilus_trader` tearsheet schema upgrade.

## Context and Problem Statement

Under the Tier-0 Upstream-Only Mandate (`ADR-0036`, `ADR-0037`, `ADR-0038`, `native-agent-stack/ADR-0047`, `ADR-0049`), self-built custom HTML/JS backtest cockpits violate anti-self-invention governance and create maintenance waste. Furthermore, hosted OpenBB Cloud (`pro.openbb.co`) redirects to the waitlisted BrightQuery portal, making local hosting of the official Apache-2.0 `openbq-org/finos-workspace` UI + official widget plugins + pure unmodified upstream tearsheets the canonical SOTA path for US equities and options research.

## Decision Outcome

1. **Zero Custom HTML/JS — Pure Official Upstream Tearsheet Exporter (`scripts/export_upstream_tearsheets.py`):**
   - Invokes ONLY unmodified official upstream report generators on split-adjusted pre-wall (`<= 2026-10-15`) US equity data with fail-closed `ADR-0022` (`0.05% <= stop_pct <= 2.00%`) and `end_date <= 2026-10-15` guards:
     - `nautilus_trader.analysis.tearsheet.create_tearsheet(..., config=TearsheetConfig(theme="nautilus_dark"))` (`nautilus_trader v2.0.0rc9`, real `Equity` instruments + `Strategy`, fail-closed on `iterations <= 0` or `fills <= 0`).
     - `vectorbt.Portfolio.from_signals(...).plot(template="plotly_dark").write_html(...)` (`vectorbt v1.2.0`, `2x` cost stress).
     - `quantstats.reports.html(...)` from immutable read-only `~/Documents/upstream/quantstats` (`ranaroussi/quantstats`) via ephemeral `uv run --no-project` with an in-memory fail-closed blocker sentinel preventing banned external feed imports without mutating `~/Documents/upstream/`.
   - Outputs standalone HTML artifacts exclusively to Disk `Z:/sota-data/tearsheets/` (served locally via stdlib `http.server` on `127.0.0.1:8793`).
2. **Official Hosted `openbq-org/finos-workspace` + Plugins (`L6`):**
   - Hosts official `openbq-org/finos-workspace@a0047d5` (`Apache-2.0`) locally via Docker Compose on `http://localhost:1420` (backend API `:8000`) connected to the official `openbq-org/backends-for-openbb` FastAPI contract on `http://127.0.0.1:6900`.
   - Deploys verified official `plotly-charts@2.5.0` (`openbq-org/finos-workspace-plugins@d9b57e90`) and `ag-grid@2.3.0` (`openbq-org/finos-workspace/docs/examples/ag-grid`) plugins validated by `scripts.plugin_check --digest`.
3. **Permanent Removal of Custom UI Waste:**
   - Any custom HTML/JS backtest UI files are permanently prohibited; `scripts/export_upstream_tearsheets.py` is explicitly mapped under `nautilus_trader` (`how = "integrate"`) in `upstream-map.toml` and verified by `tests/test_adr0039_upstream_tearsheets.py`.

## Alternatives Considered

1. **Custom FastAPI + Hand-Written HTML/JS Dashboard:** Rejected and reverted under Tier-0 Upstream-Only Mandate (`ADR-0047` / `ADR-0049`).
2. **Hosted `pro.openbb.co` Cloud Portal:** Unavailable/waitlisted (redirects to BrightQuery coming-soon page).

## Consequences

- **Positive:** 100% authentic institutional tearsheets and interactive workspace UI directly from pinned upstream open-source projects with zero custom frontend maintenance debt and strict pre-wall / `ADR-0022` guardrails.
- **Negative / Safeguards:** Requires local Docker Desktop (WSL2 data disk isolated to `Z:/sota-cache/docker-wsl`) for `finos-workspace` and `meridian/.venv` + `uv` for full-resolution tearsheet regeneration; both fail loudly if prerequisites are unavailable.

## Verification Witness

- **Real Pre-Wall Tearsheets on Disk `Z:/sota-data/tearsheets/`:** `nautilus_v2_prewall_tearsheet.html` (`4,074` iterations, `148` fills, `74` positions), `vectorbt_prewall_portfolio.html` (`2x` Sharpe `1.3796`), `quantstats_prewall_tearsheet.html` (`status=OK`).
- **Deterministic Gates:** `tests/test_adr0039_upstream_tearsheets.py`, full `pytest`, `ruff check`, `scripts/architecture_catalog.py --check`, `scripts/merge_gate.py`, and `gitleaks 0`.
