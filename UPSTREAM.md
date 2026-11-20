# Upstream SOTA Frameworks & 18-Pin Quantitative Shelf Lineage (`v3.12.0`)

Repository: `https://github.com/native-agent-stack/sota-compounding-lab` · Plane & Lane: `T3 / T4_T5_COMPOUNDING_GUARDIAN_VAULT` · DataFeed Contract: `sota-ranking-datafeeds.v3.10` (`DF1..DF12` + `DF1..DF9` compat) · Shelf Lock: `sota-upstream-shelf-lock` (`ADR-0001..0022`, `QUALITY_AND_SOTA_ONLY_NO_LICENSE_EXCLUSION`).

## 1. Multi-Model Runtime Workers & Active Orchestration Stack (`v3.12.0`, Zero Stale / Zero Kimi; `goose` & `grok` Retired)

- **Cross-Machine Supervisor (`stablyai/orca` `v1.4.263`+):** Symmetric `:6770` `Mac M5 Max (100.93.211.101) <-> Noesis (100.110.115.22)` E2EE mesh, `sota-orca-lane-hub.v2` `trading-lanes` / `harness-lanes` folder hubs, `worker-start --on 100.110.115.22`, and RTH broker contention shield (`singleActiveBrokerHostOnly = true`).
- **Sovereign Lead Orchestrator (`anthropics/claude-code` + `Yeachan-Heo/oh-my-claudecode`):** `claude-fable-5-2` (`xhigh`, `OMC_RUNTIME_V2=1`, `oneModelPerConversation=true`) + `claude-opus-5-5` (`high`) + `claude-sonnet-5-5` (`high`).
- **Cross-Family Adversarial Critic (`openai/codex` + `Yeachan-Heo/oh-my-codex` + `max-sixty/worktrunk`):** `gpt-6.2-astra` (`xhigh` critic) & `gpt-6.2-sol` (`high` implementation/pytest) in isolated git worktrees.
- **8-Slot Lifecycle & Peer Spine (`ComposioHQ/agent-orchestrator` + `aannoo/hcom` + `steveyegge/beads` + `omniroute`):** 8-slot cross-vendor `AO` convergence, SQLite WAL `peer_bus.db`, `hcom` PTY push & `bd` DAG tracking.
- **Schema-Typed Python DAGs (`google-antigravity/antigravity-sdk-python`):** `21` canonical `run_workflow` Python DAGs (`gemini-4-neon-xhigh` / `gemini-4-carbon-high` 1M context) + Native Swarm `workspace:share`.
- **Non-GPT Polyglot Graph Runtime (`code-yeongyu/oh-my-openagent`):** `claude-sonnet-5-5` / `deepseek-v4.1-flash` `Hephaestus` (`forbids_gpt_on_hephaestus=true`).
- **Role-Selected Thinking Workers (`can1357/oh-my-pi`):** `gpt-6.2-sol` (`--thinking=high`) explicit workers (`goose` and `grok` retired from the active required fleet).
- **Open-Weights Pre-Wall Evaluator (`native-agent-stack/dsh-sota-harness`):** `deepseek-v4.1-flash` (`deepseek-flash`, 1M context) 10-layer offline probes & `<= 2026-10-15` pre-wall batch evaluator.

## 2. 18 Immutable `40`-Hex Quantitative & Execution Upstream Shelf Pins (`DF1..DF12`, `ADR-0001`)

- `alpacahq/alpaca-py` @ `c42add7c0a7026e2ead3ed64e76f4def2b99fda4` (`v0.45.0` — `DF1` SIP, `DF2` BOATS / Dec-6 `23/5`)
- `alpacahq/alpaca-mcp-server` @ `63cc4278d87b31f3ed4efcfaa55624095c6d7be8` (`v2.4.1`)
- `nautechsystems/nautilus_trader` @ `da6b57aefdd62b44f675e434ffcfefca7f4025dd` (`v2.0.0rc9` — `Stage 1` / `:4002` simulation)
- `gmatrunich/nautilus-alpaca` @ `59685d56631b981018a189b5ced1a7000a50bfbd` (`main`)
- `pola-rs/polars` @ `7f902bd3ffd27ff82d14278a09a295ff2f7aed92` (`py-2.1.0` — `DF3` `RAW`-vs-`ALL`, `DF10` 20-session normal-rate lake, `64`-thread `Noesis-Carbon`)
- `duckdb/duckdb` @ `d54cbbe3df17e8e5ce68312f483b4155836112aa` (`v2.0.0` — `DF7` FINRA RegSHO `T-1`)
- `dgunning/edgartools` @ `574c2bd15f6c629bceef03b6550ea8ac8abf3473` (`v6.2.0` — `DF5` SEC `acceptanceDateTime`, `DF6` XBRL public float)
- `ml4t/diagnostic` @ `19e76072e64a18c9ba56130e4a7bcc0bc871f57e` (`v0.2.0` — `DSR` / `PBO` / `MinTRL`)
- `skfolio/skfolio` @ `90e3fb907fab39d244d899f3caf4e9676889309c` (`v1.12.7` — `CombinatorialPurgedCV`)
- `nkaz001/hftbacktest` @ `5f3ec40b2afb764e0fea112f941ed85523ef4e88` (`main` — `RiskAdverseQueueModel` `100x` round-lot queue, `DF11`)
- `ib-api-reloaded/ib_async` @ `ab629f34c1823ea4c1542f32f07377208a86bdfd` (`v2.1.0` — `DF9` IBKR `:4002` Paper `L1`)
- `finos/perspective` @ `0264ec13cd28ef4a2a136c8ef021c3c112bc8990` (`v5.9.2`)
- `tradingview/lightweight-charts` @ `c29c51e1e5bc61981da5832c97797cb8283d120f` (`master`)
- `ranaroussi/quantstats` @ `d1a214fb906e76f974f45cdccf923b601c67a756` (`main` — `DF12` tear-sheet)
- `polakowo/vectorbt` @ `a52a984b631624bb9daf41cb19f532fe559cf0ef` (`v1.2.0`)
- `unionai-oss/pandera` @ `02b375ab43f39f8c5c5108695c1791fa04f99a26` (`v0.35.0`)
- `gerrymanoim/exchange_calendars` @ `dbe38b1f6887434bbdd1a7d2df6ff8f1742a048a` (`4.13.2` — `DF4` `XNYS` `RVOL_tod`, `DF8` Rule 4560 `T+8`)
- `statsmodels/statsmodels` @ `278ff9950636cdd4939b4055e339a8e681d79cab` (`v0.15.0` — date-clustered HAC SEs)

## 3. Quality-First Frontier SOTA Upstreams (`sota-upstream-shelf-lock` `ADR-0020..0022`)

- `bashtage/arch` @ `d3e858fc1c99932bb6936cbaf0d1b45b136266e5` (`NCSA` — Hansen `SPA` + `HLN MCS`)
- `microsoft/qlib` @ `54355232463878d2eebb91fe0ee5fa7fa1f5976c` (`v0.9.8` `MIT` — AST-hardened `Alpha158` / `DF12`)
- `openbq-org/OpenBB` @ `5a7517ae8bc74ecdcd1a549ed26df8ad131b02d0` (`v5.0.0` `Apache-2.0`) & `openbq-org/backends-for-openbb` @ `a6293707576e16edda8305adda95b07b6a4b968b` (`MIT`)
- `Nixtla/conformalforecast` @ `59922b75adbc377342f68c453ff0d8fc6dcc7532` & `scikit-learn-contrib/MAPIE` @ `e4c4f5f7daeaa7c8b09323852ad1997e506891a2` (Conformal PID stops `<= 2.00%`)
- `NVIDIA/cudf` @ `ece1ff92c8e5a61ac87f6cc43ef5cf763659377d` (Noesis-Carbon RTX 4090 GPU acceleration)
- Quality-first experimental shelf (`ADR-0022` / `ADR-0031`): `cvxgrp/cvxportfolio`, `man-group/ArcticDB`, `google-research/timesfm`, `xtdb/xtdb`, `ml4t/backtest` (strict technical-quality rejections retained: `yfinance`, `backtrader`, `backtesting.py`, abandoned `mlfinlab`).
