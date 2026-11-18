# `sota-compounding-lab` — 5-Stage Quantitative Strategy Compounding & Paper-Broker Manifest Engine

Open-core, zero-secret strategy lifecycle and Empirical-Bayes fine-tuning engine for [`github.com/native-agent-stack`](https://github.com/native-agent-stack), uniting multi-model LLM-native research (`AGY`, `Claude Code`, `Codex`, `DeepSeek Harness`, `Orca`) with deterministic statistical promotion into **Alpaca Paper (`<= 200 req/min`)** and **IBKR Paper (`:4002`, `<= 3,000 msg/min`)**.

## 5-Stage Fail-Closed Lifecycle (`src/sota_compounding/lifecycle.py`)

| Stage | Gate | Enforced Invariants |
| :--- | :--- | :--- |
| **1. `SIM`** | `SimStageMetrics` | Causal features (`causal_no_lookahead`), `>= 50` events, positive gross bps |
| **2. `PREWALL_BACKTEST`** | `PrewallBacktestMetrics` | Strictly `<= 2026-10-15`, Harvey–Liu–Zhu `t*(N)`, `5`-path `CPCV` (`PBO <= 0.30`), native `DSR >= 0.95`, `>= 200` events / `>= 60` dates, `1x` & `2x` net bps `> 0` |
| **3. `PAPER_EXECUTION`** | `PaperExecutionMetrics` | Pure `REAL_BROKER_ONLY` attribution (`0` synthetic `paper_sim` splicing), `>= 30` closed RTs across `>= 10` sessions, `ADR-0022 <= 2.00%` Reg NMS Rule 612 tick stops, `Dec 6 2026` `23/5` SIP regime split |
| **4. `EMPIRICAL_BAYES_FINETUNE`** | `tune_empirical_bayes_ucb` | Shrunk expectancy `eb_r` + UCB priority in `[0.40, 1.65]` (`never_gated`), realized-`R` damping on losing lanes, `stop_atr_mult <= 1.98` |
| **5. `BROKER_MANIFEST_EXPORT`** | `export_broker_strategy_manifest` | Deterministic SHA-256 signed manifest for `paper-api.alpaca.markets` (`PA...`) & `127.0.0.1:4002` (`DU...`); hard-refuses `api.alpaca.markets` and live ports `4001` / `7496` |

## Public vs. Private Organization Firewall (`scripts/org_visibility_doctor.py`)

- **Public Open-Core Tier (Unlimited GitHub Actions CI):** `.github`, `native-agent-stack`, `sota-compounding-lab`, `sota-multi-model-workflows`, `sota-harness-foundation`, `dsh-sota-harness`, `sota-upstream-shelf-lock`.
- **Private Proprietary / Execution / Vault Tier (`100% Private`):** `sota-mover-ranker`, `sota-dynamic-trail-c`, `cc-sota-harness`, `sota-private-vault`, `ecosystem-records`, `trading-sota`.

## Quick Verification

```sh
uv sync --locked
uv run ruff check src scripts tests
uv run python scripts/architecture_catalog.py --check
uv run python scripts/org_visibility_doctor.py --offline
uv run pytest -v
```
