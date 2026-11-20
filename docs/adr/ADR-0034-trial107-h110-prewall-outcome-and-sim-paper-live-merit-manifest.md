# ADR-0034: Pre-Registered `Trial #107` (`H110`) Real `700`-Session Pre-Wall Residual Lead-Lag & EOD LETF Flow Outcome + Unified Sim -> Paper -> Live Real Performance & Merit Resolution Manifest

- **Status:** Accepted
- **Date:** 2026-11-20
- **Layer:** Advanced Pre-Wall Sim (`S1/S2`, `<= 2026-10-15`) -> Real Paper Crucible (`P1–P4`, `W5`) -> Stage-5 Live (`:8798`/`:8796`) Merit Resolution Governance (`PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`)
- **Evidence Grade:** E4 (`scripts/run_h110_trial107_residual_leadlag_and_letf_closeflow.py`, `docs/TRIAL107_H110_PREWALL_RECEIPT.json` SHA `0da4011809ca9c48...`, `src/sota_compounding/sim_to_paper_to_live_merit_manifest.py`, `scripts/generate_sim_paper_live_merit_manifest.py`, `docs/SOTA_SIM_TO_PAPER_TO_LIVE_MERIT_MANIFEST.{json,md}`, `tests/test_adr0034_trial107_h110_and_sim_paper_live_merit_manifest.py`, `sota-bus` Trial `#107` pre-registration SHA `21c9dd22bf96...`, E4 lesson `429b115446412e05`, `pytest`, `ruff`, `ty`, `org_visibility_doctor.py`, `gitleaks 0`)
- **GitHub Evidence:** https://github.com/native-agent-stack/sota-compounding-lab, https://github.com/native-agent-stack/sota-mover-ranker, https://github.com/native-agent-stack/sota-upstream-shelf-lock
- **Revisit Trigger:** Pre-registration of `Trial #108` (`HLZ t*(108) = 4.2148`) on full-universe (`~5,000`-symbol) multi-day selective swing / PEAD-VCP-IBS / low-turnover beta-neutral rebalance strategies, or `W5` Paper cohort clearing `PF >= 1.25` and `Payoff >= 1.10x`.

## Context and Problem Statement

1. **Executing Pre-Registered `Trial #107` (`H110`) on the Full `700`-Session Pre-Wall Lake:** Following `Trial #105` (`H109`, `ADR-0032`) and `Trial #106` (`H112`, `ADR-0033`), `H110_RESIDUAL_DIRECTED_LEAD_LAG_AND_EOD_LETF_FLOW` (`ADR-0029`, Cartea-Cucuringu-Shi 2026 + Baltussen et al. 2021 JFE + Cheng-Madhavan 2009) passed the `ADIA Lab #19` `MinTRL/MDE` power gate (`D = 675`, `SR_min_ann = 2.574 <= 5.0`, `t*(107) = 4.2127`) and was immutably pre-registered (`sota-bus` SHA `21c9dd22bf96...`) before reading a single return.
2. **Diagnosing the Live & Paper Merit Gap (`2026-11-20 11:11 ET`):** Real broker telemetry showed live `P2` `W5` concentrated strategies crossing `34 >= 30` closed RTs at **`52.9%` WR, `PF 0.59`, `-11.9 bps` mean, `0.46x` payoff**, while `:8798` tripped its `-1.50%` Stage-5 daily drawdown circuit breaker (`-3.577%`, `:8796` `untracked = 0`), and `:8787/api/merit_farm_status` still carried `n=0`/`n=1` `STAGE5_LOSSLESS_FOUNDATION_CANARY_10PCT` exemptions (`live_8798_allowed = true`) without clearing either Pre-Wall HLZ or `W5 >= 30` Paper merit.

## Decision Outcome

1. **Honest Empirical Outcome of `Trial #107` (`H110`, Receipt `0da4011809ca9c48...`, `700` Pre-Wall SIP 1m Sessions `2024-01-02..2026-10-15`, `0` Post-Wall Bytes):**
   - **Primary `H110_PRIMARY_BETA_NEUTRAL_DIRECTED_LEADLAG_PORTFOLIO` (`6,740` events / `337` dates, `14/14` hygiene PASS):** Ultra-low beta-neutral daily volatility (`33.67 bps/d`), `62.98%` 1x WR, and real positive gross edge across all `5/5` grid configs (`+2.40` to `+4.70 bps/d`; canonical `+3.47 bps/d`), yet swamped by daily intraday round-trip costs (`9.65 bps` `1x` / `19.30 bps` `2x`): Net `1x` **`-6.19 bps/d`** (`t = -3.3729`), Net `2x` **`-15.84 bps/d`** (`t = -8.6340`, `DSR = 0.0`, `0/5` CPCV positive, `SPA p = 1.0`). Logged honestly to `sota-bus` as **`FAILED`**.
   - **Diagnostic Arm B (`H110_DIAG_EOD_LETF_CLOSEFLOW_1530_1558`, `1,171` events / `402` dates across `10` liquid ETFs):** Negative even at zero cost (**`-1.09 bps/d` gross**; `-3.88 bps/d` at `2 bps` ETF spread, `-9.88 bps/d` at strict `8 bps`), with exact monotonic secular decay from `+1.34 bps` (`2024`) -> `-4.57 bps` (`2025`) -> **`-8.02 bps` (`2026_PREWALL`)**, confirming `2024–2026` crowding out of mechanical `15:26–15:58` EOD momentum.
   - **Diagnostic Arm C (`H110_DIAG_T1_TO_T0_RESIDUAL_LEADLAG_RTH`, `10,480` events / `524` dates):** Confirmed `69.7%` 1x WR and `+3.73 bps/d` gross (`36.29 bps/d` std), again swamped by daily `9.79 bps` `1x` turnover (`-6.06 bps/d` 1x / `-15.86 bps/d` 2x).
2. **Ship Fail-Closed Unified Sim -> Paper -> Live Real Performance & Merit Resolution Engine (`src/sota_compounding/sim_to_paper_to_live_merit_manifest.py`):**
   - Reconciles all `19` real pre-wall strategies (`Trial #105 H109`, `Trial #106 H112`, `Trial #107 H110` Primary + Diag B/C, and all `14` canonical TradingView Pine v6 `700d` strategies `S01..S14`) alongside live `W5` Paper (`34` RTs) and read-only Live `:8798`/`:8796` state into SHA-256 sealed `docs/SOTA_SIM_TO_PAPER_TO_LIVE_MERIT_MANIFEST.{json,md}`.
   - Enforces **strict governance separation**: **`OWNER_ORDER_LOSSLESS_ADOPTION`** stays always ON to protect every owner-placed Live order (`ADR-0022 <= 2.00%`, `>= 10%` exit reserve, uncapped Chandelier), whereas **`AUTONOMOUS_STRATEGY_LIVE_ENTRY`** strictly requires BOTH Pre-Wall Sim (`HLZ`/`DSR >= 0.95`/`CPCV >= 0.70`/`PBO < 0.25`/`SPA p < 0.05`) AND Real Paper `W5` (`>= 30 RTs`, `>= 10 sessions`, `PF >= 1.25`, `Payoff >= 1.10x`, `0` tail breaches) — **hard-refusing `n=0`/`n=1` seed-canary live exemptions (`0/19` autonomous live eligible today)**.
   - Proves the **Intraday Cost Wall Law**: `15/15` high-turnover intraday strategies (`1m`/`5m`) are net-negative `1x/2x`, while the only `3` positive-`1x` real-data candidates (`S04_MINERVINI_VCP_QULLAMAGGIE` `+16.24` 1x / `+3.72` 2x, `S09_CONNORS_IBS_CONFORMAL_MR` `+13.26` 1x / `+1.54` 2x, `S12_PEAD_DRIFT_COIL_BREAKOUT` `+4.39` 1x) are selective multi-day swing strategies — directing **`Trial #108+`** (`t*(108) = 4.2148`) to full-universe multi-day swing & lower-turnover beta-neutral rebalancing.

## Alternatives Considered

- **Lowering transaction cost assumptions below `8 bps` to force `H110` positive:** Rejected — violates `ADR-0030` `14`-point Low-Quality Purge and contradicts real `P2` Paper TCA (`-6.43 bps` realized entry slippage alone).
- **Retaining `n=0`/`n=1` autonomous live seed canaries:** Rejected — violates Stage-5 Invariant #1 (Paper-Merit Provenance) and directly exposes live capital to unverified negative-EV drift.

## Verification Witness

- `scripts/run_h110_trial107_residual_leadlag_and_letf_closeflow.py` (`700` real pre-wall sessions, `0` post-wall bytes, receipt SHA `0da4011809ca9c48...`, logged `FAILED` to `sota-bus`, E4 lesson `429b115446412e05`).
- `scripts/generate_sim_paper_live_merit_manifest.py` (`19` real strategies reconciled into `docs/SOTA_SIM_TO_PAPER_TO_LIVE_MERIT_MANIFEST.{json,md}`).
- `scripts/architecture_catalog.py --write && scripts/architecture_catalog.py --check` (`34/34` ADRs, `0` drift).
- `pytest`, `ruff check .`, `ruff format --check .`, `scripts/org_visibility_doctor.py`, and `gitleaks protect --staged --no-banner` (`0` leaks).
