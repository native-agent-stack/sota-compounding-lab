# ADR-0002: Five-Stage Sim -> Pre-Wall Backtest -> Paper -> Empirical-Bayes -> Broker Pipeline

- **Status:** Accepted
- **Date:** 2026-11-17
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/lifecycle.py`, `tests/test_compounding_lifecycle.py`

## Context

Multi-model agent research (`AGY`, `Claude Code`, `Codex`, `DSH`, `Orca`) generates candidate factors rapidly. Promoting strategies directly from small-sample simulations or single-day unrealized MTM ticks introduces severe multiple-testing and execution-illusion risk.

## Decision

Enforce a sequential 5-stage state machine in `evaluate_full_lifecycle`:
1. **`SIM`:** Causal simulation (`causal_no_lookahead`, `>= 50` events, positive gross expectancy).
2. **`PREWALL_BACKTEST`:** Historical discovery strictly `<= 2026-10-15` passing Harvey–Liu–Zhu `t*(N)`, `5`-path `CPCV` (`PBO <= 0.30`), native `DSR >= 0.95`, and `2x` transaction + market-impact costs.
3. **`PAPER_EXECUTION`:** Pure `REAL_BROKER_ONLY` paper fills (`>= 30` closed round-trips across `>= 10` sessions with positive realized `R`).
4. **`EMPIRICAL_BAYES_FINETUNE`:** Shrunk `eb_r` + UCB priority weight (`[0.40, 1.65]`) with realized-`R` damping on losing lanes (`stop_atr_mult <= 1.98`).
5. **`BROKER_MANIFEST_EXPORT`:** SHA-256 signed JSON manifest targeting sanctioned paper venues only.
