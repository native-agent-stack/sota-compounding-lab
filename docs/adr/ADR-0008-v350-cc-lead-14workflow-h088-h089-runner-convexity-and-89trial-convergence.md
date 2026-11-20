# ADR-0008: `v3.5.0` `cc-lead` 14-Workflow Hub, `H088`/`H089` Asymmetric `50/50` Runner Convexity & 89-Trial Lifecycle Convergence

- **Status:** Accepted
- **Date:** 2026-11-18
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/lifecycle.py`, `src/sota_compounding/strategy_template.py`, `scripts/architecture_catalog.py`, `scripts/org_visibility_doctor.py`

## Context

Following `ADR-0007` (`dirhash` snapshot locks, `56`-thread `Noesis` `CPCV` offload, and `learnharness` postmortem distillation), two pre-registered nanosecond `SIP` pre-wall (`<= 2026-10-15`) causal experiments (`Trial #88 H088` in `sota-harness-foundation` `ADR-0024` and `Trial #89 H089` in `agy-sota-harness` `ADR-0017`) alongside the `14`-workflow `v3.5.0` `cc-lead` central multi-runtime hub (`sota-multi-model-workflows` `ADR-0012` `@ 2cc19d52cfb5`) completed full empirical synthesis across `14,750` symbol-day files (`587` dates).

## Decision

1. **Two-Stage Hybrid Entry × Post-Gain-Lock `50/50` Runner Convexity (`H088` -> `H089` -> Stage-4 Paper):**
   - Single-tranche micro Gain-Locks (`+60 bps`) cut `CVaR_95` by `-44.55%` (`H088`) but censor `97.8%` of right-tail RTH trend runners; pairing Two-Stage Hybrid Entry (`50%` passive `RiskAdverseQueueModel` bid + `50%` passive depletion or `120s` runaway join, `95.50%` participation) with `50/50` Core-Harvest + Zero-Loss Breakeven Runner Convexity (`H089`) expands runner survival `23x` (`2.20% -> 50.75%`) and compresses date-clustered `SE` `-42.4%` (`9.70 -> 5.588 bps`) while keeping every initial stop strictly `<= 2.00%` (`ADR-0022`).
   - Stage-3/4 Paper strategies (`sota-mover-ranker` `ADR-0011` `:8787` and `sota-dynamic-trail-c` `ADR-0010` `:8799`) promote `50/50` Core-Harvest + `+8 bps` breakeven-floored Runner Convexity strictly after green Gain-Lock arming, accruing clean post-wall (`> 2026-10-15`) `PostWallCohortLedger` fills toward `N >= 30` (`Stage 3`) and `N >= 250` (`Stage 5` Empirical-Bayes).
2. **`cc-lead` Central 5-Channel Multi-Runtime Orchestration & Cross-OS `15/15` Parity (`v3.5.0`):**
   - Bind Stage-1..5 research and promotion workflows to the `14`-workflow `v3.5.0` suite (`cc_lead_multi_runtime_workers_workflow.py` `#14` + `org_compounding_promotion_workflow.py` `#5`), enforcing `oneModelPerConversation = true` (`claude-fable-5-2` `xhigh` + `OMC` Runtime v2), `>= 3` concrete frontier vendors (`anthropic`, `openai`, `google`) plus non-GPT `kimi-k3.1-pro` `Hephaestus`, cross-OS `darwin`/`win32` `15/15` binary parity across `Mac M5 Max` and `Noesis`, and dynamic HLZ multiplicity accounting (`N = 89 -> t*(90) = 4.1735`, `52` compiled `E1–E4` lessons).

## Verification & Evidence
- Backfilled under `sota-upstream-shelf-lock ADR-0023` (2026-11-20): anchored to repository `UPSTREAM.md`, `upstream-map.toml`, and git history of `adr0023-19adr-evidence` (`shelf_lock.hard_upstream_rule` verified).
