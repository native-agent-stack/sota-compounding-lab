# ADR-0026 — Closed-loop paper TCA collar recalibration, temporal firewall and Trial 104

- **Status:** Accepted
- **Date:** 2026-11-19
- **Evidence Grade:** E3
- **Decision owners:** Compounding research and paper execution
- **Enforced by:** `src/sota_compounding/collar_recalibration.py` and `tests/test_adr0026_collar_recalibration.py`
- **Related decision:** [ADR-0025 queue parity reconciler](ADR-0025-live-paper-tca-vs-nautilus-queue-parity-reconciler.md)

## Context and decision drivers

ADR-0025 compares a frozen queue simulation against paper execution. A fleet can report green with zero matched observations, and a lane can report red on a single adverse fill. A status alone cannot establish a useful direction for an entry-price adjustment. Independent samples, a fixed pre-wall anchor and a finite lifetime adjustment budget are necessary. The research ledger has 103 completed trials. A paper-execution recommendation must have no path into factor discovery or trial registration.

## Decision outcome

Adopt `sota.collar-recalibration.v1`, a standard-library paper advisory with a pure state transition and an atomic private journal. A recommendation adjusts only the permitted adverse **entry-price offset**, denominated in basis points. The public contract contains governance bounds and synthetic tests. Operators retain baseline values, session evidence and decision receipts in the private paper namespace.

### Private inputs and source of the anchor

The caller registers a `CollarStratum(strategy_id, account_lane, execution_session, side, liquidity_regime, collar_version)` and a `CollarBaseline`. The baseline comes from the private immutable snapshot of the pre-wall queue simulation and entry policy. Registration requires its collar, simulation engine, snapshot SHA-256 and maximum event date, bounded by 2026-10-15. The importer verifies the immutable snapshot upstream. A date or a well-formed digest alone cannot establish its contents. The accepted paper lanes are `:8787`, `:8799`, `:8797` and `:4002`. Side is BUY or SELL.

| Execution session | Floor | Ceiling |
|---|---:|---:|
| RTH | 5 bps | 25 bps |
| OVERNIGHT_BOATS | 5 bps | 35 bps |
| OVERNIGHT_23X5_SIP | 5 bps | 35 bps |

An optional positive `max_collar_bps` further limits the upper bound. For example, the DF12 handoff has `MAX_COLLAR_BPS = 10.0` in `symbol_ranking_v310.py`. For every finite anchor outside the effective interval, return HOLD with `OBSERVATIONAL_OUT_OF_DOMAIN` and preserve the original number. Existing 2–3 bps opening/core configurations fall into this case. A replacement anchor requires a separate commissioning action using a new policy version and the previous active version. Historical versions cannot be recycled.

### Matched evidence and statistical contract

Build the ADR-0025 inputs separately for each six-field stratum. Feed its `reconcile_fleet_parity()` output to `CollarLedger.consume(epoch, fleet)`. Carry the matched lane receipt SHA-256 and frozen baseline hash in the epoch. Check the actual lane-row checksum, schema, policy and cutoff. The fleet’s worst-lane summary and a snapshot total cannot stand in for a match. The producer owns economic-fill deduplication, assignment to disjoint completed session blocks, denominators including all eligible attempts and misses, and the direction check including missed-fill cost. Mixed cohorts, censored horizons or an unproved session assignment must be marked incomplete.

Each epoch needs ≥30 unique fills, ≥3 distinct completed trading sessions and ≥3 consecutive observation windows with complete denominators. Sample counts must equal the count of supplied positive finite fill weights. The sampling interval must be positive and ≤60 seconds, and the 60-second markout may be at most 6 seconds late. Sub-resolution 1s/10s markouts never trigger a step. Inputs must be fresh, and the observed version and actual collar must acknowledge the current policy and previous recommendation. The caller must establish independence between session blocks. Correlated blocks without a validated grouping are ineligible. An invalid numeric input, boolean in a numeric field or missing evidence gives HOLD. Invalid policy identities and anchors are refused at registration.

Let wᵢ be fill weights, Wⱼ their totals by independent session block, and ρᵤ an upper bound on within-block correlation. The effective-sample requirement is

    n_eff = (Σwᵢ)² / [(1 − ρᵤ)Σwᵢ² + ρᵤΣWⱼ²] ≥ 20.

The default ρᵤ is 1. Positive finite weights and 0 ≤ ρᵤ ≤ 1 are mandatory. Twenty unit-weight fills divided 7/7/6 give n_eff of 20.0000, 12.7389 and 2.9851 at ρᵤ of 0, 0.1 and 1 respectively. ADR-0025 continues to use its own raw count for empirical-Bayes shrinkage.

Let G be the number of independent session blocks and X the number whose mean matched slippage drift strictly exceeds 10 bps. Ties count against adversity. On prospective epoch k ≥ 1, the tightening evidence must satisfy

    p = Σ(j=X..G) choose(G,j) / 2ᴳ ≤ αₖ = 1 / [20 k(k+1)].

The comparison uses rational arithmetic. Three breaches in three blocks give p = 0.125 and cannot qualify. Ten in ten yield 0.0009765625. Since Σαₖ = 0.05, the lifetime bound applies per predeclared stream under the independent-block sign null. A fleet-wide claim additionally requires a prospective allocation across streams. Duplicate receipts add no tests. Every new consumed epoch advances k, including HOLDs. The journal carries k and its cohort watermark through restarts and policy changes.

### Hysteresis, execution and receipts

| Input | Effect on the collar |
|---|---|
| Qualified adverse | Require ADR-0025 THROTTLE_RED, slippage drift >10 bps, absolute fill-rate gap ≤0.15, 60s markout ≥−3 bps, EB-shrunk net edge >0, direction supported and all evidence gates including p ≤ αₖ. Two consecutive qualifying epochs confirm one tightening and reset the adverse counter. |
| Eligible PARITY_GREEN | Hold. Two consecutive greens clear the adverse alarm. The adverse sign test is specific to tightening. |
| DRIFT_WARN, unsupported RED, missing, stale or overlapping data | Hold and break confirmation. An active alarm persists until eligible recovery. |
| Prior recommendation unacknowledged | Hold. A recommendation alone is insufficient evidence of the collar used to generate the next fill. |

For eligible anchors c₀, previous recommendation cₖ, L = 5 bps and Aₖ ∈ {0,1} indicating a confirmed step,

    cₖ₊₁ = max(L, c₀ − 4, cₖ − 2Aₖ).

Consequently L ≤ cₖ₊₁ ≤ cₖ ≤ c₀ ≤ U, 0 ≤ cₖ − cₖ₊₁ ≤ 2, and total variation telescopes to c₀ − c_final ≤ 4 bps **per baseline policy version**. These inequalities rule out oscillation of the recommendation for every sequence of inputs. A green recovery never automatically expands the collar. The 4 bps budget constrains tightening even if the independent-block assumption fails. It gives no bound on realized slippage or profitability.

`CollarLedger` uses a dedicated caller-supplied SQLite connection. The normal sequence is register the baseline, assemble a `CollarEpoch`, and consume the matching fleet reconciliation. Begin an immediate transaction, check the receipt identity and prospective index, reserve at most one tightening per lane and session, then append the decision and commit. Distinct receipts racing for the same k cannot both win. An exact replay returns the original typed decision even after a later epoch or restart. A stale writer must reload its private journal. The policy anchor, original receipt and prior decisions remain immutable. Every decision carries the six-field stratum, k, source references, c₀, c_prev, c_out, action/reason, statistical diagnostics, counter/alarm state and the last consumed session. Receipts use sorted compact JSON with finite numbers, canonical zero and SHA-256 excluding the digest field itself. All carry `paper_policy="NEVER_GATED_ADVISORY"`, `halts_paper_farm=false`, `stop_distance_mutated=false` and `provenance_tag="POST_WALL_LIVE_PAPER_ADVISORY"`.

The exhaustive fixture runs four possible observations across six epochs, five initial collars and two session domains. **40,960 paths and 245,760 production transition calls** gave a largest step of **2.0 bps**, largest cumulative tightening of **4.0 bps** and **zero bound violations**. Additional integration tests exercise the actual ADR-0025 producer, effective upper caps, independent writers, restart, economic direction, sample size, hysteresis and temporal isolation. E3 applies to the reproducible implementation and numerical contracts. Realized execution performance still requires a separate paper comparison.

### Temporal firewall

The exclusive study boundary is **T = 2026-10-16T04:00:00Z**, midnight in New York following October 15. `validate_prewall_temporal_boundary` requires an explicitly historical or simulated source, an event at or after Unix epoch, and event ≤ label maturity < T. Accepted inputs are aware datetimes, explicit-offset ISO timestamps with up to nine fractional digits, or integer Unix nanoseconds. It refuses naive instants, event T, T+1ns, and an event T−1ns whose label matures at T. When receipt and decision timestamps are supplied, require event ≤ receipt ≤ decision ≤ maturity. A supplied trade date must be pre-wall. A private loader must supply both knowledge timestamps and the exchange trade date for actionable PIT evidence.

Validate provenance and timing before joins, adjustments, feature fitting, forward shifts and label generation. Discovery readers accept `PRE_WALL_HISTORICAL` and `PRE_WALL_SIMULATION` and reject typed post-wall decisions and tagged records, including nested upstream records. Discovery processes must read only frozen allowlisted historical snapshots, with no permission to read paper data. Paper processes may read the approved baseline and write only the private paper journal. No lake writer, discovery loader invocation, broker client or order submission is part of this controller. Access controls and snapshot verification belong to the storage owner. Paper evidence cannot change a pre-wall weight, label, feature selection, baseline or trial count.

### Trial 104 power and pre-wall feasibility

| Repository and function | Argument for the next trial | Returned hurdle |
|---|---:|---:|
| compounding-lab `tca_queue_parity.compute_hlz_t_hurdle` | 103 completed | 4.2063 |
| upstream-shelf-lock `validator.compute_hlz_t_hurdle` | 104 including candidate | 4.2063 |
| multi-model-workflows `promotion_gate.compute_hlz_t_hurdle` | 104 including candidate | 4.2063 |

The prior trial faced 4.2041. Trial **#104 was not consumed** by this decision, its fixtures or its controller. A new pre-registration uses `evaluate_preregistration_power_gate(cumulative_trials_n=104, ...)`. Once registered, the trial counts regardless of outcome. At 104 completed trials the next hurdle is t*(105) = 4.2084.

Let q = round(Φ⁻¹(1 − 0.00135/104), 4) = 4.2063, D the number of independent date-level observations, and σ the standard deviation of the intraday strategy’s **date-level net return**, expressed in bps. The significance boundary is q√(252/D) and the 80%-power MDE is (q + 0.8416212336)σ/√D.

| Dates D | Minimum annualized SR | MDE at σ = 40 bps | MDE at σ = 85 bps | MDE at σ = 150 bps | Power gate |
|---:|---:|---:|---:|---:|---|
| 60 | 8.6203 | 26.0674 | 55.3931 | 97.7526 | Reject |
| 120 | 6.0955 | 18.4324 | 39.1689 | 69.1215 | Reject |
| 252 | 4.2063 | 12.7196 | 27.0291 | 47.6984 | Eligible |

At plausible annual SR 3, iid returns and a zero benchmark, MinTRL = ceil[1 + 252(q/3)²] = **497 dates**. **D = 179** first clears the preregistration rule SR_min ≤5.00. Achieving 80% power at true SR 3 takes 714 dates under this model. Separate shelf-lock and normal-tail calculations agreed on all nine cells.

Pre-wall Parquet metadata supports **700** one-minute date directories, **685** ORB dates across 14,750 matched trade/quote windows (14,749 nonempty on both sides), **109** distinct first-days dates and **612,671** clean SEC rows. The physical SEC acceptance field is `accepted`. Metadata alone cannot establish a tradable catalyst timestamp, complete L2 depth or realized overlay P&L. Split-Half VWAP, Catalyst-AVWAP and the BJZZ strip default to `VISUAL_TELEMETRY_OR_GOVERNANCE`. Any strategy consuming them must independently pass pre-registration, date-clustered HLZ, DSR, purged CPCV, stressed costs and paper breadth requirements.

Alpaca switched CTA/UTP quote-size reporting from round lots to shares on **2025-11-03**. The queue replay at `scripts/run_optb_prewall_factory.py:838,849` multiplies quote sizes by 100. Verify the ingestion unit and effective date before converting, or a post-change raw share count can be inflated 100-fold. [Alpaca quote-size change](https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change)

A paper consumer should apply an absolute recommendation only to its acknowledged version and epoch. It must verify the executable tick-rounded price still satisfies the collar and record an unrepresentable price as a hold. A resumed journal checks the stored decision digest before using its counters or returning an earlier receipt.

## Considered alternatives and consequences

A bidirectional controller responding immediately to raw fleet severity could loosen on noise or tighten into underfill. It also leaves the number of repeated statistical looks uncontrolled. Fixed collars avoid feedback entirely but cannot express a reproducible paper comparison of persistent execution drift. The chosen one-sided rule supports such a comparison with a bounded and inspectable recommendation. Conservative defaults can leave a lane observational for long periods. Brokers and consumer-side tick quantization remain outside this module.

Roll back the advisory by disabling consumption and reverting this atomic change. Preserve private receipts and the lifetime spending index. Any subsequent return to a prior effective collar is a separately acknowledged operational change.
