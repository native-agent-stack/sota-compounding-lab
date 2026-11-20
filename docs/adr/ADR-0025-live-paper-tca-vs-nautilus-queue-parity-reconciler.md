# ADR-0025: Live-Paper TCA vs. Nautilus/hftbacktest Queue-Parity Reconciler & Empirical-Bayes Decay Gate

- **Status:** Accepted
- **Date:** 2026-11-19
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/tca_queue_parity.py`, `tests/test_adr0025_tca_queue_parity.py`

## Context and Problem Statement

Stage 1 (`SIM`) fills are produced by pre-wall (`<= 2026-10-15`) `NautilusTrader` / `hftbacktest` `RiskAdverseQueueModel` queue-position simulations. Stage 3 (`PAPER_EXECUTION`) produces real paper fills on the Alpaca Paper lanes (`:8787`, `:8799`, `:8797`) and the IBKR Paper gateway (`:4002`). Until now nothing compared the two: if the simulated queue model overstated fill rates or understated slippage, a strategy's paper edge could decay without its Empirical-Bayes priority reacting until the realized-`R` damping in `tune_empirical_bayes_ucb` caught up. We need a deterministic, auditable closed loop from live paper transaction-cost analysis (TCA) back to the simulated baseline. It must not stop the paper farm.

## Decision Drivers

- Dimensional safety: every quantity carries its unit suffix (`_bps`, `_ratio`, `_n`, `_date`) and passes a `math.isfinite` guard.
- Pre-wall hygiene: simulated baselines dated after `2026-10-15` are refused (`ValueError`, `Study-Wall Violation`).
- Paper `NEVER_GATED` policy: the reconciler is advisory only. It recommends collar recalibration and never halts paper order flow.
- Live-capital firewall: live lanes (`:8798`, `:4001`, `:7496`) are refused as `account_lane` values.
- Determinism: output is plain JSON, carries no timestamps and does not depend on input order, and is stamped with a SHA-256 receipt over canonical JSON (`sort_keys`, compact separators, `allow_nan=False`).
- Zero runtime dependencies: Python standard library only, so it is unit-testable anywhere.

## Considered Options

1. **Pure-stdlib reconciler module with deterministic advisory classification** (chosen).
2. Embed the comparison inside `tune_empirical_bayes_ucb`: rejected, because it couples execution-quality drift to realized-`R` tuning and makes each harder to audit separately.
3. Hard-gate paper execution on parity breach: rejected, because it violates the `NEVER_GATED` paper policy and removes the very data needed to recalibrate.

## Decision Outcome

Chosen option 1: `src/sota_compounding/tca_queue_parity.py` (`schema_version = "sota.tca-queue-parity.v1"`).

### Data contracts

- `QueueParityBaseline`: `strategy_id`, `prewall_cutoff_date`, `sim_fill_rate_ratio` (`[0, 1]`), `sim_expected_slippage_bps`, `sim_gross_edge_bps`, `sim_sample_n` (`>= 0`), `hlz_trials_n` (`>= 0`).
- `LivePaperTcaObservation`: `strategy_id`, `account_lane` (one of `:8787`, `:8799`, `:8797`, `:4002`), `live_sample_n` (`>= 0`), `live_fill_rate_ratio` (`[0, 1]`), `live_implementation_shortfall_bps`, `live_markout_{1s,10s,60s,300s}_bps`, `live_gross_edge_bps`.
- Both are frozen, slotted dataclasses that validate in `__post_init__`. They reject `NaN`/`Inf`, booleans used as numbers, out-of-range ratios and negative counts.

### Harvey-Liu-Zhu hurdle (`compute_hlz_t_hurdle`)

Delegates to the repo's exact `lifecycle.hlz_dynamic_t_hurdle` formula (one-sided `p = 0.00135 / N`, `t* = Phi^-1(1 - p)`, 4 dp). It uses the **next-trial** convention already used by `converged_strategy_factory.build_noesis_prewall_factory_plan`: `compute_hlz_t_hurdle(trials_n) = t*(trials_n + 1)`. With the ledger at `103` cumulative trials, this gives `compute_hlz_t_hurdle(103) = t*(104) = 4.2063`. For reference, `t*(103) = 4.2041` was the hurdle Trial `#103` itself faced (ADR-0024). `trials_n = 0` maps to `t*(1) = 3.0`. The function is non-decreasing in `trials_n`.

The optional keyword `hlz_t_stat` is finite-guarded and reported as `hlz_t_stat`, `hlz_t_hurdle` and `hlz_hurdle_cleared` (`None` when absent). It is informational and does not feed the parity status.

### Reconciliation (`reconcile_live_vs_sim_parity`)

- `slippage_drift_bps = live_implementation_shortfall_bps - sim_expected_slippage_bps`
- `fill_rate_gap_ratio = live_fill_rate_ratio - sim_fill_rate_ratio`
- `prior_edge_bps = sim_gross_edge_bps - sim_expected_slippage_bps`
- `live_net_edge_bps = live_gross_edge_bps - live_implementation_shortfall_bps`
- `eb_live_weight_ratio = live_sample_n / (live_sample_n + shrinkage_k)` with `shrinkage_k > 0` (default `20.0`)
- `eb_shrunk_net_edge_bps = w * live_net_edge_bps + (1 - w) * prior_edge_bps`, clamped to `[min(prior, live_net), max(prior, live_net)]`

Derived values are rounded to 6 dp before threshold comparison. This stops IEEE-754 artefacts (for example `0.65 - 0.5 = 0.15000000000000002`) from flipping a boundary verdict.

### Status classification

The classification is total and checks in order, so every input maps to exactly one status:

1. `THROTTLE_RED` if any severe breach holds: `slippage_drift_bps > 10.0`, or `|fill_rate_gap_ratio| > 0.30`, or `live_markout_60s_bps < -8.0`, or `eb_shrunk_net_edge_bps <= 0.0` with `live_sample_n >= 10`. Action: `THROTTLE_OR_RECALIBRATE_COLLAR`.
2. `PARITY_GREEN` if all of the following hold: `slippage_drift_bps <= 4.0`, `|fill_rate_gap_ratio| <= 0.15`, `live_markout_60s_bps >= -3.0` and `eb_shrunk_net_edge_bps > 0.0`. Action: `MAINTAIN_COLLAR`.
3. `DRIFT_WARN` otherwise. This covers moderate drift `(4, 10]` bps, fill gap `(0.15, 0.30]`, markout `[-8, -3)` bps, and non-positive EB edge with `live_sample_n < 10`. Action: `MONITOR_AND_TIGHTEN_COLLAR`.

Every record carries `paper_policy = "NEVER_GATED_ADVISORY"` and `halts_paper_farm = false`.

### Fleet summary (`reconcile_fleet_parity`)

`reconcile_fleet_parity` returns a cockpit panel (`cockpit_panel = "tca_queue_parity"`) intended for `:8787/api/unified_cockpit_snapshot`. It contains:

- the worst-lane `fleet_status`;
- per-status counts;
- weighted means of drift, fill gap and EB edge, weighted by `live_sample_n`;
- the list of `THROTTLE_RED` lanes;
- `unmatched_observations` and `baselines_without_observations`;
- per-lane rows sorted by `(strategy_id, account_lane)`;
- a SHA-256 receipt.

Duplicate baselines, and duplicate `(strategy_id, account_lane)` observations, raise `ValueError`.

### Numeric domain hardening (cross-vendor falsifier audit of the initial commit)

The following inputs now fail closed with `ValueError` and never leak `OverflowError`, `StatisticsError` or a non-finite receipt:

- **Huge integer inputs.** Integers beyond float64 range (for example `10**400`) are rejected.
- **`shrinkage_k`.** Validated unconditionally, including for empty or fully unmatched fleets.
- **`hlz_t_stats`.** Must be a `Mapping` with non-empty string keys and finite values. Every entry is validated upfront.
- **HLZ trial counts.** Counts are rejected where the one-sided `p` underflows (`trials_n >= 24,319,437,987,800`), at construction and at hurdle time.
- **Derived floats.** Every derived float (`slippage_drift_bps`, `fill_rate_gap_ratio`, `prior_edge_bps`, `live_net_edge_bps`, the raw EB value, `eb_shrunk_net_edge_bps`, the fleet weighted means) is `math.isfinite`-guarded before classification (`Derived-Domain Violation`).

Two computations were changed so they cannot overflow or change the receipt:

- **EB weight.** `eb_live_weight_ratio` is computed scale-invariantly, as `1 / (1 + k/n)` when `n >= k` and `(n/k) / (1 + n/k)` otherwise. It is `1.0` when `float(n)` overflows. So `n = 10**308, k = 1e308` gives `0.5`, the same as `n = k = 20`; previously `n + k = inf` collapsed the weight to `0.0` and turned a `THROTTLE_RED` lane `PARITY_GREEN`.
- **Weighted means.** Fleet means use convex weights `n_i / N`, so no `value * n_i` product is formed.

Signed zero is canonicalized (`-0.0 -> 0.0`) on validated and rounded floats, so `-0.0` and `+0.0` inputs produce identical `receipt_sha256` values.

### Consequences

- Good: queue-model optimism (fill-rate overstatement, slippage understatement, adverse selection at 60 s) now shows up per lane before EB realized-`R` damping catches it, with a reproducible receipt.
- Good: zero runtime dependencies and pure functions, so it runs anywhere and can be replayed from receipts.
- Neutral: thresholds are fixed constants. Retuning them requires a new ADR and a schema bump.
- Bad: the reconciler consumes pre-aggregated lane TCA. Producing those aggregates from raw paper fills is out of scope here, and raw fills, account identifiers and tuned parameters never enter this public repo.

## Verification

- `uv run pytest tests/test_adr0025_tca_queue_parity.py -q`: unit tests for `PARITY_GREEN`, `DRIFT_WARN`, `THROTTLE_RED`, exact float boundaries, post-wall rejection, `NaN`/`Inf` rejection, live-lane refusal, fleet aggregation and permutation invariance, and SHA-256 determinism. Hypothesis properties cover EB convex-combination bounds, monotonic slippage-drift degradation and a non-decreasing HLZ hurdle.
- `uv run pytest -q`, `uv run ruff check src scripts tests`, `uv run ruff format --check src scripts tests`, `python scripts/architecture_catalog.py --check`.

## Verification & Evidence
- Backfilled under `sota-upstream-shelf-lock ADR-0023` (2026-11-20): anchored to repository `UPSTREAM.md`, `upstream-map.toml`, and git history of `adr0023-19adr-evidence` (`shelf_lock.hard_upstream_rule` verified).
