"""ADR-0025: Live-Paper TCA vs. Stage-1 Nautilus/hftbacktest Queue-Parity Reconciler & EB Decay Gate.

Closes the feedback loop from live **paper** fills (`:8787`, `:8799`, `:8797` Alpaca Paper lanes and the
`:4002` IBKR Paper gateway) back into the pre-wall (`<= 2026-10-15`) Stage-1 simulation baseline produced by
`NautilusTrader` / `hftbacktest` `RiskAdverseQueueModel` queue-position fills.

Design invariants:

- Pure Python standard library, deterministic, no I/O and no wall-clock timestamps in any hashed payload.
- Explicit dimensional suffixes on every quantity: `_bps` (basis points), `_ratio` (unitless in `[0, 1]` or a
  difference of such ratios), `_n` (non-negative integer counts), `_date` (ISO-8601 calendar date).
- Fail-closed input validation: every numeric input passes `math.isfinite`, booleans are never accepted as
  numbers, fill rates must lie in `[0.0, 1.0]`, counts must be non-negative integers and simulated baselines
  must be cut off on or before the `2026-10-15` study wall.
- Strictly advisory: a `THROTTLE_RED` verdict recommends `THROTTLE_OR_RECALIBRATE_COLLAR` and never halts the
  paper farm (`NEVER_GATED` paper policy). Live-capital lanes (`:8798`, `:4001`, `:7496`) are refused outright.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sota_compounding.lifecycle import STUDY_WALL_DATE, hlz_dynamic_t_hurdle

TCA_QUEUE_PARITY_SCHEMA: str = "sota.tca-queue-parity.v1"

PARITY_GREEN: str = "PARITY_GREEN"
DRIFT_WARN: str = "DRIFT_WARN"
THROTTLE_RED: str = "THROTTLE_RED"
PARITY_STATUS_SEVERITY: dict[str, int] = {PARITY_GREEN: 0, DRIFT_WARN: 1, THROTTLE_RED: 2}
RECOMMENDED_ACTIONS: dict[str, str] = {
    PARITY_GREEN: "MAINTAIN_COLLAR",
    DRIFT_WARN: "MONITOR_AND_TIGHTEN_COLLAR",
    THROTTLE_RED: "THROTTLE_OR_RECALIBRATE_COLLAR",
}

PAPER_ACCOUNT_LANES: tuple[str, ...] = (":8787", ":8799", ":8797", ":4002")
FORBIDDEN_LIVE_LANES: frozenset[str] = frozenset({":8798", ":4001", ":7496"})
COCKPIT_SNAPSHOT_ENDPOINT: str = ":8787/api/unified_cockpit_snapshot"

DEFAULT_SHRINKAGE_K: float = 20.0

# Classification thresholds (dimensional suffixes are part of the contract).
GREEN_MAX_SLIPPAGE_DRIFT_BPS: float = 4.0
WARN_MAX_SLIPPAGE_DRIFT_BPS: float = 10.0
GREEN_MAX_ABS_FILL_RATE_GAP_RATIO: float = 0.15
WARN_MAX_ABS_FILL_RATE_GAP_RATIO: float = 0.30
GREEN_MIN_MARKOUT_60S_BPS: float = -3.0
WARN_MIN_MARKOUT_60S_BPS: float = -8.0
RED_NEGATIVE_EB_MIN_LIVE_SAMPLE_N: int = 10

# Derived quantities are rounded before threshold comparison so that IEEE-754 artefacts such as
# `0.65 - 0.5 == 0.15000000000000002` never flip a boundary verdict.
_ROUND_DP: int = 6


def _require_finite_real(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a real number (got {value!r})")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{field_name} must be finite (got {value!r})")
    return fval


def _require_ratio(value: Any, field_name: str) -> float:
    fval = _require_finite_real(value, field_name)
    if fval < 0.0 or fval > 1.0:
        raise ValueError(f"{field_name} must lie in [0.0, 1.0] (got {value!r})")
    return fval


def _require_count(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer count (got {value!r})")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0 (got {value!r})")
    return int(value)


def _require_label(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string (got {value!r})")
    return value


def _require_prewall_date(value: Any, field_name: str) -> str:
    if isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a calendar date, not a datetime (got {value!r})")
    if isinstance(value, date):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 date (got {value!r})") from exc
    else:
        raise ValueError(f"{field_name} must be an ISO-8601 date string (got {value!r})")
    if parsed > STUDY_WALL_DATE:
        raise ValueError(
            f"Study-Wall Violation: {field_name}={parsed.isoformat()} > {STUDY_WALL_DATE.isoformat()} "
            "(simulated baselines must be strictly pre-wall)"
        )
    return parsed.isoformat()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class QueueParityBaseline:
    """Pre-wall Stage-1 `NautilusTrader` / `hftbacktest` `RiskAdverseQueueModel` baseline statistics."""

    strategy_id: str
    prewall_cutoff_date: str
    sim_fill_rate_ratio: float
    sim_expected_slippage_bps: float
    sim_gross_edge_bps: float
    sim_sample_n: int
    hlz_trials_n: int

    def __post_init__(self) -> None:
        _require_label(self.strategy_id, "strategy_id")
        object.__setattr__(
            self, "prewall_cutoff_date", _require_prewall_date(self.prewall_cutoff_date, "prewall_cutoff_date")
        )
        object.__setattr__(self, "sim_fill_rate_ratio", _require_ratio(self.sim_fill_rate_ratio, "sim_fill_rate_ratio"))
        object.__setattr__(
            self,
            "sim_expected_slippage_bps",
            _require_finite_real(self.sim_expected_slippage_bps, "sim_expected_slippage_bps"),
        )
        object.__setattr__(
            self, "sim_gross_edge_bps", _require_finite_real(self.sim_gross_edge_bps, "sim_gross_edge_bps")
        )
        object.__setattr__(self, "sim_sample_n", _require_count(self.sim_sample_n, "sim_sample_n"))
        object.__setattr__(self, "hlz_trials_n", _require_count(self.hlz_trials_n, "hlz_trials_n"))


@dataclass(frozen=True, slots=True)
class LivePaperTcaObservation:
    """Live **paper** transaction-cost analysis (implementation shortfall + markouts) for one account lane."""

    strategy_id: str
    account_lane: str
    live_sample_n: int
    live_fill_rate_ratio: float
    live_implementation_shortfall_bps: float
    live_markout_1s_bps: float
    live_markout_10s_bps: float
    live_markout_60s_bps: float
    live_markout_300s_bps: float
    live_gross_edge_bps: float

    def __post_init__(self) -> None:
        _require_label(self.strategy_id, "strategy_id")
        lane = _require_label(self.account_lane, "account_lane")
        if lane in FORBIDDEN_LIVE_LANES:
            raise ValueError(f"Live-Capital Firewall: account_lane {lane!r} is a live-capital lane (paper only)")
        if lane not in PAPER_ACCOUNT_LANES:
            raise ValueError(f"account_lane must be one of {PAPER_ACCOUNT_LANES} (got {lane!r})")
        object.__setattr__(self, "live_sample_n", _require_count(self.live_sample_n, "live_sample_n"))
        object.__setattr__(
            self, "live_fill_rate_ratio", _require_ratio(self.live_fill_rate_ratio, "live_fill_rate_ratio")
        )
        for name in (
            "live_implementation_shortfall_bps",
            "live_markout_1s_bps",
            "live_markout_10s_bps",
            "live_markout_60s_bps",
            "live_markout_300s_bps",
            "live_gross_edge_bps",
        ):
            object.__setattr__(self, name, _require_finite_real(getattr(self, name), name))


def compute_hlz_t_hurdle(trials_n: int) -> float:
    """Harvey-Liu-Zhu hurdle a *new* trial must clear after `trials_n` cumulative ledger trials.

    Delegates to the repo's exact `lifecycle.hlz_dynamic_t_hurdle` formula using the next-trial convention
    (`t*(trials_n + 1)`, as in `converged_strategy_factory.build_noesis_prewall_factory_plan`): `0 -> 3.0`,
    `102 -> 4.2041`, `103 -> 4.2063`. Non-decreasing in `trials_n` (rounded to 4 decimals).
    """
    n_trials = _require_count(trials_n, "trials_n")
    return hlz_dynamic_t_hurdle(n_trials + 1)


def _classify(
    *,
    slippage_drift_bps: float,
    fill_rate_gap_ratio: float,
    live_markout_60s_bps: float,
    eb_shrunk_net_edge_bps: float,
    live_sample_n: int,
) -> tuple[str, list[str]]:
    """Total, ordered classification: RED if any severe breach, else GREEN if all clean, else WARN."""
    abs_gap = abs(fill_rate_gap_ratio)
    red: list[str] = []
    if slippage_drift_bps > WARN_MAX_SLIPPAGE_DRIFT_BPS:
        red.append(f"SLIPPAGE_DRIFT_SEVERE:{slippage_drift_bps}>{WARN_MAX_SLIPPAGE_DRIFT_BPS}")
    if abs_gap > WARN_MAX_ABS_FILL_RATE_GAP_RATIO:
        red.append(f"FILL_RATE_GAP_SEVERE:{abs_gap}>{WARN_MAX_ABS_FILL_RATE_GAP_RATIO}")
    if live_markout_60s_bps < WARN_MIN_MARKOUT_60S_BPS:
        red.append(f"ADVERSE_SELECTION_SEVERE:{live_markout_60s_bps}<{WARN_MIN_MARKOUT_60S_BPS}")
    if eb_shrunk_net_edge_bps <= 0.0 and live_sample_n >= RED_NEGATIVE_EB_MIN_LIVE_SAMPLE_N:
        red.append(f"EB_NET_EDGE_NON_POSITIVE:{eb_shrunk_net_edge_bps}<=0.0@n={live_sample_n}")
    if red:
        return THROTTLE_RED, red

    warn: list[str] = []
    if slippage_drift_bps > GREEN_MAX_SLIPPAGE_DRIFT_BPS:
        warn.append(f"SLIPPAGE_DRIFT_MODERATE:{slippage_drift_bps}>{GREEN_MAX_SLIPPAGE_DRIFT_BPS}")
    if abs_gap > GREEN_MAX_ABS_FILL_RATE_GAP_RATIO:
        warn.append(f"FILL_RATE_GAP_MODERATE:{abs_gap}>{GREEN_MAX_ABS_FILL_RATE_GAP_RATIO}")
    if live_markout_60s_bps < GREEN_MIN_MARKOUT_60S_BPS:
        warn.append(f"ADVERSE_SELECTION_MILD:{live_markout_60s_bps}<{GREEN_MIN_MARKOUT_60S_BPS}")
    if eb_shrunk_net_edge_bps <= 0.0:
        warn.append(f"EB_NET_EDGE_NON_POSITIVE_LOW_N:{eb_shrunk_net_edge_bps}<=0.0@n={live_sample_n}")
    if warn:
        return DRIFT_WARN, warn
    return PARITY_GREEN, []


def reconcile_live_vs_sim_parity(
    baseline: QueueParityBaseline,
    live_obs: LivePaperTcaObservation,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    *,
    hlz_t_stat: float | None = None,
) -> dict[str, Any]:
    """Reconcile one live paper lane against its pre-wall queue-parity baseline (advisory, never halts paper).

    `eb_shrunk_net_edge_bps = w * live_net + (1 - w) * prior_edge_bps` with `w = live_sample_n / (live_sample_n
    + shrinkage_k)`, `live_net = live_gross_edge_bps - live_implementation_shortfall_bps` and
    `prior_edge_bps = sim_gross_edge_bps - sim_expected_slippage_bps`; the result is clamped to
    `[min(prior, live_net), max(prior, live_net)]` after rounding so the convex-combination bound is exact.
    """
    if not isinstance(baseline, QueueParityBaseline):
        raise ValueError(f"baseline must be a QueueParityBaseline (got {type(baseline).__name__})")
    if not isinstance(live_obs, LivePaperTcaObservation):
        raise ValueError(f"live_obs must be a LivePaperTcaObservation (got {type(live_obs).__name__})")
    if baseline.strategy_id != live_obs.strategy_id:
        raise ValueError(f"strategy_id mismatch: baseline={baseline.strategy_id!r} vs live={live_obs.strategy_id!r}")
    k_val = _require_finite_real(shrinkage_k, "shrinkage_k")
    if k_val <= 0.0:
        raise ValueError(f"shrinkage_k must be > 0.0 (got {shrinkage_k!r})")
    t_stat = None if hlz_t_stat is None else _require_finite_real(hlz_t_stat, "hlz_t_stat")

    slippage_drift_bps = round(
        live_obs.live_implementation_shortfall_bps - baseline.sim_expected_slippage_bps, _ROUND_DP
    )
    fill_rate_gap_ratio = round(live_obs.live_fill_rate_ratio - baseline.sim_fill_rate_ratio, _ROUND_DP)
    prior_edge_bps = round(baseline.sim_gross_edge_bps - baseline.sim_expected_slippage_bps, _ROUND_DP)
    live_net_edge_bps = round(live_obs.live_gross_edge_bps - live_obs.live_implementation_shortfall_bps, _ROUND_DP)
    eb_live_weight_ratio = live_obs.live_sample_n / (live_obs.live_sample_n + k_val)
    eb_raw = eb_live_weight_ratio * live_net_edge_bps + (1.0 - eb_live_weight_ratio) * prior_edge_bps
    eb_lo, eb_hi = min(prior_edge_bps, live_net_edge_bps), max(prior_edge_bps, live_net_edge_bps)
    eb_shrunk_net_edge_bps = min(eb_hi, max(eb_lo, round(eb_raw, _ROUND_DP)))

    status, breach_reasons = _classify(
        slippage_drift_bps=slippage_drift_bps,
        fill_rate_gap_ratio=fill_rate_gap_ratio,
        live_markout_60s_bps=live_obs.live_markout_60s_bps,
        eb_shrunk_net_edge_bps=eb_shrunk_net_edge_bps,
        live_sample_n=live_obs.live_sample_n,
    )

    hlz_t_hurdle = compute_hlz_t_hurdle(baseline.hlz_trials_n)
    record: dict[str, Any] = {
        "schema_version": TCA_QUEUE_PARITY_SCHEMA,
        "record_type": "lane_reconciliation",
        "strategy_id": baseline.strategy_id,
        "account_lane": live_obs.account_lane,
        "prewall_cutoff_date": baseline.prewall_cutoff_date,
        "sim_sample_n": baseline.sim_sample_n,
        "live_sample_n": live_obs.live_sample_n,
        "sim_fill_rate_ratio": baseline.sim_fill_rate_ratio,
        "live_fill_rate_ratio": live_obs.live_fill_rate_ratio,
        "fill_rate_gap_ratio": fill_rate_gap_ratio,
        "sim_expected_slippage_bps": baseline.sim_expected_slippage_bps,
        "live_implementation_shortfall_bps": live_obs.live_implementation_shortfall_bps,
        "slippage_drift_bps": slippage_drift_bps,
        "live_markout_1s_bps": live_obs.live_markout_1s_bps,
        "live_markout_10s_bps": live_obs.live_markout_10s_bps,
        "live_markout_60s_bps": live_obs.live_markout_60s_bps,
        "live_markout_300s_bps": live_obs.live_markout_300s_bps,
        "sim_gross_edge_bps": baseline.sim_gross_edge_bps,
        "live_gross_edge_bps": live_obs.live_gross_edge_bps,
        "prior_edge_bps": prior_edge_bps,
        "live_net_edge_bps": live_net_edge_bps,
        "shrinkage_k": k_val,
        "eb_live_weight_ratio": round(eb_live_weight_ratio, _ROUND_DP),
        "eb_shrunk_net_edge_bps": eb_shrunk_net_edge_bps,
        "hlz_trials_n": baseline.hlz_trials_n,
        "hlz_t_hurdle": hlz_t_hurdle,
        "hlz_t_stat": t_stat,
        "hlz_hurdle_cleared": None if t_stat is None else bool(t_stat >= hlz_t_hurdle),
        "parity_status": status,
        "breach_reasons": breach_reasons,
        "recommended_action": RECOMMENDED_ACTIONS[status],
        "paper_policy": "NEVER_GATED_ADVISORY",
        "halts_paper_farm": False,
    }
    record["receipt_sha256"] = _canonical_sha256(record)
    return record


def reconcile_fleet_parity(
    baselines: Sequence[QueueParityBaseline],
    observations: Sequence[LivePaperTcaObservation],
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    *,
    hlz_t_stats: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Reconcile every paper lane and emit a deterministic cockpit-ready fleet summary.

    Output is independent of input ordering (rows sorted by `(strategy_id, account_lane)`); observations with no
    matching baseline are listed under `unmatched_observations` instead of raising.
    """
    by_strategy: dict[str, QueueParityBaseline] = {}
    for b in baselines:
        if not isinstance(b, QueueParityBaseline):
            raise ValueError(f"baselines must contain QueueParityBaseline (got {type(b).__name__})")
        if b.strategy_id in by_strategy:
            raise ValueError(f"duplicate baseline strategy_id: {b.strategy_id!r}")
        by_strategy[b.strategy_id] = b

    t_stats = dict(hlz_t_stats or {})
    observations = list(observations)
    seen_keys: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, str]] = []
    for obs in observations:
        if not isinstance(obs, LivePaperTcaObservation):
            raise ValueError(f"observations must contain LivePaperTcaObservation (got {type(obs).__name__})")
    for obs in sorted(observations, key=lambda o: (o.strategy_id, o.account_lane)):
        key = (obs.strategy_id, obs.account_lane)
        if key in seen_keys:
            raise ValueError(f"duplicate observation for strategy_id={key[0]!r} account_lane={key[1]!r}")
        seen_keys.add(key)
        base = by_strategy.get(obs.strategy_id)
        if base is None:
            unmatched.append({"strategy_id": obs.strategy_id, "account_lane": obs.account_lane})
            continue
        rows.append(reconcile_live_vs_sim_parity(base, obs, shrinkage_k, hlz_t_stat=t_stats.get(obs.strategy_id)))

    observed_strategies = {r["strategy_id"] for r in rows}
    status_counts = {s: 0 for s in PARITY_STATUS_SEVERITY}
    for r in rows:
        status_counts[r["parity_status"]] += 1
    fleet_status = (
        max((r["parity_status"] for r in rows), key=PARITY_STATUS_SEVERITY.__getitem__) if rows else PARITY_GREEN
    )

    total_live_n = sum(r["live_sample_n"] for r in rows)

    def _weighted_mean(field: str) -> float | None:
        if total_live_n <= 0:
            return None
        return round(sum(r[field] * r["live_sample_n"] for r in rows) / total_live_n, _ROUND_DP)

    summary: dict[str, Any] = {
        "schema_version": TCA_QUEUE_PARITY_SCHEMA,
        "record_type": "fleet_summary",
        "cockpit_endpoint": COCKPIT_SNAPSHOT_ENDPOINT,
        "cockpit_panel": "tca_queue_parity",
        "fleet_status": fleet_status,
        "fleet_recommended_action": RECOMMENDED_ACTIONS[fleet_status],
        "status_counts": status_counts,
        "lanes_n": len(rows),
        "strategies_n": len(observed_strategies),
        "total_live_sample_n": total_live_n,
        "weighted_slippage_drift_bps": _weighted_mean("slippage_drift_bps"),
        "weighted_fill_rate_gap_ratio": _weighted_mean("fill_rate_gap_ratio"),
        "weighted_eb_shrunk_net_edge_bps": _weighted_mean("eb_shrunk_net_edge_bps"),
        "throttle_red_lanes": [
            f"{r['strategy_id']}@{r['account_lane']}" for r in rows if r["parity_status"] == THROTTLE_RED
        ],
        "unmatched_observations": unmatched,
        "baselines_without_observations": sorted(set(by_strategy) - observed_strategies),
        "paper_policy": "NEVER_GATED_ADVISORY",
        "halts_paper_farm": False,
        "rows": rows,
    }
    summary["receipt_sha256"] = _canonical_sha256(summary)
    return summary


__all__ = [
    "COCKPIT_SNAPSHOT_ENDPOINT",
    "DEFAULT_SHRINKAGE_K",
    "DRIFT_WARN",
    "FORBIDDEN_LIVE_LANES",
    "PAPER_ACCOUNT_LANES",
    "PARITY_GREEN",
    "PARITY_STATUS_SEVERITY",
    "RECOMMENDED_ACTIONS",
    "TCA_QUEUE_PARITY_SCHEMA",
    "THROTTLE_RED",
    "LivePaperTcaObservation",
    "QueueParityBaseline",
    "compute_hlz_t_hurdle",
    "reconcile_fleet_parity",
    "reconcile_live_vs_sim_parity",
]
