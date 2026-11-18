"""Executable 5-Stage Quantitative Compounding Lifecycle (`Sim -> Pre-Wall -> Paper -> EB Fine-Tune -> Broker`).

Implements the fail-closed promotion pipeline across `https://github.com/native-agent-stack`:
1. `SIM`: Causal simulation check (zero lookahead, finite metrics, positive gross expectancy).
2. `PREWALL_BACKTEST`: Strictly `<= 2026-10-15` cutoff, Harvey-Liu-Zhu dynamic t-hurdle `t*(N)`,
   `5`-path recombined `CPCV` (`PBO <= 0.30`), native-frequency `DSR >= 0.95`, `>= 200` events / `>= 60` dates,
   and positive net return at `1x` and `2x` transaction + market-impact costs.
3. `PAPER_EXECUTION`: Pure real-broker attribution (`REAL_BROKER_ONLY`, free of synthetic `paper_sim` splicing)
   across `<= 200 req/min` Alpaca Paper and `<= 3,000 msg/min` IBKR `:4002` Paper, `ADR-0022 <= 2.00%` Reg NMS
   Rule 612 tick-quantized stops, `>= 30` closed round-trips across `>= 10` distinct sessions, and explicit
   `Dec 6, 2026` `23/5` SIP cutover regime tagging.
4. `EMPIRICAL_BAYES_FINETUNE`: Shrunk expectancy `eb_r` + UCB priority inside `[0.40, 1.65]` with realized-`R`
   damping so losing closed lanes never outrank realized winners via single-lot unrealized MTM.
5. `BROKER_MANIFEST_EXPORT`: Deterministic SHA-256 signed JSON manifest targeting sanctioned paper venues while
   hard-refusing `api.alpaca.markets`, live IBKR ports `4001`/`7496`, or `live` execution modes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from statistics import NormalDist
from typing import Literal

STUDY_WALL_DATE = date(2026, 10, 15)
DEC6_23X5_CUTOVER_DATE = date(2026, 12, 6)
ADR0022_MAX_STOP_PCT: float = 2.00
ALPACA_PAPER_MAX_RPM: int = 200
ALPACA_EXIT_RESERVE_RPM: int = 15
IBKR_PAPER_PORT: int = 4002
IBKR_PAPER_MAX_MSG_PER_MIN: int = 3000
FORBIDDEN_LIVE_PORTS: frozenset[int] = frozenset({4001, 7496})
FORBIDDEN_LIVE_HOSTS: frozenset[str] = frozenset({"api.alpaca.markets"})

_HLZ_BASE_ONE_SIDED_P: float = 0.00135
_STD_NORM = NormalDist(0.0, 1.0)


def _require_finite(value: float, field_name: str) -> float:
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{field_name} must be finite (got {value!r})")
    return fval


def hlz_dynamic_t_hurdle(cumulative_trials_n: int) -> float:
    """Compute Harvey, Liu & Zhu (2016) multiple-testing t-hurdle `t*(N)` (`3.000` at `N=1`)."""
    if cumulative_trials_n < 1:
        raise ValueError(f"cumulative_trials_n must be >= 1 (got {cumulative_trials_n})")
    p_target = _HLZ_BASE_ONE_SIDED_P / float(cumulative_trials_n)
    return round(float(_STD_NORM.inv_cdf(1.0 - p_target)), 4)


def quantize_nms_stop(
    entry_px: float,
    requested_stop_pct: float,
    *,
    side: Literal["long", "short"] = "long",
) -> tuple[Decimal, Decimal, float]:
    """Quantize `(entry_px, stop_px, realized_stop_pct)` to US Reg NMS Rule 612 ticks inside `ADR-0022 <= 2.00%`."""
    e_val = _require_finite(entry_px, "entry_px")
    s_pct = _require_finite(requested_stop_pct, "requested_stop_pct")
    if e_val < 0.05:
        raise ValueError(f"entry_px must be >= $0.05 (got {entry_px!r})")
    if s_pct <= 0.0 or s_pct > ADR0022_MAX_STOP_PCT:
        raise ValueError(f"ADR-0022 violation: requested_stop_pct={s_pct} outside (0.0, {ADR0022_MAX_STOP_PCT}]%")
    if side not in ("long", "short"):
        raise ValueError(f"Unsupported side: {side!r}")

    d_entry = Decimal(str(e_val))
    e_tick = Decimal("0.01") if d_entry >= Decimal("1.00") else Decimal("0.0001")
    e_round = ROUND_FLOOR if side == "long" else ROUND_CEILING
    q_entry = (d_entry / e_tick).to_integral_value(rounding=e_round) * e_tick

    ratio = Decimal(str(s_pct)) / Decimal("100")
    raw_stop = q_entry * (Decimal("1") - ratio) if side == "long" else q_entry * (Decimal("1") + ratio)
    s_tick = Decimal("0.01") if raw_stop >= Decimal("1.00") else Decimal("0.0001")
    s_round = ROUND_CEILING if side == "long" else ROUND_FLOOR
    q_stop = (raw_stop / s_tick).to_integral_value(rounding=s_round) * s_tick

    if side == "long" and q_stop >= q_entry:
        step = Decimal("0.01") if q_entry > Decimal("1.00") else Decimal("0.0001")
        q_stop = q_entry - step
    elif side == "short" and q_stop <= q_entry:
        step = Decimal("0.01") if q_entry >= Decimal("1.00") else Decimal("0.0001")
        q_stop = q_entry + step

    realized_pct = float(abs(q_entry - q_stop) / q_entry) * 100.0
    if realized_pct <= 0.0 or realized_pct > ADR0022_MAX_STOP_PCT + 1e-9:
        raise ValueError(f"NMS tick quantization exceeded ADR-0022 <=2.00% at entry={q_entry} ({realized_pct:.4f}%)")
    return q_entry, q_stop, round(realized_pct, 6)


def resolve_cutover_regime(eval_date: date | str, *, enable_dec6_23x5_sip: bool = False) -> str:
    """Partition forward paper evaluation cleanly across the Dec 6, 2026 `23/5` SIP cutover."""
    d = date.fromisoformat(eval_date) if isinstance(eval_date, str) else eval_date
    if enable_dec6_23x5_sip and d >= DEC6_23X5_CUTOVER_DATE:
        return "post_2026_12_06_23x5_sip"
    return "pre_2026_12_06_24x5_boats"


@dataclass(frozen=True, slots=True)
class SimStageMetrics:
    causal_no_lookahead: bool
    gross_expectancy_bps: float
    simulated_events: int


@dataclass(frozen=True, slots=True)
class PrewallBacktestMetrics:
    sample_max_date: str
    cumulative_trials_n: int
    events_count: int
    distinct_dates: int
    t_stat_clustered: float
    dsr_probability: float
    cpcv_n_paths: int
    cpcv_pbo: float
    net_return_1x_bps: float
    net_return_2x_bps: float


@dataclass(frozen=True, slots=True)
class PaperExecutionMetrics:
    attribution_source: Literal["REAL_BROKER_ONLY", "PAPER_SIM_FALLBACK"]
    closed_roundtrips: int
    distinct_sessions: int
    closed_mean_r: float
    combined_expectancy_r: float
    open_mtm_pct: float
    mean_slippage_bps: float
    max_stop_pct_observed: float
    alpaca_order_rpm_cap: int
    ibkr_port: int
    eval_date: str


@dataclass(frozen=True, slots=True)
class EmpiricalBayesTuningResult:
    strategy_id: str
    tuning_regime: str
    eb_expectancy_r: float
    ucb_priority_weight: float
    stop_atr_mult: float
    min_close_pos: float
    min_ofi_pct: float
    alloc_multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class LifecycleEvaluation:
    strategy_id: str
    highest_stage_passed: str
    sim_passed: bool
    prewall_passed: bool
    paper_passed: bool
    required_hlz_t_hurdle: float
    cutover_regime: str
    blocking_reasons: tuple[str, ...]
    tuning: EmpiricalBayesTuningResult


@dataclass(frozen=True, slots=True)
class BrokerStrategyManifest:
    manifest_version: str
    generated_at_utc: str
    organization: str
    execution_firewall: str
    alpaca_paper_lane: dict[str, object]
    ibkr_paper_lane: dict[str, object]
    strategies: list[dict[str, object]]
    manifest_sha256: str


def tune_empirical_bayes_ucb(
    strategy_id: str,
    *,
    closed_roundtrips: int,
    closed_mean_r: float,
    combined_samples: int,
    combined_expectancy_r: float,
    open_mtm_pct: float,
    mean_slippage_bps: float,
    total_book_samples: int = 100,
    is_major_core: bool = False,
    base_stop_atr_mult: float = 1.80,
    base_min_close_pos: float = 0.60,
    base_min_ofi_pct: float = 0.0,
) -> EmpiricalBayesTuningResult:
    """Compute Empirical-Bayes shrunk `R` + UCB priority weight (`[0.40, 1.65]`, never gated, `<= 1.98` ATR stop)."""
    prior_n = 8.0
    prior_r = 0.25 if is_major_core else 0.05
    n_comb = max(0, int(combined_samples))
    n_closed = max(0, int(closed_roundtrips))
    comb_r = _require_finite(combined_expectancy_r, "combined_expectancy_r")
    closed_r = _require_finite(closed_mean_r, "closed_mean_r")
    _require_finite(open_mtm_pct, "open_mtm_pct")
    slip_bps = _require_finite(mean_slippage_bps, "mean_slippage_bps")

    eb_r = ((prior_n * prior_r) + (n_comb * comb_r)) / (prior_n + n_comb)
    ucb_bonus = 0.30 * math.sqrt(math.log(max(total_book_samples, 1) + 2.0) / (n_comb + 1.0))
    core_boost = 0.14 if is_major_core else 0.0
    min_floor = 1.18 if is_major_core else 0.40
    ucb_weight = min(1.65, max(min_floor, 1.0 + 0.45 * eb_r + ucb_bonus + core_boost))

    stop_atr = min(1.98, max(0.50, float(base_stop_atr_mult)))
    min_cp = float(base_min_close_pos)
    min_ofi = float(base_min_ofi_pct)
    alloc_mult = 1.0
    regime = "MAJOR_PORTFOLIO_CORE_ACTIVE" if is_major_core else "EXPLORING_ACTIVE"

    if n_closed >= 2 and comb_r < -0.25:
        severity = min(1.0, abs(comb_r) / 1.40)
        min_cp = round(min(0.76, min_cp + 0.10 * severity), 3)
        min_ofi = round(min(4.5, min_ofi + 4.5 * severity), 2)
        stop_atr = round(min(1.98, stop_atr + 0.10 * severity), 3)
        if n_closed >= 10 and eb_r <= -0.25 and closed_r < -0.25:
            min_ofi = round(max(min_ofi, 25.0), 2)
            ucb_weight = max(0.35, min(ucb_weight, 0.45))
            alloc_mult = 0.15
            regime = "PROBATION_NEGATIVE_EB_THROTTLE"
        else:
            damped_cap = 1.18 if is_major_core else max(0.40, round(1.0 - 0.25 * severity, 3))
            ucb_weight = max(min_floor, min(ucb_weight, damped_cap))
            alloc_mult = round(max(0.48, 0.88 - 0.50 * severity), 2)
            regime = "TIGHTENED_NOISE_FILTER"
    elif n_comb >= 1 and comb_r >= 0.15 and (n_closed == 0 or closed_r >= 0.0):
        if slip_bps <= 0.0:
            ucb_weight = min(1.65, ucb_weight + 0.08)
        alloc_mult = round(min(1.42, 1.15 + 0.45 * max(0.0, comb_r)), 2)
        regime = "PROMOTED_ALPHA_LEADER"

    return EmpiricalBayesTuningResult(
        strategy_id=strategy_id,
        tuning_regime=regime,
        eb_expectancy_r=round(eb_r, 4),
        ucb_priority_weight=round(ucb_weight, 3),
        stop_atr_mult=round(min(1.98, stop_atr), 3),
        min_close_pos=round(min_cp, 3),
        min_ofi_pct=round(min_ofi, 2),
        alloc_multiplier=alloc_mult,
    )


def evaluate_full_lifecycle(
    strategy_id: str,
    sim: SimStageMetrics,
    prewall: PrewallBacktestMetrics,
    paper: PaperExecutionMetrics,
    *,
    is_major_core: bool = False,
    enable_dec6_23x5_sip: bool = False,
) -> LifecycleEvaluation:
    """Evaluate `Sim -> Pre-Wall Backtest -> Paper Execution -> Empirical-Bayes Fine-Tuning` promotion gates."""
    reasons: list[str] = []

    sim_ok = bool(sim.causal_no_lookahead and sim.simulated_events >= 50 and sim.gross_expectancy_bps > 0.0)
    if not sim_ok:
        reasons.append("SIM_GATE_FAILED: requires causal_no_lookahead, >=50 events, and gross_expectancy_bps > 0")

    max_dt = date.fromisoformat(prewall.sample_max_date)
    if max_dt > STUDY_WALL_DATE:
        raise ValueError(f"Study-Wall Violation: {max_dt.isoformat()} > {STUDY_WALL_DATE.isoformat()}")

    t_hurdle = hlz_dynamic_t_hurdle(prewall.cumulative_trials_n)
    prewall_ok = bool(
        sim_ok
        and prewall.events_count >= 200
        and prewall.distinct_dates >= 60
        and prewall.t_stat_clustered >= t_hurdle
        and prewall.dsr_probability >= 0.95
        and prewall.cpcv_n_paths >= 5
        and prewall.cpcv_pbo <= 0.30
        and prewall.net_return_1x_bps > 0.0
        and prewall.net_return_2x_bps > 0.0
    )
    if sim_ok and not prewall_ok:
        reasons.append(
            f"PREWALL_GATE_FAILED: t={prewall.t_stat_clustered:.2f} vs t*={t_hurdle:.2f}, "
            f"DSR={prewall.dsr_probability:.3f}, PBO={prewall.cpcv_pbo:.2f}, 2x_bps={prewall.net_return_2x_bps:.2f}"
        )

    if paper.ibkr_port in FORBIDDEN_LIVE_PORTS:
        raise ValueError(f"Live-Capital Firewall Violation: forbidden live IBKR port {paper.ibkr_port}")
    if paper.alpaca_order_rpm_cap > ALPACA_PAPER_MAX_RPM:
        raise ValueError(
            f"Broker Pacer Violation: alpaca_order_rpm_cap={paper.alpaca_order_rpm_cap} > {ALPACA_PAPER_MAX_RPM}"
        )
    if paper.max_stop_pct_observed > ADR0022_MAX_STOP_PCT + 1e-9:
        raise ValueError(f"ADR-0022 Stop Violation: observed stop {paper.max_stop_pct_observed}% > 2.00%")

    paper_ok = bool(
        prewall_ok
        and paper.attribution_source == "REAL_BROKER_ONLY"
        and paper.closed_roundtrips >= 30
        and paper.distinct_sessions >= 10
        and paper.closed_mean_r > 0.0
    )
    if prewall_ok and not paper_ok:
        reasons.append(
            f"PAPER_GATE_FAILED: source={paper.attribution_source}, "
            f"closed_rt={paper.closed_roundtrips}/30, "
            f"sessions={paper.distinct_sessions}/10, R={paper.closed_mean_r:+.2f}"
        )

    tuning = tune_empirical_bayes_ucb(
        strategy_id,
        closed_roundtrips=paper.closed_roundtrips,
        closed_mean_r=paper.closed_mean_r,
        combined_samples=max(paper.closed_roundtrips, 1),
        combined_expectancy_r=paper.combined_expectancy_r,
        open_mtm_pct=paper.open_mtm_pct,
        mean_slippage_bps=paper.mean_slippage_bps,
        is_major_core=is_major_core,
    )

    if paper_ok:
        highest = "STAGE_4_EMPIRICAL_BAYES_PROMOTED"
    elif prewall_ok:
        highest = "STAGE_2_PREWALL_PASSED_PAPER_ACCUMULATING"
    elif sim_ok:
        highest = "STAGE_1_SIM_ONLY"
    else:
        highest = "STAGE_0_REJECTED"

    return LifecycleEvaluation(
        strategy_id=strategy_id,
        highest_stage_passed=highest,
        sim_passed=sim_ok,
        prewall_passed=prewall_ok,
        paper_passed=paper_ok,
        required_hlz_t_hurdle=t_hurdle,
        cutover_regime=resolve_cutover_regime(paper.eval_date, enable_dec6_23x5_sip=enable_dec6_23x5_sip),
        blocking_reasons=tuple(reasons),
        tuning=tuning,
    )


def export_broker_strategy_manifest(
    evaluations: list[LifecycleEvaluation],
    *,
    alpaca_endpoint_host: str = "paper-api.alpaca.markets",
    ibkr_port: int = IBKR_PAPER_PORT,
    execution_mode: str = "paper_only",
) -> BrokerStrategyManifest:
    """Export deterministic SHA-256 signed strategy manifest for Alpaca Paper (`200/min`) & IBKR Paper (`:4002`)."""
    if execution_mode != "paper_only":
        raise ValueError(f"Live-Capital Firewall: execution_mode must be 'paper_only' (got {execution_mode!r})")
    if alpaca_endpoint_host in FORBIDDEN_LIVE_HOSTS or not alpaca_endpoint_host.startswith("paper-"):
        raise ValueError(f"Live-Capital Firewall: forbidden non-paper Alpaca host {alpaca_endpoint_host!r}")
    if ibkr_port in FORBIDDEN_LIVE_PORTS or ibkr_port != IBKR_PAPER_PORT:
        raise ValueError(f"Live-Capital Firewall: IBKR port must be paper :4002 (got {ibkr_port})")

    strat_rows = [asdict(ev) for ev in sorted(evaluations, key=lambda x: x.strategy_id)]
    alpaca_cfg: dict[str, object] = {
        "host": alpaca_endpoint_host,
        "account_prefix_required": "PA",
        "max_order_rpm": ALPACA_PAPER_MAX_RPM,
        "protective_exit_reserve_rpm": ALPACA_EXIT_RESERVE_RPM,
        "max_stop_adr0022_pct": ADR0022_MAX_STOP_PCT,
        "paper_never_gated": True,
    }
    ibkr_cfg: dict[str, object] = {
        "host": "127.0.0.1",
        "port": ibkr_port,
        "account_prefix_required": "DU",
        "max_msg_per_min": IBKR_PAPER_MAX_MSG_PER_MIN,
        "outside_rth_trail_limit": True,
        "max_stop_adr0022_pct": ADR0022_MAX_STOP_PCT,
    }
    canonical_body = json.dumps(
        {
            "organization": "native-agent-stack",
            "execution_firewall": "PAPER_ONLY_HARD_ENFORCED",
            "alpaca_paper_lane": alpaca_cfg,
            "ibkr_paper_lane": ibkr_cfg,
            "strategies": strat_rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
    return BrokerStrategyManifest(
        manifest_version="1.0.0",
        generated_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        organization="https://github.com/native-agent-stack",
        execution_firewall="PAPER_ONLY_HARD_ENFORCED (live ports 4001/7496 & api.alpaca.markets hard-refused)",
        alpaca_paper_lane=alpaca_cfg,
        ibkr_paper_lane=ibkr_cfg,
        strategies=strat_rows,
        manifest_sha256=digest,
    )


def evaluate_dry_powder_15s_collar_policy(
    *,
    ofi_entry: float,
    iso_f_buy_share: float = 0.0,
    bjzz_inst_dark_share: float = 0.0,
    resting_passive_bid_0_15s: bool = False,
    bid_survived_15s: bool = True,
    ask_drift_15s_bps: float = 0.0,
    spread_15s_bps: float = 5.0,
) -> dict[str, object]:
    """Enforce `ADR-0014` (`H096`) pure dry-powder `15s` bid-survival + informed-flow joint gate."""
    import math

    if not all(
        math.isfinite(x) for x in (ofi_entry, iso_f_buy_share, bjzz_inst_dark_share, ask_drift_15s_bps, spread_15s_bps)
    ):
        raise ValueError("Non-finite inputs in evaluate_dry_powder_15s_collar_policy")

    informed_flow_ok = ofi_entry >= 0.25 and (iso_f_buy_share >= 0.15 or bjzz_inst_dark_share >= 0.12)
    dry_powder_ok = (not resting_passive_bid_0_15s) and bid_survived_15s
    collar_ok = ask_drift_15s_bps <= 10.0 and 0.0 < spread_15s_bps <= 10.0
    approved = informed_flow_ok and dry_powder_ok and collar_ok
    return {
        "approved_for_paper_collar": approved,
        "informed_flow_ok": informed_flow_ok,
        "dry_powder_bid_survived_15s": dry_powder_ok,
        "collar_within_10bps": collar_ok,
        "promotion_track": "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "standalone_alpha_claim": False,
    }
