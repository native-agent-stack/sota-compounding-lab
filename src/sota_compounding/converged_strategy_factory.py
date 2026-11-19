"""Converged Multi-Strategy Foundation Factory, Retired-Trap Firewall & Compounding Learning Reflection (`ADR-0021`).

Unifies all research-converged strategy archetypes, empirical retired-trap denylists (`H092`/`H093`/`H094`/`H096`/
`H101`), starred-framework pre-wall sweep adapters (`Qlib`, `Kronos`, `TimesFM`, `NautilusTrader`, `hftbacktest`,
`skfolio`, `ml4t-diagnostic`, `VectorBT`), and the dimensionless **Compounding Learning Reflection Engine**:
1. **Six Converged Strategy Archetypes (`CONVERGED_FOUNDATION_CATALOG`):** Adds `IPO_DAY1_MICROSTRUCTURE_LOSSLESS`
   and `SLOW_COIL_VDU_ACCUMULATION` alongside the canonical four (`PRE_TRAVEL_EARLY_IGNITION`,
   `FIRST_HAND_SEC_EPISODIC_PIVOT`, `MEGA_CAP_NEAR_LOSSLESS_ORB`, `CLIMAX_VWAP_MEAN_REVERSION`).
2. **Fail-Closed Retired-Trap Firewall (`check_retired_trap_firewall`):** Programmatically blocks any automated sweep
   or template from re-entering proven negative-expectancy traps (`H101` already-`POST` chases `-23.62 bps`,
   `H092`/`H093` naive overnight carry, `H094` unshielded lit passive queue `-97.3 bps`, `09:30–09:35 ET` opening
   taker chases, or relaxed `OFI > 0` without `ISO`/`BJZZ` + `15s` dry-powder bid survival).
3. **Compounding Learning Reflection Engine (`reflect_and_finetune_batch`):** Evaluates dimensionless realized/pre-wall
   trade reflections (`realized_r`, `mfe_r`, `mae_r`, `slippage_bps`, `spread_bps`, `entry_ofi`, `stop_pct`,
   `exit_reason`), measures capture efficiency, premature-stopout rate, and Glosten–Milgrom adverse selection, runs
   `tune_empirical_bayes_ucb()`, emits strictly bounded parameter updates (`stop_pct <= 2.00%` `ADR-0022`), and
   seals a PII-free SHA-256 reflection digest for `LESSONS.md` / `sota-bus`.
4. **Stationary Workstation Pre-Wall Factory Plan (`build_noesis_prewall_factory_plan`):** Produces deterministic,
   study-wall-locked (`<= 2026-10-15`) batch contracts for `Noesis` 64c / RTX 4090 `DF10`/`DF6`/SEC enrichment,
   `H099` `C(6,2)=15` CPCV + `1x`/`2x` Almgren–Chriss replay, and GPU foundation-model feature sweeps.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any

from sota_compounding.lifecycle import (
    ADR0022_MAX_STOP_PCT,
    STUDY_WALL_DATE,
    EmpiricalBayesTuningResult,
    hlz_dynamic_t_hurdle,
    tune_empirical_bayes_ucb,
)

CONVERGED_FACTORY_SCHEMA = "sota-converged-strategy-factory.v1"
REFLECTION_DIGEST_SCHEMA = "sota-compounding-reflection-digest.v1"
NOESIS_PREWALL_BATCH_SCHEMA = "sota-noesis-prewall-factory-plan.v1"

RETIRED_TRAP_DENYLIST: dict[str, str] = {
    "POST_INCENTIVE_LATE_CHASE": "H101 pre-wall -23.62 bps 1x (t=-12.554); retire already-POST chases.",
    "NAIVE_MOC_TO_MOO_OVERNIGHT": "H092/H093 pre-wall -22.6 to -24.1 bps overnight carry trap.",
    "UNSHIELDED_LIT_PASSIVE_QUEUE": (
        "H094 pre-wall -97.3 bps Glosten-Milgrom adverse selection without dry-powder filter."
    ),
    "RELAXED_OFI_ONLY_WITHOUT_JOINT_SWEEP": "H096 ablation -12.73 bps when ISO_F/BJZZ + 15s bid-survival dropped.",
    "OPENING_BELL_0930_0935_TAKER_CHASE": "ADR-0017 / H090 opening 5-minute taker whipsaw trap.",
}

STARRED_FRAMEWORK_ADAPTERS: dict[str, dict[str, str]] = {
    "qlib": {
        "repo": "microsoft/qlib",
        "role": "Cross-sectional factor library (Alpha158/Alpha360) & rolling LightGBM/GBDT pre-wall sweeps",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "kronos": {
        "repo": "shiyu-coder/Kronos",
        "role": "Financial K-line OHLCV foundation model on RTX 4090 (pre-wall <=2026-10-15 fine-tune only)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "timesfm": {
        "repo": "google-research/timesfm",
        "role": "Zero-shot intraday bar/volatility foundation forecaster (pre-wall <=2026-10-15)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "nautilus_trader": {
        "repo": "nautechsystems/nautilus_trader",
        "role": "Event-driven Rust/Python execution engine, SpreadAnalyzer & ExitGate parity (pinned v2.0.0rc9)",
        "authority": "PINNED_UPSTREAM_PRODUCTION",
    },
    "hftbacktest": {
        "repo": "nkaz001/hftbacktest",
        "role": "RiskAdverseQueueModel L1/L2 queue & quote-sweep stop verification (pinned)",
        "authority": "PINNED_UPSTREAM_PRODUCTION",
    },
    "skfolio": {
        "repo": "skfolio/skfolio",
        "role": "CombinatorialPurgedCV (C(6,2)=15) & drawdown-constrained sizing (pinned v1.12.7)",
        "authority": "PINNED_UPSTREAM_PRODUCTION",
    },
    "ml4t_diagnostic": {
        "repo": "ml4t/diagnostic",
        "role": "Deflated Sharpe Ratio (DSR), PBO & excursion trade attribution (pinned v0.2.0)",
        "authority": "PINNED_UPSTREAM_PRODUCTION",
    },
    "vectorbt": {
        "repo": "polakowo/vectorbt",
        "role": "Vectorized multi-parameter pre-wall sweep & heatmaps (pinned v1.2.0)",
        "authority": "PINNED_UPSTREAM_PRODUCTION",
    },
}


@dataclass(frozen=True, slots=True)
class StrategyFoundationSpec:
    strategy_id: str
    archetype: str
    evidence_grade: str
    promotion_track: str
    default_stop_pct: float
    max_spread_bps: float
    min_ofi_norm: float
    tier1_arm_mfe_pct: float
    tier1_floor_gain_pct: float
    tier2_harvest_mfe_pct: float
    allowed_sessions: tuple[str, ...]
    pinned_upstreams: tuple[str, ...]
    is_major_core: bool = False


CONVERGED_FOUNDATION_CATALOG: dict[str, StrategyFoundationSpec] = {
    "SOTA_EARLY_SUDDEN_VOLUME_SHOCK_V1": StrategyFoundationSpec(
        strategy_id="SOTA_EARLY_SUDDEN_VOLUME_SHOCK_V1",
        archetype="PRE_TRAVEL_EARLY_IGNITION",
        evidence_grade="E4",
        promotion_track="PROTECTIVE_FILTER_OR_GOVERNANCE",
        default_stop_pct=1.65,
        max_spread_bps=22.0,
        min_ofi_norm=0.25,
        tier1_arm_mfe_pct=0.65,
        tier1_floor_gain_pct=0.18,
        tier2_harvest_mfe_pct=1.25,
        allowed_sessions=("PREMARKET", "RTH_CORE", "POST_CLOSE", "BOATS_24X5"),
        pinned_upstreams=("alpaca-py", "hftbacktest", "nautilus_trader", "polars"),
        is_major_core=True,
    ),
    "SOTA_SLOW_COIL_VDU_ACCUMULATION_V1": StrategyFoundationSpec(
        strategy_id="SOTA_SLOW_COIL_VDU_ACCUMULATION_V1",
        archetype="SLOW_COIL_VDU_ACCUMULATION",
        evidence_grade="E3",
        promotion_track="PROTECTIVE_FILTER_OR_GOVERNANCE",
        default_stop_pct=1.45,
        max_spread_bps=18.0,
        min_ofi_norm=0.20,
        tier1_arm_mfe_pct=0.65,
        tier1_floor_gain_pct=0.18,
        tier2_harvest_mfe_pct=1.25,
        allowed_sessions=("PREMARKET", "RTH_CORE", "POST_CLOSE", "BOATS_24X5"),
        pinned_upstreams=("alpaca-py", "polars", "skfolio"),
    ),
    "SOTA_FIRST_HAND_SEC_EPISODIC_PIVOT_V1": StrategyFoundationSpec(
        strategy_id="SOTA_FIRST_HAND_SEC_EPISODIC_PIVOT_V1",
        archetype="FIRST_HAND_SEC_EPISODIC_PIVOT",
        evidence_grade="E4",
        promotion_track="PROTECTIVE_FILTER_OR_GOVERNANCE",
        default_stop_pct=1.80,
        max_spread_bps=28.0,
        min_ofi_norm=0.25,
        tier1_arm_mfe_pct=0.80,
        tier1_floor_gain_pct=0.20,
        tier2_harvest_mfe_pct=1.60,
        allowed_sessions=("PREMARKET", "RTH_CORE", "POST_CLOSE", "BOATS_24X5"),
        pinned_upstreams=("edgartools", "alpaca-py", "ml4t-diagnostic"),
    ),
    "SOTA_MEGA_CAP_LOSSLESS_ORB_PRECURSOR_V1": StrategyFoundationSpec(
        strategy_id="SOTA_MEGA_CAP_LOSSLESS_ORB_PRECURSOR_V1",
        archetype="MEGA_CAP_NEAR_LOSSLESS_ORB",
        evidence_grade="E4",
        promotion_track="PROTECTIVE_FILTER_OR_GOVERNANCE",
        default_stop_pct=0.85,
        max_spread_bps=10.0,
        min_ofi_norm=0.25,
        tier1_arm_mfe_pct=0.35,
        tier1_floor_gain_pct=0.08,
        tier2_harvest_mfe_pct=0.52,
        allowed_sessions=("PREMARKET", "RTH_CORE", "POST_CLOSE"),
        pinned_upstreams=("alpaca-py", "nautilus_trader", "hftbacktest", "skfolio"),
        is_major_core=True,
    ),
    "IPO_DAY1_CROSS_STABILIZATION_AND_VWAP_WAVE2_V1": StrategyFoundationSpec(
        strategy_id="IPO_DAY1_CROSS_STABILIZATION_AND_VWAP_WAVE2_V1",
        archetype="IPO_DAY1_MICROSTRUCTURE_LOSSLESS",
        evidence_grade="E4",
        promotion_track="PROTECTIVE_FILTER_OR_GOVERNANCE",
        default_stop_pct=1.85,
        max_spread_bps=25.0,
        min_ofi_norm=0.25,
        tier1_arm_mfe_pct=2.00,
        tier1_floor_gain_pct=0.35,
        tier2_harvest_mfe_pct=4.00,
        allowed_sessions=("RTH_CORE",),
        pinned_upstreams=("alpaca-py", "polars", "hftbacktest", "nautilus_trader"),
    ),
    "SOTA_PARABOLIC_PEAK_CLIMAX_REVERSE_SHORT_V1": StrategyFoundationSpec(
        strategy_id="SOTA_PARABOLIC_PEAK_CLIMAX_REVERSE_SHORT_V1",
        archetype="CLIMAX_VWAP_MEAN_REVERSION",
        evidence_grade="E4",
        promotion_track="PROTECTIVE_FILTER_OR_GOVERNANCE",
        default_stop_pct=1.65,
        max_spread_bps=25.0,
        min_ofi_norm=0.25,
        tier1_arm_mfe_pct=0.65,
        tier1_floor_gain_pct=0.18,
        tier2_harvest_mfe_pct=1.25,
        allowed_sessions=("RTH_CORE", "PREMARKET", "POST_CLOSE"),
        pinned_upstreams=("ib_async", "alpaca-py", "nautilus_trader"),
    ),
}


@dataclass(frozen=True, slots=True)
class TradeReflectionRecord:
    """Dimensionless, PII-free trade reflection unit (zero USD balances or account IDs)."""

    strategy_id: str
    regime: str
    realized_r: float
    mfe_r: float
    mae_r: float
    slippage_bps: float
    spread_bps: float
    entry_ofi: float
    stop_pct: float
    exit_reason: str


@dataclass(frozen=True, slots=True)
class CompoundingReflectionDigest:
    schema_version: str
    generated_at_utc: str
    strategy_id: str
    sample_count: int
    win_rate_pct: float
    mean_realized_r: float
    profit_factor: float
    capture_efficiency_ratio: float
    premature_stopout_rate_pct: float
    mean_slippage_bps: float
    mean_spread_bps: float
    eb_tuning: EmpiricalBayesTuningResult
    recommended_stop_pct: float
    recommended_max_spread_bps: float
    recommended_min_ofi_norm: float
    recommended_tier1_arm_mfe_pct: float
    reflection_actions: tuple[str, ...]
    digest_sha256: str


def _require_finite_num(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric value (got {value!r})")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite (got {value!r})")
    return fval


def check_retired_trap_firewall(
    *,
    tags: list[str] | tuple[str, ...],
    is_post_incentive: bool = False,
    holds_overnight_moc_to_moo: bool = False,
    unshielded_lit_passive: bool = False,
    in_opening_0930_0935_taker: bool = False,
    ofi_only_without_joint_sweep: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """Fail-closed firewall blocking any strategy or batch sweep from re-testing retired negative-EV traps."""
    violations: list[str] = []
    for raw_tag in tags:
        norm = str(raw_tag or "").strip().upper()
        if norm in RETIRED_TRAP_DENYLIST:
            violations.append(f"RETIRED_TRAP:{norm}")
    if is_post_incentive:
        violations.append("RETIRED_TRAP:POST_INCENTIVE_LATE_CHASE")
    if holds_overnight_moc_to_moo:
        violations.append("RETIRED_TRAP:NAIVE_MOC_TO_MOO_OVERNIGHT")
    if unshielded_lit_passive:
        violations.append("RETIRED_TRAP:UNSHIELDED_LIT_PASSIVE_QUEUE")
    if in_opening_0930_0935_taker:
        violations.append("RETIRED_TRAP:OPENING_BELL_0930_0935_TAKER_CHASE")
    if ofi_only_without_joint_sweep:
        violations.append("RETIRED_TRAP:RELAXED_OFI_ONLY_WITHOUT_JOINT_SWEEP")
    dedup = tuple(dict.fromkeys(violations))
    return (len(dedup) == 0), dedup


def reflect_and_finetune_batch(
    strategy_id: str,
    records: list[TradeReflectionRecord] | tuple[TradeReflectionRecord, ...],
    *,
    total_book_samples: int = 100,
    now_utc: datetime | None = None,
) -> CompoundingReflectionDigest:
    """Run dimensionless compounding learning reflection & bounded parameter fine-tuning (`ADR-0022 <= 2.00%`)."""
    sid = str(strategy_id or "").strip()
    if not sid:
        raise ValueError("strategy_id must be non-empty")

    spec = CONVERGED_FOUNDATION_CATALOG.get(sid)
    base_stop_pct = spec.default_stop_pct if spec else 1.50
    base_max_spread_bps = spec.max_spread_bps if spec else 22.0
    base_min_ofi = spec.min_ofi_norm if spec else 0.25
    base_tier1_mfe = spec.tier1_arm_mfe_pct if spec else 0.65
    is_major_core = spec.is_major_core if spec else False

    ts = (now_utc or datetime.now(UTC)).astimezone(UTC).isoformat().replace("+00:00", "Z")
    n = len(records)
    if n == 0:
        eb = tune_empirical_bayes_ucb(
            sid,
            closed_roundtrips=0,
            closed_mean_r=0.0,
            combined_samples=0,
            combined_expectancy_r=0.0,
            open_mtm_pct=0.0,
            mean_slippage_bps=0.0,
            total_book_samples=total_book_samples,
            is_major_core=is_major_core,
        )
        body = {
            "schema_version": REFLECTION_DIGEST_SCHEMA,
            "generated_at_utc": ts,
            "strategy_id": sid,
            "sample_count": 0,
            "win_rate_pct": 0.0,
            "mean_realized_r": 0.0,
            "profit_factor": 1.0,
            "capture_efficiency_ratio": 0.0,
            "premature_stopout_rate_pct": 0.0,
            "mean_slippage_bps": 0.0,
            "mean_spread_bps": 0.0,
            "eb_tuning": asdict(eb),
            "recommended_stop_pct": min(ADR0022_MAX_STOP_PCT, base_stop_pct),
            "recommended_max_spread_bps": base_max_spread_bps,
            "recommended_min_ofi_norm": base_min_ofi,
            "recommended_tier1_arm_mfe_pct": base_tier1_mfe,
            "reflection_actions": ["COLLECT_MORE_REAL_PAPER_SAMPLES"],
        }
        sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        return CompoundingReflectionDigest(
            schema_version=REFLECTION_DIGEST_SCHEMA,
            generated_at_utc=ts,
            strategy_id=sid,
            sample_count=0,
            win_rate_pct=0.0,
            mean_realized_r=0.0,
            profit_factor=1.0,
            capture_efficiency_ratio=0.0,
            premature_stopout_rate_pct=0.0,
            mean_slippage_bps=0.0,
            mean_spread_bps=0.0,
            eb_tuning=eb,
            recommended_stop_pct=round(min(ADR0022_MAX_STOP_PCT, base_stop_pct), 2),
            recommended_max_spread_bps=round(base_max_spread_bps, 2),
            recommended_min_ofi_norm=round(base_min_ofi, 3),
            recommended_tier1_arm_mfe_pct=round(base_tier1_mfe, 2),
            reflection_actions=("COLLECT_MORE_REAL_PAPER_SAMPLES",),
            digest_sha256=sha,
        )

    realized_list: list[float] = []
    mfe_pos_sum = 0.0
    realized_pos_sum = 0.0
    gross_win_r = 0.0
    gross_loss_r = 0.0
    wins = 0
    premature_stops = 0
    slip_sum = 0.0
    spread_sum = 0.0

    for rec in records:
        r_val = _require_finite_num(rec.realized_r, "realized_r")
        mfe_val = _require_finite_num(rec.mfe_r, "mfe_r")
        _require_finite_num(rec.mae_r, "mae_r")
        slip_val = _require_finite_num(rec.slippage_bps, "slippage_bps")
        spr_val = _require_finite_num(rec.spread_bps, "spread_bps")
        _require_finite_num(rec.entry_ofi, "entry_ofi")
        stop_val = _require_finite_num(rec.stop_pct, "stop_pct")
        if stop_val <= 0.0 or stop_val > ADR0022_MAX_STOP_PCT:
            raise ValueError(f"ADR-0022 violation in reflection record: stop_pct={stop_val}")
        if mfe_val < 0.0 or spr_val < 0.0:
            raise ValueError(f"mfe_r ({mfe_val}) and spread_bps ({spr_val}) must be >= 0")

        realized_list.append(r_val)
        mfe_pos_sum += mfe_val
        if r_val >= 0.0:
            wins += 1
            gross_win_r += r_val
            realized_pos_sum += r_val
        else:
            gross_loss_r += abs(r_val)
            if mfe_val >= 0.50:
                premature_stops += 1
        slip_sum += slip_val
        spread_sum += spr_val

    mean_r = sum(realized_list) / n
    win_rate_pct = (wins / n) * 100.0
    if gross_win_r > 0.0 and gross_loss_r > 0.0:
        pf = min(99.0, gross_win_r / gross_loss_r)
    elif gross_win_r > 0.0:
        pf = 99.0
    elif gross_loss_r > 0.0:
        pf = 0.0
    else:
        pf = 1.0

    capture_eff = min(1.0, realized_pos_sum / max(mfe_pos_sum, 1e-6)) if mfe_pos_sum > 0.0 else 0.0
    premature_rate_pct = (premature_stops / n) * 100.0
    mean_slip_bps = slip_sum / n
    mean_spread_bps = spread_sum / n

    eb = tune_empirical_bayes_ucb(
        sid,
        closed_roundtrips=n,
        closed_mean_r=mean_r,
        combined_samples=n,
        combined_expectancy_r=mean_r,
        open_mtm_pct=0.0,
        mean_slippage_bps=mean_slip_bps,
        total_book_samples=max(total_book_samples, n),
        is_major_core=is_major_core,
        base_min_ofi_pct=base_min_ofi * 100.0,
    )

    # Bounded self-reflection fine-tuning rules (NEVER exceed ADR-0022 <= 2.00%!)
    rec_stop_pct = base_stop_pct
    rec_max_spread_bps = base_max_spread_bps
    rec_min_ofi = base_min_ofi
    rec_tier1_mfe = base_tier1_mfe
    actions: list[str] = []

    if mean_slip_bps > 5.0 or mean_spread_bps > base_max_spread_bps * 0.90:
        rec_max_spread_bps = max(6.0, round(base_max_spread_bps * 0.82, 2))
        rec_min_ofi = min(0.60, round(base_min_ofi + 0.08, 3))
        actions.append("TIGHTEN_SPREAD_COLLAR_AND_REQUIRE_PASSIVE_MICROPRICE_PEG")

    if premature_rate_pct >= 25.0:
        # Widen slightly within ADR-0022 <= 2.00% ceiling AND arm lossless breakeven earlier
        rec_stop_pct = min(ADR0022_MAX_STOP_PCT, round(base_stop_pct + 0.15, 2))
        rec_tier1_mfe = max(0.30, round(base_tier1_mfe * 0.85, 2))
        actions.append("ARM_TIER1_LOSSLESS_FLOOR_EARLIER_AND_CALIBRATE_GK_VOL_STOP")
    elif capture_eff < 0.40 and wins >= 2:
        rec_tier1_mfe = max(0.30, round(base_tier1_mfe * 0.85, 2))
        actions.append("IMPROVE_MFE_CAPTURE_VIA_EARLIER_TIER1_LOCK_AND_LULD_HARVEST")

    if eb.tuning_regime == "PROBATION_NEGATIVE_EB_THROTTLE":
        rec_min_ofi = min(0.65, round(max(rec_min_ofi, 0.35), 3))
        actions.append("ENFORCE_PROBATION_015X_THROTTLE_AND_AUDIT_REGIME_DRIFT")
    elif eb.tuning_regime == "PROMOTED_ALPHA_LEADER" and not actions:
        actions.append("MAINTAIN_CONVERGED_PARAMS_AND_COMPOUND_MERIT")

    if not actions:
        actions.append("HOLD_BASELINE_CONTINUE_PAPER_SAMPLE_ACCRUAL")

    rec_stop_pct = round(min(ADR0022_MAX_STOP_PCT, max(0.50, rec_stop_pct)), 2)

    digest_payload: dict[str, Any] = {
        "schema_version": REFLECTION_DIGEST_SCHEMA,
        "generated_at_utc": ts,
        "strategy_id": sid,
        "sample_count": n,
        "win_rate_pct": round(win_rate_pct, 2),
        "mean_realized_r": round(mean_r, 4),
        "profit_factor": round(pf, 3),
        "capture_efficiency_ratio": round(capture_eff, 4),
        "premature_stopout_rate_pct": round(premature_rate_pct, 2),
        "mean_slippage_bps": round(mean_slip_bps, 2),
        "mean_spread_bps": round(mean_spread_bps, 2),
        "eb_tuning": asdict(eb),
        "recommended_stop_pct": rec_stop_pct,
        "recommended_max_spread_bps": round(rec_max_spread_bps, 2),
        "recommended_min_ofi_norm": round(rec_min_ofi, 3),
        "recommended_tier1_arm_mfe_pct": round(rec_tier1_mfe, 2),
        "reflection_actions": actions,
    }
    sha = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode("utf-8")).hexdigest()

    return CompoundingReflectionDigest(
        schema_version=REFLECTION_DIGEST_SCHEMA,
        generated_at_utc=ts,
        strategy_id=sid,
        sample_count=n,
        win_rate_pct=round(win_rate_pct, 2),
        mean_realized_r=round(mean_r, 4),
        profit_factor=round(pf, 3),
        capture_efficiency_ratio=round(capture_eff, 4),
        premature_stopout_rate_pct=round(premature_rate_pct, 2),
        mean_slippage_bps=round(mean_slip_bps, 2),
        mean_spread_bps=round(mean_spread_bps, 2),
        eb_tuning=eb,
        recommended_stop_pct=rec_stop_pct,
        recommended_max_spread_bps=round(rec_max_spread_bps, 2),
        recommended_min_ofi_norm=round(rec_min_ofi, 3),
        recommended_tier1_arm_mfe_pct=round(rec_tier1_mfe, 2),
        reflection_actions=tuple(actions),
        digest_sha256=sha,
    )


def build_noesis_prewall_factory_plan(
    *,
    cumulative_trials_n: int,
    study_wall_cutoff: date = STUDY_WALL_DATE,
    frameworks: tuple[str, ...] = ("hftbacktest", "skfolio", "ml4t_diagnostic", "qlib", "kronos", "timesfm"),
    candidate_strategy_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed, study-wall-locked (`<= 2026-10-15`) Noesis 64c / RTX 4090 pre-wall batch contract."""
    if study_wall_cutoff > STUDY_WALL_DATE:
        raise ValueError(
            f"Pre-wall breach: study_wall_cutoff={study_wall_cutoff.isoformat()} > {STUDY_WALL_DATE.isoformat()}"
        )
    if isinstance(cumulative_trials_n, bool) or not isinstance(cumulative_trials_n, int) or cumulative_trials_n < 1:
        raise ValueError(f"cumulative_trials_n must be an int >= 1 (got {cumulative_trials_n!r})")

    unknown_fw = [fw for fw in frameworks if fw not in STARRED_FRAMEWORK_ADAPTERS]
    if unknown_fw:
        raise ValueError(f"Unknown framework adapter(s): {unknown_fw}")

    strat_ids = candidate_strategy_ids or tuple(CONVERGED_FOUNDATION_CATALOG.keys())
    unknown_strats = [s for s in strat_ids if s not in CONVERGED_FOUNDATION_CATALOG]
    if unknown_strats:
        raise ValueError(f"Unknown strategy_id(s): {unknown_strats}")

    hlz_hurdle = hlz_dynamic_t_hurdle(cumulative_trials_n + 1)
    plan: dict[str, Any] = {
        "schema_version": NOESIS_PREWALL_BATCH_SCHEMA,
        "execution_host": "stationary_noesis_win32_64c_rtx4090",
        "study_wall_date": study_wall_cutoff.isoformat(),
        "next_trial_number": cumulative_trials_n + 1,
        "next_hlz_t_hurdle": hlz_hurdle,
        "retired_traps_blocked": sorted(RETIRED_TRAP_DENYLIST.keys()),
        "stages": [
            {
                "stage_id": "S1_DF10_20D_TOD_MATERIALIZATION",
                "target": "Z:/sota-data/prewall-lake/derived/df10_same_clock_20d_baselines.parquet",
                "contract": (
                    "ADR-0020 compute_df10_same_clock_baseline (>=20 strictly prior XNYS sessions, <=2026-10-15)"
                ),
            },
            {
                "stage_id": "S2_DF6_DATED_REGSHO_AND_SEC_CUTOFF",
                "target": "Z:/sota-data/prewall-lake/derived/df6_regsho_threshold_and_clean_sec_pit.parquet",
                "contract": (
                    "ADR-0020 validate_df6_regsho_threshold_snapshot + "
                    "filter_sec_filings_pit_prewall (purge post-wall EFFECT)"
                ),
            },
            {
                "stage_id": "S3_H099_CPCV_15PATH_AND_COST_STRESS",
                "target": "Z:/sota-data/prewall-lake/receipts/h099_cpcv_1x_2x_dsr.json",
                "contract": (
                    f"C(6,2)=15 CPCV + 1x/2x Almgren-Chriss + DSR >= 0.95 "
                    f"vs HLZ t*({cumulative_trials_n + 1})={hlz_hurdle}"
                ),
            },
            {
                "stage_id": "S4_STARRED_FRAMEWORK_GPU_SWEEP_AND_REFLECTION",
                "frameworks": [STARRED_FRAMEWORK_ADAPTERS[fw] for fw in frameworks],
                "candidate_strategies": list(strat_ids),
                "promotion_rule": (
                    "PROTECTIVE_FILTER_OR_GOVERNANCE unless t >= t*(N), DSR >= 0.95, "
                    "PBO <= 0.30, and Stage-3 >= 30 real paper roundtrips"
                ),
            },
        ],
    }
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest()
    return plan
