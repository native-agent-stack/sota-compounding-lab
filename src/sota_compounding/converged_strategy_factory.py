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
    "ADR0064_QUARANTINED_NOISY_PAPER_STRATEGIES": (
        "ADR-0064: quarantine SOTA_DUAL_CLIMAX_STOP_AND_REVERSE (PF 0.47), POST_VIX_REVERSAL (-0.53R), "
        "SOTA_DUAL_EXTREME_BALANCE_BREAK_EARLY, blind Wave-2 churn, and FLOW_VELOCITY_CAPITAL_RECYCLE."
    ),
    "SINGLE_TICKER_IPO_DAY1_OVERFIT": (
        "ADR-0028 Post-ANT consolidation: forbid single-ticker/IPO-debut hardcoding in active merit routing."
    ),
    "LEGACY_SP500_ADDITION_POP_CHURN": (
        "Greenwood & Sammon (2025 JF) proved classic S&P 500 inclusion pops disappeared; retire addition chases."
    ),
    "VOL_TIMING_LEVERAGE_ABOVE_1X": (
        "Cederburg et al. (2020 JFE): real-time vol-timing leverage > 1.0x fails OOS; cap strictly <= 1.0x crash brake."
    ),
    "EX_POST_LIVE_MOVER_SURVIVORSHIP_BACKTEST": (
        "ADR-0030: forbid selecting today's live movers and backtesting backward; require causal T-1 PIT universes."
    ),
    "TOY_HORIZON_BELOW_MIN_TRL": (
        "ADR-0030: forbid toy 30d (<60 sessions intraday / <252 sessions swing) windows below ADIA Lab #19 MinTRL."
    ),
    "FIXED_TAKE_PROFIT_CAP_CUTTING_TAIL": (
        "ADR-0030 / ADR-0063: forbid hard Take-Profit caps that truncate convex right-tail lossless Chandelier runners."
    ),
}

POST_ANT_ERA_ID = "POST_ANT_WHOLE_MARKET_MERIT"
POST_ANT_CONCENTRATED_SOTA_STRATEGIES: frozenset[str] = frozenset(
    {
        "BJZZ_SUBPENNY_INST_DARK_SWEEP",
        "VIX_SPIKE_AND_CRUSH_REGIME_SURFER",
        "ZARATTINI_2024_5M_ORB",
        "ZARATTINI_2025_NOISE_VWAP",
        "H090_TIGHT_SPREAD_HIGH_OFI_PASSIVE_BID",
        "H01_PRE_TRAVEL_POCKET_PIVOT",
        "H08_FIRST_HAND_CATALYST_IGNITION",
    }
)
POST_ANT_QUARANTINED_STRATEGIES: frozenset[str] = frozenset(
    {
        "SOTA_DUAL_CLIMAX_STOP_AND_REVERSE",
        "POST_VIX_REVERSAL",
        "POST_VIX_REVERSAL_MAJOR_GROWTH_THRUST",
        "SOTA_DUAL_EXTREME_BALANCE_BREAK_EARLY",
        "BLIND_WAVE2_REENTRY",
        "RAPID_WAVE2_REENTRY_STRESS",
        "FLOW_VELOCITY_CAPITAL_RECYCLE",
    }
)


def is_post_ant_strategy_quarantined(strategy_id: str) -> bool:
    """Canonical ADR-0064 quarantine predicate matching exact IDs and family variants."""
    sid = str(strategy_id or "").strip().upper()
    if sid in POST_ANT_QUARANTINED_STRATEGIES:
        return True
    return sid.startswith("POST_VIX_REVERSAL") or "FLOW_VELOCITY_CAPITAL_RECYCLE" in sid


POST_ANT_VALID_SESSIONS: frozenset[str] = frozenset({"PRE_MARKET", "RTH_CORE", "POST_CLOSE", "OVERNIGHT_BOATS"})

POST_ANT_STRATEGY_CONTRACTS_V2: dict[str, dict[str, Any]] = {
    "BJZZ_SUBPENNY_INST_DARK_SWEEP": {
        "allowed_sessions": frozenset({"PRE_MARKET", "RTH_CORE", "POST_CLOSE", "OVERNIGHT_BOATS"}),
        "max_spread_bps": 28.0,
        "spread_strict_lt": False,
        "max_stop_pct": 1.60,
        "allowed_symbols": None,
    },
    "VIX_SPIKE_AND_CRUSH_REGIME_SURFER": {
        "allowed_sessions": frozenset({"PRE_MARKET", "RTH_CORE", "POST_CLOSE"}),
        "max_spread_bps": 18.0,
        "spread_strict_lt": False,
        "max_stop_pct": 1.25,
        "allowed_symbols": frozenset({"SPY", "QQQ", "IWM", "UVXY", "VXX", "SVIX", "VIXY"}),
    },
    "ZARATTINI_2024_5M_ORB": {
        "allowed_sessions": frozenset({"RTH_CORE"}),
        "max_spread_bps": 25.0,
        "spread_strict_lt": False,
        "max_stop_pct": 1.60,
        "allowed_symbols": None,
    },
    "ZARATTINI_2025_NOISE_VWAP": {
        "allowed_sessions": frozenset({"RTH_CORE"}),
        "max_spread_bps": 3.0,
        "spread_strict_lt": True,
        "max_stop_pct": 0.50,
        "allowed_symbols": frozenset({"SPY", "QQQ", "IWM"}),
    },
    "H090_TIGHT_SPREAD_HIGH_OFI_PASSIVE_BID": {
        "allowed_sessions": frozenset({"RTH_CORE"}),
        "max_spread_bps": 10.0,
        "spread_strict_lt": False,
        "max_stop_pct": 1.25,
        "allowed_symbols": None,
    },
    "H01_PRE_TRAVEL_POCKET_PIVOT": {
        "allowed_sessions": frozenset({"PRE_MARKET", "RTH_CORE", "POST_CLOSE", "OVERNIGHT_BOATS"}),
        "max_spread_bps": 28.0,
        "spread_strict_lt": False,
        "max_stop_pct": 1.60,
        "allowed_symbols": None,
    },
    "H08_FIRST_HAND_CATALYST_IGNITION": {
        "allowed_sessions": frozenset({"PRE_MARKET", "RTH_CORE", "POST_CLOSE", "OVERNIGHT_BOATS"}),
        "max_spread_bps": 28.0,
        "spread_strict_lt": False,
        "max_stop_pct": 1.60,
        "allowed_symbols": None,
    },
}


STARRED_FRAMEWORK_ADAPTERS: dict[str, dict[str, str]] = {
    "qlib": {
        "repo": "microsoft/qlib",
        "role": "Cross-sectional factor library (Alpha158/360) & LightGBM/GBDT pre-wall sweeps (v0.9.8 AST hardened)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "kronos": {
        "repo": "shiyu-coder/Kronos",
        "role": "Financial K-line OHLCV foundation model on RTX 4090 (pre-wall <=2026-10-15 fine-tune only)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "timesfm": {
        "repo": "google-research/timesfm",
        "role": "LEGACY REF ONLY: v3.0 weights non-commercial; superseded by Apache-2.0 chronos2",
        "authority": "NON_COMMERCIAL_WEIGHTS_RESTRICTED_USE_CHRONOS2",
    },
    "chronos2": {
        "repo": "amazon-science/chronos-forecasting",
        "role": "Apache-2.0 (code + weights) zero-shot multivariate/covariate probabilistic forecaster on RTX 4090",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "arch": {
        "repo": "bashtage/arch",
        "role": "Studentized Hansen SPA, White RC, Model Confidence Set (MCS), Block Bootstrap & HAR-RV (NCSA)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "linearmodels": {
        "repo": "bashtage/linearmodels",
        "role": "Fama-MacBeth cross-sectional regressions & two-way entity+date clustered panel SEs (NCSA)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "databento_dbn": {
        "repo": "databento/dbn",
        "role": "Zero-copy L2/L3 MBO/MBP-10 & NOII schema enforcing bitemporal ts_event_ns <= ts_recv_ns (Apache-2.0)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "delta_rs": {
        "repo": "delta-io/delta-rs",
        "role": "Immutable ACID Parquet time-travel snapshots for <=2026-10-15 pre-wall lake parity (Apache-2.0)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "ducklake": {
        "repo": "duckdb/ducklake",
        "role": "Native SQL-catalog ACID lakehouse & snapshot time-travel inside pinned DuckDB v2.0.0 (MIT)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "mapie": {
        "repo": "scikit-learn-contrib/MAPIE",
        "role": "Finite-sample Conformal Prediction (CQR, EnbPI, ACI) for stop <=2.00% tail bounds (BSD-3-Clause)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "river": {
        "repo": "online-ml/river",
        "role": "Streaming ADWIN/PageHinkley/KSWIN alpha-decay & Dec 6 2026 23/5 SIP drift alarms (BSD-3-Clause)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "numpyro": {
        "repo": "pyro-ppl/numpyro",
        "role": "JAX-JIT Hierarchical Empirical-Bayes partial pooling across Merit Farm strategy clusters (Apache-2.0)",
        "authority": "RESEARCH_CANDIDATE_PREWALL_ONLY",
    },
    "cvxpy_clarabel": {
        "repo": "cvxpy/cvxpy",
        "role": "Permissive Apache-2.0 convex optimization + Clarabel.rs conic solver replacing GPL-3.0 cvxportfolio",
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
        "role": "Deflated Sharpe Ratio (DSR), RAS, MinTRL, PBO & Triple-Barrier trade attribution (pinned v0.2.0)",
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
    symbol: str | None = None


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
    tags: list[str] | tuple[str, ...] | str,
    is_post_incentive: bool = False,
    holds_overnight_moc_to_moo: bool = False,
    unshielded_lit_passive: bool = False,
    in_opening_0930_0935_taker: bool = False,
    ofi_only_without_joint_sweep: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """Fail-closed firewall blocking any strategy or batch sweep from re-testing retired negative-EV traps."""
    violations: list[str] = []
    if isinstance(tags, str):
        norm_tags: tuple[str, ...] | list[str] = (tags,)
    elif isinstance(tags, bytes):
        norm_tags = (tags.decode("utf-8", errors="ignore"),)
    else:
        norm_tags = tags
    for raw_tag in norm_tags:
        norm = str(raw_tag or "").strip().upper()
        if norm in RETIRED_TRAP_DENYLIST:
            violations.append(f"RETIRED_TRAP:{norm}")
        if is_post_ant_strategy_quarantined(norm):
            violations.append(f"QUARANTINED_ADR0064:{norm}")
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
    raw_sid = str(strategy_id or "").strip()
    if not raw_sid:
        raise ValueError("strategy_id must be non-empty")
    sid = raw_sid.upper()
    if is_post_ant_strategy_quarantined(sid):
        raise ValueError(f"QUARANTINED_ADR0064: reflection tuning refused for quarantined strategy {raw_sid!r}")

    spec = CONVERGED_FOUNDATION_CATALOG.get(sid)
    contract = POST_ANT_STRATEGY_CONTRACTS_V2.get(sid)
    contract_max_stop = (
        min(ADR0022_MAX_STOP_PCT, float(contract["max_stop_pct"])) if contract is not None else ADR0022_MAX_STOP_PCT
    )
    base_stop_pct = min(spec.default_stop_pct if spec else contract_max_stop, contract_max_stop)
    base_max_spread_bps = (
        spec.max_spread_bps if spec else (float(contract["max_spread_bps"]) if contract is not None else 22.0)
    )
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
            "recommended_stop_pct": min(contract_max_stop, base_stop_pct),
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
            recommended_stop_pct=round(min(contract_max_stop, base_stop_pct), 2),
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
        if rec.strategy_id != strategy_id:
            if is_post_ant_strategy_quarantined(str(rec.strategy_id or "")):
                raise ValueError(
                    f"QUARANTINED_ADR0064: reflection record belongs to quarantined strategy {rec.strategy_id!r}"
                )
            raise ValueError(
                f"STRATEGY_ID_MISMATCH: record strategy_id={rec.strategy_id!r} != batch strategy_id={strategy_id!r}"
            )
        if contract is not None and contract["allowed_symbols"] is not None:
            rec_sym = str(rec.symbol or "").strip().upper()
            if not rec_sym or rec_sym not in contract["allowed_symbols"]:
                allowed_list = sorted(contract["allowed_symbols"])
                raise ValueError(
                    f"STRATEGY_SYMBOL_NOT_ALLOWED: {sid} requires symbol in {allowed_list}, got {rec.symbol!r}"
                )
        r_val = _require_finite_num(rec.realized_r, "realized_r")
        mfe_val = _require_finite_num(rec.mfe_r, "mfe_r")
        _require_finite_num(rec.mae_r, "mae_r")
        slip_val = _require_finite_num(rec.slippage_bps, "slippage_bps")
        spr_val = _require_finite_num(rec.spread_bps, "spread_bps")
        _require_finite_num(rec.entry_ofi, "entry_ofi")
        stop_val = _require_finite_num(rec.stop_pct, "stop_pct")
        if stop_val <= 0.0 or stop_val > contract_max_stop:
            raise ValueError(
                f"ADR-0022/contract violation in reflection record: stop_pct={stop_val} > {contract_max_stop}"
            )
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
        tightened_spr = round(base_max_spread_bps * 0.82, 2)
        rec_max_spread_bps = min(base_max_spread_bps, max(min(6.0, tightened_spr), tightened_spr))
        rec_min_ofi = min(0.60, round(base_min_ofi + 0.08, 3))
        actions.append("TIGHTEN_SPREAD_COLLAR_AND_REQUIRE_PASSIVE_MICROPRICE_PEG")

    if premature_rate_pct >= 25.0:
        # Widen slightly within contract/ADR-0022 ceiling AND arm lossless breakeven earlier
        rec_stop_pct = min(contract_max_stop, round(base_stop_pct + 0.15, 2))
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

    rec_stop_pct = round(min(contract_max_stop, max(min(0.50, contract_max_stop), rec_stop_pct)), 2)

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


def evaluate_post_ant_symbol_merit_contract(
    *,
    symbol: str,
    strategy_id: str,
    merit_score: float,
    df_pillars_manifested: int,
    spread_bps: float,
    requested_stop_pct: float = 1.60,
    sec_dilution_clean: bool = False,
    session_mode: str = "RTH_CORE",
    study_wall_cutoff: date = STUDY_WALL_DATE,
) -> dict[str, Any]:
    """Validate a Post-ANT whole-market merited symbol against ADR-0064 quarantine, ADR-0022 stop & pre-wall rules."""
    if study_wall_cutoff > STUDY_WALL_DATE:
        raise ValueError(
            f"Pre-wall breach: study_wall_cutoff={study_wall_cutoff.isoformat()} > {STUDY_WALL_DATE.isoformat()}"
        )
    sym = str(symbol or "").strip().upper()
    sid = str(strategy_id or "").strip().upper()
    sess = str(session_mode or "").strip().upper()
    if not sym or not sym.replace(".", "").isalnum():
        raise ValueError(f"Invalid symbol: {symbol!r}")
    if sess not in POST_ANT_VALID_SESSIONS:
        raise ValueError(f"Unsupported session_mode: {session_mode!r}")
    if isinstance(df_pillars_manifested, bool) or not isinstance(df_pillars_manifested, int):
        raise ValueError("df_pillars_manifested must be an int")
    if not isinstance(sec_dilution_clean, bool):
        raise ValueError("sec_dilution_clean must be a bool")
    if any(isinstance(x, bool) for x in (merit_score, spread_bps, requested_stop_pct)) or not (
        math.isfinite(merit_score) and math.isfinite(spread_bps) and math.isfinite(requested_stop_pct)
    ):
        raise ValueError("Numeric parameters must be finite non-bool numbers")

    reasons: list[str] = []
    contract = POST_ANT_STRATEGY_CONTRACTS_V2.get(sid)
    if is_post_ant_strategy_quarantined(sid):
        reasons.append(f"QUARANTINED_ADR0064:{sid}")
    elif contract is None:
        reasons.append(f"NON_ACTIVE_SOTA_STRATEGY:{sid}")
    else:
        if sess not in contract["allowed_sessions"]:
            reasons.append(f"STRATEGY_SESSION_NOT_ALLOWED:{sid}:{sess}")
        allowed_syms = contract["allowed_symbols"]
        if allowed_syms is not None and sym not in allowed_syms:
            reasons.append(f"STRATEGY_SYMBOL_NOT_ALLOWED:{sid}:{sym}")
        c_max_spr = float(contract["max_spread_bps"])
        if spread_bps > 0.0 and (
            (contract["spread_strict_lt"] and spread_bps >= c_max_spr)
            or (not contract["spread_strict_lt"] and spread_bps > c_max_spr)
        ):
            reasons.append(f"STRATEGY_SPREAD_EXCEEDS_CONTRACT:{sid}:{spread_bps}>{c_max_spr}")
        c_max_stop = float(contract["max_stop_pct"])
        if 0.0 < requested_stop_pct <= ADR0022_MAX_STOP_PCT and requested_stop_pct > c_max_stop:
            reasons.append(f"STRATEGY_STOP_EXCEEDS_CONTRACT:{sid}:{requested_stop_pct}>{c_max_stop}")

    if not sec_dilution_clean:
        reasons.append("SEC_90D_DILUTION_UNVERIFIED_OR_BLOCKED")
    if requested_stop_pct <= 0.0 or requested_stop_pct > ADR0022_MAX_STOP_PCT:
        reasons.append(f"ADR0022_STOP_EXCEEDS_CAP:{requested_stop_pct}")
    if spread_bps <= 0.0 or spread_bps > 35.0:
        reasons.append(f"SPREAD_ABOVE_BOATS_CAP:{spread_bps}")
    if df_pillars_manifested < 0 or df_pillars_manifested > 6:
        reasons.append(f"INVALID_DF_PILLARS_COUNT:{df_pillars_manifested}")
    elif df_pillars_manifested < 3:
        reasons.append(f"INSUFFICIENT_DF_PILLARS:{df_pillars_manifested}")

    admitted = len(reasons) == 0 and merit_score >= 38.0
    merit_tier = (
        "MERIT_TIER_1_CORE"
        if (admitted and merit_score >= 58.0 and spread_bps <= 15.0 and df_pillars_manifested >= 4)
        else ("MERIT_TIER_2_ACTIVE" if admitted else "REJECTED_OR_WATCH_ONLY")
    )
    return {
        "era": POST_ANT_ERA_ID,
        "schema_version": "sota.post-ant-merit-contract.v1",
        "study_wall_date": study_wall_cutoff.isoformat(),
        "symbol": sym,
        "strategy_id": sid,
        "merit_tier": merit_tier,
        "admitted": admitted,
        "rejection_reasons": reasons,
        "adr0022_max_stop_pct": ADR0022_MAX_STOP_PCT,
        "take_profit_cap_enabled": False,
        "promotion_track": "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "standalone_alpha_claim": False,
    }
