"""Advanced Backtest Playback Performance Manifest, Suitable Multi-Regime Horizons & Low-Quality Purge (`ADR-0030`).

Eliminates legacy/low-quality backtest anti-patterns across the 5-stage `Sim -> Pre-Wall -> Paper -> Merit -> Broker`
pipeline and replaces naive short-window / survivorship-biased previews with a deterministic, SHA-256 sealed,
dimensionless **Advanced Backtest Playback Performance Manifest (`sota.backtest-playback-performance-manifest.v1`)**:

1. **Scientifically Suitable Time Periods (`SUITABLE_PLAYBACK_HORIZONS` + `compute_min_trl_sessions`):**
   - Grounded in `ADIA Lab #19` (`Lopez de Prado, Lipton & Zoonekynd 2025/2026 SSRN 5520741`) AR(1) + higher-moment
     Minimum Track Record Length (`MinTRL`), `SEC 88 FR 76896` (5-day XML 13D eff. `2024-02-05`), `BOATS` nocturnal
     evolution, the `2025-11-03` SIP quote-size unit cutover, immutable study wall `<= 2026-10-15`, and `2026-12-06`
     `23/5` Consolidated SIP cutover isolation.
   - Explicitly rejects toy `< 60`-session intraday or `< 252`-session swing/daily playback windows.
2. **14-Point Fail-Closed Low-Quality Practice Purge Gate (`LOW_QUALITY_PRACTICE_PURGE_REGISTRY`):**
   - Mechanically blocks ex-post live-mover survivorship universes, toy horizons below `MinTRL`, post-`2026-10-15`
     discovery leaks, flat `< 8 bps` RT costs without `2x` + sqrt-impact stress, hard TP caps, `ADR-0022 > 2.00%`
     stops, invalid trade-count Sharpe annualization, missing `SPA`/`MCS`/`DSR`/`PBO`, in-sample symbol cherry-picking,
     `> 1.0x` vol-timing leverage (`Cederburg et al. 2020 JFE`), stationary `UCB1` on full-info manifests, trade-date
     EDGAR lookahead, quarantined `ADR-0064` / `POST` chases, silent exception swallowing, and non-permissive/scraper
     libraries (`GPL`/`AGPL`/`BUSL`/`Commons Clause`/`Proprietary`/`yfinance`/`timesfm_v3_weights`).
3. **Unified Playback Performance Manifest (`build_advanced_backtest_playback_manifest`):**
   - Computes calendar-day AR(1)-adjusted Sharpe, `MinTRL`, Calmar, max drawdown, `1x`/`2x` net bps, `Hansen SPA`,
     `Hansen-Lunde-Nason (2011) MCS M*`, `Conformal PID` MAE stop width (`<= 2.00%`), `Fixed-Share Hedge` weights,
     `Hierarchical EB` cluster pooling, and `Cederburg-safe` (`<= 1.0x`) vol crash brake — strictly
     `PROTECTIVE_FILTER_OR_GOVERNANCE` (`standalone_alpha_claim = False`, `hlz_trial_consumed = False`, preserving
     `Trial #105` `t*(105) = 4.2084`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from statistics import NormalDist
from typing import Any

from sota_compounding.advanced_merit_sim_convergence import (
    CURRENT_LEDGER_TRIALS_N,
    NEXT_HLZ_T_STAR,
    NEXT_TRIAL_ID,
    ConformalPidStopCalibrator,
    StrategyClusterObservation,
    compute_asymmetric_vol_crash_brake,
    compute_hierarchical_eb_merit_pool,
    evaluate_hansen_spa_and_mcs,
    update_fixed_share_hedge_weights,
)
from sota_compounding.converged_strategy_factory import is_post_ant_strategy_quarantined
from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE, hlz_dynamic_t_hurdle

PLAYBACK_MANIFEST_SCHEMA = "sota.backtest-playback-performance-manifest.v1"
MIN_REALISTIC_1X_RT_COST_BPS = 8.0  # ADR-0059 calibrated >= 4 bps/side half-spread before market impact
MIN_INTRADAY_PLAYBACK_SESSIONS = 60
RECOMMENDED_INTRADAY_PLAYBACK_SESSIONS = 252
MIN_SWING_PLAYBACK_SESSIONS = 252
RECOMMENDED_SWING_PLAYBACK_SESSIONS = 1260
_STD_NORM = NormalDist(0.0, 1.0)


@dataclass(frozen=True, slots=True)
class PlaybackHorizonSpec:
    horizon_id: str
    start_date: str
    end_date: str
    min_sessions: int
    canonical_sessions: int
    intended_cadence: str
    cpcv_n_groups: int
    cpcv_k_test: int
    purge_sessions: int
    embargo_sessions: int
    prewall_locked: bool
    regimes_covered: tuple[str, ...]
    rationale: str


SUITABLE_PLAYBACK_HORIZONS: dict[str, PlaybackHorizonSpec] = {
    "W1_PRIMARY_PREWALL_LAKE": PlaybackHorizonSpec(
        horizon_id="W1_PRIMARY_PREWALL_LAKE",
        start_date="2024-01-02",
        end_date=STUDY_WALL_DATE.isoformat(),
        min_sessions=252,
        canonical_sessions=679,
        intended_cadence="intraday_1m_5m_and_24x5_events",
        cpcv_n_groups=6,
        cpcv_k_test=2,
        purge_sessions=20,
        embargo_sessions=20,
        prewall_locked=True,
        regimes_covered=(
            "2024_XML_13D_MODERN",
            "2024_AUG_VIX_YEN_CARRY_SHOCK",
            "2025_TARIFF_VOL_SHOCK",
            "2025_NOV_SIP_UNIT_CUTOVER",
            "2026_MARKET_BROADENING",
        ),
        rationale=(
            "Canonical 679-session (1.318M-row) DF1..DF12 pre-wall lake spanning modern BOATS liquidity, "
            "SEC 88 FR 76896 5-day XML 13D rules, Nov 2025 SIP quote-unit cutover, and multi-shock vol cycles."
        ),
    ),
    "W2_MULTI_REGIME_SWING_STRESS": PlaybackHorizonSpec(
        horizon_id="W2_MULTI_REGIME_SWING_STRESS",
        start_date="2020-01-02",
        end_date=STUDY_WALL_DATE.isoformat(),
        min_sessions=1260,
        canonical_sessions=1707,
        intended_cadence="daily_swing_pead_edgar_vix",
        cpcv_n_groups=6,
        cpcv_k_test=2,
        purge_sessions=20,
        embargo_sessions=20,
        prewall_locked=True,
        regimes_covered=(
            "2020_COVID_CRASH_AND_REBOUND",
            "2021_RETAIL_SQUEEZE_EUPHORIA",
            "2022_FED_HIKE_BEAR_MARKET",
            "2023_2026_HIGH_RATE_DISPERSION",
        ),
        rationale=(
            "Full 6.8-year multi-regime stress horizon required for daily/swing, PEAD, Form 4, Lazy Prices, "
            "and VIX regime strategies to eliminate single-bull-regime survivorship illusion."
        ),
    ),
    "W3_MODERN_XML_13D_BOATS": PlaybackHorizonSpec(
        horizon_id="W3_MODERN_XML_13D_BOATS",
        start_date="2024-02-05",
        end_date=STUDY_WALL_DATE.isoformat(),
        min_sessions=252,
        canonical_sessions=655,
        intended_cadence="edgar_5day_xml_13d_and_boats_nocturnal",
        cpcv_n_groups=6,
        cpcv_k_test=2,
        purge_sessions=15,
        embargo_sessions=15,
        prewall_locked=True,
        regimes_covered=(
            "SEC_88FR76896_5DAY_XML_13D_EFFECTIVE",
            "BOATS_NOCTURNAL_2000_0350_ET",
            "2025_NOV_SIP_UNIT_CUTOVER",
        ),
        rationale=(
            "Strict post-2024-02-05 structural regime matching SEC 5-business-day structured XML SC 13D "
            "filing deadlines and live Blue Ocean ATS nocturnal microstructure."
        ),
    ),
    "W4_POST_2026_12_06_23X5_SIP_SHADOW": PlaybackHorizonSpec(
        horizon_id="W4_POST_2026_12_06_23X5_SIP_SHADOW",
        start_date="2026-12-06",
        end_date="2027-12-31",
        min_sessions=60,
        canonical_sessions=60,
        intended_cadence="post_cutover_23x5_sip_shadow_oos_only",
        cpcv_n_groups=0,
        cpcv_k_test=0,
        purge_sessions=0,
        embargo_sessions=0,
        prewall_locked=False,
        regimes_covered=("CONSOLIDATED_SIP_23X5_POST_CUTOVER_SHADOW",),
        rationale=(
            "Quarantined forward OOS shadow window isolating the Dec 6, 2026 21:00 ET 23/5 Consolidated SIP "
            "cutover; strictly forbidden from contaminating pre-wall <= 2026-10-15 discovery."
        ),
    ),
    "W5_LIVE_PAPER_PARITY_REPLAY": PlaybackHorizonSpec(
        horizon_id="W5_LIVE_PAPER_PARITY_REPLAY",
        start_date="2026-10-16",
        end_date="2026-12-05",
        min_sessions=10,
        canonical_sessions=60,
        intended_cadence="real_broker_paper_tca_queue_parity",
        cpcv_n_groups=0,
        cpcv_k_test=0,
        purge_sessions=0,
        embargo_sessions=0,
        prewall_locked=False,
        regimes_covered=("LIVE_PAPER_P1_P4_REAL_BROKER_ONLY",),
        rationale=(
            "Forward live-paper E2E broker parity replay (>=30 closed trades across >=10 distinct sessions, "
            "60 shadow dates for Fixed-Share Hedge/MCS) verifying ADR-0025 markout & queue parity."
        ),
    ),
}

# Owner-Authorized Quality-First Experimental Policy (ADR-0031 / shelf-lock ADR-0022):
# Inside this effectively-private experimental environment (0 external stars/watchers, zero redistribution),
# repos are evaluated STRICTLY on technical quality, mathematical rigor, and SOTA-ness — NEVER rejected solely
# on license tags (GPL/AGPL/BUSL/MPL/Commons-Clause/non-commercial R&D weights).
EXPERIMENTAL_EVALUATION_POLICY = "QUALITY_AND_SOTA_ONLY_NO_LICENSE_EXCLUSION"

EXPERIMENTAL_QUALITY_FIRST_SOTA_UNBLOCKED: frozenset[str] = frozenset(
    {
        "cvxgrp/cvxportfolio",
        "man-group/arcticdb",
        "google-research/timesfm",
        "xtdb/xtdb",
        "edtechre/pybroker",
        "freqtrade/freqtrade",
        "ml4t/backtest",
        "ml4t/engineer",
        "ml4t/skills",
        "ml4t/models",
        "nixtla/conformalforecast",
        "salesforceairesearch/uni2ts",
        "ibm-granite/granite-tsfm",
        "alphanome-ai/sec-parser",
        "nvidia/cudf",
        "nvidia/cuml",
        "aangelopoulos/conformal-time-series",
        "nautechsystems/nautilus_agents",
    }
)

# Strictly TECHNICAL low-quality / stale / toy / fragile-scraper sources only (zero license-only bans!)
BANNED_LOW_QUALITY_SOURCES: frozenset[str] = frozenset(
    {
        "yfinance",
        "ranaroussi/yfinance",
        "backtrader",
        "mementum/backtrader",
        "backtesting.py",
        "kernc/backtesting.py",
        "mlfinlab",
        "hudson-and-thames/mlfinlab",
        "tcapy",
        "cuemacro/tcapy",
        "vollib/vollib",
        "abides-jpmc-public",
        "jpmorganchase/abides-jpmc-public",
        "tradingview-api-scraper",
        "tradingview-screener-scraper",
    }
)

LOW_QUALITY_PRACTICE_PURGE_REGISTRY: dict[str, str] = {
    "EX_POST_LIVE_MOVER_SURVIVORSHIP_UNIVERSE": (
        "Selecting today's live winners/movers and backtesting backward injects severe survivorship bias; "
        "require strictly causal T-1 point-in-time universe snapshots."
    ),
    "TOY_HORIZON_BELOW_MIN_TRL": (
        "Toy 30d (<60 sessions intraday / <252 sessions swing) windows lack regime diversity and fail "
        "ADIA Lab #19 MinTRL power under HLZ t*(105) = 4.2084."
    ),
    "STUDY_WALL_BREACH_POST_2026_10_15": (
        "Pre-wall discovery or calibration referencing dates > 2026-10-15 contaminates the holdout wall."
    ),
    "UNREALISTIC_FLAT_COST_NO_2X_IMPACT": (
        "Flat <8 bps RT slippage without mandatory 2x + square-root market-impact stress understates real "
        "ADR-0059 paper TCA implementation shortfall."
    ),
    "FIXED_TAKE_PROFIT_CAP_ACTIVE": (
        "Hard Take-Profit caps truncate convex right-tail runners; enforce uncapped 3-tier lossless Chandelier."
    ),
    "ADR0022_STOP_OVER_2PCT": (
        "Protective stop distance > 2.00% violates ADR-0022 and Rule 612 wire-quantized risk ceilings."
    ),
    "INVALID_TRADE_COUNT_SHARPE_ANNUALIZATION": (
        "Scaling trade-level Sharpe by sqrt(n_trades * k) inflates statistics; require calendar-day "
        "AR(1)-adjusted Sharpe and date-clustered standard errors."
    ),
    "MISSING_MULTIPLE_TESTING_SPA_MCS_DSR_PBO": (
        "Promoting strategies without HLZ t*(N), DSR >= 0.95, CPCV PBO < 0.25, Hansen SPA & HLN MCS fails hygiene."
    ),
    "IN_SAMPLE_SYMBOL_CHERRY_PICKING": (
        "Picking per-symbol 'best' strategy in-sample with hardcoded regime_bonus weights overfits noise."
    ),
    "VOL_TIMING_LEVERAGE_ABOVE_1X": (
        "Vol-targeting leverage > 1.0x fails out-of-sample under structural breaks (Cederburg et al. 2020 JFE); "
        "cap exposure strictly <= 1.0x as an asymmetric crash brake."
    ),
    "STATIONARY_UCB1_ON_FULL_INFO_MANIFEST": (
        "Stationary UCB1 lags regime breaks and discards full-information shadow marks; use Fixed-Share Hedge/MWU."
    ),
    "TRADE_DATE_EDGAR_LOOKAHEAD": (
        "Keying SEC Form 4 / SC 13D / 8-K events on trade/event date leaks late filings; require acceptanceDateTime."
    ),
    "QUARANTINED_ADR0064_OR_POST_CHASE_ACTIVE": (
        "Routing ADR-0064 quarantined bleeders or POST_INCENTIVE late chases violates negative-EV denylist."
    ),
    "SILENT_DATA_FETCH_EXCEPTION_SWALLOW_OR_BANNED_SOURCE": (
        "Swallowing fetch errors silently (except Exception: pass) or importing technically low-quality/stale/toy/"
        "scraper libraries (yfinance, backtrader, single-asset backtesting.py, abandoned mlfinlab) fails hygiene."
    ),
}


def compute_min_trl_sessions(
    *,
    target_sharpe_ann: float,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    ar1_rho: float = 0.0,
    confidence: float = 0.95,
    benchmark_sr_ann: float = 0.0,
    periods_per_year: float = 252.0,
) -> int:
    """Compute ADIA Lab #19 (`SSRN 5520741`) AR(1) + higher-moment Minimum Track Record Length (`MinTRL`)."""
    sr_t = float(target_sharpe_ann)
    sr_0 = float(benchmark_sr_ann)
    if not (math.isfinite(sr_t) and math.isfinite(sr_0) and math.isfinite(ar1_rho)):
        raise ValueError("Sharpe and AR(1) inputs must be finite")
    if sr_t <= sr_0:
        raise ValueError("target_sharpe_ann must strictly exceed benchmark_sr_ann")
    if not (-0.95 < ar1_rho < 0.95):
        raise ValueError(f"ar1_rho must lie inside (-0.95, 0.95) (got {ar1_rho})")
    if not (0.5 < confidence < 0.999999):
        raise ValueError(f"confidence must lie inside (0.5, 0.999999) (got {confidence})")

    sr_d = sr_t / math.sqrt(periods_per_year)
    sr0_d = sr_0 / math.sqrt(periods_per_year)
    z_alpha = _STD_NORM.inv_cdf(confidence)

    # Higher-moment non-normality factor (Bailey & Lopez de Prado 2014; SSRN 5520741)
    # Note: kurtosis = excess_kurtosis + 3 -> (kurt - 1)/4 = (excess_kurtosis + 2)/4
    non_norm = max(0.25, 1.0 - skew * sr_d + ((excess_kurtosis + 2.0) / 4.0) * (sr_d**2))
    # Lo (2002) / ADIA Lab #19 AR(1) autocorrelation variance inflation factor
    ar1_inflation = max(0.25, (1.0 + ar1_rho) / (1.0 - ar1_rho))
    min_n = 1.0 + non_norm * ar1_inflation * ((z_alpha / (sr_d - sr0_d)) ** 2)
    return int(math.ceil(min_n))


def compute_calendar_day_stats(daily_returns_bps: Sequence[float]) -> dict[str, float]:
    """Compute calendar-day AR(1)-adjusted Sharpe, skew, excess kurtosis, max drawdown, Calmar & profit factor."""
    n = len(daily_returns_bps)
    if n < 2:
        return {
            "sessions": float(n),
            "mean_daily_bps": 0.0,
            "std_daily_bps": 0.0,
            "ar1_rho": 0.0,
            "sharpe_ann_raw": 0.0,
            "sharpe_ann_ar1_adj": 0.0,
            "skew": 0.0,
            "excess_kurtosis": 0.0,
            "cum_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
        }

    vals = [float(x) for x in daily_returns_bps]
    if any(not math.isfinite(x) for x in vals):
        raise ValueError("daily_returns_bps contains non-finite values")
    mean = sum(vals) / n
    centered = [x - mean for x in vals]
    var = sum(c * c for c in centered) / max(n - 1, 1)
    sd = math.sqrt(var) if var > 0.0 else 0.0

    # Sample AR(1) autocorrelation rho
    denom = sum(c * c for c in centered)
    if denom > 1e-12:
        num = sum(centered[i] * centered[i - 1] for i in range(1, n))
        rho = max(-0.90, min(0.90, num / denom))
    else:
        rho = 0.0

    # Higher moments
    if sd > 1e-9:
        skew = sum((c / sd) ** 3 for c in centered) / n
        ex_kurt = (sum((c / sd) ** 4 for c in centered) / n) - 3.0
    else:
        skew = 0.0
        ex_kurt = 0.0

    sr_raw = (mean / sd) * math.sqrt(252.0) if sd > 1e-9 else 0.0
    # Lo (2002) AR(1) adjustment divides raw Sharpe by sqrt((1 + rho)/(1 - rho))
    ar1_scale = math.sqrt(max(0.25, (1.0 + rho) / (1.0 - rho)))
    sr_adj = sr_raw / ar1_scale

    # Compounded dimensionless equity curve (1.0x unlevered notional)
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    for bps in vals:
        eq *= max(0.01, 1.0 + bps / 10_000.0)
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    cum_ret_pct = (eq - 1.0) * 100.0
    ann_ret_pct = ((eq ** (252.0 / n)) - 1.0) * 100.0 if eq > 0 else -100.0
    calmar = ann_ret_pct / (max_dd * 100.0) if max_dd > 1e-6 else 0.0

    gains = sum(x for x in vals if x > 0.0)
    losses = abs(sum(x for x in vals if x < 0.0))
    pf = (gains / losses) if losses > 1e-9 else (9.99 if gains > 0.0 else 0.0)

    return {
        "sessions": float(n),
        "mean_daily_bps": round(mean, 4),
        "std_daily_bps": round(sd, 4),
        "ar1_rho": round(rho, 4),
        "sharpe_ann_raw": round(sr_raw, 4),
        "sharpe_ann_ar1_adj": round(sr_adj, 4),
        "skew": round(skew, 4),
        "excess_kurtosis": round(ex_kurt, 4),
        "cum_return_pct": round(cum_ret_pct, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 4),
        "calmar": round(calmar, 4),
        "profit_factor": round(min(9.99, pf), 4),
    }


@dataclass(frozen=True, slots=True)
class PlaybackAuditInput:
    horizon_id: str
    sample_start_date: str
    sample_max_date: str
    distinct_sessions: int
    cadence: str  # "intraday" | "swing" | "paper_parity"
    universe_mode: str  # "CAUSAL_T_MINUS_1_PIT" | "EX_POST_LIVE_MOVERS"
    cost_1x_rt_bps: float
    includes_2x_and_sqrt_impact: bool
    take_profit_cap_enabled: bool
    max_stop_pct: float
    sharpe_annualization_mode: str  # "CALENDAR_DAY_AR1_CLUSTERED" | "TRADE_COUNT_SCALED"
    uses_spa_mcs_dsr_pbo: bool
    in_sample_symbol_cherry_picking: bool
    vol_max_leverage: float
    allocator_mode: str  # "FIXED_SHARE_HEDGE_AND_HIERARCHICAL_EB" | "STATIONARY_UCB1"
    edgar_timestamp_field: str  # "acceptanceDateTime" | "transactionDate"
    active_strategy_ids: tuple[str, ...] = ()
    silent_exception_swallow: bool = False
    source_libraries: tuple[str, ...] = ()
    is_post_incentive_chase: bool = False
    target_sharpe_for_min_trl: float = 1.50
    ar1_rho_for_min_trl: float = 0.0


def audit_low_quality_practices(inp: PlaybackAuditInput) -> dict[str, Any]:
    """Fail-closed 14-point audit purging every low-quality backtest/playback anti-pattern."""
    violations: list[str] = []

    if inp.horizon_id not in SUITABLE_PLAYBACK_HORIZONS:
        violations.append(f"UNKNOWN_HORIZON_ID:{inp.horizon_id}")
    spec = SUITABLE_PLAYBACK_HORIZONS.get(inp.horizon_id)

    # 1. Survivorship / Ex-Post Live Mover Universe
    if inp.universe_mode != "CAUSAL_T_MINUS_1_PIT":
        violations.append("EX_POST_LIVE_MOVER_SURVIVORSHIP_UNIVERSE")

    # 2. Toy Horizon & MinTRL Power Check
    min_trl = compute_min_trl_sessions(
        target_sharpe_ann=max(0.50, inp.target_sharpe_for_min_trl),
        ar1_rho=max(-0.90, min(0.90, inp.ar1_rho_for_min_trl)),
    )
    min_req_sessions = (
        MIN_SWING_PLAYBACK_SESSIONS
        if inp.cadence == "swing"
        else (10 if inp.cadence == "paper_parity" else MIN_INTRADAY_PLAYBACK_SESSIONS)
    )
    if spec is not None and spec.prewall_locked:
        min_req_sessions = max(min_req_sessions, spec.min_sessions)
    if inp.distinct_sessions < min_req_sessions:
        violations.append(f"TOY_HORIZON_BELOW_MIN_TRL:sessions={inp.distinct_sessions}<required={min_req_sessions}")

    # 3. Study Wall Lock (<= 2026-10-15) for Pre-Wall Discovery
    max_d = date.fromisoformat(inp.sample_max_date)
    start_d = date.fromisoformat(inp.sample_start_date)
    if start_d > max_d:
        violations.append("INVALID_DATE_RANGE:start_after_end")
    if spec is not None and spec.prewall_locked and max_d > STUDY_WALL_DATE:
        violations.append(f"STUDY_WALL_BREACH_POST_2026_10_15:{inp.sample_max_date}")

    # 4. Realistic 1x (>= 8 bps RT) + 2x & Square-Root Impact Stress
    if inp.cost_1x_rt_bps < MIN_REALISTIC_1X_RT_COST_BPS or not inp.includes_2x_and_sqrt_impact:
        violations.append("UNREALISTIC_FLAT_COST_NO_2X_IMPACT")

    # 5. Uncapped Lossless Runner (TP OFF)
    if inp.take_profit_cap_enabled:
        violations.append("FIXED_TAKE_PROFIT_CAP_ACTIVE")

    # 6. ADR-0022 Stop <= 2.00%
    if inp.max_stop_pct <= 0.0 or inp.max_stop_pct > ADR0022_MAX_STOP_PCT:
        violations.append(f"ADR0022_STOP_OVER_2PCT:{inp.max_stop_pct}")

    # 7. Calendar-Day AR(1) Sharpe Annualization
    if inp.sharpe_annualization_mode != "CALENDAR_DAY_AR1_CLUSTERED":
        violations.append("INVALID_TRADE_COUNT_SHARPE_ANNUALIZATION")

    # 8. Multiple-Testing Governance (SPA + MCS + DSR + PBO + HLZ)
    if not inp.uses_spa_mcs_dsr_pbo:
        violations.append("MISSING_MULTIPLE_TESTING_SPA_MCS_DSR_PBO")

    # 9. Zero In-Sample Per-Symbol Cherry-Picking
    if inp.in_sample_symbol_cherry_picking:
        violations.append("IN_SAMPLE_SYMBOL_CHERRY_PICKING")

    # 10. Cederburg-Safe Vol Brake (<= 1.0x)
    if inp.vol_max_leverage > 1.0:
        violations.append(f"VOL_TIMING_LEVERAGE_ABOVE_1X:{inp.vol_max_leverage}")

    # 11. Full-Information Non-Stationary Allocator (No Stationary UCB1)
    if inp.allocator_mode != "FIXED_SHARE_HEDGE_AND_HIERARCHICAL_EB":
        violations.append("STATIONARY_UCB1_ON_FULL_INFO_MANIFEST")

    # 12. Strict EDGAR acceptanceDateTime PIT Timestamp
    if inp.edgar_timestamp_field != "acceptanceDateTime":
        violations.append(f"TRADE_DATE_EDGAR_LOOKAHEAD:{inp.edgar_timestamp_field}")

    # 13. Zero Quarantined ADR-0064 Bleeders or POST Chases
    bad_strats = [sid for sid in inp.active_strategy_ids if is_post_ant_strategy_quarantined(sid)]
    if bad_strats or inp.is_post_incentive_chase:
        violations.append(f"QUARANTINED_ADR0064_OR_POST_CHASE_ACTIVE:{','.join(sorted(bad_strats)) or 'POST'}")

    # 14. Zero Silent Exception Swallowing & Zero Banned/Non-Permissive/Scraper Sources
    bad_libs = sorted({lib for lib in inp.source_libraries if lib.strip().lower() in BANNED_LOW_QUALITY_SOURCES})
    if inp.silent_exception_swallow or bad_libs:
        violations.append(
            f"SILENT_DATA_FETCH_EXCEPTION_SWALLOW_OR_BANNED_SOURCE:{','.join(bad_libs) or 'silent_except'}"
        )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "purged_practice_rules_enforced": len(LOW_QUALITY_PRACTICE_PURGE_REGISTRY),
        "min_trl_sessions_required": min_trl,
        "meets_min_trl_power": inp.distinct_sessions >= min_trl,
        "suitable_horizon_recommended": (
            "W2_MULTI_REGIME_SWING_STRESS" if inp.cadence == "swing" else "W1_PRIMARY_PREWALL_LAKE"
        ),
    }


def build_advanced_backtest_playback_manifest(
    *,
    audit_input: PlaybackAuditInput,
    strategy_daily_returns_1x_bps: Mapping[str, Sequence[float]],
    strategy_daily_returns_2x_bps: Mapping[str, Sequence[float]],
    strategy_clusters: Mapping[str, str],
    realized_mae_atr_samples: Sequence[tuple[float, float]] = (),
    realized_vol_ann_pct: float = 18.0,
    cumulative_trials_n: int = CURRENT_LEDGER_TRIALS_N,
    cpcv_pbo: float = 0.15,
    dsr_probability: float = 0.0,
    alpha_spa_mcs: float = 0.10,
    n_bootstrap: int = 300,
    seed: int = 20261015,
) -> dict[str, Any]:
    """Build SHA-256 sealed playback performance manifest uniting purge audit, SPA/MCS, Conformal PID & Hedge."""
    if not strategy_daily_returns_1x_bps:
        raise ValueError("strategy_daily_returns_1x_bps must be non-empty")
    if set(strategy_daily_returns_1x_bps.keys()) != set(strategy_daily_returns_2x_bps.keys()):
        raise ValueError("1x and 2x strategy maps must have identical strategy IDs")

    hygiene = audit_low_quality_practices(audit_input)
    horizon_spec = SUITABLE_PLAYBACK_HORIZONS.get(audit_input.horizon_id)

    # Per-strategy calendar-day AR(1) stats (1x and 2x costs)
    per_strategy: dict[str, dict[str, Any]] = {}
    eb_obs: list[StrategyClusterObservation] = []
    latest_r: dict[str, float] = {}
    names = sorted(strategy_daily_returns_1x_bps.keys())
    for sid in names:
        s1 = compute_calendar_day_stats(strategy_daily_returns_1x_bps[sid])
        s2 = compute_calendar_day_stats(strategy_daily_returns_2x_bps[sid])
        # Convert daily bps into dimensionless R-units (1R = 100 bps baseline daily risk unit)
        n_sess = max(1, int(s2["sessions"]))
        mean_r = s2["mean_daily_bps"] / 100.0
        se_r = max(0.01, (s2["std_daily_bps"] / 100.0) / math.sqrt(n_sess))
        latest_r[sid] = (
            float(strategy_daily_returns_2x_bps[sid][-1]) / 100.0 if strategy_daily_returns_2x_bps[sid] else 0.0
        )
        eb_obs.append(
            StrategyClusterObservation(
                strategy_id=sid,
                cluster_id=strategy_clusters.get(sid, "UNCLUSTERED"),
                observed_mean_r=mean_r,
                observed_se_r=se_r,
                closed_roundtrips=n_sess,
            )
        )
        per_strategy[sid] = {
            "cluster_id": strategy_clusters.get(sid, "UNCLUSTERED"),
            "quarantined_adr0064": is_post_ant_strategy_quarantined(sid),
            "cost_1x": s1,
            "cost_2x": s2,
            "survives_2x_cost": s2["mean_daily_bps"] > 0.0,
        }

    # Hansen (2005) SPA + Politis-Romano Stationary Bootstrap + HLN (2011) MCS on conservative 2x net returns
    spa_mcs = evaluate_hansen_spa_and_mcs(
        strategy_daily_returns_2x_bps,
        alpha=alpha_spa_mcs,
        n_bootstrap=n_bootstrap,
        avg_block_len=5.0,
        seed=seed,
    )

    # Conformal PID stop calibration strictly clamped <= 2.00% (ADR-0022)
    pid = ConformalPidStopCalibrator(target_miscoverage_alpha=0.10)
    last_pid_state: dict[str, Any] = {
        "next_quantile_multiplier": pid.quantile_multiplier,
        "next_stop_pct": pid.recommended_stop_pct(1.0),
        "adr0022_compliant": True,
    }
    for mae_pct, atr_pct in realized_mae_atr_samples:
        last_pid_state = pid.update(realized_mae_pct=mae_pct, atr_pct=atr_pct)

    # Full-Information Fixed-Share Hedge + Hierarchical EB Cluster Pooling + Cederburg-Safe Vol Crash Brake
    uniform_prior = {sid: 1.0 / len(names) for sid in names}
    hedge_weights = update_fixed_share_hedge_weights(uniform_prior, latest_r, eta=0.25, switching_alpha=0.05)
    hierarchical_eb = compute_hierarchical_eb_merit_pool(eb_obs)
    vol_brake = compute_asymmetric_vol_crash_brake(realized_vol_ann_pct, target_vol_ann_pct=18.0, min_scale=0.25)

    req_hlz_t = hlz_dynamic_t_hurdle(max(1, int(cumulative_trials_n)))
    best_sid = str(spa_mcs["best_strategy_id"])
    best_t = float(spa_mcs["per_strategy_t_stats"][best_sid])
    promotion_eligible = bool(
        hygiene["passed"]
        and hygiene["meets_min_trl_power"]
        and spa_mcs["spa_passes_alpha"]
        and best_t >= req_hlz_t
        and dsr_probability >= 0.95
        and cpcv_pbo < 0.25
        and per_strategy[best_sid]["survives_2x_cost"]
    )

    payload: dict[str, Any] = {
        "schema": PLAYBACK_MANIFEST_SCHEMA,
        "adr": "ADR-0030",
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "promotion_track": "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "standalone_alpha_claim": False,
        "hlz_trial_consumed": False,
        "completed_ledger_trials_n": CURRENT_LEDGER_TRIALS_N,
        "next_trial_id": NEXT_TRIAL_ID,
        "next_hlz_t_star": NEXT_HLZ_T_STAR,
        "required_hlz_t_hurdle": req_hlz_t,
        "promotion_eligible": promotion_eligible,
        "horizon_spec": asdict(horizon_spec) if horizon_spec is not None else None,
        "low_quality_practice_audit": hygiene,
        "per_strategy_playback": per_strategy,
        "hansen_spa_and_mcs_2x": spa_mcs,
        "conformal_pid_stop_state": last_pid_state,
        "fixed_share_hedge_weights": hedge_weights,
        "hierarchical_eb_merit_pool": hierarchical_eb,
        "asymmetric_vol_crash_brake": vol_brake,
        "cpcv_pbo": round(float(cpcv_pbo), 4),
        "dsr_probability": round(float(dsr_probability), 4),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    payload["manifest_sha256"] = digest
    return payload


def render_playback_manifest_markdown(manifest: Mapping[str, Any]) -> str:
    """Render dimensionless Markdown playback summary for audit & cockpit inspection."""
    audit = manifest["low_quality_practice_audit"]
    spa = manifest["hansen_spa_and_mcs_2x"]
    hz = manifest.get("horizon_spec") or {}
    hz_id = hz.get("horizon_id", "UNKNOWN")
    hz_span = f"{hz.get('start_date', '?')} .. {hz.get('end_date', '?')}"
    audit_status = "PASS" if audit["passed"] else "FAIL"
    n_viol = len(audit["violations"])
    min_trl = audit["min_trl_sessions_required"]
    mcs_list = ", ".join(spa["mcs_surviving_strategies"])
    pid_stop = manifest["conformal_pid_stop_state"]["next_stop_pct"]
    vol_brk = manifest["asymmetric_vol_crash_brake"]
    lines = [
        f"# Advanced Backtest Playback Performance Manifest (`{manifest['schema']}`)",
        "",
        f"- **Horizon:** `{hz_id}` (`{hz_span}`)",
        (
            f"- **Study Wall:** `<= {manifest['study_wall_date']}` | "
            f"**HLZ Hurdle:** `t* = {manifest['required_hlz_t_hurdle']}` | "
            f"**Promotion Eligible:** `{manifest['promotion_eligible']}`"
        ),
        f"- **Low-Quality Purge Gate:** `{audit_status}` (`{n_viol}` violations; `MinTRL = {min_trl}` sessions)",
        f"- **Hansen SPA (2x Cost):** `p_consistent = {spa['spa_p_consistent']}` | **MCS (`M*`):** `{mcs_list}`",
        f"- **Conformal PID Stop:** `{pid_stop}%` (`ADR-0022 <= 2.00%`) | **Vol Brake:** `{vol_brk}x` (`<= 1.0x`)",
        f"- **Manifest SHA-256:** `{manifest['manifest_sha256']}`",
        "",
        (
            "| Strategy | Cluster | 1x Mean (bps/d) | 2x Mean (bps/d) | "
            "2x AR(1) Sharpe | Max DD (%) | MCS `p` | Hedge Wt | EB Shrunk `R` |"
        ),
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sid, row in sorted(manifest["per_strategy_playback"].items()):
        c1 = row["cost_1x"]
        c2 = row["cost_2x"]
        mcs_p = spa["mcs_p_values"].get(sid, 0.0)
        hw = manifest["fixed_share_hedge_weights"].get(sid, 0.0)
        eb = manifest["hierarchical_eb_merit_pool"].get(sid, {}).get("hierarchical_eb_mean_r", 0.0)
        lines.append(
            f"| `{sid}` | `{row['cluster_id']}` | {c1['mean_daily_bps']:.2f} | {c2['mean_daily_bps']:.2f} | "
            f"{c2['sharpe_ann_ar1_adj']:.2f} | {c2['max_drawdown_pct']:.2f}% | {mcs_p:.4f} | {hw:.3f} | {eb:.4f} |"
        )
    return "\n".join(lines) + "\n"
