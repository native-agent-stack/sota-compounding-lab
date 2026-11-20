"""Advanced Backtest Playback Performance Manifest, Suitable Multi-Regime Horizons & Low-Quality Purge (`ADR-0030`).

Eliminates legacy/low-quality backtest anti-patterns across the 5-stage `Sim -> Pre-Wall -> Paper -> Merit -> Broker`
pipeline and replaces naive short-window / survivorship-biased previews with a deterministic, SHA-256 sealed,
dimensionless **Advanced Backtest Playback Performance Manifest (`sota.backtest-playback-performance-manifest.v1`)**:

1. **Scientifically Suitable Time Periods (`SUITABLE_PLAYBACK_HORIZONS` + `compute_min_trl_sessions`):**
   - Exact `ADIA Lab #19` (`Lopez de Prado, Lipton & Zoonekynd 2025/2026 SSRN 5520741` Eq. 11) AR(1) + higher-moment
     `MinTRL` in 80-digit `Decimal` with exact `Fraction` Pearson feasibility (`excess_kurtosis + 2 >= skew^2`,
     zero underflow even at `1e-200`) and overflow-safe evaluation.
   - Official NYSE (`XNYS`) trading-session calendar (weekends, Rule 7.2 US equity holidays & special closures;
     exact `W2 = 1706` sessions), mandatory contiguous NYSE `session_dates` for promotion (no omitted interior
     sessions), chronological NYSE `mae_atr_dates`, cadence-horizon alignment, and forward shadow isolation (`W4`/`W5`).
2. **14-Point Fail-Closed Low-Quality Practice Purge Gate (`LOW_QUALITY_PRACTICE_PURGE_REGISTRY`):**
   - Strict boolean, finite-real, non-scalar sequence validation across `active_strategy_ids`, `candidate_strategy_ids`,
     and `source_libraries`; `W5` requires `>= 30` closed roundtrips across `>= 10` distinct trade sessions bounded by
     `observed_trade_sessions <= min(sessions, roundtrips)` per strategy.
3. **Unified Playback Performance Manifest (`build_advanced_backtest_playback_manifest`):**
   - Exact `Fraction` rational compounded equity (zero float64/Decimal rounding drift on exact 1.0 break-even series)
     with `2^-1074` absorbing underflow bankruptcy, 80-digit `Decimal` log-equity drawdown/Calmar & scale-safe
     `profit_factor`, log-space tail-safe HLZ hurdle (`N >= 10^310`), exact **Bailey & López de Prado (2014) Eq. 2**
     PSD observed-Sharpe `DSR` (plus strictly monotone `ADIA Lab #19` null-Sharpe companion), infinite-Sharpe-aware
     cross-trial dispersion bound, scale-invariant Sharpe-only exact-purge `C(6, 2) = 15` CPCV PBO requiring all
     declared `active_strategy_ids`, all-strategy 2x cost drag with explicit `roundtrip_notional_fraction`, adaptive
     audited-cap Conformal PID (down to `0.0001%`, failing closed below `0.0001%`), and JSON-safe raw-evidence
     SHA-256 sealing (`allow_nan=False`) including `seed`, `n_bootstrap`, `alpha_spa_mcs`, and `audit_input`.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal, Overflow, localcontext
from fractions import Fraction
from functools import lru_cache
from statistics import NormalDist
from typing import Any, TypedDict

from sota_compounding.advanced_merit_sim_convergence import (
    CURRENT_LEDGER_TRIALS_N,
    NEXT_HLZ_T_STAR,
    NEXT_TRIAL_ID,
    ConformalPidStopCalibrator,
    StrategyClusterObservation,
    _exact_bartlett_hac_t_key,
    _exact_t_cmp,
    _to_exact_dyadic_ints,
    compute_asymmetric_vol_crash_brake,
    compute_hierarchical_eb_merit_pool,
    evaluate_hansen_spa_and_mcs,
    run_fixed_share_hedge_sequence,
)
from sota_compounding.converged_strategy_factory import is_post_ant_strategy_quarantined
from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE

PLAYBACK_MANIFEST_SCHEMA = "sota.backtest-playback-performance-manifest.v1"
MIN_REALISTIC_1X_RT_COST_BPS = 8.0  # ADR-0059 calibrated >= 4 bps/side half-spread before market impact
MIN_REPRESENTABLE_STOP_PCT = 0.0001
MIN_ROUNDTRIP_NOTIONAL_FRACTION = 0.01
MIN_INTRADAY_PLAYBACK_SESSIONS = 60
RECOMMENDED_INTRADAY_PLAYBACK_SESSIONS = 252
MIN_SWING_PLAYBACK_SESSIONS = 252
RECOMMENDED_SWING_PLAYBACK_SESSIONS = 1260
W5_MIN_CLOSED_ROUNDTRIPS = 30
W5_MIN_TRADE_SESSIONS = 10
_STD_NORM = NormalDist(0.0, 1.0)
_HLZ_BASE_ONE_SIDED_P = 0.00135
_EULER_MASCHERONI = 0.5772156649015329
_MAX_STATIONARY_RHO = 0.99
_MAX_JSON_PCT_CAP = sys.float_info.max
_BINARY64_MIN_SUBNORMAL_FRAC = Fraction(1, 1 << 1074)
_DEC_80_ONE = Decimal(1)
_DEC_80_ZERO = Decimal(0)
_DEC_80_TEN_K = Decimal(10_000)

# Special non-recurring NYSE market closures (national days of mourning / emergency closures)
_NYSE_SPECIAL_CLOSURES: frozenset[date] = frozenset(
    {
        date(2018, 12, 5),  # President George H.W. Bush National Day of Mourning
        date(2025, 1, 9),  # President Jimmy Carter National Day of Mourning
    }
)


def _is_finite_real(val: Any) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    try:
        f_v = float(val)
    except OverflowError:
        return False
    if not math.isfinite(f_v):
        return False
    if isinstance(val, int) and int(f_v) != val:
        return False
    return True


def _require_finite_real(val: Any, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite non-boolean number (got {val!r})")
    try:
        f_v = float(val)
    except OverflowError as exc:
        raise ValueError(f"{name} overflows IEEE-754 binary64 (got {val!r})") from exc
    if not math.isfinite(f_v):
        raise ValueError(f"{name} must be a finite non-boolean number (got {val!r})")
    if isinstance(val, int) and int(f_v) != val:
        raise ValueError(f"{name} integer is not exactly representable in IEEE-754 binary64 (got {val!r})")
    return f_v


def _require_nonbool_int(val: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be a non-boolean integer (got {val!r})")
    if minimum is not None and val < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {val})")
    return val


def _safe_num_str(val: Any) -> str:
    """Format arbitrary numbers/objects without triggering CPython's 4,300-digit int-str limit."""
    if isinstance(val, int) and not isinstance(val, bool) and val.bit_length() > 14_000:
        return hex(val)
    return repr(val) if isinstance(val, (str, bool, type(None))) else str(val)


def _json_safe_evidence(obj: Any) -> Any:
    """Canonicalize arbitrary audit/evidence values so `json.dumps(..., allow_nan=False)` NEVER crashes."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        # Guard CPython's 4,300-digit int-str JSON limit (`10**5000` serializes deterministically via hex!)
        return obj if obj.bit_length() <= 14_000 else {"__big_int_hex__": hex(obj)}
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else {"__non_finite__": repr(obj)}
    if isinstance(obj, bytes):
        return {"__bytes_hex__": obj.hex()}
    if isinstance(obj, (set, frozenset)):
        items = [_json_safe_evidence(x) for x in obj]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"__set__": items}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_evidence(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): _json_safe_evidence(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    return {"__repr__": repr(obj)}


def _inv_norm_upper_from_log_p(log_p: float) -> float:
    """Invert `log(Phi(-z)) = log_p` to full float64 precision (immune to subnormal underflow past `N = 10^310`)."""
    if not math.isfinite(log_p) or log_p >= 0.0:
        raise ValueError(f"log_p must be finite and < 0 (got {log_p!r})")
    if log_p > -700.0:
        return float(-_STD_NORM.inv_cdf(math.exp(log_p)))
    # Asymptotic Mills-ratio Newton refinement: exact to < 1 ULP for all log_p <= -700 (z >= 37.3)
    t = -2.0 * log_p
    z = math.sqrt(t - math.log(math.pi * t))
    half_log_2pi = 0.5 * math.log(2.0 * math.pi)
    for _ in range(5):
        inv_z2 = 1.0 / (z * z)
        # Asymptotic expansion of z * sqrt(2*pi) * exp(z^2/2) * Phi(-z)
        poly = 1.0 - inv_z2 * (1.0 - inv_z2 * (3.0 - inv_z2 * (15.0 - inv_z2 * (105.0 - inv_z2 * 945.0))))
        f_val = -0.5 * z * z - math.log(z) - half_log_2pi + math.log(poly) - log_p
        df_val = -z - (1.0 / z)
        step = f_val / df_val
        z -= step
        if abs(step) <= 1e-15 * z:
            break
    return z


def _hlz_t_hurdle_raw(cumulative_trials_n: int) -> float:
    """Exact tail-safe HLZ Bonferroni quantile via log-space inversion (accurate past `N = 10^310`)."""
    n_i = _require_nonbool_int(cumulative_trials_n, "cumulative_trials_n", minimum=1)
    log_p = math.log(_HLZ_BASE_ONE_SIDED_P) - math.log(n_i)
    return _inv_norm_upper_from_log_p(log_p)


def _gregorian_easter(year: int) -> date:
    """Anonymous Gregorian (Meeus/Jones/Butcher) algorithm for exact Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    next_month = date(year + (month // 12), (month % 12) + 1, 1)
    last_day = next_month - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _observed_fixed_holiday(d: date, *, observe_saturday_friday: bool = True) -> date | None:
    if d.weekday() == 5:
        return (d - timedelta(days=1)) if observe_saturday_friday else None
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def _nyse_holidays_for_year(year: int) -> frozenset[date]:
    """Return full-day NYSE (XNYS) Rule 7.2 US equity market holidays for `year`."""
    hols: set[date] = set()
    nyd = _observed_fixed_holiday(date(year, 1, 1), observe_saturday_friday=False)
    if nyd is not None:
        hols.add(nyd)
    hols.add(_nth_weekday_of_month(year, 1, 0, 3))
    hols.add(_nth_weekday_of_month(year, 2, 0, 3))
    hols.add(_gregorian_easter(year) - timedelta(days=2))
    hols.add(_last_weekday_of_month(year, 5, 0))
    if year >= 2022:
        jt = _observed_fixed_holiday(date(year, 6, 19))
        if jt is not None:
            hols.add(jt)
    ind = _observed_fixed_holiday(date(year, 7, 4))
    if ind is not None:
        hols.add(ind)
    hols.add(_nth_weekday_of_month(year, 9, 0, 1))
    hols.add(_nth_weekday_of_month(year, 11, 3, 4))
    xmas = _observed_fixed_holiday(date(year, 12, 25))
    if xmas is not None:
        hols.add(xmas)
    for sp in _NYSE_SPECIAL_CLOSURES:
        if sp.year == year:
            hols.add(sp)
    return frozenset(hols)


def _is_nyse_trading_day(d: date) -> bool:
    """Return True iff `d` is an official NYSE (XNYS) full-or-early equity trading session."""
    if d.weekday() >= 5:
        return False
    return d not in _nyse_holidays_for_year(d.year)


def _count_nyse_sessions_inclusive(start_d: date, end_d: date) -> int:
    """Count official NYSE (XNYS) trading sessions inclusively between `start_d` and `end_d`."""
    total_days = (end_d - start_d).days + 1
    if total_days <= 0:
        return 0
    full_weeks, rem = divmod(total_days, 7)
    start_wd = start_d.weekday()
    weekdays = full_weeks * 5 + sum(1 for offset in range(rem) if (start_wd + offset) % 7 < 5)
    holidays = 0
    for yr in range(start_d.year, end_d.year + 1):
        for h in _nyse_holidays_for_year(yr):
            if start_d <= h <= end_d and h.weekday() < 5:
                holidays += 1
    return max(0, weekdays - holidays)


def _exact_fraction_product(factors: Sequence[Fraction]) -> Fraction:
    """Compute exact rational product via balanced divide-and-conquer (zero rounding drift)."""
    n = len(factors)
    if n == 0:
        return Fraction(1, 1)
    if n == 1:
        return factors[0]
    mid = n // 2
    return _exact_fraction_product(factors[:mid]) * _exact_fraction_product(factors[mid:])


def _scaled_sum(vals: Sequence[float]) -> float:
    """Overflow-safe power-of-two pre-scaled sum (handles `[1e308]*400` without `OverflowError`)."""
    if not vals:
        return 0.0
    m = max(abs(x) for x in vals)
    if m == 0.0:
        return 0.0
    _, exp = math.frexp(m)
    s_u = math.fsum(math.ldexp(x, -exp) for x in vals)
    try:
        return math.ldexp(s_u, exp)
    except OverflowError:
        return math.copysign(math.inf, s_u)


def _scaled_mean(vals: Sequence[float]) -> float:
    """Overflow-safe power-of-two pre-scaled mean (returns finite `1e308` on `[1e308]*400`)."""
    n = len(vals)
    if n == 0:
        return 0.0
    if min(vals) == max(vals):
        return float(vals[0])
    m = max(abs(x) for x in vals)
    if m == 0.0:
        return 0.0
    _, exp = math.frexp(m)
    mean_u = math.fsum(math.ldexp(x, -exp) for x in vals) / n
    return math.ldexp(mean_u, exp)


def _dec_log1p_bps(v: float, d_ln_10k: Decimal) -> Decimal:
    """Compute 80-digit `ln(1 + v/10000)` using exact `10000 + v` and Taylor series for tiny `|v| < 1e-15`."""
    d_v = Decimal.from_float(v)
    if abs(v) < 1e-15:
        s = d_v / _DEC_80_TEN_K
        return s - (s * s) / Decimal(2) + (s * s * s) / Decimal(3)
    num_d = _DEC_80_TEN_K + d_v
    return num_d.ln() - d_ln_10k


def _apply_absorbing_bankruptcy(returns_bps: Sequence[float]) -> tuple[list[float], bool, bool, int]:
    """Zero out post-wipeout returns at absorbing equity zero; return `(eff_returns, bankrupt, eq_gt_one, live_len)`.

    `live_len` is the number of initial sessions that started with strictly positive live capital (so if
    equity hits zero on session 0, `live_len == 1` and sessions `1..T-1` have zero capital and cannot fund
    post-bankruptcy trades). Tracks running log-equity in 80-digit `Decimal` via exact `10000 + v`,
    invoking exact `Fraction` divide-and-conquer within `1e-55` of `0` or `ln(2^-1074)`.
    """
    bankrupt = False
    eff: list[float] = []
    live_vals: list[float] = []
    live_len = 0
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.traps[Overflow] = False
        d_ln_10k = _DEC_80_TEN_K.ln()
        d_log_sub_floor = Decimal(-1074) * Decimal(2).ln()
        d_tol = Decimal("1e-55")
        d_log_eq = _DEC_80_ZERO
        for i, x in enumerate(returns_bps):
            v = _require_finite_real(x, f"returns_bps[{i}]")
            if bankrupt:
                eff.append(0.0)
                continue
            live_len += 1
            if v <= -10_000.0:
                bankrupt = True
                eff.append(-10_000.0)
                live_vals = []
                continue
            d_log_eq += _dec_log1p_bps(v, d_ln_10k)
            live_vals.append(v)
            eff.append(v)
            if d_log_eq < d_log_sub_floor - d_tol:
                bankrupt = True
                live_vals = []
            elif d_log_eq <= d_log_sub_floor + d_tol:
                cur_eq = _exact_fraction_product([1 + Fraction(val) / 10_000 for val in live_vals])
                if cur_eq < _BINARY64_MIN_SUBNORMAL_FRAC:
                    bankrupt = True
                    live_vals = []

        if bankrupt:
            return eff, True, False, live_len
        if d_log_eq > d_tol:
            return eff, False, True, live_len
        if d_log_eq < -d_tol:
            return eff, False, False, live_len
    final_eq = _exact_fraction_product([1 + Fraction(val) / 10_000 for val in live_vals])
    return eff, False, final_eq > Fraction(1, 1), live_len


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
        canonical_sessions=1706,
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
            "Full 6.8-year multi-regime stress horizon (exact 1,706 official XNYS sessions including the "
            "2025-01-09 National Day of Mourning) required for daily/swing, PEAD, Form 4, Lazy Prices, and VIX."
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
        end_date="2027-01-15",
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
            "Forward live-paper E2E broker parity replay (>=30 closed trades across >=10 distinct trade sessions, "
            "60 shadow dates inside a 63-session NYSE window) verifying ADR-0025 markout & queue parity."
        ),
    ),
}

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


def _pre_scaled_moments(vals: Sequence[float]) -> tuple[float, float, float, float]:
    """Compute overflow-free, shift-centered `(sd_sample, raw_rho, raw_skew, raw_ex_kurt)` obeying Pearson's bound.

    Shifts by `anchor = vals[0]` BEFORE centering so ultra-tight variations like `[1, 1 - 2**-53]*200` never
    lose their `±2**-54` deviations to `1.0` mean rounding, yielding exact `rho = -399/400`.
    """
    n = len(vals)
    if n < 2 or min(vals) == max(vals):
        return 0.0, 0.0, 0.0, 0.0
    # Shift by anchor when all values share finite differences; otherwise pre-scale first to avoid `±1e308` overflow
    anchor = float(vals[0])
    shifted_ok = all(math.isfinite(x - anchor) for x in vals)
    if shifted_ok:
        shifted = [x - anchor for x in vals]
        sh_scale = max(abs(x) for x in shifted)
        if sh_scale == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        s_u = [x / sh_scale for x in shifted]
        mean_u = math.fsum(s_u) / n
        centered_u = [x - mean_u for x in s_u]
        dev_scale = max(abs(c) for c in centered_u)
        if dev_scale == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        u = [c / dev_scale for c in centered_u]
        denom_u = math.fsum(x * x for x in u)
        if denom_u <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        sd_sample = sh_scale * dev_scale * math.sqrt(denom_u / (n - 1))
    else:
        pre_scale = max(abs(x) for x in vals)
        s_u = [x / pre_scale for x in vals]
        mean_u = math.fsum(s_u) / n
        centered_u = [x - mean_u for x in s_u]
        dev_scale = max(abs(c) for c in centered_u)
        if dev_scale == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        u = [c / dev_scale for c in centered_u]
        denom_u = math.fsum(x * x for x in u)
        if denom_u <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        sd_sample = pre_scale * dev_scale * math.sqrt(denom_u / (n - 1))

    num_u = math.fsum(u[i] * u[i - 1] for i in range(1, n))
    raw_rho = max(-1.0 + 1e-12, min(1.0 - 1e-12, num_u / denom_u))
    sd_pop_u = math.sqrt(denom_u / n)
    raw_skew = math.fsum((x / sd_pop_u) ** 3 for x in u) / n if sd_pop_u > 0.0 else 0.0
    raw_kurt_4 = math.fsum((x / sd_pop_u) ** 4 for x in u) / n if sd_pop_u > 0.0 else 3.0
    raw_ex_kurt = max(-2.0, raw_skew * raw_skew - 2.0, raw_kurt_4 - 3.0)
    while Fraction(raw_ex_kurt) + 2 < Fraction(raw_skew) * Fraction(raw_skew):
        raw_ex_kurt = math.nextafter(raw_ex_kurt, math.inf)
    return sd_sample, raw_rho, raw_skew, raw_ex_kurt


def _series_sharpe_ratio_unit(vals: Sequence[float]) -> float:
    """Return scale-invariant overflow-free sample Sharpe ratio (`mean / sample_sd`) alone (no mean tie-break)."""
    n = len(vals)
    if n == 0:
        return 0.0
    if min(vals) == max(vals):
        v0 = float(vals[0])
        return math.inf if v0 > 0.0 else (-math.inf if v0 < 0.0 else 0.0)
    pre_scale = max(abs(x) for x in vals)
    _, exp = math.frexp(pre_scale)
    s_u = [math.ldexp(x, -exp) for x in vals]
    mean_u = math.fsum(s_u) / n
    anchor = s_u[0]
    sh = [x - anchor for x in s_u]
    sh_mean = math.fsum(sh) / n
    centered = [x - sh_mean for x in sh]
    dev_scale = max(abs(c) for c in centered)
    if dev_scale == 0.0 or n < 2:
        return math.inf if mean_u > 0.0 else (-math.inf if mean_u < 0.0 else 0.0)
    u = [c / dev_scale for c in centered]
    var_u = math.fsum(x * x for x in u) / (n - 1)
    sd_u = dev_scale * math.sqrt(var_u)
    return mean_u / sd_u


def _exact_sharpe_key(vals: Sequence[float]) -> tuple[int, int]:
    """Return exact integer `(sum_q, ss_c)` for zero-epsilon, scale-invariant Sharpe ranking (`_exact_t_cmp`).

    Since `SR = (sum_q * sqrt(n - 1)) / sqrt(ss_c)` for common `n`, comparing `(sum_q, ss_c)` via `_exact_t_cmp`
    gives bit-exact equality on `A` vs `k*A` (`2*A`) while NEVER merging genuinely distinct Sharpes (`A` vs
    `A - 2^-40`) or opposite signs (`-4e-13` vs `+4e-13`).
    """
    n = len(vals)
    if n == 0:
        return 0, 0
    q, _ = _to_exact_dyadic_ints(vals)
    sum_q = sum(q)
    if n < 2 or min(q) == max(q):
        return sum_q, 0
    c = [n * qi - sum_q for qi in q]
    ss_c = sum(ci * ci for ci in c)
    return sum_q, ss_c


def _compute_empirical_cpcv_pbo(
    returns_by_strategy: Mapping[str, Sequence[float]],
    *,
    n_groups: int = 6,
    k_test: int = 2,
    purge_sessions: int = 0,
    selector: str = "sharpe",
    hac_max_lag: int = 5,
) -> float | None:
    """Compute padding-monotone, exact-purge, zero-epsilon, conservative-tie `C(6, 2) = 15` CPCV PBO.

    Supports BOTH `selector="sharpe"` (exact scale-invariant Sharpe ratio) and `selector="hac_t"` (exact
    Bartlett HAC t-statistic matching the promotion winner selector), never shortens `purge_sessions`,
    and marks a split overfit whenever any tied IS winner fails to strictly beat every competitor OOS OR
    any full-sample winner `u` flips OOS rank against any competitor `v != u` (`IS(u) >= IS(v)` and
    `OOS(v) >= OOS(u)`, or vice versa). Strictly >= Bailey et al. (2014) panel PBO and mathematically
    monotone under arbitrary panel augmentation (`S -> S union P` can NEVER lower PBO via control padding!).
    """
    names = sorted(returns_by_strategy.keys())
    if len(names) < 2:
        return None
    # Reject panels consisting solely of all-zero flat cash strategies (no active strategy comparison!)
    if all(all(x == 0.0 for x in returns_by_strategy[nm]) for nm in names):
        return None
    t_len = len(returns_by_strategy[names[0]])
    if t_len < n_groups * 2:
        return None

    groups: list[list[int]] = []
    for g in range(n_groups):
        lo = (g * t_len) // n_groups
        hi = ((g + 1) * t_len) // n_groups
        groups.append(list(range(lo, hi)))

    eff_purge = max(0, int(purge_sessions))
    full_idx = list(range(t_len))
    if selector == "hac_t":
        full_keys = {
            nm: _exact_bartlett_hac_t_key(returns_by_strategy[nm], hac_max_lag, indices=full_idx) for nm in names
        }
    else:
        full_keys = {nm: _exact_sharpe_key(returns_by_strategy[nm]) for nm in names}
    full_best_key = full_keys[names[0]]
    for nm in names[1:]:
        if _exact_t_cmp(full_keys[nm][0], full_keys[nm][1], full_best_key[0], full_best_key[1]) > 0:
            full_best_key = full_keys[nm]
    full_best_names = [
        nm for nm in names if _exact_t_cmp(full_keys[nm][0], full_keys[nm][1], full_best_key[0], full_best_key[1]) == 0
    ]

    n_splits = 0
    overfit_splits = 0
    for test_tuple in itertools.combinations(range(n_groups), k_test):
        test_set = set(test_tuple)
        test_idx: list[int] = []
        train_idx: list[int] = []
        for g in range(n_groups):
            blk = groups[g]
            if g in test_set:
                test_idx.extend(blk)
            else:
                left_trim = eff_purge if (g - 1) in test_set else 0
                right_trim = eff_purge if (g + 1) in test_set else 0
                hi_idx = len(blk) - right_trim
                if left_trim < hi_idx:
                    train_idx.extend(blk[left_trim:hi_idx])
        if len(train_idx) < 2 or len(test_idx) < 2:
            return 1.0

        if selector == "hac_t":
            is_keys = {
                nm: _exact_bartlett_hac_t_key(
                    [returns_by_strategy[nm][i] for i in train_idx], hac_max_lag, indices=train_idx
                )
                for nm in names
            }
            oos_keys = {
                nm: _exact_bartlett_hac_t_key(
                    [returns_by_strategy[nm][i] for i in test_idx], hac_max_lag, indices=test_idx
                )
                for nm in names
            }
        else:
            is_keys = {nm: _exact_sharpe_key([returns_by_strategy[nm][i] for i in train_idx]) for nm in names}
            oos_keys = {nm: _exact_sharpe_key([returns_by_strategy[nm][i] for i in test_idx]) for nm in names}

        best_key = is_keys[names[0]]
        for nm in names[1:]:
            if _exact_t_cmp(is_keys[nm][0], is_keys[nm][1], best_key[0], best_key[1]) > 0:
                best_key = is_keys[nm]
        tied_is_winners = [
            nm for nm in names if _exact_t_cmp(is_keys[nm][0], is_keys[nm][1], best_key[0], best_key[1]) == 0
        ]

        split_overfit = False
        for win_nm in tied_is_winners:
            w_oos = oos_keys[win_nm]
            if any(
                nm != win_nm and _exact_t_cmp(oos_keys[nm][0], oos_keys[nm][1], w_oos[0], w_oos[1]) >= 0 for nm in names
            ):
                split_overfit = True
                break
        if not split_overfit:
            for u in full_best_names:
                for v in names:
                    if v == u:
                        continue
                    cmp_is = _exact_t_cmp(is_keys[u][0], is_keys[u][1], is_keys[v][0], is_keys[v][1])
                    cmp_oos = _exact_t_cmp(oos_keys[u][0], oos_keys[u][1], oos_keys[v][0], oos_keys[v][1])
                    if cmp_is == 0 or (cmp_is > 0 and cmp_oos <= 0) or (cmp_is < 0 and cmp_oos >= 0):
                        split_overfit = True
                        break
                if split_overfit:
                    break
        if split_overfit:
            overfit_splits += 1
        n_splits += 1

    if n_splits == 0:
        return None
    return overfit_splits / float(n_splits)


def _min_feasible_cross_trial_sr_std(
    returns_by_strategy: Mapping[str, Sequence[float]],
    cumulative_trials_n: int,
) -> float:
    """Compute unavoidable lower bound on N-trial Sharpe SD (tail-safe past `N = 10^310`, `inf` on `±inf` SR)."""
    if cumulative_trials_n <= 1:
        return 0.0
    srs = [_series_sharpe_ratio_unit(series) for series in returns_by_strategy.values()]
    if any(not math.isfinite(s) for s in srs):
        return math.inf
    if len(srs) < 2:
        return 0.0
    m = len(srs)
    scale = max((abs(s) for s in srs), default=0.0)
    if scale == 0.0:
        return 0.0
    s_u = [s / scale for s in srs]
    mean_u = math.fsum(s_u) / m
    ss_u = math.fsum((s - mean_u) ** 2 for s in s_u)
    if ss_u <= 0.0:
        return 0.0
    # Log-domain division avoids `float(cumulative_trials_n - 1)` OverflowError for `N >= 10^309`!
    log_res = math.log(scale) + 0.5 * (math.log(ss_u) - math.log(cumulative_trials_n - 1))
    if log_res < -745.0:
        return 0.0
    if log_res > 709.78:
        return math.inf
    return math.exp(log_res)


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
    """Compute exact ADIA Lab #19 (SSRN 5520741 / zoonek/2025-sharpe-ratio Eq. 11) AR(1) + higher-moment MinTRL.

    Enforces exact rational Pearson feasibility (`Fraction(ex_k) + 2 >= Fraction(skew)**2`, zero underflow at
    `1e-200`) and evaluates `var_0` and `(sr_t - sr_0)` in 80-digit `Decimal` to prevent catastrophic cancellation
    and `ZeroDivisionError` when `sr_t` and `sr_0` differ by 1 ULP.
    """
    sr_t = _require_finite_real(target_sharpe_ann, "target_sharpe_ann")
    sr_0 = _require_finite_real(benchmark_sr_ann, "benchmark_sr_ann")
    g3 = _require_finite_real(skew, "skew")
    ex_k = _require_finite_real(excess_kurtosis, "excess_kurtosis")
    rho = _require_finite_real(ar1_rho, "ar1_rho")
    conf = _require_finite_real(confidence, "confidence")
    ppy = _require_finite_real(periods_per_year, "periods_per_year")

    if ppy <= 0.0:
        raise ValueError(f"periods_per_year must be > 0 (got {periods_per_year!r})")
    if sr_t <= sr_0:
        raise ValueError("target_sharpe_ann must strictly exceed benchmark_sr_ann")
    if ex_k < -2.0:
        raise ValueError(f"excess_kurtosis must be >= -2.0 (got {excess_kurtosis!r})")
    # Exact rational Pearson bound (immune to `1e-200**2 -> 0.0` underflow!)
    if Fraction(ex_k) + 2 < Fraction(g3) * Fraction(g3):
        raise ValueError(f"Infeasible moments violating Pearson inequality (excess_kurtosis={ex_k} + 2 < skew^2)")
    if not (-_MAX_STATIONARY_RHO < rho < _MAX_STATIONARY_RHO):
        raise ValueError(f"ar1_rho must lie inside (-{_MAX_STATIONARY_RHO}, {_MAX_STATIONARY_RHO}) (got {ar1_rho})")
    if not (0.5 < conf < 0.999999):
        raise ValueError(f"confidence must lie inside (0.5, 0.999999) (got {confidence})")

    z_alpha = _STD_NORM.inv_cdf(conf)
    with localcontext() as ctx:
        ctx.prec = 80
        d_one = Decimal(1)
        d_two = Decimal(2)
        d_four = Decimal(4)
        d_ppy = Decimal.from_float(ppy)
        d_sqrt_ppy = d_ppy.sqrt()
        d_sr_t = Decimal.from_float(sr_t)
        d_sr_0 = Decimal.from_float(sr_0)
        d_delta = (d_sr_t - d_sr_0) / d_sqrt_ppy
        d_sr0_d = d_sr_0 / d_sqrt_ppy
        d_g3 = Decimal.from_float(g3)
        d_ex_k = Decimal.from_float(ex_k)
        d_rho = Decimal.from_float(rho)
        d_z = Decimal.from_float(z_alpha)

        d_b_term = d_rho / (d_one - d_rho)
        d_c_term = (d_rho * d_rho) / (d_one - d_rho * d_rho)
        d_a_coef = d_one + d_two * d_b_term
        d_b_coef = d_one + d_b_term + d_c_term
        d_c_coef = d_one + d_two * d_c_term
        d_gamma4_m1 = d_ex_k + d_two
        try:
            d_var0 = d_a_coef - d_b_coef * d_g3 * d_sr0_d + d_c_coef * (d_gamma4_m1 / d_four) * (d_sr0_d * d_sr0_d)
            if not d_var0.is_finite() or d_var0 <= 0:
                raise ValueError(f"Invalid moment/AR(1) combination yields non-positive/non-finite variance ({d_var0})")
            d_ratio = d_z / d_delta
            d_min_n = d_var0 * (d_ratio * d_ratio)
            if not d_min_n.is_finite():
                raise ValueError("MinTRL overflows finite horizon")
            # Exact ceiling of finite Decimal (guard gigantic > 10^100 horizon cleanly)
            if d_min_n > Decimal("1e100"):
                raise ValueError("MinTRL overflows practical finite horizon")
            int_part = int(d_min_n)
            ceil_n = int_part if d_min_n == Decimal(int_part) else int_part + 1
            return max(2, ceil_n)
        except (Overflow, OverflowError, ZeroDivisionError) as exc:
            raise ValueError("MinTRL overflows finite representable horizon") from exc


def _compute_empirical_dsr_suite(
    daily_returns_bps: Sequence[float],
    *,
    cumulative_trials_n: int,
    ar1_rho: float = 0.0,
    cross_trial_sr_std: float | None = None,
) -> tuple[float, float, float | None]:
    """Compute `(dsr_ar1_obs_psd, dsr_blp2014_iid, dsr_adia19_ar1_null)` in 80-digit `Decimal`.

    - `dsr_blp2014_iid`: Exact Bailey & López de Prado (2014) Eq. 2 IID observed-Sharpe DSR (`a=b=c=1`),
      computed independently of the AR(1) stationarity cutoff (`rho >= 0.99`).
    - `dsr_ar1_obs_psd`: AR(1)-adjusted observed-Sharpe PSD DSR across stationary `rho < 0.99` (returns `0.0`
      when `rho >= 0.99`).
    - `dsr_adia19_ar1_null`: Exact ADIA Lab #19 (SSRN 5520741) AR(1) null-Sharpe `V(SR_0)` DSR, returning
      `None` whenever `rho >= 0.99` OR the raw ADIA variance `v_adia19_raw <= 0` (never substituting PSD!).
    """
    n = len(daily_returns_bps)
    if n < 3:
        return 0.0, 0.0, None
    vals = [_require_finite_real(x, f"dsr[{i}]") for i, x in enumerate(daily_returns_bps)]
    rho_f = _require_finite_real(ar1_rho, "ar1_rho")
    ar1_stationary = rho_f < _MAX_STATIONARY_RHO
    rho_eval = max(-_MAX_STATIONARY_RHO + 1e-6, min(_MAX_STATIONARY_RHO - 1e-6, rho_f))

    n_trials = _require_nonbool_int(cumulative_trials_n, "cumulative_trials_n", minimum=1)
    with localcontext() as ctx:
        ctx.prec = 80
        d_one = Decimal(1)
        d_two = Decimal(2)
        d_four = Decimal(4)
        d_rho = Decimal.from_float(rho_eval)
        d_bt = d_rho / (d_one - d_rho)
        d_ct = (d_rho * d_rho) / (d_one - d_rho * d_rho)
        d_a_ar = d_one + d_two * d_bt
        d_b_ar = d_one + d_bt + d_ct
        d_c_ar = d_one + d_two * d_ct
        d_b_psd = min(d_b_ar, (d_a_ar * d_c_ar).sqrt()) if (d_a_ar > 0 and d_c_ar > 0) else d_b_ar

        if n_trials == 1:
            d_sr_star_iid = Decimal(0)
            d_sr_star_ar1 = Decimal(0)
        else:
            log_n = math.log(n_trials)
            z1 = _inv_norm_upper_from_log_p(-log_n)
            z2 = _inv_norm_upper_from_log_p(-log_n - 1.0)
            e_max_z = (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
            d_emz = Decimal.from_float(e_max_z)
            if cross_trial_sr_std is not None:
                ct_sd = _require_finite_real(cross_trial_sr_std, "cross_trial_sr_std")
                if ct_sd <= 0.0:
                    raise ValueError(f"cross_trial_sr_std must be > 0 when cumulative_trials_n > 1 (got {ct_sd!r})")
                # Evaluate `ct_sd * e_max_z` strictly in 80-digit Decimal (no float64 early-return at `1e308`!)
                d_sr_star_iid = Decimal.from_float(ct_sd) * d_emz
                d_sr_star_ar1 = d_sr_star_iid
            else:
                d_sr_star_iid = (d_one / Decimal(n - 1)).sqrt() * d_emz
                d_sr_star_ar1 = (max(d_one, d_a_ar) / Decimal(n - 1)).sqrt() * d_emz
            if not d_sr_star_iid.is_finite():
                return 0.0, 0.0, None

        if min(vals) == max(vals):
            if vals[0] <= 0.0:
                return 0.0, 0.0, (0.0 if ar1_stationary else None)
            # Constant positive returns -> SR = +inf, beats any finite sr_star unless sr_star overflows binary64
            v_const = 0.0 if (cross_trial_sr_std is not None and not math.isfinite(float(d_sr_star_iid))) else 1.0
            return (v_const if ar1_stationary else 0.0), v_const, (v_const if ar1_stationary else None)

        # Exact 80-digit scale-invariant sample moments (eliminates float64 near-Pearson cancellation!)
        pre_scale = max(abs(x) for x in vals)
        _, exp = math.frexp(pre_scale)
        d_vals = [Decimal.from_float(math.ldexp(x, -exp)) for x in vals]
        d_n = Decimal(n)
        d_mean = sum(d_vals, Decimal(0)) / d_n
        d_c = [x - d_mean for x in d_vals]
        d_ss = sum((x * x for x in d_c), Decimal(0))
        if d_ss <= 0:
            v_inf = 1.0 if d_mean > 0 else 0.0
            return (v_inf if ar1_stationary else 0.0), v_inf, (v_inf if ar1_stationary else None)
        d_sd_sample = (d_ss / Decimal(n - 1)).sqrt()
        d_sr_obs = d_mean / d_sd_sample
        d_sd_pop = (d_ss / d_n).sqrt()
        d_z = [x / d_sd_pop for x in d_c]
        d_skew = sum((z * z * z for z in d_z), Decimal(0)) / d_n
        d_kurt_4 = sum((z * z * z * z for z in d_z), Decimal(0)) / d_n
        d_gamma4_m1 = max(d_skew * d_skew, d_kurt_4 - d_one)

        d_sq_nm1 = Decimal(n - 1).sqrt()
        d_num_iid = (d_sr_obs - d_sr_star_iid) * d_sq_nm1
        d_num_ar1 = (d_sr_obs - d_sr_star_ar1) * d_sq_nm1

        def _cdf_from_var(d_num_val: Decimal, d_var: Decimal) -> float:
            if not d_var.is_finite() or d_var <= 0:
                return 0.0
            z_stat = d_num_val / d_var.sqrt()
            if z_stat > Decimal(40):
                return 1.0
            if z_stat < Decimal(-40):
                return 0.0
            return float(_STD_NORM.cdf(float(z_stat)))

        v_blp2014 = d_one - d_skew * d_sr_obs + (d_gamma4_m1 / d_four) * (d_sr_obs * d_sr_obs)
        dsr_iid = _cdf_from_var(d_num_iid, v_blp2014)
        if not ar1_stationary:
            return 0.0, dsr_iid, None
        v_ar1_obs = d_a_ar - d_b_psd * d_skew * d_sr_obs + d_c_ar * (d_gamma4_m1 / d_four) * (d_sr_obs * d_sr_obs)
        v_adia19_raw = (
            d_a_ar - d_b_ar * d_skew * d_sr_star_ar1 + d_c_ar * (d_gamma4_m1 / d_four) * (d_sr_star_ar1 * d_sr_star_ar1)
        )
        dsr_adia: float | None = (
            _cdf_from_var(d_num_ar1, v_adia19_raw) if (v_adia19_raw.is_finite() and v_adia19_raw > 0) else None
        )
        return _cdf_from_var(d_num_ar1, v_ar1_obs), dsr_iid, dsr_adia


def _compute_empirical_dsr(
    daily_returns_bps: Sequence[float],
    *,
    cumulative_trials_n: int,
    ar1_rho: float = 0.0,
    cross_trial_sr_std: float | None = None,
) -> float:
    """Compute AR(1)-adjusted observed-Sharpe PSD Deflated Sharpe Ratio."""
    dsr_ar1, _, _ = _compute_empirical_dsr_suite(
        daily_returns_bps,
        cumulative_trials_n=cumulative_trials_n,
        ar1_rho=ar1_rho,
        cross_trial_sr_std=cross_trial_sr_std,
    )
    return dsr_ar1


class CalendarDayStats(TypedDict):
    sessions: float
    mean_daily_bps: float
    std_daily_bps: float
    ar1_rho: float
    sharpe_ann_raw: float
    sharpe_ann_ar1_adj: float
    sharpe_status: str
    skew: float
    excess_kurtosis: float
    cum_return_pct: float
    max_drawdown_pct: float
    calmar: float
    profit_factor: float | None


def compute_calendar_day_stats(daily_returns_bps: Sequence[float]) -> CalendarDayStats:
    """Compute calendar-day AR(1)-adjusted Sharpe, moments, 80-digit log-equity max DD, Calmar & scale-safe PF."""
    n = len(daily_returns_bps)
    if n == 0:
        return {
            "sessions": 0.0,
            "mean_daily_bps": 0.0,
            "std_daily_bps": 0.0,
            "ar1_rho": 0.0,
            "sharpe_ann_raw": 0.0,
            "sharpe_ann_ar1_adj": 0.0,
            "sharpe_status": "ZERO_VARIANCE_CASH",
            "skew": 0.0,
            "excess_kurtosis": 0.0,
            "cum_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
        }

    # Truncate post-wipeout returns at absorbing zero equity so standalone stats never report zombie gains
    vals, bankrupt, _, _ = _apply_absorbing_bankruptcy(daily_returns_bps)

    # 80-digit Decimal log-equity + Taylor series for tiny losses (`[10000, -1e-80] -> Calmar ~1e84`)
    # and unclipped binary64 cumulative returns (`[1e155, 2e158] -> 2e307%`).
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.traps[Overflow] = False
        d_one = _DEC_80_ONE
        d_zero = _DEC_80_ZERO
        d_ln_10k = _DEC_80_TEN_K.ln()
        d_tiny = Decimal("1e-20")
        d_log_eq = d_zero
        d_u_dd = d_zero
        d_max_dd = d_zero
        wiped = False
        for bps in vals:
            if wiped:
                dd_d = d_one
            else:
                if bps <= -10_000.0:
                    wiped = True
                    dd_d = d_one
                else:
                    step_log = _dec_log1p_bps(bps, d_ln_10k)
                    d_log_eq += step_log
                    # Incremental drawdown depth `u_dd = max(0, u_dd - step_log)` never loses tiny losses
                    # like `-1e-80 bps` to subtraction from `ln(2)` (exact Calmar even down to `1e-300`!).
                    d_u_dd = d_u_dd - step_log
                    if d_u_dd <= d_zero:
                        d_u_dd = d_zero
                        dd_d = d_zero
                    elif d_u_dd < d_tiny:
                        dd_d = d_u_dd - (d_u_dd * d_u_dd) / Decimal(2) + (d_u_dd * d_u_dd * d_u_dd) / Decimal(6)
                    else:
                        dd_d = d_one - (-d_u_dd).exp()
            if dd_d > d_max_dd:
                d_max_dd = dd_d
        if bankrupt:
            wiped = True
            d_max_dd = d_one

        max_dd = float(d_max_dd)
        if wiped:
            cum_ret_pct = -100.0
            ann_ret_pct_d = Decimal(-100)
        else:
            if d_log_eq > Decimal(715):
                cum_ret_pct = _MAX_JSON_PCT_CAP
            elif abs(d_log_eq) < d_tiny:
                cum_d = (d_log_eq + (d_log_eq * d_log_eq) / Decimal(2) + (d_log_eq**3) / Decimal(6)) * Decimal(100)
                cum_ret_pct = float(cum_d)
            else:
                cum_d = (d_log_eq.exp() - d_one) * Decimal(100)
                cum_f = float(cum_d) if cum_d.is_finite() else math.inf
                cum_ret_pct = cum_f if math.isfinite(cum_f) else _MAX_JSON_PCT_CAP
            ann_log_d = d_log_eq * (Decimal(252) / Decimal(n))
            if ann_log_d > Decimal(715):
                ann_ret_pct_d = Decimal.from_float(_MAX_JSON_PCT_CAP)
            elif abs(ann_log_d) < d_tiny:
                ann_ret_pct_d = (
                    ann_log_d + (ann_log_d * ann_log_d) / Decimal(2) + (ann_log_d**3) / Decimal(6)
                ) * Decimal(100)
            else:
                ann_ret_pct_d = (ann_log_d.exp() - d_one) * Decimal(100)
        calmar_d = (ann_ret_pct_d / (d_max_dd * Decimal(100))) if d_max_dd > d_zero else d_zero
        calmar_f = float(calmar_d) if calmar_d.is_finite() else 0.0
        calmar = calmar_f if math.isfinite(calmar_f) else 0.0

    # Pre-scaled profit factor prevents `math.fsum` overflow on `[1e308, 1e308, -10000]` -> exact `2e304`,
    # and returns `None` (unbounded) for all-gain zero-loss series or overflow ratios (never capped at `999.99`!)
    pf_scale = max((abs(x) for x in vals), default=0.0)
    pf_val: float | None
    if pf_scale == 0.0:
        pf_val = 0.0
    else:
        gains_u = math.fsum(x / pf_scale for x in vals if x > 0.0)
        losses_u = abs(math.fsum(x / pf_scale for x in vals if x < 0.0))
        if losses_u == 0.0:
            pf_val = None if gains_u > 0.0 else 0.0
        else:
            raw_pf = gains_u / losses_u
            pf_val = round(raw_pf, 4) if math.isfinite(raw_pf) else None

    if n == 1:
        st_1 = (
            "ZERO_VARIANCE_POSITIVE"
            if vals[0] > 0.0
            else ("ZERO_VARIANCE_NEGATIVE" if vals[0] < 0.0 else "ZERO_VARIANCE_CASH")
        )
        return {
            "sessions": 1.0,
            "mean_daily_bps": round(vals[0], 4),
            "std_daily_bps": 0.0,
            "ar1_rho": 0.0,
            "sharpe_ann_raw": 0.0,
            "sharpe_ann_ar1_adj": 0.0,
            "sharpe_status": st_1,
            "skew": 0.0,
            "excess_kurtosis": 0.0,
            "cum_return_pct": round(cum_ret_pct, 4),
            "max_drawdown_pct": round(max_dd * 100.0, 4),
            "calmar": round(calmar, 4),
            "profit_factor": pf_val,
        }

    if min(vals) == max(vals):
        mean = vals[0]
        sd_sample = 0.0
        rho = 0.0
        skew = 0.0
        ex_kurt = 0.0
        sr_raw = 0.0
        sr_adj = 0.0
        sharpe_status = (
            "ZERO_VARIANCE_POSITIVE"
            if mean > 0.0
            else ("ZERO_VARIANCE_NEGATIVE" if mean < 0.0 else "ZERO_VARIANCE_CASH")
        )
    else:
        mean = _scaled_mean(vals)
        sd_sample, rho, skew, ex_kurt = _pre_scaled_moments(vals)
        sr_daily = _series_sharpe_ratio_unit(vals)
        if not math.isfinite(sr_daily):
            sr_raw = 0.0
            sr_adj = 0.0
            sharpe_status = (
                "ZERO_VARIANCE_POSITIVE"
                if mean > 0.0
                else ("ZERO_VARIANCE_NEGATIVE" if mean < 0.0 else "ZERO_VARIANCE_CASH")
            )
        else:
            sr_raw = sr_daily * math.sqrt(252.0)
            ar1_scale = math.sqrt((1.0 + rho) / (1.0 - rho))
            sr_adj = sr_raw / ar1_scale
            sharpe_status = "FINITE"

    return {
        "sessions": float(n),
        "mean_daily_bps": round(mean, 4),
        "std_daily_bps": round(sd_sample, 4),
        "ar1_rho": round(rho, 4),
        "sharpe_ann_raw": round(sr_raw, 4),
        "sharpe_ann_ar1_adj": round(sr_adj, 4),
        "sharpe_status": sharpe_status,
        "skew": round(skew, 4),
        "excess_kurtosis": round(ex_kurt, 4),
        "cum_return_pct": round(cum_ret_pct, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 4),
        "calmar": round(calmar, 4),
        "profit_factor": pf_val,
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


def _normalize_str_sequence(raw: Any, field_name: str, violations: list[str], code_prefix: str) -> list[str]:
    """Prevent scalar string iteration bypass (`'yfinance' -> 'y','f',...`) by rejecting non-collection strings."""
    if isinstance(raw, (str, bytes)):
        violations.append(f"{code_prefix}:scalar_string_not_sequence:{field_name}")
        return [raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw]
    if not isinstance(raw, (tuple, list, set, frozenset)):
        violations.append(f"{code_prefix}:invalid_sequence_type:{field_name}")
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            violations.append(f"{code_prefix}:non_str_element:{field_name}")
        else:
            out.append(item)
    return out


def audit_low_quality_practices(
    inp: PlaybackAuditInput,
    *,
    observed_sessions: int | None = None,
    candidate_strategy_ids: Sequence[str] = (),
    observed_closed_roundtrips: int | None = None,
    observed_trade_sessions: int | None = None,
) -> dict[str, Any]:
    """Fail-closed 14-point audit purging every low-quality backtest/playback anti-pattern."""
    violations: list[str] = []

    if inp.horizon_id not in SUITABLE_PLAYBACK_HORIZONS:
        violations.append(f"UNKNOWN_HORIZON_ID:{_safe_num_str(inp.horizon_id)}")
    spec = SUITABLE_PLAYBACK_HORIZONS.get(inp.horizon_id)

    # Validate strict boolean flags (reject truthy non-bool strings like "false")
    if not isinstance(inp.includes_2x_and_sqrt_impact, bool):
        violations.append("UNREALISTIC_FLAT_COST_NO_2X_IMPACT:non_bool_flag")
    if not isinstance(inp.take_profit_cap_enabled, bool):
        violations.append("FIXED_TAKE_PROFIT_CAP_ACTIVE:non_bool_flag")
    if not isinstance(inp.uses_spa_mcs_dsr_pbo, bool):
        violations.append("MISSING_MULTIPLE_TESTING_SPA_MCS_DSR_PBO:non_bool_flag")
    if not isinstance(inp.in_sample_symbol_cherry_picking, bool):
        violations.append("IN_SAMPLE_SYMBOL_CHERRY_PICKING:non_bool_flag")
    if not isinstance(inp.silent_exception_swallow, bool):
        violations.append("SILENT_DATA_FETCH_EXCEPTION_SWALLOW_OR_BANNED_SOURCE:non_bool_flag")
    if not isinstance(inp.is_post_incentive_chase, bool):
        violations.append("QUARANTINED_ADR0064_OR_POST_CHASE_ACTIVE:non_bool_flag")

    # 1. Survivorship / Ex-Post Live Mover Universe
    if inp.universe_mode != "CAUSAL_T_MINUS_1_PIT":
        violations.append("EX_POST_LIVE_MOVER_SURVIVORSHIP_UNIVERSE")

    # 2. Toy Horizon, Cadence-Horizon Alignment & Overflow-Safe Exact MinTRL Power Check
    min_trl_valid = True
    if not _is_finite_real(inp.target_sharpe_for_min_trl) or not _is_finite_real(inp.ar1_rho_for_min_trl):
        violations.append("TOY_HORIZON_BELOW_MIN_TRL:non_finite_min_trl_inputs")
        min_trl = MIN_SWING_PLAYBACK_SESSIONS
        min_trl_valid = False
    else:
        t_sr = float(inp.target_sharpe_for_min_trl)
        rho_in = float(inp.ar1_rho_for_min_trl)
        if t_sr <= 0.0 or not (-_MAX_STATIONARY_RHO < rho_in < _MAX_STATIONARY_RHO):
            violations.append("TOY_HORIZON_BELOW_MIN_TRL:invalid_min_trl_inputs")
            min_trl = MIN_SWING_PLAYBACK_SESSIONS
            min_trl_valid = False
        else:
            try:
                min_trl = compute_min_trl_sessions(target_sharpe_ann=t_sr, ar1_rho=rho_in)
            except ValueError:
                violations.append("TOY_HORIZON_BELOW_MIN_TRL:min_trl_overflow_or_invalid")
                min_trl = MIN_SWING_PLAYBACK_SESSIONS
                min_trl_valid = False

    obs_valid_int: int | None = None
    if observed_sessions is not None:
        if isinstance(observed_sessions, bool) or not isinstance(observed_sessions, int) or observed_sessions < 0:
            violations.append(f"TOY_HORIZON_BELOW_MIN_TRL:invalid_observed_sessions={_safe_num_str(observed_sessions)}")
            obs_valid_int = 0
        else:
            obs_valid_int = observed_sessions

    if (
        isinstance(inp.distinct_sessions, bool)
        or not isinstance(inp.distinct_sessions, int)
        or inp.distinct_sessions < 0
    ):
        violations.append(f"TOY_HORIZON_BELOW_MIN_TRL:invalid_distinct_sessions={_safe_num_str(inp.distinct_sessions)}")
        eff_sessions = 0
    else:
        if obs_valid_int is not None and obs_valid_int != inp.distinct_sessions:
            violations.append(
                f"TOY_HORIZON_BELOW_MIN_TRL:session_count_mismatch_obs={_safe_num_str(obs_valid_int)}"
                f"!=decl={_safe_num_str(inp.distinct_sessions)}"
            )
        eff_sessions = inp.distinct_sessions if obs_valid_int is None else min(inp.distinct_sessions, obs_valid_int)

    if inp.cadence not in {"intraday", "swing", "paper_parity"}:
        violations.append(f"TOY_HORIZON_BELOW_MIN_TRL:invalid_cadence={_safe_num_str(inp.cadence)}")
    if inp.cadence == "swing" and inp.horizon_id != "W2_MULTI_REGIME_SWING_STRESS":
        violations.append(f"TOY_HORIZON_BELOW_MIN_TRL:swing_requires_W2_got_{_safe_num_str(inp.horizon_id)}")
    if inp.cadence == "paper_parity" and inp.horizon_id != "W5_LIVE_PAPER_PARITY_REPLAY":
        violations.append(f"TOY_HORIZON_BELOW_MIN_TRL:paper_parity_requires_W5_got_{_safe_num_str(inp.horizon_id)}")
    if inp.horizon_id == "W5_LIVE_PAPER_PARITY_REPLAY":
        rt_ok = (
            observed_closed_roundtrips is not None
            and not isinstance(observed_closed_roundtrips, bool)
            and isinstance(observed_closed_roundtrips, int)
            and observed_closed_roundtrips >= W5_MIN_CLOSED_ROUNDTRIPS
        )
        if not rt_ok:
            violations.append(
                f"TOY_HORIZON_BELOW_MIN_TRL:w5_closed_roundtrips="
                f"{_safe_num_str(observed_closed_roundtrips)}<{W5_MIN_CLOSED_ROUNDTRIPS}"
            )
        ts_ok = (
            observed_trade_sessions is not None
            and not isinstance(observed_trade_sessions, bool)
            and isinstance(observed_trade_sessions, int)
            and observed_trade_sessions >= W5_MIN_TRADE_SESSIONS
        )
        if not ts_ok:
            violations.append(
                f"TOY_HORIZON_BELOW_MIN_TRL:w5_trade_sessions="
                f"{_safe_num_str(observed_trade_sessions)}<{W5_MIN_TRADE_SESSIONS}"
            )
        elif rt_ok:
            assert observed_trade_sessions is not None and observed_closed_roundtrips is not None
            max_feasible_ts = min(eff_sessions, observed_closed_roundtrips)
            if observed_trade_sessions > max_feasible_ts:
                violations.append(
                    f"TOY_HORIZON_BELOW_MIN_TRL:w5_impossible_trade_sessions="
                    f"{_safe_num_str(observed_trade_sessions)}>{_safe_num_str(max_feasible_ts)}"
                )

    min_req_sessions = (
        RECOMMENDED_SWING_PLAYBACK_SESSIONS
        if inp.cadence == "swing"
        else (10 if inp.cadence == "paper_parity" else MIN_INTRADAY_PLAYBACK_SESSIONS)
    )
    if spec is not None:
        min_req_sessions = max(min_req_sessions, spec.min_sessions)
    if spec is None or spec.prewall_locked:
        min_req_sessions = max(min_req_sessions, min_trl)
    if eff_sessions < min_req_sessions:
        violations.append(
            f"TOY_HORIZON_BELOW_MIN_TRL:sessions={_safe_num_str(eff_sessions)}<required={min_req_sessions}"
        )

    # 3. Date Range, Official NYSE (XNYS) Session Count Plausibility, Endpoint Bounds & Study Wall Lock
    try:
        max_d = date.fromisoformat(inp.sample_max_date)
        start_d = date.fromisoformat(inp.sample_start_date)
    except (TypeError, ValueError):
        violations.append("INVALID_DATE_RANGE:malformed_iso_date")
        max_d = None
        start_d = None

    if start_d is not None and max_d is not None:
        if start_d > max_d:
            violations.append("INVALID_DATE_RANGE:start_after_end")
        else:
            nyse_sessions = _count_nyse_sessions_inclusive(start_d, max_d)
            if eff_sessions > nyse_sessions:
                violations.append(
                    f"INVALID_DATE_RANGE:sessions={_safe_num_str(eff_sessions)}>nyse_sessions={nyse_sessions}"
                )
        if spec is not None:
            spec_start = date.fromisoformat(spec.start_date)
            spec_end = date.fromisoformat(spec.end_date)
            if start_d < spec_start:
                violations.append(f"HORIZON_START_BOUND_BREACH:{inp.sample_start_date}<{spec.start_date}")
            if max_d > spec_end:
                if spec.prewall_locked and max_d > STUDY_WALL_DATE:
                    violations.append(f"STUDY_WALL_BREACH_POST_2026_10_15:{inp.sample_max_date}")
                else:
                    violations.append(f"HORIZON_END_BOUND_BREACH:{inp.sample_max_date}>{spec.end_date}")
            elif spec.prewall_locked and max_d > STUDY_WALL_DATE:
                violations.append(f"STUDY_WALL_BREACH_POST_2026_10_15:{inp.sample_max_date}")

    # 4. Realistic 1x (>= 8 bps RT) + 2x & Square-Root Impact Stress
    if (
        not _is_finite_real(inp.cost_1x_rt_bps)
        or float(inp.cost_1x_rt_bps) < MIN_REALISTIC_1X_RT_COST_BPS
        or inp.includes_2x_and_sqrt_impact is not True
    ):
        violations.append("UNREALISTIC_FLAT_COST_NO_2X_IMPACT")

    # 5. Uncapped Lossless Runner (TP OFF)
    if inp.take_profit_cap_enabled is not False:
        violations.append("FIXED_TAKE_PROFIT_CAP_ACTIVE")

    # 6. ADR-0022 Stop strictly in [0.0001%, 2.00%] (aligned with representable calibrator floor)
    if (
        not _is_finite_real(inp.max_stop_pct)
        or float(inp.max_stop_pct) < MIN_REPRESENTABLE_STOP_PCT
        or float(inp.max_stop_pct) > ADR0022_MAX_STOP_PCT
    ):
        violations.append(f"ADR0022_STOP_OVER_2PCT:{_safe_num_str(inp.max_stop_pct)}")

    # 7. Calendar-Day AR(1) Sharpe Annualization
    if inp.sharpe_annualization_mode != "CALENDAR_DAY_AR1_CLUSTERED":
        violations.append("INVALID_TRADE_COUNT_SHARPE_ANNUALIZATION")

    # 8. Multiple-Testing Governance (SPA + MCS + DSR + PBO + HLZ)
    if inp.uses_spa_mcs_dsr_pbo is not True:
        violations.append("MISSING_MULTIPLE_TESTING_SPA_MCS_DSR_PBO")

    # 9. Zero In-Sample Per-Symbol Cherry-Picking
    if inp.in_sample_symbol_cherry_picking is not False:
        violations.append("IN_SAMPLE_SYMBOL_CHERRY_PICKING")

    # 10. Cederburg-Safe Vol Brake strictly in [0.0, 1.0]
    if (
        not _is_finite_real(inp.vol_max_leverage)
        or float(inp.vol_max_leverage) < 0.0
        or float(inp.vol_max_leverage) > 1.0
    ):
        violations.append(f"VOL_TIMING_LEVERAGE_ABOVE_1X:{_safe_num_str(inp.vol_max_leverage)}")

    # 11. Full-Information Non-Stationary Allocator (No Stationary UCB1)
    if inp.allocator_mode != "FIXED_SHARE_HEDGE_AND_HIERARCHICAL_EB":
        violations.append("STATIONARY_UCB1_ON_FULL_INFO_MANIFEST")

    # 12. Strict EDGAR acceptanceDateTime PIT Timestamp
    if inp.edgar_timestamp_field != "acceptanceDateTime":
        violations.append(f"TRADE_DATE_EDGAR_LOOKAHEAD:{_safe_num_str(inp.edgar_timestamp_field)}")

    # 13. Zero Quarantined ADR-0064 Bleeders or POST Chases across BOTH active AND candidate strategy IDs
    active_ids = _normalize_str_sequence(
        inp.active_strategy_ids,
        "active_strategy_ids",
        violations,
        "QUARANTINED_ADR0064_OR_POST_CHASE_ACTIVE",
    )
    cand_ids = _normalize_str_sequence(
        candidate_strategy_ids,
        "candidate_strategy_ids",
        violations,
        "QUARANTINED_ADR0064_OR_POST_CHASE_ACTIVE",
    )
    all_strats = sorted(set(active_ids) | set(cand_ids))
    bad_strats = [sid for sid in all_strats if is_post_ant_strategy_quarantined(sid)]
    if bad_strats or inp.is_post_incentive_chase is not False:
        violations.append(f"QUARANTINED_ADR0064_OR_POST_CHASE_ACTIVE:{','.join(sorted(bad_strats)) or 'POST'}")

    # 14. Zero Silent Exception Swallowing & Zero Banned/Non-Permissive/Scraper Sources
    src_libs = _normalize_str_sequence(
        inp.source_libraries,
        "source_libraries",
        violations,
        "SILENT_DATA_FETCH_EXCEPTION_SWALLOW_OR_BANNED_SOURCE",
    )
    bad_libs = sorted({lib for lib in src_libs if lib.strip().lower() in BANNED_LOW_QUALITY_SOURCES})
    if inp.silent_exception_swallow is not False or bad_libs:
        violations.append(
            f"SILENT_DATA_FETCH_EXCEPTION_SWALLOW_OR_BANNED_SOURCE:{','.join(bad_libs) or 'silent_except'}"
        )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "purged_practice_rules_enforced": len(LOW_QUALITY_PRACTICE_PURGE_REGISTRY),
        "min_trl_sessions_required": min_trl,
        "effective_sessions_checked": eff_sessions,
        "meets_min_trl_power": min_trl_valid and (eff_sessions >= min_trl),
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
    cumulative_trials_n: int = NEXT_TRIAL_ID,
    cpcv_pbo: float = 1.0,
    dsr_probability: float = 0.0,
    alpha_spa_mcs: float = 0.10,
    n_bootstrap: int = 300,
    seed: int = 20261015,
    session_dates: Sequence[str] | None = None,
    mae_atr_dates: Sequence[str] | None = None,
    closed_roundtrips_by_strategy: Mapping[str, int] | None = None,
    closed_roundtrip_sessions_by_strategy: Mapping[str, int] | None = None,
    daily_roundtrips_by_strategy: Mapping[str, Sequence[int]] | None = None,
    cross_trial_sr_std: float | None = None,
    risk_unit_bps: float = 100.0,
    roundtrip_notional_fraction: float = 1.0,
) -> dict[str, Any]:
    """Build SHA-256 sealed playback performance manifest uniting purge audit, SPA/MCS, Conformal PID & Hedge."""
    if not strategy_daily_returns_1x_bps:
        raise ValueError("strategy_daily_returns_1x_bps must be non-empty")
    if set(strategy_daily_returns_1x_bps.keys()) != set(strategy_daily_returns_2x_bps.keys()):
        raise ValueError("1x and 2x strategy maps must have identical strategy IDs")

    names = sorted(strategy_daily_returns_1x_bps.keys())
    trials_n = _require_nonbool_int(cumulative_trials_n, "cumulative_trials_n", minimum=1)
    if len(names) > trials_n:
        raise ValueError(f"cumulative_trials_n={trials_n} cannot be less than candidate count={len(names)}")
    pbo_val = _require_finite_real(cpcv_pbo, "cpcv_pbo")
    if not (0.0 <= pbo_val <= 1.0):
        raise ValueError(f"cpcv_pbo must lie in [0.0, 1.0] (got {cpcv_pbo!r})")
    dsr_in = _require_finite_real(dsr_probability, "dsr_probability")
    if not (0.0 <= dsr_in <= 1.0):
        raise ValueError(f"dsr_probability must lie in [0.0, 1.0] (got {dsr_probability!r})")
    rv_clean = _require_finite_real(realized_vol_ann_pct, "realized_vol_ann_pct")
    if rv_clean <= 0.0:
        raise ValueError(f"realized_vol_ann_pct must be > 0 (got {realized_vol_ann_pct!r})")
    ct_sr_sd_clean: float | None = None
    if cross_trial_sr_std is not None:
        ct_sr_sd_clean = _require_finite_real(cross_trial_sr_std, "cross_trial_sr_std")
        if trials_n > 1 and ct_sr_sd_clean <= 0.0:
            raise ValueError(f"cross_trial_sr_std must be > 0 when cumulative_trials_n > 1 (got {ct_sr_sd_clean!r})")
        if ct_sr_sd_clean < 0.0:
            raise ValueError(f"cross_trial_sr_std must be >= 0 (got {cross_trial_sr_std!r})")
    r_unit_bps = _require_finite_real(risk_unit_bps, "risk_unit_bps")
    if r_unit_bps <= 0.0:
        raise ValueError(f"risk_unit_bps must be > 0 (got {risk_unit_bps!r})")
    rt_notional_frac = _require_finite_real(roundtrip_notional_fraction, "roundtrip_notional_fraction")
    if not (MIN_ROUNDTRIP_NOTIONAL_FRACTION <= rt_notional_frac <= 1.0):
        raise ValueError(
            f"roundtrip_notional_fraction must lie in [{MIN_ROUNDTRIP_NOTIONAL_FRACTION}, 1.0] "
            f"(got {roundtrip_notional_fraction!r})"
        )
    seed_i = _require_nonbool_int(seed, "seed")

    # Unconditionally validate every `(mae_pct, atr_pct)` sample even when `mae_atr_dates` is omitted
    validated_mae_samples: list[tuple[float, float]] = []
    for idx_m, pair in enumerate(realized_mae_atr_samples):
        if len(pair) != 2:
            raise ValueError(f"realized_mae_atr_samples[{idx_m}] must be a 2-tuple (mae_pct, atr_pct)")
        mae_v = _require_finite_real(pair[0], f"realized_mae_atr_samples[{idx_m}].mae_pct")
        atr_v = _require_finite_real(pair[1], f"realized_mae_atr_samples[{idx_m}].atr_pct")
        if mae_v < 0.0:
            raise ValueError(f"realized_mae_atr_samples[{idx_m}].mae_pct must be >= 0 (got {mae_v!r})")
        if atr_v <= 0.0:
            raise ValueError(f"realized_mae_atr_samples[{idx_m}].atr_pct must be > 0 (got {atr_v!r})")
        validated_mae_samples.append((mae_v, atr_v))

    len_1x = {len(strategy_daily_returns_1x_bps[k]) for k in names}
    len_2x = {len(strategy_daily_returns_2x_bps[k]) for k in names}
    if len(len_1x) != 1 or len(len_2x) != 1 or len_1x != len_2x:
        raise ValueError(
            f"All 1x and 2x strategy series must share identical aligned length T "
            f"(1x={sorted(len_1x)}, 2x={sorted(len_2x)})"
        )
    obs_t = next(iter(len_2x))
    if obs_t < 10:
        raise ValueError(f"Require >= 10 aligned sessions across 1x & 2x series (got {obs_t})")

    horizon_spec = SUITABLE_PLAYBACK_HORIZONS.get(audit_input.horizon_id)
    obs_start_date = audit_input.sample_start_date
    obs_max_date = audit_input.sample_max_date
    canonical_session_dates: list[str] | None = None
    if session_dates is not None:
        decl_start_d = date.fromisoformat(audit_input.sample_start_date)
        decl_max_d = date.fromisoformat(audit_input.sample_max_date)
        if len(session_dates) != obs_t:
            raise ValueError(f"session_dates length {len(session_dates)} != observed sessions {obs_t}")
        parsed_dates = [date.fromisoformat(d) for d in session_dates]
        for i, d_item in enumerate(parsed_dates):
            if not _is_nyse_trading_day(d_item):
                raise ValueError(
                    f"session_dates[{i}]={d_item.isoformat()} is a weekend or NYSE holiday; require XNYS sessions"
                )
            if i > 0 and d_item <= parsed_dates[i - 1]:
                raise ValueError("session_dates must be strictly increasing unique ISO dates")
        if parsed_dates[0] < decl_start_d or parsed_dates[-1] > decl_max_d:
            raise ValueError(
                f"session_dates [{parsed_dates[0].isoformat()}, {parsed_dates[-1].isoformat()}] "
                f"exceeds declared sample window [{audit_input.sample_start_date}, {audit_input.sample_max_date}]"
            )
        # Require contiguous official NYSE trading sessions so interior wipeout sessions can NEVER be omitted!
        expected_nyse_count = _count_nyse_sessions_inclusive(parsed_dates[0], parsed_dates[-1])
        if len(parsed_dates) != expected_nyse_count:
            raise ValueError(
                f"session_dates has omitted interior NYSE trading sessions "
                f"(supplied={len(parsed_dates)} != contiguous_nyse={expected_nyse_count})"
            )
        canonical_session_dates = [d.isoformat() for d in parsed_dates]
        obs_start_date = canonical_session_dates[0]
        obs_max_date = canonical_session_dates[-1]

    canonical_mae_dates: list[str] | None = None
    if mae_atr_dates is not None:
        if len(mae_atr_dates) != len(validated_mae_samples):
            raise ValueError("mae_atr_dates length must match realized_mae_atr_samples")
        parsed_mae_dates = [date.fromisoformat(d) for d in mae_atr_dates]
        for i, d_m in enumerate(parsed_mae_dates):
            if not _is_nyse_trading_day(d_m):
                raise ValueError(
                    f"mae_atr_dates[{i}]={d_m.isoformat()} is a weekend or NYSE holiday; require XNYS sessions"
                )
            if i > 0 and d_m < parsed_mae_dates[i - 1]:
                raise ValueError("mae_atr_dates must be chronologically non-decreasing")
        span_start_d = date.fromisoformat(obs_start_date)
        span_max_d = date.fromisoformat(obs_max_date)
        if any(d < span_start_d or d > span_max_d for d in parsed_mae_dates):
            raise ValueError(f"mae_atr_dates must lie within observed sample span [{obs_start_date}, {obs_max_date}]")
        if horizon_spec is not None and horizon_spec.prewall_locked:
            if any(d > STUDY_WALL_DATE for d in parsed_mae_dates):
                raise ValueError(f"mae_atr_dates breaches pre-wall lock <= {STUDY_WALL_DATE.isoformat()}")
        canonical_mae_dates = [d.isoformat() for d in parsed_mae_dates]

    # Exact key parity on `daily_roundtrips_by_strategy`, `closed_roundtrips_by_strategy`, & closing-session counts
    daily_rt_clean: dict[str, list[int]] | None = None
    if daily_roundtrips_by_strategy is not None:
        if set(daily_roundtrips_by_strategy.keys()) != set(names):
            missing_dr = sorted(set(names) - set(daily_roundtrips_by_strategy.keys()))
            extra_dr = sorted(set(daily_roundtrips_by_strategy.keys()) - set(names))
            raise ValueError(f"daily_roundtrips_by_strategy key mismatch (missing={missing_dr}, extra={extra_dr})")
        daily_rt_clean = {}
        for sid in names:
            seq_dr = daily_roundtrips_by_strategy[sid]
            if len(seq_dr) != obs_t:
                raise ValueError(f"daily_roundtrips[{sid}] length {len(seq_dr)} != observed sessions {obs_t}")
            daily_rt_clean[sid] = [
                _require_nonbool_int(c, f"daily_roundtrips[{sid}][{t_i}]", minimum=0) for t_i, c in enumerate(seq_dr)
            ]

    rt_by_strat: dict[str, int] = {}
    rt_supplied = (closed_roundtrips_by_strategy is not None) or (daily_rt_clean is not None)
    if closed_roundtrips_by_strategy is not None:
        if set(closed_roundtrips_by_strategy.keys()) != set(names):
            missing_rt = sorted(set(names) - set(closed_roundtrips_by_strategy.keys()))
            extra_rt = sorted(set(closed_roundtrips_by_strategy.keys()) - set(names))
            raise ValueError(f"closed_roundtrips_by_strategy key mismatch (missing={missing_rt}, extra={extra_rt})")
        for sid in names:
            rt_by_strat[sid] = _require_nonbool_int(
                closed_roundtrips_by_strategy[sid], f"closed_roundtrips[{sid}]", minimum=0
            )
            if daily_rt_clean is not None and sum(daily_rt_clean[sid]) != rt_by_strat[sid]:
                daily_sum = sum(daily_rt_clean[sid])
                raise ValueError(f"closed_roundtrips[{sid}]={rt_by_strat[sid]} != sum(daily_roundtrips)={daily_sum}")
        total_closed_rt: int | None = sum(rt_by_strat.values())
    elif daily_rt_clean is not None:
        for sid in names:
            rt_by_strat[sid] = sum(daily_rt_clean[sid])
        total_closed_rt = sum(rt_by_strat.values())
    else:
        for sid in names:
            rt_by_strat[sid] = 0
        total_closed_rt = None

    # Validate per-strategy trade-session counts:
    # - `0 <= ts_v <= min(obs_t, rt_by_strat[sid])` AND `ts_v >= 1` whenever `rt_by_strat[sid] > 0`!
    # - When `daily_roundtrips_by_strategy` is given, compute exact portfolio union of closing sessions!
    rt_sessions_by_strat: dict[str, int] | None = None
    total_trade_sessions: int | None = None
    if closed_roundtrip_sessions_by_strategy is not None:
        if set(closed_roundtrip_sessions_by_strategy.keys()) != set(names):
            raise ValueError("closed_roundtrip_sessions_by_strategy keys must match strategy IDs")
        rt_sessions_by_strat = {}
        for sid in names:
            ts_v = _require_nonbool_int(
                closed_roundtrip_sessions_by_strategy[sid], f"closed_roundtrip_sessions[{sid}]", minimum=0
            )
            max_ts_sid = min(obs_t, rt_by_strat[sid]) if rt_supplied else obs_t
            if ts_v > max_ts_sid:
                raise ValueError(f"closed_roundtrip_sessions[{sid}]={ts_v} exceeds feasible bound {max_ts_sid}")
            if rt_supplied and rt_by_strat[sid] > 0 and ts_v == 0:
                raise ValueError(
                    f"closed_roundtrip_sessions[{sid}]=0 is impossible for {rt_by_strat[sid]} closed roundtrips"
                )
            if daily_rt_clean is not None:
                actual_ts_sid = sum(1 for c in daily_rt_clean[sid] if c > 0)
                if ts_v != actual_ts_sid:
                    raise ValueError(
                        f"closed_roundtrip_sessions[{sid}]={ts_v} != daily active closing sessions {actual_ts_sid}"
                    )
            rt_sessions_by_strat[sid] = ts_v
        if daily_rt_clean is not None:
            total_trade_sessions = sum(1 for t_i in range(obs_t) if any(daily_rt_clean[sid][t_i] > 0 for sid in names))
        else:
            total_trade_sessions = max(rt_sessions_by_strat.values(), default=0)
    elif daily_rt_clean is not None:
        rt_sessions_by_strat = {sid: sum(1 for c in daily_rt_clean[sid] if c > 0) for sid in names}
        total_trade_sessions = sum(1 for t_i in range(obs_t) if any(daily_rt_clean[sid][t_i] > 0 for sid in names))

    # Apply absorbing-bankruptcy truncation + exact Fraction break-even capital survival + live session count
    # BEFORE W5 parity audit & cost reconciliation so dead post-bankruptcy closes can NEVER pass W5!
    eff_1x: dict[str, list[float]] = {}
    eff_2x: dict[str, list[float]] = {}
    bankrupt_1x: dict[str, bool] = {}
    bankrupt_2x: dict[str, bool] = {}
    eq_gt_one_1x: dict[str, bool] = {}
    eq_gt_one_2x: dict[str, bool] = {}
    live_len_1x: dict[str, int] = {}
    live_len_2x: dict[str, int] = {}
    raw_1x_clean: dict[str, list[float]] = {}
    raw_2x_clean: dict[str, list[float]] = {}
    for sid in names:
        r1_c = [_require_finite_real(x, f"1x[{sid}][{i}]") for i, x in enumerate(strategy_daily_returns_1x_bps[sid])]
        r2_c = [_require_finite_real(x, f"2x[{sid}][{i}]") for i, x in enumerate(strategy_daily_returns_2x_bps[sid])]
        raw_1x_clean[sid] = r1_c
        raw_2x_clean[sid] = r2_c
        eff_1x[sid], bankrupt_1x[sid], eq_gt_one_1x[sid], live_len_1x[sid] = _apply_absorbing_bankruptcy(r1_c)
        eff_2x[sid], bankrupt_2x[sid], eq_gt_one_2x[sid], live_len_2x[sid] = _apply_absorbing_bankruptcy(r2_c)

    # Enforce pointwise `2x <= 1x` in exact 80-digit Decimal + exact cadence-aware live-capital cost reconciliation:
    # - Same-day intraday cadence enforces whole-trip `floor(d_t / d_unit_drag)` per session (no overnight carry!).
    # - Multi-session/swing/parity cadences enforce causal two-leg (`d_half_drag`) entry<=exit reconciliation!
    # - Per-day representation fuzz strictly `< 1e-9` of a single half-trip leg (aggregate `< 1e-6` of a half-leg),
    #   so `8 * 10**12` bps can NEVER create an extra whole trip (`10**12 + 1` fails cleanly!).
    # - Exact arbitrary-precision `int` capacities + `Overflow` trap disabled -> safe even at `rt = 10**1_000_000`!
    cost_1x_bps = (
        float(audit_input.cost_1x_rt_bps)
        if (_is_finite_real(audit_input.cost_1x_rt_bps) and float(audit_input.cost_1x_rt_bps) > 0.0)
        else MIN_REALISTIC_1X_RT_COST_BPS
    )
    cadence_low = str(audit_input.cadence or "").strip().lower()
    is_same_day_intraday = (
        ("intraday" in cadence_low)
        and ("swing" not in cadence_low)
        and ("overnight" not in cadence_low)
        and ("multi" not in cadence_low)
    )

    def _max_causal_closing_sessions_and_caps(legs_seq: Sequence[int]) -> tuple[int, list[int]]:
        """Compute exact maximum causal closing sessions & per-day exit-leg caps for multi-session swing/parity.

        A session `t` can close a trade only when `legs_seq[t] >= 1` AND cumulative legs up through `t` cover
        at least `2` legs (`cum_legs >= 2`, preventing a leading entry-only `4 bps` session from counting as a
        closing session!). Distinct closing sessions are greedily bounded by `cum_legs >= 2 * (used + 1)`.
        """
        cum_lg = 0
        used_sessions = 0
        exit_caps: list[int] = []
        for lg in legs_seq:
            cum_lg += lg
            if lg >= 1 and cum_lg >= 2:
                exit_caps.append(min(lg, cum_lg // 2))
                if cum_lg >= 2 * (used_sessions + 1):
                    used_sessions += 1
            else:
                exit_caps.append(0)
        return used_sessions, exit_caps

    def _min_causal_closing_sessions(legs_seq: Sequence[int], target_rt: int) -> int | None:
        """Exact O(T log T) backward max-heap minimum closing sessions required to causally realize `target_rt`."""
        if target_rt <= 0:
            return 0
        prefix_u: list[int] = []
        cum_lg = 0
        for lg in legs_seq:
            cum_lg += lg
            prefix_u.append(cum_lg // 2)
        if not prefix_u or target_rt > prefix_u[-1]:
            return None
        rem = target_rt
        max_heap: list[int] = []
        days_used = 0
        for t_idx in range(len(legs_seq) - 1, -1, -1):
            if legs_seq[t_idx] > 0:
                heapq.heappush(max_heap, -legs_seq[t_idx])
            prev_u = prefix_u[t_idx - 1] if t_idx > 0 else 0
            while rem > prev_u:
                if not max_heap:
                    return None
                rem -= -heapq.heappop(max_heap)
                days_used += 1
                if rem <= 0:
                    return days_used
        return days_used

    all_strategies_cost_ok = True
    live_cap_by_strat: dict[str, int] = {}
    legs_per_day_by_strat: dict[str, list[int]] = {}
    trips_per_day_by_strat: dict[str, list[int]] = {}
    max_closing_sessions_by_strat: dict[str, int] = {}
    max_rt_cap_by_strat: dict[str, int] = {}
    pointwise_nonneg_by_strat: dict[str, bool] = {}
    with localcontext() as cost_ctx:
        cost_ctx.prec = 80
        cost_ctx.traps[Overflow] = False
        d_unit_drag = Decimal.from_float(cost_1x_bps) * Decimal.from_float(rt_notional_frac)
        d_half_drag = d_unit_drag / Decimal(2)
        f_unit_drag = Fraction(cost_1x_bps) * Fraction(rt_notional_frac)
        f_half_drag = f_unit_drag / 2
        d_half_day_fuzz_cap = d_half_drag * Decimal("1e-9")
        d_agg_fuzz_cap = d_half_drag * Decimal("1e-6")
        d_div_thresh = d_half_drag * Decimal("1e70")
        for sid in names:
            r1_s = raw_1x_clean[sid]
            r2_s = raw_2x_clean[sid]
            d_diffs = [Decimal.from_float(a) - Decimal.from_float(b) for a, b in zip(r1_s, r2_s, strict=True)]
            pw_ok = not any(d < 0 for d in d_diffs)
            pointwise_nonneg_by_strat[sid] = pw_ok
            live_cap_sessions = min(live_len_1x[sid], live_len_2x[sid])
            live_cap_by_strat[sid] = live_cap_sessions
            legs_per_day: list[int] = []
            d_live_diffs: list[Decimal] = []
            d_sum_day_tol = Decimal(0)
            for t_i in range(obs_t):
                if t_i >= live_cap_sessions or not pw_ok:
                    d_live_diffs.append(Decimal(0))
                    legs_per_day.append(0)
                    continue
                d_t = d_diffs[t_i]
                d_live_diffs.append(d_t)
                if d_t <= 0:
                    legs_per_day.append(0)
                    continue
                mag_t = max(abs(Decimal.from_float(r1_s[t_i])), abs(Decimal.from_float(r2_s[t_i])))
                d_day_tol = min(d_half_day_fuzz_cap, mag_t * Decimal("1e-14"))
                d_sum_day_tol += d_day_tol
                if d_t < d_div_thresh:
                    legs_per_day.append(int((d_t + d_day_tol) // d_half_drag))
                else:
                    f_d_t = Fraction(r1_s[t_i]) - Fraction(r2_s[t_i])
                    legs_per_day.append(int((f_d_t + Fraction(d_day_tol)) // f_half_drag))
            legs_per_day_by_strat[sid] = legs_per_day
            trips_per_day = [lg // 2 for lg in legs_per_day]
            trips_per_day_by_strat[sid] = trips_per_day
            if is_same_day_intraday:
                max_closing_sess_sid = sum(1 for tr in trips_per_day if tr >= 1)
                day_exit_caps_sid = trips_per_day
            else:
                max_closing_sess_sid, day_exit_caps_sid = _max_causal_closing_sessions_and_caps(legs_per_day)
            max_closing_sessions_by_strat[sid] = max_closing_sess_sid
            max_rt_cap = sum(trips_per_day) if is_same_day_intraday else (sum(legs_per_day) // 2)
            max_rt_cap_by_strat[sid] = max_rt_cap

            if not pw_ok:
                all_strategies_cost_ok = False
                continue

            is_flat_cash = all(x == 0.0 for x in r1_s) and all(x == 0.0 for x in r2_s)
            d_total_drag = sum(d_live_diffs, Decimal(0))
            d_agg_tol = min(d_sum_day_tol, d_agg_fuzz_cap)
            if rt_supplied and rt_by_strat[sid] == 0:
                if not is_flat_cash and not (d_total_drag > 0 and d_total_drag + d_agg_tol >= d_half_drag):
                    all_strategies_cost_ok = False
                    continue
            elif rt_supplied:
                rt_target = rt_by_strat[sid]
                if max_rt_cap < rt_target:
                    all_strategies_cost_ok = False
                    continue
                if rt_target < 10**70:
                    d_min_req = Decimal(rt_target) * d_unit_drag
                    if not (d_total_drag > 0 and d_total_drag + d_agg_tol >= d_min_req):
                        all_strategies_cost_ok = False
                        continue
            else:
                if max_rt_cap < 1 or not (d_total_drag > 0 and d_total_drag + d_agg_tol >= d_unit_drag):
                    all_strategies_cost_ok = False
                    continue

            if daily_rt_clean is not None:
                cum_legs = 0
                cum_closes = 0
                for t_i, c_t in enumerate(daily_rt_clean[sid]):
                    if t_i < live_cap_sessions:
                        cum_legs += legs_per_day[t_i]
                    if c_t > 0:
                        if t_i >= live_cap_sessions:
                            all_strategies_cost_ok = False
                            break
                        if is_same_day_intraday:
                            if trips_per_day[t_i] < c_t:
                                all_strategies_cost_ok = False
                                break
                        else:
                            # Multi-session swing/parity: exit day pays >= `c_t` legs & causal prefix >= `2*cum_closes`!
                            cum_closes += c_t
                            if legs_per_day[t_i] < c_t or cum_legs < 2 * cum_closes:
                                all_strategies_cost_ok = False
                                break
            elif rt_supplied and rt_by_strat[sid] > 0 and rt_sessions_by_strat is not None:
                rt_target = rt_by_strat[sid]
                ts_sid = rt_sessions_by_strat[sid]
                if ts_sid > live_cap_sessions or ts_sid > max_closing_sess_sid:
                    all_strategies_cost_ok = False
                    continue
                if ts_sid > 0:
                    if is_same_day_intraday:
                        top_ts_cap = sum(sorted(day_exit_caps_sid, reverse=True)[:ts_sid])
                        if top_ts_cap < rt_target:
                            all_strategies_cost_ok = False
                            continue
                    else:
                        min_req_days = _min_causal_closing_sessions(legs_per_day, rt_target)
                        if min_req_days is None or ts_sid < min_req_days:
                            all_strategies_cost_ok = False
                            continue

    # Compute strictly live-capital, cost-backed W5 parity counts BEFORE issuing audit results
    # (blocking post-bankruptcy zombie closes & leading entry-only sessions from ever certifying W5 parity!).
    w5_closed_rt: int | None = total_closed_rt
    w5_trade_sessions: int | None = total_trade_sessions
    if daily_rt_clean is not None:
        live_rt_total = 0
        live_active_day_flags = [False] * obs_t
        for sid in names:
            live_cap = live_cap_by_strat[sid]
            legs_s = legs_per_day_by_strat[sid]
            trips_s = trips_per_day_by_strat[sid]
            cum_legs = 0
            cum_credited = 0
            for t_i in range(live_cap):
                cum_legs += legs_s[t_i]
                c_t = daily_rt_clean[sid][t_i]
                if c_t > 0:
                    if is_same_day_intraday:
                        credited_t = min(c_t, trips_s[t_i])
                    else:
                        avail_pairs = max(0, (cum_legs - 2 * cum_credited) // 2)
                        credited_t = min(c_t, legs_s[t_i], avail_pairs)
                    if credited_t > 0:
                        cum_credited += credited_t
                        live_active_day_flags[t_i] = True
            live_rt_total += cum_credited
        w5_closed_rt = live_rt_total
        w5_trade_sessions = sum(1 for flag in live_active_day_flags if flag)
    elif rt_supplied:
        credited_by_sid: dict[str, int] = {}
        for sid in names:
            credited_by_sid[sid] = (
                min(rt_by_strat[sid], max_rt_cap_by_strat[sid]) if pointwise_nonneg_by_strat[sid] else 0
            )
        w5_closed_rt = sum(credited_by_sid.values())
        if rt_sessions_by_strat is not None:
            w5_trade_sessions = max(
                (
                    min(
                        rt_sessions_by_strat[sid],
                        live_cap_by_strat[sid],
                        max_closing_sessions_by_strat[sid],
                        credited_by_sid[sid],
                    )
                    for sid in names
                ),
                default=0,
            )

    # Audit declared `audit_input` directly (never mask a declared `2026-10-16` breach using earlier `session_dates`!)
    hygiene_decl = audit_low_quality_practices(
        audit_input,
        observed_sessions=obs_t,
        candidate_strategy_ids=names,
        observed_closed_roundtrips=w5_closed_rt,
        observed_trade_sessions=w5_trade_sessions,
    )
    if canonical_session_dates is not None and (
        obs_start_date != audit_input.sample_start_date or obs_max_date != audit_input.sample_max_date
    ):
        obs_audit_input = PlaybackAuditInput(
            horizon_id=audit_input.horizon_id,
            sample_start_date=obs_start_date,
            sample_max_date=obs_max_date,
            distinct_sessions=audit_input.distinct_sessions,
            cadence=audit_input.cadence,
            universe_mode=audit_input.universe_mode,
            cost_1x_rt_bps=audit_input.cost_1x_rt_bps,
            includes_2x_and_sqrt_impact=audit_input.includes_2x_and_sqrt_impact,
            take_profit_cap_enabled=audit_input.take_profit_cap_enabled,
            max_stop_pct=audit_input.max_stop_pct,
            sharpe_annualization_mode=audit_input.sharpe_annualization_mode,
            uses_spa_mcs_dsr_pbo=audit_input.uses_spa_mcs_dsr_pbo,
            in_sample_symbol_cherry_picking=audit_input.in_sample_symbol_cherry_picking,
            vol_max_leverage=audit_input.vol_max_leverage,
            allocator_mode=audit_input.allocator_mode,
            edgar_timestamp_field=audit_input.edgar_timestamp_field,
            active_strategy_ids=audit_input.active_strategy_ids,
            silent_exception_swallow=audit_input.silent_exception_swallow,
            source_libraries=audit_input.source_libraries,
            is_post_incentive_chase=audit_input.is_post_incentive_chase,
            target_sharpe_for_min_trl=audit_input.target_sharpe_for_min_trl,
            ar1_rho_for_min_trl=audit_input.ar1_rho_for_min_trl,
        )
        hygiene_obs = audit_low_quality_practices(
            obs_audit_input,
            observed_sessions=obs_t,
            candidate_strategy_ids=names,
            observed_closed_roundtrips=w5_closed_rt,
            observed_trade_sessions=w5_trade_sessions,
        )
        merged_v = list(dict.fromkeys([*hygiene_decl["violations"], *hygiene_obs["violations"]]))
        hygiene: dict[str, Any] = {
            **hygiene_obs,
            "violations": merged_v,
            "passed": len(merged_v) == 0,
            "meets_min_trl_power": bool(hygiene_decl["meets_min_trl_power"] and hygiene_obs["meets_min_trl_power"]),
        }
    else:
        hygiene = hygiene_decl

    # Validate cross_trial_sr_std against unavoidable mathematical lower bound (including infinite-Sharpe constants!)
    min_feasible_sd = _min_feasible_cross_trial_sr_std(eff_2x, trials_n)
    if (
        ct_sr_sd_clean is not None
        and min_feasible_sd > 0.0
        and (not math.isfinite(min_feasible_sd) or ct_sr_sd_clean + 1e-12 < min_feasible_sd)
    ):
        raise ValueError(
            f"Infeasible cross_trial_sr_std={ct_sr_sd_clean:.6f} < mathematical lower bound "
            f"{min_feasible_sd} implied by submitted candidate Sharpes across N={trials_n} trials"
        )

    # Per-strategy calendar-day AR(1) stats (1x and 2x costs) + exact dyadic Fraction R-unit EB inputs
    # (100% permutation-invariant even at `[2^-300, 2^-600, -2^-300]` AND preserves subnormal `2^-1074 / 100`!)
    per_strategy: dict[str, dict[str, Any]] = {}
    eb_obs: list[StrategyClusterObservation] = []
    f_r_unit = Fraction(r_unit_bps)
    with localcontext() as r_ctx:
        r_ctx.prec = 80
        for sid in names:
            s1 = compute_calendar_day_stats(eff_1x[sid])
            s2 = compute_calendar_day_stats(eff_2x[sid])
            n_sess = max(1, int(s2["sessions"] or 0))
            q_2x, shift_2x = _to_exact_dyadic_ints(eff_2x[sid])
            sum_q_2x = sum(q_2x)
            f_mean_r = Fraction(sum_q_2x, n_sess << shift_2x) / f_r_unit
            try:
                mean_r = float(f_mean_r)
            except OverflowError as exc:
                raise ValueError(
                    f"mean return in R units overflows finite float for {sid} with risk_unit_bps={r_unit_bps}"
                ) from exc
            if not math.isfinite(mean_r):
                raise ValueError(
                    f"mean return in R units overflows finite float for {sid} with risk_unit_bps={r_unit_bps}"
                )
            _, raw_rho_2x, _, _ = _pre_scaled_moments(eff_2x[sid])
            if n_sess < 2 or min(q_2x) == max(q_2x):
                f_se_sq_r = Fraction(0, 1)
                se_r = 0.0
            else:
                c_2x = [n_sess * qi - sum_q_2x for qi in q_2x]
                ss_c_2x = sum(ci * ci for ci in c_2x)
                if ss_c_2x <= 0:
                    f_se_sq_r = Fraction(0, 1)
                    se_r = 0.0
                else:
                    rho_for_se = max(-(_MAX_STATIONARY_RHO - 1e-9), min(_MAX_STATIONARY_RHO - 1e-9, raw_rho_2x))
                    ar1_var_mult = max(1.0, (1.0 + rho_for_se) / (1.0 - rho_for_se))
                    f_var_bps = Fraction(ss_c_2x, (n_sess * n_sess * (n_sess - 1)) << (2 * shift_2x))
                    f_se_sq_r = (f_var_bps / (n_sess * f_r_unit * f_r_unit)) * Fraction(ar1_var_mult)
                    d_se_sq_r = Decimal(f_se_sq_r.numerator) / Decimal(f_se_sq_r.denominator)
                    d_se_r = d_se_sq_r.sqrt()
                    se_r = float(d_se_r)
                    if not math.isfinite(se_r):
                        raise ValueError(
                            f"SE in R units overflows finite float for {sid} with risk_unit_bps={r_unit_bps}"
                        )

            # Exact integer sum sign prevents subnormal underflow (`[0, 2^-1074]*200` survives!)
            sum_1x_pos = sum(_to_exact_dyadic_ints(eff_1x[sid])[0]) > 0
            sum_2x_pos = sum_q_2x > 0

            rt_count = rt_by_strat[sid]
            eb_obs.append(
                StrategyClusterObservation(
                    strategy_id=sid,
                    cluster_id=strategy_clusters.get(sid, "UNCLUSTERED"),
                    observed_mean_r=mean_r,
                    observed_se_r=se_r,
                    closed_roundtrips=rt_count,
                    exact_mean_r=f_mean_r,
                    exact_se_sq_r=f_se_sq_r,
                )
            )
            per_strategy[sid] = {
                "cluster_id": strategy_clusters.get(sid, "UNCLUSTERED"),
                "quarantined_adr0064": is_post_ant_strategy_quarantined(sid),
                "sessions_observed": n_sess,
                "closed_roundtrips": rt_count,
                "bankrupt_1x": bankrupt_1x[sid],
                "bankrupt_2x": bankrupt_2x[sid],
                "cost_1x": s1,
                "cost_2x": s2,
                "survives_1x_cost": (not bankrupt_1x[sid]) and sum_1x_pos and eq_gt_one_1x[sid],
                "survives_2x_cost": (not bankrupt_2x[sid]) and sum_2x_pos and eq_gt_one_2x[sid],
            }

    # Hansen (2005) SPA + Politis-Romano Stationary Bootstrap + HLN (2011) MCS on absorbing-equity 2x net returns
    spa_mcs = evaluate_hansen_spa_and_mcs(
        eff_2x,
        alpha=alpha_spa_mcs,
        n_bootstrap=n_bootstrap,
        avg_block_len=5.0,
        seed=seed_i,
    )

    # Conformal PID stop calibration strictly bounded by audited `max_stop_pct` in [0.0001%, 2.00%];
    # fails closed (NEVER emits 1.25% / adr0022_compliant=True) when submitted max_stop_pct is invalid!
    audit_stop_valid = (
        _is_finite_real(audit_input.max_stop_pct)
        and MIN_REPRESENTABLE_STOP_PCT <= float(audit_input.max_stop_pct) <= ADR0022_MAX_STOP_PCT
    )
    if audit_stop_valid:
        eff_max_stop = float(audit_input.max_stop_pct)
        # Adaptive lower floor for `6-14 bps` stops & non-grid caps like `0.00015` / `0.10005`
        eff_min_stop = max(MIN_REPRESENTABLE_STOP_PCT, min(0.15, math.floor(eff_max_stop * 5_000.0) / 10_000.0))
        pid = ConformalPidStopCalibrator(
            target_miscoverage_alpha=0.10,
            min_stop_pct=eff_min_stop,
            max_stop_pct=eff_max_stop,
        )
        last_pid_state: dict[str, Any] = {
            "next_quantile_multiplier": pid.quantile_multiplier,
            "next_stop_pct": pid.recommended_stop_pct(1.0),
            "adr0022_compliant": True,
        }
        if canonical_mae_dates is not None:
            for mae_pct, atr_pct in validated_mae_samples:
                last_pid_state = pid.update(realized_mae_pct=mae_pct, atr_pct=atr_pct)
    else:
        last_pid_state = {
            "next_quantile_multiplier": 0.0,
            "next_stop_pct": 0.0,
            "adr0022_compliant": False,
            "error": "invalid_audited_max_stop_pct",
        }

    # Chronological Full-Information Fixed-Share Hedge carry-through strictly in log-weight space across t = 0..obs_t-1
    # Pre-clip returns to `±3 * r_unit_bps` before dividing so tiny `r_unit_bps < 1` never overflows `1e308`!
    hedge_bps_cap = 3.0 * r_unit_bps
    hedge_reward_seq = [
        {sid: max(-hedge_bps_cap, min(hedge_bps_cap, eff_2x[sid][t_idx])) / r_unit_bps for sid in names}
        for t_idx in range(obs_t)
    ]
    hedge_weights = run_fixed_share_hedge_sequence(
        {sid: 1.0 / len(names) for sid in names},
        hedge_reward_seq,
        eta=0.25,
        switching_alpha=0.05,
    )

    hierarchical_eb = compute_hierarchical_eb_merit_pool(eb_obs)
    eff_vol_cap = (
        min(1.0, max(0.0, float(audit_input.vol_max_leverage)))
        if _is_finite_real(audit_input.vol_max_leverage)
        else 1.0
    )
    if eff_vol_cap == 0.0:
        vol_brake = 0.0
    else:
        raw_brake = compute_asymmetric_vol_crash_brake(
            rv_clean,
            target_vol_ann_pct=18.0,
            min_scale=min(0.25, eff_vol_cap),
        )
        vol_brake = min(raw_brake, eff_vol_cap)

    req_hlz_t_raw = _hlz_t_hurdle_raw(trials_n)
    req_hlz_t = round(req_hlz_t_raw, 4)
    best_sid = str(spa_mcs["best_strategy_id"])
    best_t_raw = float(spa_mcs["per_strategy_t_stats_raw"][best_sid])

    # Distinctly reported 80-digit DSR suite: AR(1) observed-SR PSD (gated), exact BLP (2014) IID, & ADIA #19 null-SR
    _, best_rho_raw, best_skew_raw, best_ex_kurt_raw = _pre_scaled_moments(eff_2x[best_sid])
    computed_dsr_2x, computed_dsr_2x_blp2014_iid, computed_dsr_2x_adia19_null = _compute_empirical_dsr_suite(
        eff_2x[best_sid],
        cumulative_trials_n=trials_n,
        ar1_rho=best_rho_raw,
        cross_trial_sr_std=ct_sr_sd_clean,
    )

    # Built-in exact-purge, zero-epsilon, conservative-tie C(6, 2) = 15 CPCV PBO for BOTH Sharpe AND HAC-t selectors!
    purge_cfg = horizon_spec.purge_sessions if horizon_spec is not None else 0
    computed_cpcv_pbo_sharpe = _compute_empirical_cpcv_pbo(
        eff_2x,
        n_groups=6,
        k_test=2,
        purge_sessions=purge_cfg,
        selector="sharpe",
    )
    computed_cpcv_pbo_hac = _compute_empirical_cpcv_pbo(
        eff_2x,
        n_groups=6,
        k_test=2,
        purge_sessions=purge_cfg,
        selector="hac_t",
        hac_max_lag=5,
    )
    if computed_cpcv_pbo_sharpe is not None and computed_cpcv_pbo_hac is not None:
        computed_cpcv_pbo: float | None = max(computed_cpcv_pbo_sharpe, computed_cpcv_pbo_hac)
    else:
        computed_cpcv_pbo = computed_cpcv_pbo_sharpe
    effective_pbo = max(pbo_val, computed_cpcv_pbo) if computed_cpcv_pbo is not None else pbo_val

    # Empirical MinTRL evaluated on exact requested target_sharpe, unrounded moments & unclipped rho < +0.99
    eff_sessions_checked = int(hygiene["effective_sessions_checked"])
    t_sr_raw = audit_input.target_sharpe_for_min_trl
    emp_target_sr = float(t_sr_raw) if (_is_finite_real(t_sr_raw) and float(t_sr_raw) > 0.0) else 1.50
    empirical_min_trl_2x: int | None
    if best_rho_raw < _MAX_STATIONARY_RHO:
        rho_for_min_trl = max(-_MAX_STATIONARY_RHO + 1e-6, best_rho_raw)
        try:
            empirical_min_trl_2x = compute_min_trl_sessions(
                target_sharpe_ann=emp_target_sr,
                skew=best_skew_raw,
                excess_kurtosis=best_ex_kurt_raw,
                ar1_rho=rho_for_min_trl,
            )
            req_emp_min_trl = max(int(hygiene["min_trl_sessions_required"]), empirical_min_trl_2x)
            meets_empirical_min_trl = eff_sessions_checked >= req_emp_min_trl
        except ValueError:
            empirical_min_trl_2x = None
            meets_empirical_min_trl = False
    else:
        empirical_min_trl_2x = None
        meets_empirical_min_trl = False

    # Enforce exact predeclared comparison roster (no omitted strategies, no undeclared controls!)
    decl_active_ids: list[str] = []
    if isinstance(audit_input.active_strategy_ids, (tuple, list, set, frozenset)):
        decl_active_ids = [s for s in audit_input.active_strategy_ids if isinstance(s, str)]
    active_panel_complete_ok = (len(decl_active_ids) > 0) and (set(names) == set(decl_active_ids))

    dated_sessions_ok = canonical_session_dates is not None
    mae_calibration_dated_ok = (len(validated_mae_samples) == 0) or (canonical_mae_dates is not None)
    trial_dispersion_bound_ok = (trials_n == 1) or (ct_sr_sd_clean is not None)
    cpcv_comparison_ok = (trials_n == 1) or (computed_cpcv_pbo is not None)
    roundtrips_bound_ok = rt_supplied and (rt_by_strat[best_sid] >= 1)
    issued_stop_within_audit_ok = audit_stop_valid and (
        0.0 < float(last_pid_state["next_stop_pct"]) <= float(audit_input.max_stop_pct)
    )

    is_prewall_horizon = bool(horizon_spec is not None and horizon_spec.prewall_locked)
    evaluation_mode = "PREWALL_DISCOVERY_PROMOTION_GATE" if is_prewall_horizon else "FORWARD_SHADOW_OR_PARITY_ONLY"

    promotion_eligible = bool(
        is_prewall_horizon
        and hygiene["passed"]
        and hygiene["meets_min_trl_power"]
        and meets_empirical_min_trl
        and dated_sessions_ok
        and active_panel_complete_ok
        and roundtrips_bound_ok
        and issued_stop_within_audit_ok
        and trials_n >= NEXT_TRIAL_ID
        and trial_dispersion_bound_ok
        and cpcv_comparison_ok
        and mae_calibration_dated_ok
        and spa_mcs["spa_passes_alpha"]
        and best_sid in spa_mcs["mcs_surviving_strategies"]
        and best_t_raw >= req_hlz_t_raw
        and not per_strategy[best_sid]["quarantined_adr0064"]
        and not per_strategy[best_sid]["bankrupt_1x"]
        and not per_strategy[best_sid]["bankrupt_2x"]
        and per_strategy[best_sid]["survives_1x_cost"]
        and per_strategy[best_sid]["survives_2x_cost"]
        and all_strategies_cost_ok
        and dsr_in >= 0.95
        and computed_dsr_2x >= 0.95
        and effective_pbo < 0.25
    )

    input_evidence_digest = hashlib.sha256(
        json.dumps(
            _json_safe_evidence(
                {
                    "audit_input": asdict(audit_input),
                    "cumulative_trials_n": trials_n,
                    "strategy_clusters": {sid: strategy_clusters.get(sid, "UNCLUSTERED") for sid in names},
                    "sample_start_date": obs_start_date,
                    "sample_max_date": obs_max_date,
                    "session_dates": canonical_session_dates,
                    "observed_sessions": obs_t,
                    "declared_sessions": audit_input.distinct_sessions,
                    "realized_vol_ann_pct": rv_clean,
                    "cross_trial_sr_std": ct_sr_sd_clean,
                    "risk_unit_bps": r_unit_bps,
                    "roundtrip_notional_fraction": rt_notional_frac,
                    "seed": seed_i,
                    "n_bootstrap": int(n_bootstrap),
                    "alpha_spa_mcs": float(alpha_spa_mcs),
                    "raw_cpcv_pbo": pbo_val,
                    "raw_dsr_probability": dsr_in,
                    "closed_roundtrips_by_strategy": None if not rt_supplied else rt_by_strat,
                    "closed_roundtrip_sessions_by_strategy": rt_sessions_by_strat,
                    "daily_roundtrips_by_strategy": daily_rt_clean,
                    "realized_mae_atr_samples": [[a, b] for a, b in validated_mae_samples],
                    "mae_atr_dates": canonical_mae_dates,
                    "returns_1x": raw_1x_clean,
                    "returns_2x": raw_2x_clean,
                }
            ),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    payload: dict[str, Any] = {
        "schema": PLAYBACK_MANIFEST_SCHEMA,
        "adr": "ADR-0030",
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "evaluation_mode": evaluation_mode,
        "promotion_track": "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "standalone_alpha_claim": False,
        "hlz_trial_consumed": False,
        "completed_ledger_trials_n": CURRENT_LEDGER_TRIALS_N,
        "evaluated_trials_n": trials_n,
        "next_trial_id": NEXT_TRIAL_ID,
        "next_hlz_t_star": NEXT_HLZ_T_STAR,
        "required_hlz_t_hurdle": req_hlz_t,
        "empirical_min_trl_2x": empirical_min_trl_2x,
        "risk_unit_bps": r_unit_bps,
        "roundtrip_notional_fraction": rt_notional_frac,
        "bootstrap_seed": seed_i,
        "all_strategies_cost_ok": all_strategies_cost_ok,
        "promotion_eligible": promotion_eligible,
        "observed_sample_span": {
            "start_date": obs_start_date,
            "max_date": obs_max_date,
            "observed_sessions": obs_t,
            "declared_sessions": _json_safe_evidence(audit_input.distinct_sessions),
            "dated_axis_supplied": session_dates is not None,
            "mae_dates_supplied": canonical_mae_dates is not None,
            "input_evidence_sha256": input_evidence_digest,
        },
        "horizon_spec": asdict(horizon_spec) if horizon_spec is not None else None,
        "low_quality_practice_audit": hygiene,
        "per_strategy_playback": per_strategy,
        "hansen_spa_and_mcs_2x": spa_mcs,
        "conformal_pid_stop_state": last_pid_state,
        "fixed_share_hedge_weights": hedge_weights,
        "hierarchical_eb_merit_pool": hierarchical_eb,
        "asymmetric_vol_crash_brake": vol_brake,
        "cpcv_pbo": round(pbo_val, 4),
        "computed_cpcv_pbo_2x": None if computed_cpcv_pbo is None else round(computed_cpcv_pbo, 4),
        "computed_cpcv_pbo_2x_sharpe": (
            None if computed_cpcv_pbo_sharpe is None else round(computed_cpcv_pbo_sharpe, 4)
        ),
        "computed_cpcv_pbo_2x_hac": None if computed_cpcv_pbo_hac is None else round(computed_cpcv_pbo_hac, 4),
        "effective_cpcv_pbo": round(effective_pbo, 4),
        "dsr_probability": round(dsr_in, 4),
        "computed_dsr_2x": round(computed_dsr_2x, 4),
        "computed_dsr_2x_blp2014_iid": round(computed_dsr_2x_blp2014_iid, 4),
        "computed_dsr_2x_adia19_null_var": (
            None if computed_dsr_2x_adia19_null is None else round(computed_dsr_2x_adia19_null, 4)
        ),
        "dsr_variance_estimator": "AR1_ADJUSTED_OBSERVED_SR_PSD",
        "cross_trial_sr_std_input": None if ct_sr_sd_clean is None else round(ct_sr_sd_clean, 6),
        "dsr_trial_sr_std_source": (
            "EXPLICIT_CROSS_TRIAL_LEDGER" if ct_sr_sd_clean is not None else "NULL_H0_SAMPLING_BOUND"
        ),
    }
    safe_payload = _json_safe_evidence(payload)
    digest = hashlib.sha256(
        json.dumps(safe_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    safe_payload["manifest_sha256"] = digest
    return safe_payload


def render_playback_manifest_markdown(manifest: Mapping[str, Any]) -> str:
    """Render dimensionless Markdown playback summary for audit & cockpit inspection."""
    audit = manifest["low_quality_practice_audit"]
    spa = manifest["hansen_spa_and_mcs_2x"]
    hz = manifest.get("horizon_spec") or {}
    hz_id = hz.get("horizon_id", "UNKNOWN")
    hz_span = f"{hz.get('start_date', '?')} .. {hz.get('end_date', '?')}"
    obs = manifest.get("observed_sample_span") or {}
    obs_start = obs.get("start_date", "?")
    obs_max = obs.get("max_date", "?")
    obs_cnt = obs.get("observed_sessions", "?")
    obs_span = f"{obs_start} .. {obs_max} ({obs_cnt} sessions)"
    audit_status = "PASS" if audit["passed"] else "FAIL"
    n_viol = len(audit["violations"])
    min_trl = audit["min_trl_sessions_required"]
    mcs_list = ", ".join(spa["mcs_surviving_strategies"])
    pid_stop = manifest["conformal_pid_stop_state"]["next_stop_pct"]
    vol_brk = manifest["asymmetric_vol_crash_brake"]
    lines = [
        f"# Advanced Backtest Playback Performance Manifest (`{manifest['schema']}`)",
        "",
        f"- **Horizon Spec:** `{hz_id}` (`{hz_span}`) | **Mode:** `{manifest.get('evaluation_mode', 'PREWALL')}`",
        f"- **Observed Sample Span:** `{obs_span}`",
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
