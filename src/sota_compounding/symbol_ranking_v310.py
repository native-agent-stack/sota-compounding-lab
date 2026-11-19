"""Finalized `v3.10.3` 12-Feed Symbol Ranking (`DF1..DF12`) & Fail-Closed Bitemporal Handoff (`ADR-0016`/`ADR-0017`)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE
from sota_compounding.ranking_and_datafeeds import CANONICAL_7_PIT_FEEDS

SYMBOL_RANKING_V310_SCHEMA = "sota-symbol-ranking-datafeeds.v3.10"
DF10_TO_DF12_FEEDS: tuple[str, ...] = (
    "DF10_OWN_HISTORY_TOD_ABNORMAL_RATE_AND_UCURVE",
    "DF11_BITEMPORAL_DATABENTO_PIT_AND_H096_COLLAR",
    "DF12_QLIB_RANK_IC_AND_KAPPA40_POSTWALL_EB_FEEDBACK",
)
DF10_TO_DF12_FEEDS_NORMALIZED: tuple[str, ...] = (
    "DF10_own_history_tod_abnormal_rate_and_ucurve",
    "DF11_bitemporal_databento_pit_and_h096_collar",
    "DF12_qlib_rank_ic_and_kappa40_postwall_eb_feedback",
)
CANONICAL_12_RANKING_FEEDS: tuple[str, ...] = (
    *CANONICAL_7_PIT_FEEDS,
    "DF8_finra_rule4560_settled_short_interest_t8",
    "DF9_nyse_ucurve_and_market_cap_liquidity_tier",
    *DF10_TO_DF12_FEEDS,
)
CANONICAL_12_RANKING_FEEDS_NORMALIZED: tuple[str, ...] = (
    *CANONICAL_7_PIT_FEEDS,
    "DF8_finra_rule4560_settled_short_interest_t8",
    "DF9_nyse_ucurve_and_market_cap_liquidity_tier",
    *DF10_TO_DF12_FEEDS_NORMALIZED,
)
POSTWALL_EB_KAPPA = 40
MIN_RVOL_TOD = 2.20
MIN_VOL_Z_TOD = 2.30
MIN_DRY_POWDER_SEC = 15.0
MAX_COLLAR_BPS = 10.0
MIN_HARD_STOP_PCT = 0.05  # 5.0 bps realistic tick/spread floor (ADR-0019)
WAVE2_COOLDOWN_SEC = 900.0
MIN_EPOCH_NS_2000 = 946_684_800_000_000_000  # 2000-01-01T00:00:00Z


def _to_ny_datetime(dt_utc: dt.datetime) -> dt.datetime:
    """Convert aware UTC datetime to America/New_York (cross-OS fallback when Windows lacks tzdata)."""
    try:
        return dt_utc.astimezone(ZoneInfo("America/New_York"))
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        y = dt_utc.year
        if y < 2007:
            apr1_wd = dt.date(y, 4, 1).weekday()
            dst_start = dt.datetime(y, 4, 1 + ((6 - apr1_wd) % 7), 7, 0, tzinfo=dt.UTC)
            oct31_wd = dt.date(y, 10, 31).weekday()
            dst_end = dt.datetime(y, 10, 31 - ((oct31_wd + 1) % 7), 6, 0, tzinfo=dt.UTC)
        else:
            mar1_wd = dt.date(y, 3, 1).weekday()
            dst_start = dt.datetime(y, 3, 1 + ((6 - mar1_wd) % 7) + 7, 7, 0, tzinfo=dt.UTC)
            nov1_wd = dt.date(y, 11, 1).weekday()
            dst_end = dt.datetime(y, 11, 1 + ((6 - nov1_wd) % 7), 6, 0, tzinfo=dt.UTC)
        is_edt = dst_start <= dt_utc < dst_end
        return dt_utc.astimezone(dt.timezone(dt.timedelta(hours=-4 if is_edt else -5), name="EDT" if is_edt else "EST"))


def _safe_finite(val: object) -> tuple[bool, float]:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False, float("nan")
    try:
        f = float(val)
    except OverflowError:
        return False, float("inf")
    return math.isfinite(f), f


def evaluate_df12_bitemporal_symbol_ranking_handoff(
    *,
    symbol: str,
    study_date: str,
    ts_event_ns: int,
    ts_recv_ns: int,
    decision_ns: int,
    rvol_tod: float,
    vol_z_tod: float,
    ofi: float,
    dry_powder_sec: float,
    spread_collar_bps: float,
    hard_stop_pct: float,
    is_wave2_reentry: bool = False,
    cooldown_elapsed_sec: float = 0.0,
) -> dict[str, Any]:
    """Fail-closed DF1..DF12 bitemporal ordering, IEEE-754 finite checks, decision_ns wall & ADR-0022 bounds."""
    violations: list[str] = []
    try:
        study_d = dt.date.fromisoformat(study_date)
    except ValueError:
        study_d = None
        violations.append(f"BAD_STUDY_DATE: {study_date!r}")
    if study_d is not None and study_d > STUDY_WALL_DATE:
        violations.append(f"PREWALL_VIOLATION: {study_date} > {STUDY_WALL_DATE.isoformat()}")

    ts_ok = all(
        isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in (ts_event_ns, ts_recv_ns, decision_ns)
    )
    if not ts_ok or not (ts_event_ns <= ts_recv_ns <= decision_ns):
        violations.append("BITEMPORAL_PIT_VIOLATION: require non-negative int ts_event_ns <= ts_recv_ns <= decision_ns")
    elif decision_ns >= MIN_EPOCH_NS_2000:
        try:
            # Exact integer nanosecond division + America/New_York session date (ADR-0019: BOATS 20:00-23:59 ET)
            dec_d = _to_ny_datetime(dt.datetime.fromtimestamp(decision_ns // 1_000_000_000, tz=dt.UTC)).date()
            if dec_d > STUDY_WALL_DATE:
                violations.append(f"PREWALL_DECISION_NS_VIOLATION: {dec_d.isoformat()} > {STUDY_WALL_DATE.isoformat()}")
            elif study_d is not None and dec_d != study_d:
                violations.append(f"DECISION_NS_DATE_MISMATCH: {dec_d.isoformat()} != {study_d.isoformat()}")
        except (OverflowError, OSError, ValueError):
            violations.append("DECISION_NS_OVERFLOW")

    ok_rv, f_rv = _safe_finite(rvol_tod)
    if not ok_rv or not (f_rv >= MIN_RVOL_TOD):
        violations.append(f"RVOL_TOD_BELOW_MIN: {rvol_tod!r} < {MIN_RVOL_TOD:.2f}")
    ok_vz, f_vz = _safe_finite(vol_z_tod)
    if not ok_vz or not (f_vz >= MIN_VOL_Z_TOD):
        violations.append(f"VOL_Z_TOD_BELOW_MIN: {vol_z_tod!r} < {MIN_VOL_Z_TOD:.2f}")
    ok_ofi, f_ofi = _safe_finite(ofi)
    if not ok_ofi or not (f_ofi >= 0.25):
        violations.append(f"OFI_BELOW_MIN: {ofi!r} < 0.25")
    ok_dp, f_dp = _safe_finite(dry_powder_sec)
    if not ok_dp or not (f_dp >= MIN_DRY_POWDER_SEC):
        violations.append(f"DRY_POWDER_BELOW_15S: {dry_powder_sec!r} < {MIN_DRY_POWDER_SEC:.1f}")
    ok_sc, f_sc = _safe_finite(spread_collar_bps)
    if not ok_sc or not (0.0 < f_sc <= MAX_COLLAR_BPS):
        violations.append(f"COLLAR_BPS_OUT_OF_RANGE: {spread_collar_bps!r}")
    ok_hs, f_hs = _safe_finite(hard_stop_pct)
    if not ok_hs or not (MIN_HARD_STOP_PCT <= f_hs <= ADR0022_MAX_STOP_PCT):
        violations.append(f"ADR0022_STOP_EXCEEDED: {hard_stop_pct!r}")
    ok_cd, f_cd = _safe_finite(cooldown_elapsed_sec)
    if is_wave2_reentry and (not ok_cd or not (f_cd >= WAVE2_COOLDOWN_SEC)):
        violations.append(f"WAVE2_COOLDOWN_ACTIVE: {cooldown_elapsed_sec!r} < {WAVE2_COOLDOWN_SEC:.1f}")

    digest = hashlib.sha256(
        json.dumps(
            {
                "symbol": symbol.upper(),
                "study_date": study_date,
                "ts_event_ns": ts_event_ns,
                "ts_recv_ns": ts_recv_ns,
                "decision_ns": decision_ns,
                "feeds": list(CANONICAL_12_RANKING_FEEDS),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SYMBOL_RANKING_V310_SCHEMA,
        "feeds_count": len(CANONICAL_12_RANKING_FEEDS),
        "eb_kappa": POSTWALL_EB_KAPPA,
        "approved": len(violations) == 0,
        "violations": violations,
        "handoff_sha256": digest,
    }
