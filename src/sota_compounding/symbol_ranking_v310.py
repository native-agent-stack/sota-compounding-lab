"""Finalized `v3.10.0` 12-Feed Symbol Ranking (`DF1..DF12`) & Bitemporal Strategy Handoff (`ADR-0016`)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any

from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE
from sota_compounding.ranking_and_datafeeds import CANONICAL_7_PIT_FEEDS

SYMBOL_RANKING_V310_SCHEMA = "sota-symbol-ranking-datafeeds.v3.10"
DF10_TO_DF12_FEEDS: tuple[str, ...] = (
    "DF10_OWN_HISTORY_TOD_ABNORMAL_RATE_AND_UCURVE",
    "DF11_BITEMPORAL_DATABENTO_PIT_AND_H096_COLLAR",
    "DF12_QLIB_RANK_IC_AND_KAPPA40_POSTWALL_EB_FEEDBACK",
)
CANONICAL_12_RANKING_FEEDS: tuple[str, ...] = (
    *CANONICAL_7_PIT_FEEDS,
    "DF8_finra_rule4560_settled_short_interest_t8",
    "DF9_nyse_ucurve_and_market_cap_liquidity_tier",
    *DF10_TO_DF12_FEEDS,
)
POSTWALL_EB_KAPPA = 40
MIN_RVOL_TOD = 2.20
MIN_VOL_Z_TOD = 2.30
MIN_DRY_POWDER_SEC = 15.0
MAX_COLLAR_BPS = 10.0
WAVE2_COOLDOWN_SEC = 900.0


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
    """Validate DF1..DF12 bitemporal ordering, RVOL_tod size-bias purge & ADR-0022 bounds."""
    violations: list[str] = []
    if dt.date.fromisoformat(study_date) > STUDY_WALL_DATE:
        violations.append(f"PREWALL_VIOLATION: {study_date} > {STUDY_WALL_DATE.isoformat()}")
    if not (ts_event_ns <= ts_recv_ns <= decision_ns):
        violations.append("BITEMPORAL_PIT_VIOLATION: require ts_event_ns <= ts_recv_ns <= decision_ns")
    if rvol_tod < MIN_RVOL_TOD:
        violations.append(f"RVOL_TOD_BELOW_MIN: {rvol_tod:.2f} < {MIN_RVOL_TOD:.2f}")
    if vol_z_tod < MIN_VOL_Z_TOD:
        violations.append(f"VOL_Z_TOD_BELOW_MIN: {vol_z_tod:.2f} < {MIN_VOL_Z_TOD:.2f}")
    if ofi < 0.25:
        violations.append(f"OFI_BELOW_MIN: {ofi:.3f} < 0.25")
    if dry_powder_sec < MIN_DRY_POWDER_SEC:
        violations.append(f"DRY_POWDER_BELOW_15S: {dry_powder_sec:.1f} < {MIN_DRY_POWDER_SEC:.1f}")
    if not (0.0 < spread_collar_bps <= MAX_COLLAR_BPS):
        violations.append(f"COLLAR_BPS_OUT_OF_RANGE: {spread_collar_bps:.2f}")
    if not (0.0 < hard_stop_pct <= ADR0022_MAX_STOP_PCT):
        violations.append(f"ADR0022_STOP_EXCEEDED: {hard_stop_pct:.2f} > {ADR0022_MAX_STOP_PCT:.2f}")
    if is_wave2_reentry and cooldown_elapsed_sec < WAVE2_COOLDOWN_SEC:
        violations.append(f"WAVE2_COOLDOWN_ACTIVE: {cooldown_elapsed_sec:.1f} < {WAVE2_COOLDOWN_SEC:.1f}")

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
