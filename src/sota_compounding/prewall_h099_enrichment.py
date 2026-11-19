"""Deterministic Pre-Wall `H099` `DF10`/`DF6`/SEC-Acceptance & Native `ts_recv_ns` Enrichment Contract (`ADR-0020`).

Closes `T3` `#377` / Lesson `94c3e0cf6bbf9fde` (`published-orb` event slices cannot supply prior-20-session
same-clock `DF10` volume baselines or dated `DF6` FINRA RegSHO Threshold List membership, and `11` SEC `EFFECT`
records crossed past the `2026-10-15` ET wall). Enforces fail-closed, zero-fabrication point-in-time gates:
1. **`DF10` Same-Clock 20-Session Intraday Baseline (`compute_df10_same_clock_baseline`):** Requires `>= 20`
   strictly prior `XNYS` session dates (`d < target_date`, `target_date <= 2026-10-15`) at the same intraday
   clock bucket (`09:30..15:59 ET`); never fabricates baselines from sparse `< 20`-session event slices.
2. **`DF6` Dated FINRA RegSHO Threshold List (`validate_df6_regsho_threshold_snapshot`):** Requires dated daily
   threshold membership snapshots (`trade_date <= 2026-10-15`, `published_ts_utc <= decision_ts_utc`); never
   substitutes daily short-volume ratio (`CNMSshvol`) for RegSHO Rule 203(c)(6) threshold membership.
3. **Strict SEC EDGAR `accepted_at_utc` Pre-Wall Cutoff (`filter_sec_filings_pit_prewall`):** Purges and counts
   any filing lacking a parseable `accepted_at_utc` or whose `America/New_York` date exceeds `2026-10-15` (or
   exceeds `decision_ts_utc`), eliminating post-wall `EFFECT` / `S-3` leakage.
4. **Unified `H099` Lake Readiness Gate (`audit_h099_prewall_lake_readiness`):** Composes `ADR-0018` `ts_recv_ns`
   with `DF10`, `DF6`, and SEC acceptance-time audits into an immutable SHA-256 readiness receipt.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from sota_compounding.bitemporal_sidecar import (
    CONSERVATIVE_SIP_LATENCY_BOUND_MS,
    BitemporalReceiptSidecarSpec,
)
from sota_compounding.lifecycle import STUDY_WALL_DATE
from sota_compounding.symbol_ranking_v310 import _to_ny_datetime

H099_ENRICHMENT_SCHEMA = "sota-prewall-h099-enrichment.v1"
DF10_MIN_PRIOR_XNYS_SESSIONS: int = 20
DF10_MIN_STD_FLOOR_RATIO: float = 1e-6


@dataclass(frozen=True, slots=True)
class Df10SameClockBaselineResult:
    symbol: str
    target_date: str
    clock_bucket_et: str
    prior_sessions_count: int
    mean_20d_vol: float
    std_20d_vol: float
    current_cum_vol: float
    rvol_tod: float
    vol_z_tod: float
    approved: bool
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Df6RegshoPitResult:
    symbol: str
    trade_date: str
    snapshot_present: bool
    on_threshold_list: bool
    approved: bool
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SecPrewallFilterResult:
    retained_filings: tuple[dict[str, Any], ...]
    purged_postwall_count: int
    purged_missing_ts_count: int
    purged_future_decision_count: int
    clean_prewall: bool


@dataclass(frozen=True, slots=True)
class H099PrewallReadinessReceipt:
    schema_version: str
    hypothesis_id: str
    spec_sha256: str
    study_wall_date: str
    readiness_tier: str
    standalone_live_promotion_eligible: bool
    df10_ready: bool
    df6_ready: bool
    sec_cutoff_clean: bool
    has_native_ts_recv_ns: bool
    missing_prerequisites: tuple[str, ...]
    receipt_sha256: str


def _parse_iso_utc(raw: Any) -> dt.datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(dt.UTC)


def compute_df10_same_clock_baseline(
    *,
    symbol: str,
    target_date: dt.date,
    clock_bucket_et: str,
    current_cum_vol: float,
    prior_same_clock_volumes: dict[dt.date, float],
) -> Df10SameClockBaselineResult:
    """Compute strictly causal 20-session same-clock `RVOL_tod` & `vol_z_tod` (zero look-ahead)."""
    violations: list[str] = []
    sym = str(symbol or "").strip().upper()
    if not sym:
        violations.append("EMPTY_SYMBOL")
    if target_date > STUDY_WALL_DATE:
        violations.append(f"PREWALL_CUTOFF_BREACH:{target_date.isoformat()}>{STUDY_WALL_DATE.isoformat()}")
    if (
        isinstance(current_cum_vol, bool)
        or not isinstance(current_cum_vol, (int, float))
        or not math.isfinite(float(current_cum_vol))
        or float(current_cum_vol) < 0.0
    ):
        violations.append(f"INVALID_CURRENT_CUM_VOL:{current_cum_vol!r}")
        cur_v = 0.0
    else:
        cur_v = float(current_cum_vol)

    # Reject any same-day or future date leakage and keep strictly prior XNYS sessions
    future_leak_dates = [d for d in prior_same_clock_volumes if d >= target_date]
    if future_leak_dates:
        violations.append(f"DF10_LOOKAHEAD_DATE_LEAKAGE:{len(future_leak_dates)}")

    valid_prior_dates = sorted(d for d in prior_same_clock_volumes if d < target_date)
    window_dates = valid_prior_dates[-DF10_MIN_PRIOR_XNYS_SESSIONS:]
    vols: list[float] = []
    for d in window_dates:
        v = prior_same_clock_volumes[d]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0.0:
            violations.append(f"INVALID_PRIOR_VOL:{d.isoformat()}")
            continue
        vols.append(float(v))

    if len(vols) < DF10_MIN_PRIOR_XNYS_SESSIONS:
        violations.append(f"DF10_INSUFFICIENT_PRIOR_SESSIONS:{len(vols)}<{DF10_MIN_PRIOR_XNYS_SESSIONS}")

    if len(vols) >= 2:
        mean_v = sum(vols) / len(vols)
        var_v = sum((x - mean_v) ** 2 for x in vols) / (len(vols) - 1)
        std_v = math.sqrt(max(var_v, 0.0))
    else:
        mean_v = 0.0
        std_v = 0.0

    if mean_v <= 0.0:
        if len(vols) >= DF10_MIN_PRIOR_XNYS_SESSIONS:
            violations.append("DF10_ZERO_MEAN_BASELINE")
        rvol = 0.0
        vol_z = 0.0
    else:
        rvol = cur_v / mean_v
        eff_std = max(std_v, mean_v * DF10_MIN_STD_FLOOR_RATIO)
        vol_z = (cur_v - mean_v) / eff_std

    return Df10SameClockBaselineResult(
        symbol=sym,
        target_date=target_date.isoformat(),
        clock_bucket_et=str(clock_bucket_et or "").strip(),
        prior_sessions_count=len(vols),
        mean_20d_vol=round(mean_v, 4),
        std_20d_vol=round(std_v, 4),
        current_cum_vol=round(cur_v, 4),
        rvol_tod=round(rvol, 4),
        vol_z_tod=round(vol_z, 4),
        approved=len(violations) == 0,
        violations=tuple(violations),
    )


def validate_df6_regsho_threshold_snapshot(
    *,
    symbol: str,
    trade_date: dt.date,
    decision_ts_utc: dt.datetime,
    dated_threshold_snapshots: dict[dt.date, dict[str, Any]],
) -> Df6RegshoPitResult:
    """Fail-closed point-in-time FINRA RegSHO Rule 203(c)(6) Threshold List lookup."""
    violations: list[str] = []
    sym = str(symbol or "").strip().upper()
    if not sym:
        violations.append("EMPTY_SYMBOL")
    if trade_date > STUDY_WALL_DATE:
        violations.append(f"PREWALL_CUTOFF_BREACH:{trade_date.isoformat()}>{STUDY_WALL_DATE.isoformat()}")

    dec_utc = decision_ts_utc.astimezone(dt.UTC) if decision_ts_utc.tzinfo else decision_ts_utc.replace(tzinfo=dt.UTC)
    snap = dated_threshold_snapshots.get(trade_date)
    if not isinstance(snap, dict):
        violations.append(f"DF6_MISSING_DATED_THRESHOLD_SNAPSHOT:{trade_date.isoformat()}")
        return Df6RegshoPitResult(
            symbol=sym,
            trade_date=trade_date.isoformat(),
            snapshot_present=False,
            on_threshold_list=False,
            approved=False,
            violations=tuple(violations),
        )

    pub_dt = _parse_iso_utc(snap.get("published_ts_utc"))
    if pub_dt is None:
        violations.append("DF6_MISSING_PUBLISHED_TS_UTC")
    elif pub_dt > dec_utc:
        violations.append("DF6_SNAPSHOT_PUBLISHED_AFTER_DECISION")

    members_raw = snap.get("threshold_symbols")
    if not isinstance(members_raw, (list, tuple, set, frozenset)):
        violations.append("DF6_INVALID_THRESHOLD_SYMBOLS_SCHEMA")
        on_list = False
    else:
        members = {str(x).strip().upper() for x in members_raw if str(x).strip()}
        on_list = sym in members

    return Df6RegshoPitResult(
        symbol=sym,
        trade_date=trade_date.isoformat(),
        snapshot_present=True,
        on_threshold_list=on_list,
        approved=len(violations) == 0,
        violations=tuple(violations),
    )


def filter_sec_filings_pit_prewall(
    filings: list[dict[str, Any]],
    *,
    decision_ts_utc: dt.datetime | None = None,
) -> SecPrewallFilterResult:
    """Enforce strict `accepted_at_utc <= 2026-10-15` ET wall & decision causality (purges post-wall `EFFECT` leaks)."""
    dec_utc = (
        (decision_ts_utc.astimezone(dt.UTC) if decision_ts_utc.tzinfo else decision_ts_utc.replace(tzinfo=dt.UTC))
        if decision_ts_utc is not None
        else None
    )
    retained: list[dict[str, Any]] = []
    purged_postwall = 0
    purged_missing = 0
    purged_future_dec = 0

    for rec in filings:
        if not isinstance(rec, dict):
            purged_missing += 1
            continue
        acc_dt = _parse_iso_utc(rec.get("accepted_at_utc") or rec.get("acceptanceDateTime"))
        if acc_dt is None:
            purged_missing += 1
            continue
        ny_date = _to_ny_datetime(acc_dt).date()
        if ny_date > STUDY_WALL_DATE:
            purged_postwall += 1
            continue
        if dec_utc is not None and acc_dt > dec_utc:
            purged_future_dec += 1
            continue
        retained.append(dict(rec))

    return SecPrewallFilterResult(
        retained_filings=tuple(retained),
        purged_postwall_count=purged_postwall,
        purged_missing_ts_count=purged_missing,
        purged_future_decision_count=purged_future_dec,
        clean_prewall=(purged_postwall == 0 and purged_missing == 0),
    )


def audit_h099_prewall_lake_readiness(
    *,
    bitemporal_spec: BitemporalReceiptSidecarSpec,
    df10_covered_symbol_days: int,
    df10_total_symbol_days: int,
    df6_dated_snapshots_available: bool,
    sec_purged_postwall_count: int,
    hypothesis_id: str = "H099_BJZZ_DARK_SWEEP_15S_COLLAR_ASYMMETRIC_BREAKEVEN_CPCV",
    spec_sha256: str = "42cc7cfe05bb3200ad8fba757951be6b0ea8c4b8eceb6bc34b2b823d69525119",
) -> H099PrewallReadinessReceipt:
    """Emit deterministic SHA-256 readiness receipt composing ADR-0018 + DF10 + DF6 + SEC acceptance wall."""
    missing: list[str] = []
    if not bitemporal_spec.approved:
        missing.append(f"BITEMPORAL_SIDECAR_NOT_APPROVED:{','.join(bitemporal_spec.violations)}")

    df10_ready = df10_total_symbol_days > 0 and df10_covered_symbol_days > 0
    if not df10_ready:
        missing.append(f"DF10_MISSING_20D_SAME_CLOCK_BASELINES({df10_covered_symbol_days}/{df10_total_symbol_days})")

    df6_ready = bool(df6_dated_snapshots_available)
    if not df6_ready:
        missing.append("DF6_MISSING_DATED_REGSHO_THRESHOLD_SNAPSHOTS")

    sec_clean = int(sec_purged_postwall_count) == 0
    if not sec_clean:
        missing.append(f"SEC_UNFILTERED_POSTWALL_RECORDS:{int(sec_purged_postwall_count)}")

    if missing:
        tier = "BLOCKED_MISSING_PIT_PREREQUISITES"
        live_eligible = False
    elif bitemporal_spec.has_native_ts_recv_ns and bitemporal_spec.standalone_live_promotion_eligible:
        tier = "E4_NATIVE_H099_LIVE_PROMOTION_ELIGIBLE"
        live_eligible = True
    else:
        tier = "E2_BOUNDED_500MS_H099_EXPLORATORY_READY"
        live_eligible = False

    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": H099_ENRICHMENT_SCHEMA,
                "hypothesis_id": hypothesis_id,
                "spec_sha256": spec_sha256,
                "study_wall_date": STUDY_WALL_DATE.isoformat(),
                "bitemporal_sha256": bitemporal_spec.sidecar_sha256,
                "has_native_ts_recv_ns": bitemporal_spec.has_native_ts_recv_ns,
                "latency_bound_ms": bitemporal_spec.latency_bound_ms or CONSERVATIVE_SIP_LATENCY_BOUND_MS,
                "df10_covered": int(df10_covered_symbol_days),
                "df10_total": int(df10_total_symbol_days),
                "df6_ready": df6_ready,
                "sec_purged_postwall_count": int(sec_purged_postwall_count),
                "tier": tier,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return H099PrewallReadinessReceipt(
        schema_version=H099_ENRICHMENT_SCHEMA,
        hypothesis_id=hypothesis_id,
        spec_sha256=spec_sha256,
        study_wall_date=STUDY_WALL_DATE.isoformat(),
        readiness_tier=tier,
        standalone_live_promotion_eligible=live_eligible,
        df10_ready=df10_ready,
        df6_ready=df6_ready,
        sec_cutoff_clean=sec_clean,
        has_native_ts_recv_ns=bitemporal_spec.has_native_ts_recv_ns,
        missing_prerequisites=tuple(missing),
        receipt_sha256=digest,
    )
