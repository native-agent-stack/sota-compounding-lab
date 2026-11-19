"""Deterministic Bitemporal `ts_recv_ns` Availability Sidecar Contract for Pre-Wall Lakes (`ADR-0018`).

Closes `T3` `#363` (`H099` honest bitemporal block) without ever fabricating native feed receipt timestamps:
1. **Strict `E4` Native Receipt Mode (`NATIVE_TS_RECV_NS_REQUIRED`):** Required for Stage-3/4 autonomous alpha
   promotion (`standalone_live_promotion_eligible = True`); fails closed whenever native `ts_recv_ns` is absent.
2. **Bounded-Latency `E2` Exploratory Sidecar Mode (`CONSERVATIVE_SIP_LATENCY_BOUND_MS = 500.0`):** When an
   existing pre-wall (`<= 2026-10-15`) lake partition carries verified event timestamps `t_ns` without native
   `ts_recv_ns`, derives an explicitly labeled worst-case conservative propagation ceiling
   `ts_recv_bound_ns = t_ns + int(latency_bound_ms * 1_000_000)` (`evidence_cap = "E2_BOUNDED_LATENCY_PROJECTION"`,
   `standalone_live_promotion_eligible = False`) with immutable SHA-256 sidecar provenance so pre-registered
   exploratory CPCV / `1x`/`2x` stress studies (`H099`) can execute honestly while failing closed on any
   `t_ns > 2026-10-15` ET or non-finite / non-positive latency bound.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from sota_compounding.lifecycle import STUDY_WALL_DATE
from sota_compounding.symbol_ranking_v310 import _to_ny_datetime

BITEMPORAL_SIDECAR_SCHEMA = "sota-bitemporal-receipt-sidecar.v1"
CONSERVATIVE_SIP_LATENCY_BOUND_MS: float = 500.0
MIN_CONSERVATIVE_LATENCY_BOUND_MS: float = 50.0
MAX_CONSERVATIVE_LATENCY_BOUND_MS: float = 5_000.0


@dataclass(frozen=True, slots=True)
class BitemporalReceiptSidecarSpec:
    partition_id: str
    row_count: int
    max_event_ts_ns: int
    has_native_ts_recv_ns: bool
    latency_bound_ms: float
    evidence_cap: str
    standalone_live_promotion_eligible: bool
    prewall_wall_date: str
    approved: bool
    violations: tuple[str, ...]
    sidecar_sha256: str


def derive_conservative_sip_receipt_ns(
    ts_event_ns: int,
    *,
    latency_bound_ms: float = CONSERVATIVE_SIP_LATENCY_BOUND_MS,
) -> int:
    """Compute conservative worst-case SIP receipt bound `ts_recv_bound_ns = ts_event_ns + bound_ns`."""
    if isinstance(ts_event_ns, bool) or not isinstance(ts_event_ns, int) or ts_event_ns <= 0:
        raise ValueError(f"ts_event_ns must be a positive integer nanosecond timestamp, got {ts_event_ns!r}")
    if (
        isinstance(latency_bound_ms, bool)
        or not isinstance(latency_bound_ms, (int, float))
        or not math.isfinite(float(latency_bound_ms))
        or not (MIN_CONSERVATIVE_LATENCY_BOUND_MS <= float(latency_bound_ms) <= MAX_CONSERVATIVE_LATENCY_BOUND_MS)
    ):
        raise ValueError(
            "latency_bound_ms must be finite in "
            f"[{MIN_CONSERVATIVE_LATENCY_BOUND_MS}, {MAX_CONSERVATIVE_LATENCY_BOUND_MS}], "
            f"got {latency_bound_ms!r}"
        )
    bound_ns = int(round(float(latency_bound_ms) * 1_000_000.0))
    return ts_event_ns + bound_ns


def validate_prewall_bitemporal_partition_contract(
    *,
    partition_id: str,
    columns: list[str] | tuple[str, ...],
    row_count: int,
    max_event_ts_ns: int,
    allow_bounded_e2_projection: bool = False,
    latency_bound_ms: float = CONSERVATIVE_SIP_LATENCY_BOUND_MS,
) -> BitemporalReceiptSidecarSpec:
    """Fail-closed audit of a pre-wall Parquet partition for native `ts_recv_ns` vs bounded E2 sidecar eligibility."""
    violations: list[str] = []
    col_set = {str(c).strip() for c in columns}
    has_event_col = "t_ns" in col_set or "ts_event_ns" in col_set
    has_native_recv = "ts_recv_ns" in col_set

    if not partition_id or not str(partition_id).strip():
        violations.append("EMPTY_PARTITION_ID")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        violations.append(f"NONPOSITIVE_ROW_COUNT:{row_count!r}")
    if not has_event_col:
        violations.append("MISSING_EVENT_TIMESTAMP_COLUMN(t_ns|ts_event_ns)")

    # Verify strict pre-wall <= 2026-10-15 ET event-time cutoff
    if isinstance(max_event_ts_ns, bool) or not isinstance(max_event_ts_ns, int) or max_event_ts_ns <= 0:
        violations.append(f"INVALID_MAX_EVENT_TS_NS:{max_event_ts_ns!r}")
    else:
        try:
            ev_dt_ny = _to_ny_datetime(dt.datetime.fromtimestamp(max_event_ts_ns // 1_000_000_000, tz=dt.UTC))
            if ev_dt_ny.date() > STUDY_WALL_DATE:
                violations.append(f"PREWALL_CUTOFF_BREACH:{ev_dt_ny.date().isoformat()}>{STUDY_WALL_DATE.isoformat()}")
        except (OverflowError, OSError, ValueError):
            violations.append("MAX_EVENT_TS_NS_OVERFLOW")

    if has_native_recv:
        evidence_cap = "E4_NATIVE_BITEMPORAL_RECEIPT"
        live_eligible = True
        eff_bound = 0.0
    else:
        if not allow_bounded_e2_projection:
            violations.append("MISSING_NATIVE_TS_RECV_NS_STRICT_MODE_BLOCKED")
            evidence_cap = "BLOCKED_MISSING_TS_RECV_NS"
            live_eligible = False
            eff_bound = 0.0
        else:
            if (
                isinstance(latency_bound_ms, bool)
                or not isinstance(latency_bound_ms, (int, float))
                or not math.isfinite(float(latency_bound_ms))
                or not (
                    MIN_CONSERVATIVE_LATENCY_BOUND_MS <= float(latency_bound_ms) <= MAX_CONSERVATIVE_LATENCY_BOUND_MS
                )
            ):
                violations.append(f"INVALID_CONSERVATIVE_LATENCY_BOUND_MS:{latency_bound_ms!r}")
                eff_bound = 0.0
            else:
                eff_bound = float(latency_bound_ms)
            evidence_cap = "E2_BOUNDED_LATENCY_PROJECTION"
            live_eligible = False

    approved = len(violations) == 0
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": BITEMPORAL_SIDECAR_SCHEMA,
                "partition_id": str(partition_id),
                "row_count": int(row_count) if isinstance(row_count, int) else 0,
                "max_event_ts_ns": int(max_event_ts_ns) if isinstance(max_event_ts_ns, int) else 0,
                "has_native_ts_recv_ns": has_native_recv,
                "latency_bound_ms": eff_bound,
                "evidence_cap": evidence_cap,
                "prewall_wall_date": STUDY_WALL_DATE.isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return BitemporalReceiptSidecarSpec(
        partition_id=str(partition_id),
        row_count=int(row_count) if isinstance(row_count, int) else 0,
        max_event_ts_ns=int(max_event_ts_ns) if isinstance(max_event_ts_ns, int) else 0,
        has_native_ts_recv_ns=has_native_recv,
        latency_bound_ms=eff_bound,
        evidence_cap=evidence_cap,
        standalone_live_promotion_eligible=live_eligible and approved,
        prewall_wall_date=STUDY_WALL_DATE.isoformat(),
        approved=approved,
        violations=tuple(violations),
        sidecar_sha256=digest,
    )


def build_h099_bitemporal_replay_contract(
    *,
    hypothesis_id: str = "H099_BJZZ_DARK_SWEEP_15S_COLLAR_ASYMMETRIC_BREAKEVEN_CPCV",
    spec_sha256: str = "42cc7cfe05bb3200ad8fba757951be6b0ea8c4b8eceb6bc34b2b823d69525119",
    partitions_audited: int = 29_500,
    has_native_ts_recv_ns: bool = False,
    latency_bound_ms: float = CONSERVATIVE_SIP_LATENCY_BOUND_MS,
) -> dict[str, Any]:
    """Emit deterministic H099 pre-wall replay contract declaring exact E2 bounded vs E4 native provenance."""
    mode = "E4_NATIVE_BITEMPORAL_RECEIPT" if has_native_ts_recv_ns else "E2_BOUNDED_LATENCY_PROJECTION"
    return {
        "schema_version": BITEMPORAL_SIDECAR_SCHEMA,
        "adr": "ADR-0018",
        "hypothesis_id": hypothesis_id,
        "spec_sha256": spec_sha256,
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "partitions_audited": int(partitions_audited),
        "has_native_ts_recv_ns": bool(has_native_ts_recv_ns),
        "conservative_sip_latency_bound_ms": 0.0 if has_native_ts_recv_ns else float(latency_bound_ms),
        "evidence_cap": mode,
        "standalone_live_promotion_eligible": bool(has_native_ts_recv_ns),
        "cpcv_splits": 15,
        "hlz_t_hurdle_99": 4.1949,
        "cost_stress_multipliers": [1.0, 2.0],
        "dsr_min": 0.95,
    }
