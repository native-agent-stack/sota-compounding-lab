"""TDD & Property Suite for ADR-0018 Bitemporal `ts_recv_ns` Pre-Wall Sidecar Contract."""

from __future__ import annotations

import datetime as dt
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sota_compounding.bitemporal_sidecar import (
    CONSERVATIVE_SIP_LATENCY_BOUND_MS,
    build_h099_bitemporal_replay_contract,
    derive_conservative_sip_receipt_ns,
    validate_prewall_bitemporal_partition_contract,
)


def _ns(year: int, month: int, day: int, hour: int = 15, minute: int = 0) -> int:
    return int(dt.datetime(year, month, day, hour, minute, tzinfo=dt.UTC).timestamp() * 1_000_000_000)


def test_strict_e4_blocks_missing_ts_recv_ns_while_bounded_e2_is_explicitly_capped() -> None:
    prewall_ns = _ns(2026, 10, 15, 19, 30)  # 15:30 ET on 2026-10-15 (inside wall)
    # 1. Strict mode (default) FAILS CLOSED when ts_recv_ns is absent (matching T3 #363)
    strict = validate_prewall_bitemporal_partition_contract(
        partition_id="published-orb/trades/2026-10-15.parquet",
        columns=["t_ns", "price", "size", "exchange"],
        row_count=125_000,
        max_event_ts_ns=prewall_ns,
        allow_bounded_e2_projection=False,
    )
    assert strict.approved is False
    assert strict.standalone_live_promotion_eligible is False
    assert "MISSING_NATIVE_TS_RECV_NS_STRICT_MODE_BLOCKED" in strict.violations

    # 2. Explicit bounded E2 projection (500ms worst-case SIP propagation ceiling) approves exploratory replay ONLY
    e2 = validate_prewall_bitemporal_partition_contract(
        partition_id="published-orb/trades/2026-10-15.parquet",
        columns=["t_ns", "price", "size", "exchange"],
        row_count=125_000,
        max_event_ts_ns=prewall_ns,
        allow_bounded_e2_projection=True,
        latency_bound_ms=CONSERVATIVE_SIP_LATENCY_BOUND_MS,
    )
    assert e2.approved is True
    assert e2.evidence_cap == "E2_BOUNDED_LATENCY_PROJECTION"
    assert e2.standalone_live_promotion_eligible is False
    assert e2.latency_bound_ms == 500.0
    assert len(e2.sidecar_sha256) == 64

    # 3. Native ts_recv_ns unlocks E4 promotion eligibility
    e4 = validate_prewall_bitemporal_partition_contract(
        partition_id="databento/trades/2026-10-15.parquet",
        columns=["ts_event_ns", "ts_recv_ns", "price", "size"],
        row_count=125_000,
        max_event_ts_ns=prewall_ns,
    )
    assert e4.approved is True
    assert e4.evidence_cap == "E4_NATIVE_BITEMPORAL_RECEIPT"
    assert e4.standalone_live_promotion_eligible is True


def test_postwall_event_ts_always_fails_closed_even_in_e2_mode() -> None:
    postwall_ns = _ns(2026, 10, 16, 14, 0)  # 10:00 ET on 2026-10-16 -> STRICTLY FORBIDDEN!
    bad = validate_prewall_bitemporal_partition_contract(
        partition_id="published-orb/trades/2026-10-16.parquet",
        columns=["t_ns", "ts_recv_ns"],
        row_count=100,
        max_event_ts_ns=postwall_ns,
        allow_bounded_e2_projection=True,
    )
    assert bad.approved is False
    assert bad.standalone_live_promotion_eligible is False
    assert any(v.startswith("PREWALL_CUTOFF_BREACH") for v in bad.violations)


@given(bound_ms=st.floats(min_value=50.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
def test_property_conservative_sip_receipt_strictly_after_event(bound_ms: float) -> None:
    ev = _ns(2026, 10, 15, 15, 0)
    recv = derive_conservative_sip_receipt_ns(ev, latency_bound_ms=bound_ms)
    assert recv > ev
    assert recv - ev == int(round(bound_ms * 1_000_000.0))


def test_nonfinite_latency_bound_rejected_and_h099_contract_valid() -> None:
    with pytest.raises(ValueError):
        derive_conservative_sip_receipt_ns(_ns(2026, 10, 15, 15, 0), latency_bound_ms=math.nan)
    c = build_h099_bitemporal_replay_contract(has_native_ts_recv_ns=False)
    assert c["evidence_cap"] == "E2_BOUNDED_LATENCY_PROJECTION"
    assert c["standalone_live_promotion_eligible"] is False
    assert c["hlz_t_hurdle_99"] == 4.1949
    assert c["study_wall_date"] == "2026-10-15"
