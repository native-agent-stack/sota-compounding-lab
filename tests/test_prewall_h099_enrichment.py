"""Deterministic & Property TDD Suite for ADR-0020 Pre-Wall H099 DF10/DF6/SEC-Cutoff Enrichment."""

from __future__ import annotations

import datetime as dt

from hypothesis import given
from hypothesis import strategies as st

from sota_compounding.bitemporal_sidecar import validate_prewall_bitemporal_partition_contract
from sota_compounding.prewall_h099_enrichment import (
    DF10_MIN_PRIOR_XNYS_SESSIONS,
    audit_h099_prewall_lake_readiness,
    compute_df10_same_clock_baseline,
    filter_sec_filings_pit_prewall,
    validate_df6_regsho_threshold_snapshot,
)


def _prior_20_dates(anchor: dt.date) -> dict[dt.date, float]:
    out: dict[dt.date, float] = {}
    d = anchor - dt.timedelta(days=1)
    idx = 0
    while len(out) < DF10_MIN_PRIOR_XNYS_SESSIONS:
        if d.weekday() < 5:
            out[d] = 100_000.0 + (idx % 5) * 5_000.0
            idx += 1
        d -= dt.timedelta(days=1)
    return out


def test_df10_rejects_sparse_event_slices_and_computes_20d_baseline() -> None:
    target = dt.date(2026, 10, 15)
    # T3 #377 empirical failure mode: <=5 prior sessions MUST fail closed
    sparse = {dt.date(2026, 10, 14) - dt.timedelta(days=i): 100_000.0 for i in range(5)}
    bad = compute_df10_same_clock_baseline(
        symbol="NVDA",
        target_date=target,
        clock_bucket_et="09:45",
        current_cum_vol=300_000.0,
        prior_same_clock_volumes=sparse,
    )
    assert bad.approved is False
    assert any("DF10_INSUFFICIENT_PRIOR_SESSIONS" in v for v in bad.violations)

    # Full 20 prior sessions passes with positive RVOL_tod & vol_z_tod
    prior20 = _prior_20_dates(target)
    ok = compute_df10_same_clock_baseline(
        symbol="NVDA",
        target_date=target,
        clock_bucket_et="09:45",
        current_cum_vol=300_000.0,
        prior_same_clock_volumes=prior20,
    )
    assert ok.approved is True
    assert ok.prior_sessions_count == 20
    assert ok.rvol_tod >= 2.20
    assert ok.vol_z_tod >= 2.30

    # Same-day or post-wall date leakage fails closed
    leaked = dict(prior20)
    leaked[target] = 120_000.0
    leak_res = compute_df10_same_clock_baseline(
        symbol="NVDA",
        target_date=target,
        clock_bucket_et="09:45",
        current_cum_vol=300_000.0,
        prior_same_clock_volumes=leaked,
    )
    assert leak_res.approved is False
    assert any("DF10_LOOKAHEAD_DATE_LEAKAGE" in v for v in leak_res.violations)


def test_df6_regsho_threshold_snapshot_and_sec_postwall_effect_purge() -> None:
    td = dt.date(2026, 10, 15)
    dec_ts = dt.datetime(2026, 10, 15, 14, 0, tzinfo=dt.UTC)

    # Missing dated snapshot fails closed (T3 #377)
    miss = validate_df6_regsho_threshold_snapshot(
        symbol="XYZ",
        trade_date=td,
        decision_ts_utc=dec_ts,
        dated_threshold_snapshots={},
    )
    assert miss.approved is False
    assert any("DF6_MISSING_DATED_THRESHOLD_SNAPSHOT" in v for v in miss.violations)

    # Dated snapshot published before decision passes
    snaps = {
        td: {
            "published_ts_utc": "2026-10-15T04:30:00Z",
            "threshold_symbols": ["MEME", "XYZ"],
        }
    }
    hit = validate_df6_regsho_threshold_snapshot(
        symbol="XYZ",
        trade_date=td,
        decision_ts_utc=dec_ts,
        dated_threshold_snapshots=snaps,
    )
    assert hit.approved is True
    assert hit.on_threshold_list is True

    # SEC filter purges the 11 post-wall EFFECT records caught by T3 #377
    filings = [
        {"form": "S-3", "symbol": "AAA", "accepted_at_utc": "2026-10-15T13:30:00Z"},
        {"form": "EFFECT", "symbol": "BBB", "accepted_at_utc": "2026-10-16T14:00:00Z"},
        {"form": "EFFECT", "symbol": "CCC", "accepted_at_utc": ""},
    ]
    sec_res = filter_sec_filings_pit_prewall(filings, decision_ts_utc=dec_ts)
    assert len(sec_res.retained_filings) == 1
    assert sec_res.purged_postwall_count == 1
    assert sec_res.purged_missing_ts_count == 1
    assert sec_res.clean_prewall is False


def test_h099_readiness_tiers_blocked_e2_and_e4() -> None:
    ts_prewall_ns = int(dt.datetime(2026, 10, 15, 14, 45, tzinfo=dt.UTC).timestamp() * 1_000_000_000)

    e2_spec = validate_prewall_bitemporal_partition_contract(
        partition_id="published-orb/trades/2026-10-15",
        columns=["symbol", "t_ns", "p", "s"],
        row_count=53_063_777,
        max_event_ts_ns=ts_prewall_ns,
        allow_bounded_e2_projection=True,
    )
    # Current T3 empirical state -> BLOCKED_MISSING_PIT_PREREQUISITES
    blocked = audit_h099_prewall_lake_readiness(
        bitemporal_spec=e2_spec,
        df10_covered_symbol_days=0,
        df10_total_symbol_days=10_669,
        df6_dated_snapshots_available=False,
        sec_purged_postwall_count=11,
    )
    assert blocked.readiness_tier == "BLOCKED_MISSING_PIT_PREREQUISITES"
    assert blocked.standalone_live_promotion_eligible is False
    assert len(blocked.missing_prerequisites) == 3

    # Enriched with DF10 + DF6 + clean SEC under bounded E2 -> exploratory ready, NOT live eligible
    e2_ready = audit_h099_prewall_lake_readiness(
        bitemporal_spec=e2_spec,
        df10_covered_symbol_days=10_669,
        df10_total_symbol_days=10_669,
        df6_dated_snapshots_available=True,
        sec_purged_postwall_count=0,
    )
    assert e2_ready.readiness_tier == "E2_BOUNDED_500MS_H099_EXPLORATORY_READY"
    assert e2_ready.standalone_live_promotion_eligible is False

    # Native ts_recv_ns + DF10 + DF6 + clean SEC -> E4 live promotion eligible
    e4_spec = validate_prewall_bitemporal_partition_contract(
        partition_id="databento/trades/2026-10-15",
        columns=["symbol", "ts_event_ns", "ts_recv_ns", "p", "s"],
        row_count=53_063_777,
        max_event_ts_ns=ts_prewall_ns,
        allow_bounded_e2_projection=False,
    )
    e4_ready = audit_h099_prewall_lake_readiness(
        bitemporal_spec=e4_spec,
        df10_covered_symbol_days=10_669,
        df10_total_symbol_days=10_669,
        df6_dated_snapshots_available=True,
        sec_purged_postwall_count=0,
    )
    assert e4_ready.readiness_tier == "E4_NATIVE_H099_LIVE_PROMOTION_ELIGIBLE"
    assert e4_ready.standalone_live_promotion_eligible is True


@given(cur_vol=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False))
def test_property_df10_monotonic_rvol(cur_vol: float) -> None:
    target = dt.date(2026, 10, 15)
    prior20 = _prior_20_dates(target)
    res = compute_df10_same_clock_baseline(
        symbol="SPY",
        target_date=target,
        clock_bucket_et="10:00",
        current_cum_vol=cur_vol,
        prior_same_clock_volumes=prior20,
    )
    assert res.approved is True
    assert res.rvol_tod >= 0.0
