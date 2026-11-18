"""Tests for v3.10.0 DF1..DF12 bitemporal symbol ranking handoff (ADR-0016)."""

from __future__ import annotations

from sota_compounding import (
    CANONICAL_12_RANKING_FEEDS,
    SYMBOL_RANKING_V310_SCHEMA,
    evaluate_df12_bitemporal_symbol_ranking_handoff,
)


def test_df12_bitemporal_handoff_valid_and_rejects_violations() -> None:
    assert len(CANONICAL_12_RANKING_FEEDS) == 12
    ok = evaluate_df12_bitemporal_symbol_ranking_handoff(
        symbol="AAPL",
        study_date="2026-10-15",
        ts_event_ns=100,
        ts_recv_ns=150,
        decision_ns=200,
        rvol_tod=2.40,
        vol_z_tod=2.60,
        ofi=0.35,
        dry_powder_sec=16.0,
        spread_collar_bps=7.0,
        hard_stop_pct=1.70,
        is_wave2_reentry=True,
        cooldown_elapsed_sec=910.0,
    )
    assert ok["schema_version"] == SYMBOL_RANKING_V310_SCHEMA
    assert ok["approved"] is True
    assert ok["eb_kappa"] == 40

    bad = evaluate_df12_bitemporal_symbol_ranking_handoff(
        symbol="AAPL",
        study_date="2026-11-01",
        ts_event_ns=250,
        ts_recv_ns=200,
        decision_ns=210,
        rvol_tod=1.10,
        vol_z_tod=0.80,
        ofi=0.10,
        dry_powder_sec=5.0,
        spread_collar_bps=15.0,
        hard_stop_pct=2.50,
        is_wave2_reentry=True,
        cooldown_elapsed_sec=120.0,
    )
    assert bad["approved"] is False
    assert len(bad["violations"]) >= 7

    # ADR-0017: IEEE-754 NaN/Inf, overflow & 2027 decision_ns under 2026-10-15 label must fail closed
    base = {
        "symbol": "AAPL",
        "study_date": "2026-10-15",
        "ts_event_ns": 1_792_074_600_000_000_000,
        "ts_recv_ns": 1_792_074_600_005_000_000,
        "decision_ns": 1_792_074_600_010_000_000,
        "rvol_tod": 2.40,
        "vol_z_tod": 2.60,
        "ofi": 0.35,
        "dry_powder_sec": 16.0,
        "spread_collar_bps": 7.0,
        "hard_stop_pct": 1.70,
        "is_wave2_reentry": True,
        "cooldown_elapsed_sec": 910.0,
    }
    assert evaluate_df12_bitemporal_symbol_ranking_handoff(**base)["approved"] is True
    bad_fields = (
        "rvol_tod",
        "vol_z_tod",
        "ofi",
        "dry_powder_sec",
        "spread_collar_bps",
        "hard_stop_pct",
        "cooldown_elapsed_sec",
    )
    for fld in bad_fields:
        for bad_v in (float("nan"), float("inf"), float("-inf")):
            assert evaluate_df12_bitemporal_symbol_ranking_handoff(**{**base, fld: bad_v})["approved"] is False
    jan_2027_ns = 1_800_000_000_000_000_000
    wall_bad = evaluate_df12_bitemporal_symbol_ranking_handoff(
        **{**base, "ts_event_ns": jan_2027_ns - 20, "ts_recv_ns": jan_2027_ns - 10, "decision_ns": jan_2027_ns}
    )
    assert wall_bad["approved"] is False
    assert any("PREWALL_DECISION_NS_VIOLATION" in v for v in wall_bad["violations"])
