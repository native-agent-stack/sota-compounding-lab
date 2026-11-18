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
