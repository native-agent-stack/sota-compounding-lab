"""Tests for ADR-0031 Quality-First Experimental SOTA Policy (zero license-only bans; technical-only purge)."""

from __future__ import annotations

from dataclasses import replace

from sota_compounding.backtest_playback_manifest import (
    BANNED_LOW_QUALITY_SOURCES,
    EXPERIMENTAL_EVALUATION_POLICY,
    EXPERIMENTAL_QUALITY_FIRST_SOTA_UNBLOCKED,
    PlaybackAuditInput,
    audit_low_quality_practices,
)


def _base_clean() -> PlaybackAuditInput:
    return PlaybackAuditInput(
        horizon_id="W1_PRIMARY_PREWALL_LAKE",
        sample_start_date="2024-01-02",
        sample_max_date="2026-10-15",
        distinct_sessions=679,
        cadence="intraday",
        universe_mode="CAUSAL_T_MINUS_1_PIT",
        cost_1x_rt_bps=8.0,
        includes_2x_and_sqrt_impact=True,
        take_profit_cap_enabled=False,
        max_stop_pct=1.60,
        sharpe_annualization_mode="CALENDAR_DAY_AR1_CLUSTERED",
        uses_spa_mcs_dsr_pbo=True,
        in_sample_symbol_cherry_picking=False,
        vol_max_leverage=1.0,
        allocator_mode="FIXED_SHARE_HEDGE_AND_HIERARCHICAL_EB",
        edgar_timestamp_field="acceptanceDateTime",
        active_strategy_ids=("BJZZ_SUBPENNY_INST_DARK_SWEEP",),
    )


def test_adr0031_quality_first_unblocks_technically_sota_libs_regardless_of_license() -> None:
    assert EXPERIMENTAL_EVALUATION_POLICY == "QUALITY_AND_SOTA_ONLY_NO_LICENSE_EXCLUSION"
    sota_unblocked = (
        "cvxgrp/cvxportfolio",
        "man-group/ArcticDB",
        "google-research/timesfm",
        "xtdb/xtdb",
        "edtechre/pybroker",
        "freqtrade/freqtrade",
        "ml4t/backtest",
        "Nixtla/conformalforecast",
        "SalesforceAIResearch/uni2ts",
        "NVIDIA/cudf",
    )
    res = audit_low_quality_practices(replace(_base_clean(), source_libraries=sota_unblocked))
    assert res["passed"] is True
    assert res["violations"] == []
    assert {"cvxgrp/cvxportfolio", "man-group/arcticdb", "google-research/timesfm", "xtdb/xtdb"} <= (
        EXPERIMENTAL_QUALITY_FIRST_SOTA_UNBLOCKED
    )


def test_adr0031_still_rejects_technically_low_quality_stale_or_scraper_sources() -> None:
    for bad in ("ranaroussi/yfinance", "mementum/backtrader", "kernc/backtesting.py", "hudson-and-thames/mlfinlab"):
        assert bad in BANNED_LOW_QUALITY_SOURCES
        res = audit_low_quality_practices(replace(_base_clean(), source_libraries=(bad,)))
        assert res["passed"] is False
        assert any("SILENT_DATA_FETCH_EXCEPTION_SWALLOW_OR_BANNED_SOURCE" in v for v in res["violations"])
