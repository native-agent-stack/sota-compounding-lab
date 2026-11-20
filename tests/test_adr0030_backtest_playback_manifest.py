"""Tests for ADR-0030 Advanced Backtest Playback Performance Manifest & 14-Point Low-Quality Purge Gate."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sota_compounding.backtest_playback_manifest import (
    LOW_QUALITY_PRACTICE_PURGE_REGISTRY,
    PLAYBACK_MANIFEST_SCHEMA,
    SUITABLE_PLAYBACK_HORIZONS,
    PlaybackAuditInput,
    audit_low_quality_practices,
    build_advanced_backtest_playback_manifest,
    compute_calendar_day_stats,
    compute_min_trl_sessions,
    render_playback_manifest_markdown,
)
from sota_compounding.converged_strategy_factory import RETIRED_TRAP_DENYLIST, STARRED_FRAMEWORK_ADAPTERS
from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT


def _clean_audit_input() -> PlaybackAuditInput:
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
        active_strategy_ids=("BJZZ_SUBPENNY_INST_DARK_SWEEP", "ZARATTINI_2024_5M_ORB"),
        silent_exception_swallow=False,
        source_libraries=("polars", "duckdb", "bashtage/arch", "scikit-learn-contrib/MAPIE"),
        is_post_incentive_chase=False,
        target_sharpe_for_min_trl=1.50,
        ar1_rho_for_min_trl=0.05,
    )


def test_suitable_playback_horizons_and_min_trl_power_math() -> None:
    assert set(SUITABLE_PLAYBACK_HORIZONS) == {
        "W1_PRIMARY_PREWALL_LAKE",
        "W2_MULTI_REGIME_SWING_STRESS",
        "W3_MODERN_XML_13D_BOATS",
        "W4_POST_2026_12_06_23X5_SIP_SHADOW",
        "W5_LIVE_PAPER_PARITY_REPLAY",
    }
    assert SUITABLE_PLAYBACK_HORIZONS["W1_PRIMARY_PREWALL_LAKE"].canonical_sessions == 679
    assert SUITABLE_PLAYBACK_HORIZONS["W2_MULTI_REGIME_SWING_STRESS"].canonical_sessions == 1707

    # Higher target Sharpe requires fewer sessions; positive AR(1) autocorrelation inflates required sessions
    n_sr1 = compute_min_trl_sessions(target_sharpe_ann=1.0, ar1_rho=0.0)
    n_sr2 = compute_min_trl_sessions(target_sharpe_ann=2.0, ar1_rho=0.0)
    n_sr1_ar1 = compute_min_trl_sessions(target_sharpe_ann=1.0, ar1_rho=0.25)
    assert n_sr2 < n_sr1 < n_sr1_ar1
    # A toy 30-day window (~21 sessions) never satisfies realistic Sharpe=1.5 MinTRL
    assert compute_min_trl_sessions(target_sharpe_ann=1.5) > 250


def test_all_14_low_quality_practices_fail_closed_individually() -> None:
    clean = _clean_audit_input()
    assert len(LOW_QUALITY_PRACTICE_PURGE_REGISTRY) == 14
    res_ok = audit_low_quality_practices(clean)
    assert res_ok["passed"] is True
    assert res_ok["violations"] == []
    assert res_ok["meets_min_trl_power"] is True

    bad_cases = [
        (replace(clean, universe_mode="EX_POST_LIVE_MOVERS"), "EX_POST_LIVE_MOVER_SURVIVORSHIP_UNIVERSE"),
        (replace(clean, distinct_sessions=21), "TOY_HORIZON_BELOW_MIN_TRL"),
        (replace(clean, sample_max_date="2026-11-20"), "STUDY_WALL_BREACH_POST_2026_10_15"),
        (replace(clean, cost_1x_rt_bps=4.0), "UNREALISTIC_FLAT_COST_NO_2X_IMPACT"),
        (replace(clean, includes_2x_and_sqrt_impact=False), "UNREALISTIC_FLAT_COST_NO_2X_IMPACT"),
        (replace(clean, take_profit_cap_enabled=True), "FIXED_TAKE_PROFIT_CAP_ACTIVE"),
        (replace(clean, max_stop_pct=3.50), "ADR0022_STOP_OVER_2PCT"),
        (replace(clean, sharpe_annualization_mode="TRADE_COUNT_SCALED"), "INVALID_TRADE_COUNT_SHARPE_ANNUALIZATION"),
        (replace(clean, uses_spa_mcs_dsr_pbo=False), "MISSING_MULTIPLE_TESTING_SPA_MCS_DSR_PBO"),
        (replace(clean, in_sample_symbol_cherry_picking=True), "IN_SAMPLE_SYMBOL_CHERRY_PICKING"),
        (replace(clean, vol_max_leverage=1.50), "VOL_TIMING_LEVERAGE_ABOVE_1X"),
        (replace(clean, allocator_mode="STATIONARY_UCB1"), "STATIONARY_UCB1_ON_FULL_INFO_MANIFEST"),
        (replace(clean, edgar_timestamp_field="transactionDate"), "TRADE_DATE_EDGAR_LOOKAHEAD"),
        (
            replace(clean, active_strategy_ids=("SOTA_DUAL_CLIMAX_STOP_AND_REVERSE",)),
            "QUARANTINED_ADR0064_OR_POST_CHASE_ACTIVE",
        ),
        (replace(clean, is_post_incentive_chase=True), "QUARANTINED_ADR0064_OR_POST_CHASE_ACTIVE"),
        (replace(clean, silent_exception_swallow=True), "SILENT_DATA_FETCH_EXCEPTION_SWALLOW_OR_BANNED_SOURCE"),
        (replace(clean, source_libraries=("yfinance",)), "SILENT_DATA_FETCH_EXCEPTION_SWALLOW_OR_BANNED_SOURCE"),
        (
            replace(clean, source_libraries=("mementum/backtrader",)),
            "SILENT_DATA_FETCH_EXCEPTION_SWALLOW_OR_BANNED_SOURCE",
        ),
    ]
    for mutated, expected_prefix in bad_cases:
        out = audit_low_quality_practices(mutated)
        assert out["passed"] is False, expected_prefix
        assert any(v.startswith(expected_prefix) for v in out["violations"]), (expected_prefix, out["violations"])


def test_build_advanced_backtest_playback_manifest_end_to_end_and_deterministic() -> None:
    n = 80
    r1_alpha = [24.0 + (2.5 if i % 2 == 0 else -1.5) for i in range(n)]
    r2_alpha = [14.0 + (2.5 if i % 2 == 0 else -1.5) for i in range(n)]
    r1_bleed = [-10.0 + (1.0 if i % 3 == 0 else -1.0) for i in range(n)]
    r2_bleed = [-20.0 + (1.0 if i % 3 == 0 else -1.0) for i in range(n)]

    m1 = build_advanced_backtest_playback_manifest(
        audit_input=_clean_audit_input(),
        strategy_daily_returns_1x_bps={"H109_BOATS": r1_alpha, "BLEED_CTRL": r1_bleed},
        strategy_daily_returns_2x_bps={"H109_BOATS": r2_alpha, "BLEED_CTRL": r2_bleed},
        strategy_clusters={"H109_BOATS": "OVERNIGHT_BIFURCATION", "BLEED_CTRL": "CONTROL"},
        realized_mae_atr_samples=[(0.8, 0.9), (1.1, 1.0), (0.4, 0.7)],
        realized_vol_ann_pct=36.0,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    m2 = build_advanced_backtest_playback_manifest(
        audit_input=_clean_audit_input(),
        strategy_daily_returns_1x_bps={"H109_BOATS": r1_alpha, "BLEED_CTRL": r1_bleed},
        strategy_daily_returns_2x_bps={"H109_BOATS": r2_alpha, "BLEED_CTRL": r2_bleed},
        strategy_clusters={"H109_BOATS": "OVERNIGHT_BIFURCATION", "BLEED_CTRL": "CONTROL"},
        realized_mae_atr_samples=[(0.8, 0.9), (1.1, 1.0), (0.4, 0.7)],
        realized_vol_ann_pct=36.0,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m1["schema"] == PLAYBACK_MANIFEST_SCHEMA
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    assert m1["hlz_trial_consumed"] is False
    assert m1["standalone_alpha_claim"] is False
    assert m1["next_hlz_t_star"] == pytest.approx(4.2084)
    assert m1["low_quality_practice_audit"]["passed"] is True
    assert m1["conformal_pid_stop_state"]["next_stop_pct"] <= ADR0022_MAX_STOP_PCT
    assert m1["asymmetric_vol_crash_brake"] == pytest.approx(0.5)
    assert "H109_BOATS" in m1["hansen_spa_and_mcs_2x"]["mcs_surviving_strategies"]
    assert "BLEED_CTRL" not in m1["hansen_spa_and_mcs_2x"]["mcs_surviving_strategies"]

    md = render_playback_manifest_markdown(m1)
    assert "Advanced Backtest Playback Performance Manifest" in md
    assert m1["manifest_sha256"] in md


def test_converged_factory_retired_traps_and_chronos2_upgrade() -> None:
    assert "LEGACY_SP500_ADDITION_POP_CHURN" in RETIRED_TRAP_DENYLIST
    assert "VOL_TIMING_LEVERAGE_ABOVE_1X" in RETIRED_TRAP_DENYLIST
    assert "EX_POST_LIVE_MOVER_SURVIVORSHIP_BACKTEST" in RETIRED_TRAP_DENYLIST
    assert STARRED_FRAMEWORK_ADAPTERS["chronos2"]["repo"] == "amazon-science/chronos-forecasting"
    assert STARRED_FRAMEWORK_ADAPTERS["timesfm"]["authority"] == "RESEARCH_CANDIDATE_PREWALL_ONLY"
    assert {
        "arch",
        "linearmodels",
        "databento_dbn",
        "delta_rs",
        "ducklake",
        "mapie",
        "river",
        "numpyro",
        "arcticdb",
        "cvxportfolio",
        "xtdb",
        "uni2ts",
        "granite_tsfm",
        "ml4t_suite",
        "conformalforecast",
        "sec_parser",
        "nvidia_rapids",
    } <= set(STARRED_FRAMEWORK_ADAPTERS)


@settings(max_examples=80, deadline=None)
@given(
    sr=st.floats(min_value=0.6, max_value=4.0, allow_nan=False, allow_infinity=False),
    rho=st.floats(min_value=-0.4, max_value=0.6, allow_nan=False, allow_infinity=False),
    stop_pct=st.floats(min_value=2.01, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_min_trl_positive_and_adr0022_stop_rejects_above_2pct(
    sr: float, rho: float, stop_pct: float
) -> None:
    n_req = compute_min_trl_sessions(target_sharpe_ann=sr, ar1_rho=rho)
    assert n_req >= 2

    bad = audit_low_quality_practices(replace(_clean_audit_input(), max_stop_pct=stop_pct))
    assert bad["passed"] is False
    assert any(v.startswith("ADR0022_STOP_OVER_2PCT") for v in bad["violations"])

    stats = compute_calendar_day_stats([10.0, -5.0, 8.0, 2.0, -3.0])
    assert 0.0 <= stats["max_drawdown_pct"] <= 100.0
