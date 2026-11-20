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
    assert SUITABLE_PLAYBACK_HORIZONS["W2_MULTI_REGIME_SWING_STRESS"].canonical_sessions == 1706
    from datetime import date

    from sota_compounding.backtest_playback_manifest import _count_nyse_sessions_inclusive, _is_nyse_trading_day

    assert _is_nyse_trading_day(date(2025, 1, 9)) is False  # President Jimmy Carter National Day of Mourning
    for spec in SUITABLE_PLAYBACK_HORIZONS.values():
        actual_nyse = _count_nyse_sessions_inclusive(
            date.fromisoformat(spec.start_date), date.fromisoformat(spec.end_date)
        )
        if spec.horizon_id == "W2_MULTI_REGIME_SWING_STRESS":
            assert actual_nyse == spec.canonical_sessions == 1706
        else:
            assert actual_nyse >= spec.canonical_sessions, (spec.horizon_id, actual_nyse, spec.canonical_sessions)

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


def _nyse_dates(start_iso: str, count: int) -> list[str]:
    from datetime import date, timedelta

    from sota_compounding.backtest_playback_manifest import _is_nyse_trading_day

    out: list[str] = []
    d = date.fromisoformat(start_iso)
    while len(out) < count:
        if _is_nyse_trading_day(d):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def test_build_advanced_backtest_playback_manifest_end_to_end_and_deterministic() -> None:
    n = 400
    r1_alpha = [24.0 + (2.5 if i % 2 == 0 else -1.5) for i in range(n)]
    r2_alpha = [14.0 + (2.5 if i % 2 == 0 else -1.5) for i in range(n)]
    r1_bleed = [-10.0 + (1.0 if i % 3 == 0 else -1.0) for i in range(n)]
    r2_bleed = [-20.0 + (1.0 if i % 3 == 0 else -1.0) for i in range(n)]
    clean_400 = replace(_clean_audit_input(), distinct_sessions=n, active_strategy_ids=("H109_BOATS", "BLEED_CTRL"))
    sess_400 = _nyse_dates("2024-01-02", n)

    m1 = build_advanced_backtest_playback_manifest(
        audit_input=clean_400,
        strategy_daily_returns_1x_bps={"H109_BOATS": r1_alpha, "BLEED_CTRL": r1_bleed},
        strategy_daily_returns_2x_bps={"H109_BOATS": r2_alpha, "BLEED_CTRL": r2_bleed},
        strategy_clusters={"H109_BOATS": "OVERNIGHT_BIFURCATION", "BLEED_CTRL": "CONTROL"},
        session_dates=sess_400,
        realized_mae_atr_samples=[(0.8, 0.9), (1.1, 1.0), (0.4, 0.7)],
        mae_atr_dates=["2024-03-01", "2024-06-03", "2025-01-15"],
        closed_roundtrips_by_strategy={"H109_BOATS": n, "BLEED_CTRL": n},
        realized_vol_ann_pct=36.0,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.0,
    )
    m2 = build_advanced_backtest_playback_manifest(
        audit_input=clean_400,
        strategy_daily_returns_1x_bps={"H109_BOATS": r1_alpha, "BLEED_CTRL": r1_bleed},
        strategy_daily_returns_2x_bps={"H109_BOATS": r2_alpha, "BLEED_CTRL": r2_bleed},
        strategy_clusters={"H109_BOATS": "OVERNIGHT_BIFURCATION", "BLEED_CTRL": "CONTROL"},
        session_dates=sess_400,
        realized_mae_atr_samples=[(0.8, 0.9), (1.1, 1.0), (0.4, 0.7)],
        mae_atr_dates=["2024-03-01", "2024-06-03", "2025-01-15"],
        closed_roundtrips_by_strategy={"H109_BOATS": n, "BLEED_CTRL": n},
        realized_vol_ann_pct=36.0,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.0,
    )
    assert m1["schema"] == PLAYBACK_MANIFEST_SCHEMA
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    assert m1["hlz_trial_consumed"] is False
    assert m1["standalone_alpha_claim"] is False
    assert m1["next_hlz_t_star"] == pytest.approx(4.2084)
    assert m1["low_quality_practice_audit"]["passed"] is True
    assert m1["promotion_eligible"] is True
    assert m1["conformal_pid_stop_state"]["next_stop_pct"] <= ADR0022_MAX_STOP_PCT
    assert m1["asymmetric_vol_crash_brake"] == pytest.approx(0.5)
    assert "H109_BOATS" in m1["hansen_spa_and_mcs_2x"]["mcs_surviving_strategies"]
    assert "BLEED_CTRL" not in m1["hansen_spa_and_mcs_2x"]["mcs_surviving_strategies"]
    assert m1["fixed_share_hedge_weights"]["H109_BOATS"] > 0.75

    md = render_playback_manifest_markdown(m1)
    assert "Advanced Backtest Playback Performance Manifest" in md
    assert m1["manifest_sha256"] in md
    assert "Observed Sample Span" in md


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


def test_redteam_playback_manifest_all_p0_p1_p2_counterexamples_closed() -> None:
    """Close Astra #1-#8 + P2s and sota-falsifier P0-4/P0-5/P1-9/P1-10."""
    clean = replace(_clean_audit_input(), distinct_sessions=400, ar1_rho_for_min_trl=0.0)
    r2_good = [9.0, 11.0] * 200
    r1_good = [17.0, 19.0] * 200

    # #8: Exact ADIA Lab #19 / zoonek/2025-sharpe-ratio Eq. 11 numerical regression case -> 14 (not 35)
    assert (
        compute_min_trl_sessions(
            target_sharpe_ann=0.036 / 0.079,
            benchmark_sr_ann=0.0,
            periods_per_year=1.0,
            skew=-2.448,
            excess_kurtosis=7.164,
            ar1_rho=0.0,
            confidence=0.95,
        )
        == 14
    )
    # P0-5: NaN skew/kurtosis must raise (never shrink MinTRL 4x)
    with pytest.raises(ValueError):
        compute_min_trl_sessions(target_sharpe_ann=1.5, skew=float("nan"), excess_kurtosis=float("nan"))

    # #7 / P0-5: NaN in max_stop_pct, cost_1x_rt_bps, vol_max_leverage must fail closed
    bad_nan = audit_low_quality_practices(
        replace(clean, max_stop_pct=float("nan"), cost_1x_rt_bps=float("nan"), vol_max_leverage=float("nan"))
    )
    assert bad_nan["passed"] is False
    assert any(v.startswith("ADR0022_STOP_OVER_2PCT") for v in bad_nan["violations"])
    assert any(v.startswith("UNREALISTIC_FLAT_COST") for v in bad_nan["violations"])
    assert any(v.startswith("VOL_TIMING_LEVERAGE_ABOVE_1X") for v in bad_nan["violations"])

    # P2 standalone purge power: 252 sessions < 304 MinTRL must fail standalone audit
    bad_pow = audit_low_quality_practices(replace(clean, distinct_sessions=252, ar1_rho_for_min_trl=0.05))
    assert bad_pow["passed"] is False
    assert bad_pow["meets_min_trl_power"] is False

    # #1: 10 observations with declared distinct_sessions=400 must fail; empty 1x must raise
    with pytest.raises(ValueError):
        build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={"A": []},
            strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 5},
            strategy_clusters={"A": "C"},
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
        )
    m_short = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [17.0, 19.0] * 5},
        strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 5},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_short["low_quality_practice_audit"]["passed"] is False
    assert m_short["promotion_eligible"] is False

    # #2 / P0-4: Forward W4 post-wall horizon NEVER earns promotion_eligible=True; W3 rejects pre-2024-02-05 start
    w4_inp = replace(
        clean,
        horizon_id="W4_POST_2026_12_06_23X5_SIP_SHADOW",
        sample_start_date="2026-12-07",
        sample_max_date="2027-03-31",
        distinct_sessions=60,
        target_sharpe_for_min_trl=4.0,
    )
    m_w4 = build_advanced_backtest_playback_manifest(
        audit_input=w4_inp,
        strategy_daily_returns_1x_bps={"A": [17.0, 19.0] * 30},
        strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 30},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_w4["evaluation_mode"] == "FORWARD_SHADOW_OR_PARITY_ONLY"
    assert m_w4["promotion_eligible"] is False

    w3_bad = audit_low_quality_practices(
        replace(clean, horizon_id="W3_MODERN_XML_13D_BOATS", sample_start_date="2024-01-02")
    )
    assert w3_bad["passed"] is False
    assert any(v.startswith("HORIZON_START_BOUND_BREACH") for v in w3_bad["violations"])

    # #3 / P1-9: Best-t strategy eliminated from MCS must NEVER earn promotion_eligible=True
    a_low_var = [12.0 / 128.0, 14.0 / 128.0] * 200
    b_high_ret = [1.0] * 400
    b_high_ret[200] = -88.5
    m_mcs_split = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in a_low_var], "B": [x + 8.0 for x in b_high_ret]},
        strategy_daily_returns_2x_bps={"A": a_low_var, "B": b_high_ret},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_mcs_split["promotion_eligible"] is False

    # #4 / P0-4: Quarantined candidate with empty active_strategy_ids OR losing 1x series must fail promotion
    m_quar = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, active_strategy_ids=()),
        strategy_daily_returns_1x_bps={"SOTA_DUAL_CLIMAX_STOP_AND_REVERSE": r1_good},
        strategy_daily_returns_2x_bps={"SOTA_DUAL_CLIMAX_STOP_AND_REVERSE": r2_good},
        strategy_clusters={"SOTA_DUAL_CLIMAX_STOP_AND_REVERSE": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_quar["low_quality_practice_audit"]["passed"] is False
    assert m_quar["promotion_eligible"] is False

    m_bad1x = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [-1.0] * 400},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_bad1x["promotion_eligible"] is False

    # #5: Impossible DSR / PBO fail closed; default cpcv_pbo is fail-closed (1.0)
    with pytest.raises(ValueError):
        build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={"A": r1_good},
            strategy_daily_returns_2x_bps={"A": r2_good},
            strategy_clusters={"A": "C"},
            dsr_probability=float("inf"),
            cpcv_pbo=-1.0,
        )
    m_default_pbo = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        dsr_probability=0.98,
    )
    assert m_default_pbo["cpcv_pbo"] == 1.0
    assert m_default_pbo["promotion_eligible"] is False

    # #6: cumulative_trials_n=0 raises; unrounded raw t < raw HLZ quantile fails even if rounded display ties
    with pytest.raises(ValueError):
        build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={"A": r1_good},
            strategy_daily_returns_2x_bps={"A": r2_good},
            strategy_clusters={"A": "C"},
            cumulative_trials_n=0,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
        )
    r_border = [1.0] * 400
    r_border[200] = -76.27194
    m_border = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in r_border]},
        strategy_daily_returns_2x_bps={"A": r_border},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_border["promotion_eligible"] is False

    # P2 / P1-10: Chronological Fixed-Share Hedge carries full history (does NOT discard t=0..T-2)
    m_hedge = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [108.0] * 9 + [8.0], "B": [-92.0] * 9 + [8.0]},
        strategy_daily_returns_2x_bps={"A": [100.0] * 9 + [0.0], "B": [-100.0] * 9 + [0.0]},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_hedge["fixed_share_hedge_weights"]["A"] > 0.85

    # P2: Calendar stats validate 1-day returns, NaN rejection, kurtosis >= -2.0, absorbing total wipeout
    with pytest.raises(ValueError):
        compute_calendar_day_stats([float("nan")])
    s_one = compute_calendar_day_stats([100.0])
    assert s_one["mean_daily_bps"] == 100.0
    assert s_one["cum_return_pct"] == 1.0
    s_two = compute_calendar_day_stats([-1.0, 1.0])
    assert s_two["excess_kurtosis"] >= -2.0
    s_wipe = compute_calendar_day_stats([-10000.0, 10000.0])
    assert s_wipe["cum_return_pct"] == -100.0
    assert s_wipe["max_drawdown_pct"] == 100.0

    # P2: Changing observed sample_start_date changes manifest_sha256 & rendered Markdown
    m_d1 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, sample_start_date="2024-01-02"),
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    m_d2 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, sample_start_date="2025-01-02"),
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_d1["manifest_sha256"] != m_d2["manifest_sha256"]
    assert "2025-01-02" in render_playback_manifest_markdown(m_d2)


def test_redteam_pass2_playback_counterexamples_all_closed() -> None:
    """Close every Pass-2 Astra (P1 #1-#8 + P2s) & Gemini counterexample in ADR-0030 playback manifest."""
    clean = replace(_clean_audit_input(), distinct_sessions=400, ar1_rho_for_min_trl=0.0)
    r1_good = [17.0, 19.0] * 200
    r2_good = [9.0, 11.0] * 200

    # P1 #1: Bankrupt winner `[-10000]+[1000]*399` NEVER promotes (absorbing equity stops post-zero zombie gains)
    r2_bankrupt = [-10000.0] + [1000.0] * 399
    r1_bankrupt = [-9992.0] + [1008.0] * 399
    m_bk = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"BK": r1_bankrupt},
        strategy_daily_returns_2x_bps={"BK": r2_bankrupt},
        strategy_clusters={"BK": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_bk["per_strategy_playback"]["BK"]["bankrupt_2x"] is True
    assert m_bk["promotion_eligible"] is False

    # P1 #2: Impossible 1-day date span `[2024-01-02, 2024-01-02]` with 400 sessions fails closed
    bad_span = audit_low_quality_practices(replace(clean, sample_start_date="2024-01-02", sample_max_date="2024-01-02"))
    assert bad_span["passed"] is False
    assert any(v.startswith("INVALID_DATE_RANGE") for v in bad_span["violations"])

    # P1 #3: MAE calibration date past observed sample_max_date raises ValueError
    with pytest.raises(ValueError, match="mae_atr_dates"):
        build_advanced_backtest_playback_manifest(
            audit_input=replace(clean, sample_max_date="2025-08-15"),
            strategy_daily_returns_1x_bps={"A": r1_good},
            strategy_daily_returns_2x_bps={"A": r2_good},
            strategy_clusters={"A": "C"},
            realized_mae_atr_samples=[(1.2, 1.0)],
            mae_atr_dates=["2026-10-15"],
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
        )

    # P1 #4: 252 declared distinct sessions with 640 rows against empirical MinTRL=514 MUST fail promotion
    r2_640 = [9.0, 9.0, 9.0, 9.0, 11.0, 11.0, 11.0, 11.0] * 80
    r1_640 = [x + 8.0 for x in r2_640]
    m_252 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=252, target_sharpe_for_min_trl=2.0, ar1_rho_for_min_trl=0.0),
        strategy_daily_returns_1x_bps={"A": r1_640},
        strategy_daily_returns_2x_bps={"A": r2_640},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_252["empirical_min_trl_2x"] >= 500
    assert m_252["promotion_eligible"] is False

    # P1 #5: Requested target_sharpe=0.10 with rho=-0.90 requires exact 3,589 sessions (never clamped up to 0.50)
    assert compute_min_trl_sessions(target_sharpe_ann=0.10, ar1_rho=-0.90) == 3589
    bad_low_sr = audit_low_quality_practices(
        replace(clean, distinct_sessions=400, target_sharpe_for_min_trl=0.10, ar1_rho_for_min_trl=-0.90)
    )
    assert bad_low_sr["min_trl_sessions_required"] == 3589
    assert bad_low_sr["passed"] is False

    # P1 #6: Unclipped empirical rho=0.9225 requires empirical MinTRL >= 470 and rejects 400 sessions
    r2_hi_rho = ([9.0] * 25 + [11.0] * 25) * 8
    r1_hi_rho = [x + 8.0 for x in r2_hi_rho]
    m_hi_rho = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=400, target_sharpe_for_min_trl=6.0, ar1_rho_for_min_trl=0.0),
        strategy_daily_returns_1x_bps={"A": r1_hi_rho},
        strategy_daily_returns_2x_bps={"A": r2_hi_rho},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_hi_rho["per_strategy_playback"]["A"]["cost_2x"]["ar1_rho"] > 0.92
    assert m_hi_rho["empirical_min_trl_2x"] >= 470
    assert m_hi_rho["promotion_eligible"] is False

    # P1 #7 + Gemini P2: Explicit cross_trial_sr_std=0.20 rejects marginal winner (<2e-7); 1e-200 scale-invariant DSR
    r2_marg = [1.25, 1.25, -0.75, -0.75] * 100
    r1_marg = [x + 8.0 for x in r2_marg]
    m_ct = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"M": r1_marg},
        strategy_daily_returns_2x_bps={"M": r2_marg},
        strategy_clusters={"M": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.20,
    )
    assert m_ct["computed_dsr_2x"] == 0.0
    assert m_ct["promotion_eligible"] is False
    assert m_ct["dsr_trial_sr_std_source"] == "EXPLICIT_CROSS_TRIAL_LEDGER"

    # Scale invariance at 1e-200 (must NOT underflow variance to 0 -> fake DSR=1.0)
    m_tiny = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"T": [x * 1e-200 for x in r1_marg]},
        strategy_daily_returns_2x_bps={"T": [x * 1e-200 for x in r2_marg]},
        strategy_clusters={"T": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.20,
    )
    assert m_tiny["computed_dsr_2x"] == m_ct["computed_dsr_2x"]

    # P1 #8 + Gemini P1: Pointwise unrounded cost check rejects `[8.99999,10.99999]`, `[5,17]`, and equal `2x == 1x`
    for bad_1x in ([8.99999, 10.99999] * 200, [5.0, 17.0] * 200, list(r2_good)):
        m_cost = build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={"A": bad_1x},
            strategy_daily_returns_2x_bps={"A": r2_good},
            strategy_clusters={"A": "C"},
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
        )
        assert m_cost["promotion_eligible"] is False

    # P2 EB playback constant `+10` bps retains exact zero SE (zero posterior SD, no artificial 0.01 floor)
    m_const = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"C": [18.0] * 400},
        strategy_daily_returns_2x_bps={"C": [10.0] * 400},
        strategy_clusters={"C": "CL"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_const["hierarchical_eb_merit_pool"]["C"]["hierarchical_eb_sd_r"] == 0.0

    # P2 PF `[1e-14, -1e-13]` returns exact 0.10 (never 9.99)
    assert compute_calendar_day_stats([1e-14, -1e-13])["profit_factor"] == pytest.approx(0.1, abs=1e-4)

    # P2 Audit & Provenance: observed_sessions=NaN, string "false" on required-true & false flags, W1 swing, dates hash
    nan_obs: object = float("nan")
    bad_flag: object = "false"
    assert audit_low_quality_practices(clean, observed_sessions=nan_obs)["passed"] is False  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    assert audit_low_quality_practices(replace(clean, take_profit_cap_enabled=bad_flag))["passed"] is False  # type: ignore[arg-type]
    assert audit_low_quality_practices(replace(clean, includes_2x_and_sqrt_impact=bad_flag))["passed"] is False  # type: ignore[arg-type]
    assert audit_low_quality_practices(replace(clean, uses_spa_mcs_dsr_pbo=bad_flag))["passed"] is False  # type: ignore[arg-type]
    assert audit_low_quality_practices(replace(clean, cadence="swing"))["passed"] is False

    # Generate two distinct contiguous official NYSE (XNYS) 400-session windows
    all_nyse = _nyse_dates("2024-01-02", 460)
    dates_1 = all_nyse[:400]
    dates_2 = all_nyse[1:401]
    m_s1 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        session_dates=dates_1,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    m_s2 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        session_dates=dates_2,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert (
        m_s1["observed_sample_span"]["input_evidence_sha256"] != m_s2["observed_sample_span"]["input_evidence_sha256"]
    )
    assert m_s1["manifest_sha256"] != m_s2["manifest_sha256"]


def test_redteam_pass3_playback_counterexamples_all_closed() -> None:
    """Close ALL 12 Astra P1s + 10 P2s & Gemini P1-1/P2-1 in ADR-0030 playback manifest."""
    import math
    from datetime import date, timedelta
    from statistics import NormalDist

    clean = replace(_clean_audit_input(), distinct_sessions=400, ar1_rho_for_min_trl=0.0)
    r1_good = [17.0, 19.0] * 200
    r2_good = [9.0, 11.0] * 200

    # Astra P1-1: Compounded capital loss (-99.9965%) & float64 underflow-to-zero bankruptcy both block!
    m_caploss = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"CL": [9008.0, -4992.0] * 200},
        strategy_daily_returns_2x_bps={"CL": [9000.0, -5000.0] * 200},
        strategy_clusters={"CL": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_caploss["per_strategy_playback"]["CL"]["survives_2x_cost"] is False
    assert m_caploss["promotion_eligible"] is False

    m_underflow_bk = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"UB": [25008.0, -9991.0] * 200},
        strategy_daily_returns_2x_bps={"UB": [25000.0, -9999.0] * 200},
        strategy_clusters={"UB": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_underflow_bk["per_strategy_playback"]["UB"]["bankrupt_2x"] is True
    assert m_underflow_bk["promotion_eligible"] is False

    # Astra P1-2: Duplicating rows (`distinct_sessions=252`, `504` rows) fails session parity & blocks promotion!
    r2_dup = [1.2, 1.2, -0.8, -0.8] * 126
    m_dup = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=252, target_sharpe_for_min_trl=2.0),
        strategy_daily_returns_1x_bps={"D": [x + 8.0 for x in r2_dup]},
        strategy_daily_returns_2x_bps={"D": r2_dup},
        strategy_clusters={"D": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.01,
    )
    assert m_dup["low_quality_practice_audit"]["passed"] is False
    assert m_dup["promotion_eligible"] is False

    # Astra P1-3: Weekend session_dates AND session_dates extending declared sample_max_date both raise ValueError!
    d0 = date(2024, 1, 2)
    wknd_dates = [(d0 + timedelta(days=i)).isoformat() for i in range(399)] + ["2025-07-15"]
    with pytest.raises(ValueError, match="weekend"):
        build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={"A": r1_good},
            strategy_daily_returns_2x_bps={"A": r2_good},
            strategy_clusters={"A": "C"},
            session_dates=wknd_dates,
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.10,
        )
    valid_wds = _nyse_dates("2024-01-02", 400)
    with pytest.raises(ValueError, match="declared sample window"):
        build_advanced_backtest_playback_manifest(
            audit_input=replace(clean, sample_max_date="2025-06-01"),
            strategy_daily_returns_1x_bps={"A": r1_good},
            strategy_daily_returns_2x_bps={"A": r2_good},
            strategy_clusters={"A": "C"},
            session_dates=valid_wds,
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.10,
        )

    # Astra P1-4: Undated MAE blocks promotion; reversed mae_atr_dates raises ValueError!
    m_undated_mae = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        realized_mae_atr_samples=[(1.2, 1.0)],
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_undated_mae["promotion_eligible"] is False
    with pytest.raises(ValueError, match="chronologically"):
        build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={"A": r1_good},
            strategy_daily_returns_2x_bps={"A": r2_good},
            strategy_clusters={"A": "C"},
            realized_mae_atr_samples=[(2.0, 1.0), (0.0, 1.0)],
            mae_atr_dates=["2025-05-02", "2025-05-01"],
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.10,
        )

    # Astra P1-5: Unclipped supported rho=295/298=0.989932886 requires exact 299 sessions at target_sharpe=21.25!
    r2_rho9899 = [9.0] * 149 + [11.0] * 149
    m_rho = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=298, target_sharpe_for_min_trl=21.25),
        strategy_daily_returns_1x_bps={"R": [x + 8.0 for x in r2_rho9899]},
        strategy_daily_returns_2x_bps={"R": r2_rho9899},
        strategy_clusters={"R": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_rho["empirical_min_trl_2x"] == 299
    assert m_rho["promotion_eligible"] is False

    # Astra P1-6: Omitting cross_trial_sr_std blocks promotion; explicit 0.01 has no artificial floor (~0.9935)!
    r2_marg = [1.25, 1.25, -0.75, -0.75] * 100
    m_omit = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"M": [x + 8.0 for x in r2_marg]},
        strategy_daily_returns_2x_bps={"M": r2_marg},
        strategy_clusters={"M": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_omit["promotion_eligible"] is False

    r2_sd01 = [1.15, 1.15, -0.85, -0.85] * 100
    m_sd01 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"S": [x + 8.0 for x in r2_sd01]},
        strategy_daily_returns_2x_bps={"S": r2_sd01},
        strategy_clusters={"S": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.01,
    )
    assert m_sd01["computed_dsr_2x"] == pytest.approx(0.9935, abs=0.002)

    # Gemini P1-1: Monotonic PSD DSR on skewed autocorrelated series (worse strategy NEVER beats strictly better one!)
    base_skew = ([0.0] * 18 + [10.0] * 2) * 20
    n_sk = len(base_skew)
    mu_sk = sum(base_skew) / n_sk
    sd_sk = math.sqrt(sum((x - mu_sk) ** 2 for x in base_skew) / (n_sk - 1))
    z1 = NormalDist().inv_cdf(1.0 - 1.0 / 105.0)
    z2 = NormalDist().inv_cdf(1.0 - 1.0 / (105.0 * math.e))
    e_max_z = (1.0 - 0.5772156649015329) * z1 + 0.5772156649015329 * z2
    worse_sk = [x + (1.3070 - (mu_sk / sd_sk)) * sd_sk for x in base_skew]
    better_sk = [x + 0.0035 * sd_sk for x in worse_sk]
    ct_sk = (1.3070 - 1e-4) / e_max_z
    m_w_sk = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"W": [x + 8.0 for x in worse_sk]},
        strategy_daily_returns_2x_bps={"W": worse_sk},
        strategy_clusters={"W": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=ct_sk,
    )
    m_b_sk = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"B": [x + 8.0 for x in better_sk]},
        strategy_daily_returns_2x_bps={"B": better_sk},
        strategy_clusters={"B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=ct_sk,
    )
    assert m_w_sk["computed_dsr_2x"] <= m_b_sk["computed_dsr_2x"]
    assert m_w_sk["promotion_eligible"] is False

    # Astra P1-7: 106 candidates with 105 trials raises; built-in C(6,2)=15 CPCV detects 15/15 overfit!
    with pytest.raises(ValueError, match="candidate count"):
        build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={f"S{i}": r1_good for i in range(106)},
            strategy_daily_returns_2x_bps={f"S{i}": r2_good for i in range(106)},
            strategy_clusters={f"S{i}": "C" for i in range(106)},
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.10,
        )
    d_offs = [-5.0, -4.0, -3.0, 1.0, 2.0, 9.0]
    a_ov: list[float] = []
    b_ov: list[float] = []
    for dg in d_offs:
        a_ov.extend([0.0] * 30 + [100.0 + dg] * 20 + [0.0] * 30)
        b_ov.extend([0.0] * 30 + [100.0 - dg] * 20 + [0.0] * 30)
    m_ov = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=480, target_sharpe_for_min_trl=20.0),
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in a_ov], "B": [x + 8.0 for x in b_ov]},
        strategy_daily_returns_2x_bps={"A": a_ov, "B": b_ov},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_ov["computed_cpcv_pbo_2x"] == 1.0
    assert m_ov["effective_cpcv_pbo"] == 1.0
    assert m_ov["promotion_eligible"] is False

    # Astra P1-8: Issued PID stop & vol brake respect audited max_stop_pct=0.50 & vol_max_leverage=0.25!
    m_cap = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, max_stop_pct=0.50, vol_max_leverage=0.25),
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        realized_vol_ann_pct=18.0,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_cap["conformal_pid_stop_state"]["next_stop_pct"] <= 0.50
    assert m_cap["asymmetric_vol_crash_brake"] <= 0.25

    # Astra P1-9 & P1-10: paper_parity on W1 fails; scalar strings bypass denylists blocked!
    assert audit_low_quality_practices(replace(clean, cadence="paper_parity"))["passed"] is False
    bad_lib_str: object = "yfinance"
    bad_strat_str: object = "SOTA_DUAL_CLIMAX_STOP_AND_REVERSE"
    assert audit_low_quality_practices(replace(clean, source_libraries=bad_lib_str))["passed"] is False  # type: ignore[arg-type]
    assert audit_low_quality_practices(replace(clean, active_strategy_ids=bad_strat_str))["passed"] is False  # type: ignore[arg-type]

    # Astra P1-12: Nominal 1e-9 bps total cost drag fails 2x cost gate!
    r2_var10 = [9.0, 11.0] * 200
    r1_nom = [9.0 + 1e-9, 11.0] + [9.0, 11.0] * 199
    m_nom = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_nom},
        strategy_daily_returns_2x_bps={"A": r2_var10},
        strategy_clusters={"A": "C"},
        closed_roundtrips_by_strategy={"A": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_nom["promotion_eligible"] is False

    # Astra P2-5..P2-10 & Gemini P2-1: Pearson bound, standalone bankruptcy, Calmar/AR1, hash, W5 & risk_unit_bps
    with pytest.raises(ValueError, match="Pearson"):
        compute_min_trl_sessions(
            target_sharpe_ann=1.5, benchmark_sr_ann=0.5, periods_per_year=1.0, skew=1.0, excess_kurtosis=-2.0
        )
    s_bk_sa = compute_calendar_day_stats([-10000.0, 20000.0])
    assert s_bk_sa["cum_return_pct"] == -100.0
    assert s_bk_sa["mean_daily_bps"] == -5000.0
    assert s_bk_sa["profit_factor"] == 0.0

    s_cal = compute_calendar_day_stats([10.0, -0.005] * 200)
    assert s_cal["calmar"] > 200_000.0
    s_neg_rho = compute_calendar_day_stats([9.0, 11.0] * 8)
    assert s_neg_rho["sharpe_ann_ar1_adj"] > 800.0

    # P2-8: Moving MAE date OR changing raw PBO 0.10001 vs 0.10002 changes both hashes!
    m_mae1 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        realized_mae_atr_samples=[(0.8, 1.0)],
        mae_atr_dates=["2024-01-03"],
        cumulative_trials_n=105,
        cpcv_pbo=0.10001,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    m_mae2 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        realized_mae_atr_samples=[(0.8, 1.0)],
        mae_atr_dates=["2025-01-03"],
        cumulative_trials_n=105,
        cpcv_pbo=0.10001,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    m_pbo2 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        realized_mae_atr_samples=[(0.8, 1.0)],
        mae_atr_dates=["2024-01-03"],
        cumulative_trials_n=105,
        cpcv_pbo=0.10002,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert (
        m_mae1["observed_sample_span"]["input_evidence_sha256"]
        != m_mae2["observed_sample_span"]["input_evidence_sha256"]
    )
    assert m_mae1["manifest_sha256"] != m_mae2["manifest_sha256"]
    assert (
        m_mae1["observed_sample_span"]["input_evidence_sha256"]
        != m_pbo2["observed_sample_span"]["input_evidence_sha256"]
    )
    assert m_mae1["manifest_sha256"] != m_pbo2["manifest_sha256"]

    # P2-9: W5 horizon contains >= 60 NYSE sessions AND requires >= 30 closed trades across >= 10 trade sessions!
    w5_spec = SUITABLE_PLAYBACK_HORIZONS["W5_LIVE_PAPER_PARITY_REPLAY"]
    w5_inp = replace(
        clean,
        horizon_id="W5_LIVE_PAPER_PARITY_REPLAY",
        sample_start_date=w5_spec.start_date,
        sample_max_date=w5_spec.end_date,
        distinct_sessions=60,
        cadence="paper_parity",
    )
    assert (
        audit_low_quality_practices(w5_inp, observed_closed_roundtrips=0, observed_trade_sessions=10)["passed"] is False
    )
    assert (
        audit_low_quality_practices(w5_inp, observed_closed_roundtrips=30, observed_trade_sessions=1)["passed"] is False
    )
    assert (
        audit_low_quality_practices(w5_inp, observed_closed_roundtrips=30, observed_trade_sessions=10)["passed"] is True
    )

    # P2-10: Explicit risk_unit_bps=50.0 vs 100.0 maps +50 bps to +1.0R/+0.5R in BOTH EB & exact Fixed-Share Hedge!
    m_r50 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [58.0] * 400, "B": [8.0] * 400},
        strategy_daily_returns_2x_bps={"A": [50.0] * 400, "B": [0.0] * 400},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        risk_unit_bps=50.0,
    )
    m_r100 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [58.0] * 400, "B": [8.0] * 400},
        strategy_daily_returns_2x_bps={"A": [50.0] * 400, "B": [0.0] * 400},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        risk_unit_bps=100.0,
    )
    assert m_r50["hierarchical_eb_merit_pool"]["A"]["observed_mean_r"] == pytest.approx(1.0, abs=1e-4)
    assert m_r50["fixed_share_hedge_weights"]["A"] == pytest.approx(0.897078, abs=1e-5)
    assert m_r100["fixed_share_hedge_weights"]["A"] == pytest.approx(0.826621, abs=1e-5)


def test_redteam_pass4_playback_counterexamples_all_closed() -> None:
    """Close ALL Pass-4 Astra (12 P1 + 16 P2) & Gemini (3 P1 + 4 P2) counterexamples in ADR-0030."""
    import math
    from statistics import NormalDist

    from sota_compounding.backtest_playback_manifest import _hlz_t_hurdle_raw

    clean = replace(_clean_audit_input(), distinct_sessions=400, ar1_rho_for_min_trl=0.0)
    r1_good = [17.0, 19.0] * 200
    r2_good = [9.0, 11.0] * 200
    sess_400 = _nyse_dates("2024-01-02", 400)

    # Astra P1-1: Exact Fraction compounding rejects `[52500, -8400]*200` (exact 1.0) & `-8400.000000000002` (loss)
    r2_flat_exact = [52500.0, -8400.0] * 200
    r2_ctrl_b = [x / 100.0 - 30.0 for x in r2_flat_exact]
    m_flat = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, active_strategy_ids=("F", "B")),
        strategy_daily_returns_1x_bps={"F": [x + 8.0 for x in r2_flat_exact], "B": [x + 8.0 for x in r2_ctrl_b]},
        strategy_daily_returns_2x_bps={"F": r2_flat_exact, "B": r2_ctrl_b},
        strategy_clusters={"F": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"F": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.50,
    )
    assert m_flat["per_strategy_playback"]["F"]["survives_2x_cost"] is False
    assert m_flat["promotion_eligible"] is False
    r2_tiny_loss = list(r2_flat_exact)
    r2_tiny_loss[-1] = -8400.000000000002
    m_tloss = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, active_strategy_ids=("F", "B")),
        strategy_daily_returns_1x_bps={"F": [x + 8.0 for x in r2_tiny_loss], "B": [x + 8.0 for x in r2_ctrl_b]},
        strategy_daily_returns_2x_bps={"F": r2_tiny_loss, "B": r2_ctrl_b},
        strategy_clusters={"F": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"F": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.50,
    )
    assert m_tloss["per_strategy_playback"]["F"]["survives_2x_cost"] is False
    assert m_tloss["promotion_eligible"] is False

    # Astra P1-2 (isolated): Passing all other gates WITH session_dates promotes, omitting session_dates ALONE blocks!
    r2_iso_b = [-19.0, -21.0] * 200
    iso_audit = replace(clean, active_strategy_ids=("A", "B"))
    m_dated_ok = build_advanced_backtest_playback_manifest(
        audit_input=iso_audit,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [x + 8.0 for x in r2_iso_b]},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_iso_b},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    m_undated_iso = build_advanced_backtest_playback_manifest(
        audit_input=iso_audit,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [x + 8.0 for x in r2_iso_b]},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_iso_b},
        strategy_clusters={"A": "C", "B": "C"},
        closed_roundtrips_by_strategy={"A": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_dated_ok["promotion_eligible"] is True
    assert m_undated_iso["promotion_eligible"] is False

    # Astra P1-3: 2024-01-02..2024-12-18 has only 244 NYSE sessions (8 US holidays) -> rejects 252!
    hol_audit = audit_low_quality_practices(
        replace(
            clean,
            sample_start_date="2024-01-02",
            sample_max_date="2024-12-18",
            distinct_sessions=252,
            target_sharpe_for_min_trl=2.0,
        )
    )
    assert hol_audit["passed"] is False
    assert any("nyse_sessions=244" in v for v in hol_audit["violations"])

    # Astra P1-4: Undated MAE NEVER alters issued PID stop; Saturday MAE date raises ValueError!
    m_base_stop = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    m_undated_mae = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        realized_mae_atr_samples=[(1.2, 1.0)],
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert (
        m_undated_mae["conformal_pid_stop_state"]["next_stop_pct"]
        == m_base_stop["conformal_pid_stop_state"]["next_stop_pct"]
    )
    with pytest.raises(ValueError, match="weekend or NYSE holiday"):
        build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={"A": r1_good},
            strategy_daily_returns_2x_bps={"A": r2_good},
            strategy_clusters={"A": "C"},
            realized_mae_atr_samples=[(2.0, 1.0)],
            mae_atr_dates=["2024-07-06"],
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.10,
        )

    # Astra P1-6: Infeasible `cross_trial_sr_std=0.01` contradicting submitted A vs B (`min SD = 0.277003`) raises!
    a_disp = [1.15, 1.15, -0.85, -0.85] * 100
    b_disp = [x - 4.0 for x in a_disp]
    with pytest.raises(ValueError, match="lower bound"):
        build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in a_disp], "B": [x + 8.0 for x in b_disp]},
            strategy_daily_returns_2x_bps={"A": a_disp, "B": b_disp},
            strategy_clusters={"A": "C", "B": "C"},
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.01,
        )

    # Astra P1-7 & Additional P1-1/P1-2: Single-column removal blocks; exact 20d purge -> PBO=1.0; name-invariant ties!
    m_single_col = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_single_col["computed_cpcv_pbo_2x"] is None
    assert m_single_col["promotion_eligible"] is False

    # Exact 20-day purge on six 50-day groups (d[15]=d[34]=+5, d[24]=-9) -> PBO == 1.0 (never shortened to 12!)
    a_p20: list[float] = []
    b_p20: list[float] = []
    for _ in range(6):
        dg = [0.0] * 50
        dg[15] = 5.0
        dg[34] = 5.0
        dg[24] = -9.0
        a_p20.extend([100.0 + x for x in dg])
        b_p20.extend([100.0 - x for x in dg])
    m_p20 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=300, target_sharpe_for_min_trl=20.0),
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in a_p20], "B": [x + 8.0 for x in b_p20]},
        strategy_daily_returns_2x_bps={"A": a_p20, "B": b_p20},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=5.0,
    )
    assert m_p20["computed_cpcv_pbo_2x"] == 1.0
    assert m_p20["promotion_eligible"] is False

    # Name-invariant CPCV IS tie-breaking: swapping A <-> B yields IDENTICAL PBO!
    mid_sums = [-6, 2, 2, 2, 10, 10]
    a_tie: list[float] = []
    b_tie: list[float] = []
    for ms in mid_sums:
        pos_cnt = (40 + ms) // 2
        neg_cnt = 40 - pos_cnt
        margin = [1.0, -1.0] * 10
        mid = [1.0] * pos_cnt + [-1.0] * neg_cnt
        seq = margin + mid + margin
        a_tie.extend([1.0 + x for x in seq])
        b_tie.extend([1.0 - x for x in seq])
    m_tie1 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=480, target_sharpe_for_min_trl=20.0),
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in a_tie], "B": [x + 8.0 for x in b_tie]},
        strategy_daily_returns_2x_bps={"A": a_tie, "B": b_tie},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=5.0,
    )
    m_tie2 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=480, target_sharpe_for_min_trl=20.0),
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in b_tie], "B": [x + 8.0 for x in a_tie]},
        strategy_daily_returns_2x_bps={"A": b_tie, "B": a_tie},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=5.0,
    )
    assert m_tie1["computed_cpcv_pbo_2x"] == m_tie2["computed_cpcv_pbo_2x"] == pytest.approx(4.0 / 15.0, abs=1e-4)

    # Astra Additional P1-4: Bailey-Lopez de Prado observed-SR variance gives DSR = 0.9449 < 0.95!
    r2_dsr_obs = [0.1583 + (1.0 if i % 20 == 4 else (-1.0 if i % 20 == 14 else 0.0)) for i in range(400)]
    z1 = -NormalDist().inv_cdf(1.0 / 105.0)
    z2 = -NormalDist().inv_cdf(1.0 / (105.0 * math.e))
    e_max_z = (1.0 - 0.5772156649015329) * z1 + 0.5772156649015329 * z2
    m_dsr_obs = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in r2_dsr_obs]},
        strategy_daily_returns_2x_bps={"A": r2_dsr_obs},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.4 / e_max_z,
    )
    assert m_dsr_obs["computed_dsr_2x"] < 0.95
    assert m_dsr_obs["promotion_eligible"] is False

    # Astra P1-8 & Gemini P1-1: Audited `max_stop_pct=0.0001%` issues exact `0.0001%` (NEVER `1.25%`!)
    m_tiny_stop = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, max_stop_pct=0.0001),
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_tiny_stop["conformal_pid_stop_state"]["next_stop_pct"] == 0.0001

    # Astra P1-12 & Gemini P1-2: Mismatched `closed_roundtrips_by_strategy` key raises; omitted roundtrips blocks!
    with pytest.raises(ValueError, match="key mismatch"):
        build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={"A": [17.0, 19.0] * 200},
            strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 200},
            strategy_clusters={"A": "C"},
            closed_roundtrips_by_strategy={"A_TYPO": 400},
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.10,
        )
    m_no_rt = build_advanced_backtest_playback_manifest(
        audit_input=iso_audit,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [x + 8.0 for x in r2_iso_b]},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_iso_b},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_no_rt["promotion_eligible"] is False

    # Gemini P1-3 & Astra Additional P2-8: `vol_max_leverage=0.0` still validates BOTH `NaN` AND `-1.0`!
    for bad_rv in (float("nan"), -1.0):
        with pytest.raises(ValueError, match="realized_vol_ann_pct"):
            build_advanced_backtest_playback_manifest(
                audit_input=replace(clean, vol_max_leverage=0.0),
                strategy_daily_returns_1x_bps={"A": r1_good},
                strategy_daily_returns_2x_bps={"A": r2_good},
                strategy_clusters={"A": "C"},
                realized_vol_ann_pct=bad_rv,
                cumulative_trials_n=105,
                cpcv_pbo=0.10,
                dsr_probability=0.98,
                cross_trial_sr_std=0.10,
            )

    # Astra P1-10, P2-5, P2-7, P2-8, Additional P2-1/P2-2/P2-7 & Gemini P2-2/P2-3/P2-4:
    bad_cand_str: object = "SOTA_DUAL_CLIMAX_STOP_AND_REVERSE"
    assert audit_low_quality_practices(clean, candidate_strategy_ids=bad_cand_str)["passed"] is False  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Pearson"):
        compute_min_trl_sessions(
            target_sharpe_ann=900001.0, benchmark_sr_ann=900000.0, periods_per_year=1.0, skew=1e-6, excess_kurtosis=-2.0
        )
    assert audit_low_quality_practices(replace(clean, target_sharpe_for_min_trl=1e-200))["passed"] is False

    m_inf_sr_star = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=1e308,
    )
    assert m_inf_sr_star["computed_dsr_2x"] == 0.0
    assert m_inf_sr_star["promotion_eligible"] is False

    assert compute_calendar_day_stats([10.0, -1e-13] * 200)["calmar"] > 1e15
    assert compute_calendar_day_stats([100000.0] * 400 + [-5000.0])["max_drawdown_pct"] == pytest.approx(50.0, abs=0.1)
    assert math.isfinite(compute_calendar_day_stats([190000.0, 190000.0])["cum_return_pct"])
    assert compute_calendar_day_stats([100.0, -0.5])["profit_factor"] == pytest.approx(200.0, abs=1e-4)
    assert _hlz_t_hurdle_raw(10**14) > 8.0

    # Hash changes on realized_vol_ann_pct (9 vs 18) AND audit_input changes; declared wall breach cannot be masked!
    m_v9 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        realized_vol_ann_pct=9.0,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    m_v18 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        realized_vol_ann_pct=18.0,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert (
        m_v9["observed_sample_span"]["input_evidence_sha256"] != m_v18["observed_sample_span"]["input_evidence_sha256"]
    )
    assert m_v9["manifest_sha256"] != m_v18["manifest_sha256"]

    m_wall_mask = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, sample_max_date="2026-10-16"),
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        session_dates=sess_400,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_wall_mask["low_quality_practice_audit"]["passed"] is False
    assert m_wall_mask["promotion_eligible"] is False


def test_redteam_pass5_playback_counterexamples_all_closed() -> None:
    """Close ALL Pass-5 Astra (8 P1 + 24 P2) & Gemini (4 P1 + 3 P2) counterexamples in ADR-0030."""
    import math
    from statistics import NormalDist

    from sota_compounding.backtest_playback_manifest import _compute_empirical_dsr, _hlz_t_hurdle_raw

    clean = replace(_clean_audit_input(), distinct_sessions=400, ar1_rho_for_min_trl=0.0)
    r1_good = [17.0, 19.0] * 200
    r2_good = [9.0, 11.0] * 200
    r2_ctrl = [-19.0, -21.0] * 200
    r1_ctrl = [x + 8.0 for x in r2_ctrl]
    sess_400 = _nyse_dates("2024-01-02", 400)
    clean_ab = replace(clean, active_strategy_ids=("A", "B"))

    # Astra P1 #1: Reordered break-even `[52500]*38 + [-8400]*38 + [52500,-8400]*162` has exact equity == 1 -> blocks!
    r2_reord = [52500.0] * 38 + [-8400.0] * 38 + [52500.0, -8400.0] * 162
    r2_reord_b = [x / 100.0 - 30.0 for x in r2_reord]
    m_reord = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in r2_reord], "B": [x + 8.0 for x in r2_reord_b]},
        strategy_daily_returns_2x_bps={"A": r2_reord, "B": r2_reord_b},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.05,
    )
    assert m_reord["per_strategy_playback"]["A"]["survives_2x_cost"] is False
    assert m_reord["promotion_eligible"] is False

    # Astra P1 #2: Constant infinite-Sharpe competitor `B=[-18]*400` rejects any finite cross_trial_sr_std!
    with pytest.raises(ValueError, match="Infeasible cross_trial_sr_std"):
        build_advanced_backtest_playback_manifest(
            audit_input=clean_ab,
            strategy_daily_returns_1x_bps={"A": r1_good, "B": [-10.0] * 400},
            strategy_daily_returns_2x_bps={"A": r2_good, "B": [-18.0] * 400},
            strategy_clusters={"A": "C", "B": "C"},
            session_dates=sess_400,
            closed_roundtrips_by_strategy={"A": 400, "B": 400},
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.01,
        )

    # Astra P1 #3: Omitting a declared `active_strategy_ids` member (`B`) NEVER allows promotion!
    a_p20: list[float] = []
    for _ in range(6):
        dg = [0.0] * 50
        dg[15] = 5.0
        dg[34] = 5.0
        dg[24] = -9.0
        a_p20.extend([100.0 + x for x in dg])
    c_ctrl = [-2.0, 0.0] * 150
    sess_300 = _nyse_dates("2024-01-02", 300)
    m_omit_active = build_advanced_backtest_playback_manifest(
        audit_input=replace(
            clean, distinct_sessions=300, target_sharpe_for_min_trl=20.0, active_strategy_ids=("A", "B", "C")
        ),
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in a_p20], "C": [x + 8.0 for x in c_ctrl]},
        strategy_daily_returns_2x_bps={"A": a_p20, "C": c_ctrl},
        strategy_clusters={"A": "C", "C": "C"},
        session_dates=sess_300,
        closed_roundtrips_by_strategy={"A": 300, "C": 300},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=6.0,
    )
    assert m_omit_active["promotion_eligible"] is False

    # Astra P1 #5 + Gemini P1-4: Exact Bailey-Lopez de Prado (2014) Eq. 2 DSR == 0.960126 >= 0.95 (no max inversion!)
    r_dsr_p5 = [(1.0 if i % 10 == 0 else 0.0) - 0.04414 for i in range(400)]
    z1 = -NormalDist().inv_cdf(1.0 / 105.0)
    z2 = -NormalDist().inv_cdf(1.0 / (105.0 * math.e))
    e_max_z = (1.0 - 0.5772156649015329) * z1 + 0.5772156649015329 * z2
    m_dsr_p5 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in r_dsr_p5]},
        strategy_daily_returns_2x_bps={"A": r_dsr_p5},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.12 / e_max_z,
    )
    assert m_dsr_p5["computed_dsr_2x_blp2014_iid"] == pytest.approx(0.9601, abs=1e-4)
    assert m_dsr_p5["computed_dsr_2x"] == pytest.approx(0.9793, abs=1e-4)
    assert m_dsr_p5["computed_dsr_2x_adia19_null_var"] == pytest.approx(0.9638, abs=1e-4)

    # Astra P1 #6: Omitted interior NYSE sessions (`1..200` + `401..600`) raise ValueError!
    all_600 = _nyse_dates("2024-01-02", 600)
    gapped_400 = all_600[:200] + all_600[400:600]
    with pytest.raises(ValueError, match="omitted interior NYSE"):
        build_advanced_backtest_playback_manifest(
            audit_input=clean_ab,
            strategy_daily_returns_1x_bps={"A": r1_good, "B": r1_ctrl},
            strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
            strategy_clusters={"A": "C", "B": "C"},
            session_dates=gapped_400,
            closed_roundtrips_by_strategy={"A": 400, "B": 400},
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=2.5,
        )

    # Astra P1 #7: Scale-invariant Sharpe-only CPCV ties `A` vs `2*A` at exact PBO == 1.0 (never 0/15!)
    m_scale_tie = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [2.0 * x + 8.0 for x in r2_good]},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": [2.0 * x for x in r2_good]},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_scale_tie["computed_cpcv_pbo_2x"] == 1.0
    assert m_scale_tie["promotion_eligible"] is False

    # Astra P1 #8: Invalid negative cost drag on comparison control `B` blocks promotion of `A`!
    m_ctrl_ok = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": r1_ctrl},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    m_ctrl_bad_cost = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [-100.0] * 400},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_ctrl_ok["promotion_eligible"] is True
    assert m_ctrl_bad_cost["promotion_eligible"] is False

    # Astra P2 #3 & #23: Isolated 400-trip drag (8 vs 3200 bps required) fails; notional_fraction=0.5 passes 2 trips!
    r1_8bps_only = [r2_good[0] + 8.0] + list(r2_good[1:])
    m_drag_short = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_8bps_only, "B": r1_ctrl},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_drag_short["promotion_eligible"] is False

    m_half_notional_ok = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_8bps_only, "B": r1_ctrl},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 2, "B": 2},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
        roundtrip_notional_fraction=0.5,
    )
    m_full_notional_fail = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_8bps_only, "B": r1_ctrl},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 2, "B": 2},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
        roundtrip_notional_fraction=1.0,
    )
    assert m_half_notional_ok["promotion_eligible"] is True
    assert m_full_notional_fail["promotion_eligible"] is False

    # Astra P2 #2, #22 & Gemini P1-1: Sub-0.15% & non-grid caps adapt cleanly; invalid 0.00005% fails closed!
    for cap in (0.00015, 0.10005, 0.10):
        m_cap_ok = build_advanced_backtest_playback_manifest(
            audit_input=replace(clean, max_stop_pct=cap),
            strategy_daily_returns_1x_bps={"A": r1_good},
            strategy_daily_returns_2x_bps={"A": r2_good},
            strategy_clusters={"A": "C"},
            realized_mae_atr_samples=[(0.02, 0.08)],
            mae_atr_dates=["2024-03-01"],
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.10,
        )
        assert 0.0001 <= m_cap_ok["conformal_pid_stop_state"]["next_stop_pct"] <= cap
        assert m_cap_ok["conformal_pid_stop_state"]["adr0022_compliant"] is True

    m_bad_stop = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, max_stop_pct=0.00005),
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_bad_stop["conformal_pid_stop_state"]["adr0022_compliant"] is False
    assert m_bad_stop["conformal_pid_stop_state"]["next_stop_pct"] == 0.0
    assert m_bad_stop["promotion_eligible"] is False

    # Gemini P1-3: Non-JSON `set` & `NaN` inside `audit_input` seal cleanly without crashing!
    bad_set: object = {"polars"}
    m_json_safe = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, source_libraries=bad_set, max_stop_pct=float("nan")),  # type: ignore[arg-type]
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert len(m_json_safe["manifest_sha256"]) == 64
    assert m_json_safe["promotion_eligible"] is False

    # Astra P2 #7: Changing `seed` on constant series changes both evidence digest & manifest hash!
    m_seed1 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [18.0] * 400, "B": [-2.0] * 400},
        strategy_daily_returns_2x_bps={"A": [10.0] * 400, "B": [-10.0] * 400},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        seed=1,
    )
    m_seed2 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [18.0] * 400, "B": [-2.0] * 400},
        strategy_daily_returns_2x_bps={"A": [10.0] * 400, "B": [-10.0] * 400},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        seed=2,
    )
    assert m_seed1["bootstrap_seed"] == 1 and m_seed2["bootstrap_seed"] == 2
    assert (
        m_seed1["observed_sample_span"]["input_evidence_sha256"]
        != m_seed2["observed_sample_span"]["input_evidence_sha256"]
    )
    assert m_seed1["manifest_sha256"] != m_seed2["manifest_sha256"]

    # Astra P2 #8 & Gemini P2-2: W5 per-strategy cross-spoof raises & impossible standalone trade_sessions fails!
    w5_spec = SUITABLE_PLAYBACK_HORIZONS["W5_LIVE_PAPER_PARITY_REPLAY"]
    w5_inp = replace(
        clean,
        horizon_id="W5_LIVE_PAPER_PARITY_REPLAY",
        sample_start_date=w5_spec.start_date,
        sample_max_date=w5_spec.end_date,
        distinct_sessions=60,
        cadence="paper_parity",
    )
    assert (
        audit_low_quality_practices(w5_inp, observed_closed_roundtrips=30, observed_trade_sessions=999)["passed"]
        is False
    )
    with pytest.raises(ValueError, match="exceeds feasible bound"):
        build_advanced_backtest_playback_manifest(
            audit_input=replace(w5_inp, distinct_sessions=10),
            strategy_daily_returns_1x_bps={"A": [17.0, 19.0] * 5, "B": [-11.0, -13.0] * 5},
            strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 5, "B": [-19.0, -21.0] * 5},
            strategy_clusters={"A": "C", "B": "C"},
            closed_roundtrips_by_strategy={"A": 30, "B": 0},
            closed_roundtrip_sessions_by_strategy={"A": 1, "B": 10},
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=2.5,
        )

    # Gemini P2-3: Undated invalid MAE raises unconditionally; 3500x 1e308 finishes rapidly without Decimal overflow!
    with pytest.raises(ValueError, match="mae_pct"):
        build_advanced_backtest_playback_manifest(
            audit_input=clean,
            strategy_daily_returns_1x_bps={"A": r1_good},
            strategy_daily_returns_2x_bps={"A": r2_good},
            strategy_clusters={"A": "C"},
            realized_mae_atr_samples=[(-5.0, -1.0)],
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.10,
        )
    s_3500 = compute_calendar_day_stats([1e308] * 3500)
    assert math.isfinite(float(s_3500["cum_return_pct"] or 0.0))

    # Astra P2 #5, #6, #10, #11, #12, #14, #16, #18, #19, #24 & Gemini P2-1: Exact numerical invariants!
    with pytest.raises(ValueError, match="Pearson"):
        compute_min_trl_sessions(
            target_sharpe_ann=2.0, benchmark_sr_ann=0.0, periods_per_year=1.0, skew=1e-200, excess_kurtosis=-2.0
        )
    assert (
        compute_min_trl_sessions(target_sharpe_ann=5.000000000000001, benchmark_sr_ann=5.0, periods_per_year=5.0) > 5e31
    )
    assert (
        compute_min_trl_sessions(
            target_sharpe_ann=1.0 + 2.0**-29,
            benchmark_sr_ann=1.0 + 2.0**-30,
            periods_per_year=1.0,
            skew=2.0,
            excess_kurtosis=2.0,
        )
        == 3
    )
    assert compute_calendar_day_stats([10000.0, -1e-13] + [0.0] * 250)["calmar"] == pytest.approx(1e17, rel=1e-4)
    assert compute_calendar_day_stats([1e306])["cum_return_pct"] == pytest.approx(1e304, rel=1e-6)
    assert compute_calendar_day_stats([1e308, 1e308, -10000.0])["profit_factor"] == pytest.approx(2e304, rel=1e-6)
    assert compute_calendar_day_stats([100.0])["profit_factor"] is None
    assert compute_calendar_day_stats([1e308, -1e-308])["profit_factor"] is None
    s_p16 = compute_calendar_day_stats([1e200] * 3 + [-9999.0] * 130)
    assert s_p16["cum_return_pct"] == pytest.approx(1e70, rel=1e-4)
    assert compute_calendar_day_stats([1.0, 1.0 - 2.0**-53] * 200)["ar1_rho"] == pytest.approx(-0.9975, abs=1e-4)
    assert math.isfinite(_hlz_t_hurdle_raw(10**310)) and _hlz_t_hurdle_raw(10**310) > 37.0
    assert math.isfinite(_compute_empirical_dsr(r2_good, cumulative_trials_n=10**308, cross_trial_sr_std=0.01))


def test_redteam_pass6_playback_counterexamples_all_closed() -> None:
    """Close ALL Pass-6 Astra (10 P1 + 14 P2) & Gemini (2 P1 + 2 P2) counterexamples in ADR-0030."""
    import math
    from statistics import NormalDist

    clean = replace(_clean_audit_input(), distinct_sessions=400, ar1_rho_for_min_trl=0.0)
    r1_good = [17.0, 19.0] * 200
    r2_good = [9.0, 11.0] * 200
    r2_ctrl = [-19.0, -21.0] * 200
    r1_ctrl = [x + 8.0 for x in r2_ctrl]
    sess_400 = _nyse_dates("2024-01-02", 400)
    clean_ab = replace(clean, active_strategy_ids=("A", "B"))

    # Astra P1 #1: `[5e19, -9999.999999999998]*200` (exact equity = 5.75e-9) MUST fail capital survival & promotion!
    r2_wipe_pair = [5e19, -9999.999999999998] * 200
    r2_b_pair = [-4.0, 2.0] * 200
    m_p1 = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": [x + 16.0 for x in r2_wipe_pair], "B": [x + 16.0 for x in r2_b_pair]},
        strategy_daily_returns_2x_bps={"A": r2_wipe_pair, "B": r2_b_pair},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.15,
    )
    assert m_p1["per_strategy_playback"]["A"]["survives_2x_cost"] is False
    assert m_p1["promotion_eligible"] is False

    # Astra P1 #4: Near-Pearson-boundary DSR variance `(1 - 0.75*SR)^2 = 2.168e-19` NEVER collapses to 0!
    z1 = -NormalDist().inv_cdf(1.0 / 105.0)
    z2 = -NormalDist().inv_cdf(1.0 / (105.0 * math.e))
    e_max_z = (1.0 - 0.5772156649015329) * z1 + 0.5772156649015329 * z2
    l_val, h_val = 0.3340012528585398, 1.3340012528585399
    r_quad = [l_val, l_val, l_val, l_val, h_val] * 80
    m_quad = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in r_quad]},
        strategy_daily_returns_2x_bps={"A": r_quad},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.12 / e_max_z,
    )
    assert m_quad["computed_dsr_2x"] == 1.0
    assert m_quad["computed_dsr_2x_blp2014_iid"] == 1.0
    assert m_quad["computed_dsr_2x_adia19_null_var"] == 1.0

    # Astra P1 #5: Adjacent pairs `I(i%20<2)-0.04414` cleanly separate BLP2014 IID (.9601) from AR1-gated (.8420 < .95)!
    r_adj_pairs = [(1.0 if (i % 20) < 2 else 0.0) - 0.04414 for i in range(400)]
    m_ar1_dsr = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in r_adj_pairs]},
        strategy_daily_returns_2x_bps={"A": r_adj_pairs},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.12 / e_max_z,
    )
    assert m_ar1_dsr["dsr_variance_estimator"] == "AR1_ADJUSTED_OBSERVED_SR_PSD"
    assert m_ar1_dsr["computed_dsr_2x_blp2014_iid"] == pytest.approx(0.9601, abs=1e-4)
    assert m_ar1_dsr["computed_dsr_2x"] == pytest.approx(0.8420, abs=1e-4)
    assert m_ar1_dsr["computed_dsr_2x_adia19_null_var"] == pytest.approx(0.8245, abs=1e-4)
    assert m_ar1_dsr["promotion_eligible"] is False

    # Astra P1 #6 & Gemini P1-2: Appending undeclared C to `A, 2*A` stays PBO=1.0; tiny eps=1e-13 stays 14/15!
    m_undef_c = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": r1_good, "C": r1_ctrl},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_good, "C": r2_ctrl},
        strategy_clusters={"A": "C", "B": "C", "C": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400, "C": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_undef_c["computed_cpcv_pbo_2x"] == 1.0
    assert m_undef_c["promotion_eligible"] is False

    d_offs = [-5.0, -4.0, -3.0, 1.0, 2.0, 9.0]
    a_eps: list[float] = []
    b_eps: list[float] = []
    c_eps: list[float] = []
    for dg in d_offs:
        a_eps.extend([9.0 + 1e-13 * dg, 11.0 + 1e-13 * dg] * 40)
        b_eps.extend([9.0 - 1e-13 * dg, 11.0 - 1e-13 * dg] * 40)
        c_eps.extend([-19.0, -21.0] * 40)
    m_eps_over = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=480, active_strategy_ids=("A", "B", "C")),
        strategy_daily_returns_1x_bps={
            "A": [x + 8.0 for x in a_eps],
            "B": [x + 8.0 for x in b_eps],
            "C": [x + 8.0 for x in c_eps],
        },
        strategy_daily_returns_2x_bps={"A": a_eps, "B": b_eps, "C": c_eps},
        strategy_clusters={"A": "C", "B": "C", "C": "C"},
        session_dates=_nyse_dates("2024-01-02", 480),
        closed_roundtrips_by_strategy={"A": 480, "B": 480, "C": 480},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=5.0,
    )
    assert m_eps_over["computed_cpcv_pbo_2x_sharpe"] == pytest.approx(14.0 / 15.0, abs=1e-3)
    assert m_eps_over["promotion_eligible"] is False

    # Astra P1 #7: CPCV validates BOTH Sharpe AND the HAC-t promotion selector (0/15 Sharpe vs 15/15 HAC -> blocks!)
    def _g_sig(n_len: int, blk: int) -> list[float]:
        return [1.0 if ((i // blk) % 2 == 0) else -1.0 for i in range(n_len)]

    a_hac = ([1.25 + x for x in (_g_sig(20, 1) + _g_sig(10, 5) + _g_sig(20, 1))]) * 6
    b_hac = ([1.0 + x for x in (_g_sig(20, 2) + _g_sig(10, 1) + _g_sig(20, 2))]) * 6
    m_hac_pbo = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean_ab, distinct_sessions=300, target_sharpe_for_min_trl=20.0),
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in a_hac], "B": [x + 8.0 for x in b_hac]},
        strategy_daily_returns_2x_bps={"A": a_hac, "B": b_hac},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=_nyse_dates("2024-01-02", 300),
        closed_roundtrips_by_strategy={"A": 300, "B": 300},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=1.0,
    )
    assert m_hac_pbo["computed_cpcv_pbo_2x_sharpe"] == 0.0
    assert m_hac_pbo["computed_cpcv_pbo_2x_hac"] == 1.0
    assert m_hac_pbo["computed_cpcv_pbo_2x"] == 1.0
    assert m_hac_pbo["promotion_eligible"] is False

    # Astra P1 #8, Gemini P1-1 & Astra P2 #14: Notional boundaries reject 0/-0.1/1.01/NaN/1e-13; zero drag blocks!
    for bad_nf in (0.0, -0.1, 1.01, float("nan"), 1e-13):
        with pytest.raises(ValueError, match="roundtrip_notional_fraction"):
            build_advanced_backtest_playback_manifest(
                audit_input=clean_ab,
                strategy_daily_returns_1x_bps={"A": r2_good, "B": r2_ctrl},
                strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
                strategy_clusters={"A": "C", "B": "C"},
                roundtrip_notional_fraction=bad_nf,
            )
    m_zero_drag = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r2_good, "B": r2_ctrl},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
        roundtrip_notional_fraction=0.01,
    )
    assert m_zero_drag["promotion_eligible"] is False

    # Astra P1 #9: Zero-trade cash control `B_1x=B_2x=[0]*400` with 0 trips owes 0 bps and does NOT block passing A!
    m_cash_ctrl = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [0.0] * 400},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": [0.0] * 400},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 0},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_cash_ctrl["promotion_eligible"] is True

    # Astra P1 #10: Stuffing 3200 bps into day 0 across 400 declared closing sessions fails cost reconciliation!
    r1_lump = [r2_good[0] + 3200.0] + list(r2_good[1:])
    m_lump_ts = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_lump, "B": r1_ctrl},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400},
        closed_roundtrip_sessions_by_strategy={"A": 400, "B": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_lump_ts["promotion_eligible"] is False
    m_lump_daily = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_lump, "B": r1_ctrl},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        daily_roundtrips_by_strategy={"A": [1] * 400, "B": [1] * 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_lump_daily["promotion_eligible"] is False

    # Astra P2 #1: Genuinely different Sharpes `A` vs `A - 2^-40` are NEVER merged (`PBO == 0.0`, NOT `1.0`)!
    b_minus_tiny = [x - 2.0**-40 for x in r2_good]
    m_distinct_sr = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [x + 8.0 for x in b_minus_tiny]},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": b_minus_tiny},
        strategy_clusters={"A": "C", "B": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_distinct_sr["computed_cpcv_pbo_2x_sharpe"] == 0.0

    # Astra P2 #2 & Gemini P2-1: Public manifest builder handles `N = 10**14` and `10**310` with 2 finite strategies!
    for huge_n in (10**14, 10**310):
        m_huge_n = build_advanced_backtest_playback_manifest(
            audit_input=clean_ab,
            strategy_daily_returns_1x_bps={"A": r1_good, "B": r1_ctrl},
            strategy_daily_returns_2x_bps={"A": r2_good, "B": r2_ctrl},
            strategy_clusters={"A": "C", "B": "C"},
            cumulative_trials_n=huge_n,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=2.5,
        )
        assert math.isfinite(m_huge_n["required_hlz_t_hurdle"])

    # Astra P2 #3 & Gemini P2-2: Public manifest builder handles `[1e308]*10` without `math.fsum` OverflowError!
    m_1e308 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=10),
        strategy_daily_returns_1x_bps={"A": [1e308] * 10},
        strategy_daily_returns_2x_bps={"A": [1e308] * 10},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_1e308["hierarchical_eb_merit_pool"]["A"]["observed_mean_r"] == pytest.approx(1e306, rel=1e-6)

    # Astra P2 #4, #5, #13: `1e-80` Calmar == `1e84`, `[1e155, 2e158]` == `2e307%`, distinct `sharpe_status`!
    assert compute_calendar_day_stats([10000.0, -1e-80] + [0.0] * 250)["calmar"] == pytest.approx(1e84, rel=1e-4)
    assert compute_calendar_day_stats([1e155, 2e158])["cum_return_pct"] == pytest.approx(2e307, rel=1e-4)
    assert compute_calendar_day_stats([-18.0] * 400)["sharpe_status"] == "ZERO_VARIANCE_NEGATIVE"
    assert compute_calendar_day_stats([0.0] * 400)["sharpe_status"] == "ZERO_VARIANCE_CASH"
    assert compute_calendar_day_stats([18.0] * 400)["sharpe_status"] == "ZERO_VARIANCE_POSITIVE"

    # Astra P2 #6: Changing `cumulative_trials_n` OR `strategy_clusters` changes `input_evidence_sha256`!
    m_ev_base = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    m_ev_n106 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=106,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    m_ev_cl_d = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": r1_good},
        strategy_daily_returns_2x_bps={"A": r2_good},
        strategy_clusters={"A": "D"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    ev0 = m_ev_base["observed_sample_span"]["input_evidence_sha256"]
    assert ev0 != m_ev_n106["observed_sample_span"]["input_evidence_sha256"]
    assert ev0 != m_ev_cl_d["observed_sample_span"]["input_evidence_sha256"]

    # Astra P2 #7 & #8: W5 disjoint daily closes pass (union=10); positive closes with 0 dates raises ValueError!
    w5_spec = SUITABLE_PLAYBACK_HORIZONS["W5_LIVE_PAPER_PARITY_REPLAY"]
    w5_10 = replace(
        clean_ab,
        horizon_id="W5_LIVE_PAPER_PARITY_REPLAY",
        sample_start_date=w5_spec.start_date,
        sample_max_date=w5_spec.end_date,
        distinct_sessions=10,
        cadence="paper_parity",
    )
    m_w5_disjoint = build_advanced_backtest_playback_manifest(
        audit_input=w5_10,
        strategy_daily_returns_1x_bps={"A": [33.0, 35.0] * 5, "B": [33.0, 35.0] * 5},
        strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 5, "B": [9.0, 11.0] * 5},
        strategy_clusters={"A": "C", "B": "C"},
        daily_roundtrips_by_strategy={"A": [3] * 5 + [0] * 5, "B": [0] * 5 + [3] * 5},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.50,
    )
    assert m_w5_disjoint["low_quality_practice_audit"]["passed"] is True
    with pytest.raises(ValueError, match="impossible for 20 closed roundtrips"):
        build_advanced_backtest_playback_manifest(
            audit_input=w5_10,
            strategy_daily_returns_1x_bps={"A": [17.0, 19.0] * 5, "B": [17.0, 19.0] * 5},
            strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 5, "B": [9.0, 11.0] * 5},
            strategy_clusters={"A": "C", "B": "C"},
            closed_roundtrips_by_strategy={"A": 20, "B": 10},
            closed_roundtrip_sessions_by_strategy={"A": 0, "B": 10},
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
            cross_trial_sr_std=0.50,
        )


def test_redteam_pass7_playback_counterexamples_all_closed() -> None:
    """Close ALL Pass-7 Astra (P1 #2-#4 + P2 #1-#3/#8) & Gemini (P1-2 + P2-2/P2-3) counterexamples in ADR-0030."""
    import math
    from statistics import NormalDist

    clean = replace(_clean_audit_input(), distinct_sessions=400, ar1_rho_for_min_trl=0.0)
    r1_good = [17.0, 19.0] * 200
    r2_good = [9.0, 11.0] * 200
    sess_400 = _nyse_dates("2024-01-02", 400)
    clean_ab = replace(clean, active_strategy_ids=("A", "B"))

    # Astra P1 #2: CPCV preserves original session distances for HAC lags -> 11/15 (NOT 2/15!) and blocks!
    a_dist = [12.0, -4.0] * 150
    b_dist = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0] * 50
    m_dist = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean_ab, distinct_sessions=300, target_sharpe_for_min_trl=20.0),
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in a_dist], "B": [x + 8.0 for x in b_dist]},
        strategy_daily_returns_2x_bps={"A": a_dist, "B": b_dist},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=_nyse_dates("2024-01-02", 300),
        closed_roundtrips_by_strategy={"A": 300, "B": 300},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.50,
    )
    assert m_dist["computed_cpcv_pbo_2x_sharpe"] == 0.0
    assert m_dist["computed_cpcv_pbo_2x_hac"] == pytest.approx(11.0 / 15.0, abs=1e-4)
    assert m_dist["promotion_eligible"] is False

    # Astra P1 #3: 600 declared trips across 400 days of 12 bps drag (floor(12/8)=1 -> 400 < 600) blocks!
    m_unfunded = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": [x + 12.0 for x in r2_good], "B": [0.0] * 400},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": [0.0] * 400},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 600, "B": 0},
        closed_roundtrip_sessions_by_strategy={"A": 400, "B": 0},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_unfunded["promotion_eligible"] is False

    # Gemini P1-2: Non-zero active strategy B (`r1 == r2`) declaring `rt=0` fails cost reconciliation!
    m_active_rt0 = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [-8.0, -12.0] * 200},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": [-8.0, -12.0] * 200},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 0},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_active_rt0["promotion_eligible"] is False

    # Astra P1 #4: Undefined raw ADIA-null variance (`-0.2058 <= 0`) publishes `None` (NEVER `1.0`!)
    z1 = -NormalDist().inv_cdf(1.0 / 105.0)
    z2 = -NormalDist().inv_cdf(1.0 / (105.0 * math.e))
    e_max_z = (1.0 - 0.5772156649015329) * z1 + 0.5772156649015329 * z2
    r_neg_adia = [(1.0 if (i % 20) < 2 else 0.0) + 0.55 for i in range(400)]
    m_neg_adia = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in r_neg_adia]},
        strategy_daily_returns_2x_bps={"A": r_neg_adia},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=1.0 / e_max_z,
    )
    assert m_neg_adia["computed_dsr_2x_adia19_null_var"] is None

    # Astra P2 #1: Non-stationary `rho = 0.9925` keeps IID diagnostic at `1.0` while gating AR(1) at `0.0`!
    r_hi_ar = [9.0] * 200 + [11.0] * 200
    m_hi_ar = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in r_hi_ar]},
        strategy_daily_returns_2x_bps={"A": r_hi_ar},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.12 / e_max_z,
    )
    assert m_hi_ar["computed_dsr_2x_blp2014_iid"] == 1.0
    assert m_hi_ar["computed_dsr_2x"] == 0.0
    assert m_hi_ar["computed_dsr_2x_adia19_null_var"] is None

    # Astra P2 #2 & #3: Tiny net-gain Calmar == `1.0`; subnormal `[0, 2^-1074]*200` has `FINITE` Sharpe `15.8546`!
    assert compute_calendar_day_stats([2e-80, -1e-80] + [0.0] * 250)["calmar"] == pytest.approx(1.0, abs=1e-4)
    s_sub = compute_calendar_day_stats([0.0, 2.0**-1074] * 200)
    assert s_sub["sharpe_status"] == "FINITE"
    assert s_sub["sharpe_ann_raw"] == pytest.approx(15.8546, abs=1e-3)

    # Astra P2 #8: Isolated extra-roster test: A + 2 flat cash controls passes with ("A","B","C"), blocks on ("A","B")!
    m_roster_ok = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, active_strategy_ids=("A", "B", "C")),
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [0.0] * 400, "C": [0.0] * 400},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": [0.0] * 400, "C": [0.0] * 400},
        strategy_clusters={"A": "C", "B": "C", "C": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 0, "C": 0},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    m_roster_extra = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, active_strategy_ids=("A", "B")),
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [0.0] * 400, "C": [0.0] * 400},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": [0.0] * 400, "C": [0.0] * 400},
        strategy_clusters={"A": "C", "B": "C", "C": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 0, "C": 0},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_roster_ok["computed_cpcv_pbo_2x"] == 0.0
    assert m_roster_ok["promotion_eligible"] is True
    assert m_roster_extra["computed_cpcv_pbo_2x"] == 0.0
    assert m_roster_extra["promotion_eligible"] is False

    # Gemini P2-2 & P2-3: Huge roundtrip int `10**309` fails cost gate cleanly; `1e308 / 0.5` raises clean ValueError!
    m_rt_huge = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=10),
        strategy_daily_returns_1x_bps={"A": [17.0, 19.0] * 5},
        strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 5},
        strategy_clusters={"A": "C"},
        closed_roundtrips_by_strategy={"A": 10**309},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_rt_huge["promotion_eligible"] is False
    m_drt_huge = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=10),
        strategy_daily_returns_1x_bps={"A": [17.0, 19.0] * 5},
        strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 5},
        strategy_clusters={"A": "C"},
        daily_roundtrips_by_strategy={"A": [10**309] + [0] * 9},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_drt_huge["promotion_eligible"] is False
    with pytest.raises(ValueError, match="overflows finite float"):
        build_advanced_backtest_playback_manifest(
            audit_input=replace(clean, distinct_sessions=10),
            strategy_daily_returns_1x_bps={"A": [1e308] * 10},
            strategy_daily_returns_2x_bps={"A": [1e308] * 10},
            strategy_clusters={"A": "C"},
            risk_unit_bps=0.5,
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
        )


def test_redteam_pass8_playback_counterexamples_all_closed() -> None:
    """Close ALL Pass-8 Astra (P1 #1-#3/#6 + P2 #1/#2/#6/#7/#9) & Gemini (P1-1) counterexamples in ADR-0030."""
    import math

    clean = replace(_clean_audit_input(), distinct_sessions=400, ar1_rho_for_min_trl=0.0)
    r1_good = [17.0, 19.0] * 200
    r2_good = [9.0, 11.0] * 200
    sess_400 = _nyse_dates("2024-01-02", 400)
    clean_ab = replace(clean, active_strategy_ids=("A", "B"))

    # Astra P1 #1 & Gemini P1-1: Exact Decimal subtraction blocks `+1e308 - (-1e308)` funding `10**309` trips
    # AND eliminates `DivisionImpossible` / `DivisionByZero` / `OverflowError` on extreme costs & returns!
    m_inf_drag = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [1e308] + [0.0] * 399},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": [-1e308] + [0.0] * 399},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        daily_roundtrips_by_strategy={"A": [1] * 400, "B": [10**309] + [0] * 399},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_inf_drag["promotion_eligible"] is False

    for bad_cost, nf_v, r1_x, r2_x in (
        (1e-80, 1.0, [18.0, 20.0] * 5, [10.0, 12.0] * 5),
        (2.0**-1074, 0.01, [18.0, 20.0] * 5, [10.0, 12.0] * 5),
        (8.0, 1.0, [1e100] * 10, [0.0] * 10),
        (8.0, 1.0, [1e308] * 10, [-1e308] * 10),
    ):
        m_ext = build_advanced_backtest_playback_manifest(
            audit_input=replace(clean, distinct_sessions=10, cost_1x_rt_bps=bad_cost),
            strategy_daily_returns_1x_bps={"A": r1_x},
            strategy_daily_returns_2x_bps={"A": r2_x},
            strategy_clusters={"A": "C"},
            closed_roundtrips_by_strategy={"A": 10},
            roundtrip_notional_fraction=nf_v,
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
        )
        assert len(m_ext["manifest_sha256"]) == 64

    # Astra P1 #2: Post-bankruptcy sessions NEVER credit zombie trades!
    m_zombie = build_advanced_backtest_playback_manifest(
        audit_input=clean_ab,
        strategy_daily_returns_1x_bps={"A": r1_good, "B": [-10000.0] + [8.0] * 399},
        strategy_daily_returns_2x_bps={"A": r2_good, "B": [-10008.0] + [0.0] * 399},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=sess_400,
        daily_roundtrips_by_strategy={"A": [1] * 400, "B": [1] * 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=2.5,
    )
    assert m_zombie["promotion_eligible"] is False

    # Astra P1 #3: Duplicating declared flat cash controls NEVER lowers HAC PBO from 11/15 to 0/15!
    a_dist = [12.0, -4.0] * 150
    b_dist = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0] * 50
    m_dup_cash = build_advanced_backtest_playback_manifest(
        audit_input=replace(
            clean, distinct_sessions=300, target_sharpe_for_min_trl=20.0, active_strategy_ids=("A", "B", "C1", "C2")
        ),
        strategy_daily_returns_1x_bps={
            "A": [x + 8.0 for x in a_dist],
            "B": [x + 8.0 for x in b_dist],
            "C1": [0.0] * 300,
            "C2": [0.0] * 300,
        },
        strategy_daily_returns_2x_bps={"A": a_dist, "B": b_dist, "C1": [0.0] * 300, "C2": [0.0] * 300},
        strategy_clusters={"A": "C", "B": "C", "C1": "C", "C2": "C"},
        session_dates=_nyse_dates("2024-01-02", 300),
        closed_roundtrips_by_strategy={"A": 300, "B": 300, "C1": 0, "C2": 0},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
    )
    assert m_dup_cash["computed_cpcv_pbo_2x_hac"] == pytest.approx(11.0 / 15.0, abs=1e-4)
    assert m_dup_cash["promotion_eligible"] is False

    # Astra P1 #6: Subnormal profitable series `[0, 2^-1074]*200` with `risk_unit_bps=2^-1074` survives & promotes!
    m_sub = 2.0**-1074
    m_sub_win = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, active_strategy_ids=("A", "B", "C")),
        strategy_daily_returns_1x_bps={"A": [8.0] * 400, "B": [0.0] * 400, "C": [0.0] * 400},
        strategy_daily_returns_2x_bps={"A": [0.0, m_sub] * 200, "B": [0.0] * 400, "C": [0.0] * 400},
        strategy_clusters={"A": "C", "B": "C", "C": "C"},
        session_dates=sess_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 0, "C": 0},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
        risk_unit_bps=m_sub,
    )
    assert m_sub_win["per_strategy_playback"]["A"]["survives_2x_cost"] is True
    assert m_sub_win["hierarchical_eb_merit_pool"]["A"]["observed_mean_r"] == pytest.approx(0.5, abs=1e-4)
    assert m_sub_win["hierarchical_eb_merit_pool"]["A"]["posterior_prob_positive"] == 1.0
    assert m_sub_win["promotion_eligible"] is True

    # Astra P2 #2, #6, #7, #9: `10**5000` JSON-safe seal, finite `2.5e307 R` SE, `1e308` ADIA-null, singleton Calmar!
    m_5000 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=10),
        strategy_daily_returns_1x_bps={"A": [17.0, 19.0] * 5},
        strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 5},
        strategy_clusters={"A": "C"},
        closed_roundtrips_by_strategy={"A": 10**5000},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert len(m_5000["manifest_sha256"]) == 64
    assert m_5000["promotion_eligible"] is False

    m_fin_se = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [1e308] + [8.0] * 399},
        strategy_daily_returns_2x_bps={"A": [1e308] + [0.0] * 399},
        strategy_clusters={"A": "C"},
        risk_unit_bps=0.01,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_fin_se["hierarchical_eb_merit_pool"]["A"]["observed_mean_r"] == pytest.approx(2.5e307, rel=1e-6)

    m_adia_big = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [9.0] + [8.0] * 399},
        strategy_daily_returns_2x_bps={"A": [1.0] + [0.0] * 399},
        strategy_clusters={"A": "C"},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
        cross_trial_sr_std=1e308,
    )
    assert m_adia_big["computed_dsr_2x_adia19_null_var"] == pytest.approx(0.0225, abs=1e-4)

    assert compute_calendar_day_stats([-100.0])["calmar"] == pytest.approx(-92.0554, abs=1e-3)
    assert compute_calendar_day_stats([-10000.0])["calmar"] == pytest.approx(-1.0, abs=1e-4)
    assert math.isfinite(compute_calendar_day_stats([100.0])["calmar"])


def test_redteam_pass9_playback_counterexamples_all_closed() -> None:
    """Close all Pass-9 Astra (5 P1 + 5 P2) & Falsifier (1 P2) counterexamples in ADR-0030."""
    import json

    from sota_compounding.backtest_playback_manifest import _compute_empirical_cpcv_pbo

    w1_dates = _nyse_dates("2024-01-02", 400)
    clean = replace(
        _clean_audit_input(),
        sample_start_date=w1_dates[0],
        sample_max_date=w1_dates[-1],
        distinct_sessions=400,
        ar1_rho_for_min_trl=0.0,
        active_strategy_ids=("A",),
    )

    # Astra P1 #1: W5 parity audit rejects post-bankruptcy closes `[0]+[3]*10` after session-0 wipeout!
    w5_spec = SUITABLE_PLAYBACK_HORIZONS["W5_LIVE_PAPER_PARITY_REPLAY"]
    w5_11 = replace(
        clean,
        horizon_id="W5_LIVE_PAPER_PARITY_REPLAY",
        sample_start_date=w5_spec.start_date,
        sample_max_date=w5_spec.end_date,
        distinct_sessions=11,
        cadence="paper_parity",
    )
    m_w5_dead = build_advanced_backtest_playback_manifest(
        audit_input=w5_11,
        strategy_daily_returns_1x_bps={"A": [-10000.0] + [24.0] * 10},
        strategy_daily_returns_2x_bps={"A": [-10000.0] + [0.0] * 10},
        strategy_clusters={"A": "C"},
        daily_roundtrips_by_strategy={"A": [0] + [3] * 10},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_w5_dead["low_quality_practice_audit"]["passed"] is False
    # Astra Pass-10 partial gem: leading entry-only `4 bps` session cannot certify a 10th closing session in W5!
    w5_10_lead = replace(w5_11, distinct_sessions=10)
    m_w5_lead = build_advanced_backtest_playback_manifest(
        audit_input=w5_10_lead,
        strategy_daily_returns_1x_bps={"A": [14.0] + [38.0] * 9},
        strategy_daily_returns_2x_bps={"A": [10.0] * 10},
        strategy_clusters={"A": "C"},
        closed_roundtrips_by_strategy={"A": 30},
        closed_roundtrip_sessions_by_strategy={"A": 10},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_w5_lead["low_quality_practice_audit"]["passed"] is False

    # Astra P1 #2: Funded multi-session swing (`4 bps` entry + `4 bps` exit across 1260 W2 sessions) PASSES!
    w2_dates = _nyse_dates("2020-01-02", 1260)
    w2_inp = replace(
        clean,
        horizon_id="W2_MULTI_REGIME_SWING_STRESS",
        sample_start_date=w2_dates[0],
        sample_max_date=w2_dates[-1],
        distinct_sessions=1260,
        cadence="swing",
        active_strategy_ids=("A", "B"),
    )
    r1_sw = [13.0, 15.0] * 630
    r2_sw = [9.0, 11.0] * 630
    m_swing = build_advanced_backtest_playback_manifest(
        audit_input=w2_inp,
        strategy_daily_returns_1x_bps={"A": r1_sw, "B": [0.0] * 1260},
        strategy_daily_returns_2x_bps={"A": r2_sw, "B": [0.0] * 1260},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=w2_dates,
        daily_roundtrips_by_strategy={"A": [0, 1] * 630, "B": [0] * 1260},
        cumulative_trials_n=105,
        cpcv_pbo=0.05,
        dsr_probability=0.98,
        cross_trial_sr_std=1.5,
    )
    assert m_swing["promotion_eligible"] is True
    # Aggregate-count swing path (`closed_roundtrips=630` without daily breakdown) ALSO passes!
    m_swing_agg = build_advanced_backtest_playback_manifest(
        audit_input=w2_inp,
        strategy_daily_returns_1x_bps={"A": r1_sw, "B": [0.0] * 1260},
        strategy_daily_returns_2x_bps={"A": r2_sw, "B": [0.0] * 1260},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=w2_dates,
        closed_roundtrips_by_strategy={"A": 630, "B": 0},
        cumulative_trials_n=105,
        cpcv_pbo=0.05,
        dsr_probability=0.98,
        cross_trial_sr_std=1.5,
    )
    assert m_swing_agg["promotion_eligible"] is True
    # Causal check: closing BEFORE any entry leg (`[1, 0]` on `[4, 4]`) still fails!
    m_swing_acausal = build_advanced_backtest_playback_manifest(
        audit_input=w2_inp,
        strategy_daily_returns_1x_bps={"A": r1_sw, "B": [0.0] * 1260},
        strategy_daily_returns_2x_bps={"A": r2_sw, "B": [0.0] * 1260},
        strategy_clusters={"A": "C", "B": "C"},
        session_dates=w2_dates,
        daily_roundtrips_by_strategy={"A": [1, 0] * 630, "B": [0] * 1260},
        cumulative_trials_n=105,
        cpcv_pbo=0.05,
        dsr_probability=0.98,
        cross_trial_sr_std=1.5,
    )
    assert m_swing_acausal["promotion_eligible"] is False

    # Astra P1 #3: Duplicating a non-cash control `L=[-1, 1]*150` NEVER lowers HAC PBO from 11/15 to 0/15!
    s_a = [12.0, -4.0] * 150
    s_b = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0] * 50
    s_l = [-1.0, 1.0] * 150
    pbo_abl = _compute_empirical_cpcv_pbo({"A": s_a, "B": s_b, "L": s_l}, purge_sessions=20, selector="hac_t")
    pbo_abll = _compute_empirical_cpcv_pbo(
        {"A": s_a, "B": s_b, "L": s_l, "L2": list(s_l)}, purge_sessions=20, selector="hac_t"
    )
    pbo_ab_scaled = _compute_empirical_cpcv_pbo(
        {"A": s_a, "B": s_b, "L": s_l, "L2": [2.0 * x for x in s_l]}, purge_sessions=20, selector="hac_t"
    )
    assert pbo_abl == pytest.approx(11.0 / 15.0, abs=1e-9)
    assert pbo_abll == pbo_abl
    assert pbo_ab_scaled == pbo_abl
    assert (
        _compute_empirical_cpcv_pbo(
            {"A": s_a, "A2": [2.0 * x for x in s_a], "L": s_l}, purge_sessions=20, selector="hac_t"
        )
        == 1.0
    )

    # Astra P1 #4: Inexact integer marks `2**56+9` vs `2**56+8` fail closed at intake (no fake trip / digest alias)!
    with pytest.raises(ValueError, match="binary64"):
        build_advanced_backtest_playback_manifest(
            audit_input=replace(clean, distinct_sessions=10),
            strategy_daily_returns_1x_bps={"A": [2**56 + 9] * 10},
            strategy_daily_returns_2x_bps={"A": [2**56 + 8] * 10},
            strategy_clusters={"A": "C"},
            closed_roundtrips_by_strategy={"A": 10},
            cumulative_trials_n=105,
            cpcv_pbo=0.10,
            dsr_probability=0.98,
        )

    # Astra P1 #5: `[0, 2^-1074]*200` with default `risk_unit_bps=100` preserves exact EB `posterior_prob_positive=1.0`!
    m_sub = 2.0**-1074
    m_sub_def = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [8.0, 8.0 + m_sub] * 200},
        strategy_daily_returns_2x_bps={"A": [0.0, m_sub] * 200},
        strategy_clusters={"A": "C"},
        session_dates=w1_dates,
        closed_roundtrips_by_strategy={"A": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.05,
        dsr_probability=0.98,
        cross_trial_sr_std=0.10,
        risk_unit_bps=100.0,
    )
    assert m_sub_def["hierarchical_eb_merit_pool"]["A"]["posterior_prob_positive"] == 1.0
    assert m_sub_def["per_strategy_playback"]["A"]["survives_2x_cost"] is True

    # Astra P2 #1 & Falsifier P2-1: Returned manifest is directly `json.dumps`-safe (`10**5000`) & `10**1_000_000` ok!
    m_5000 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=10),
        strategy_daily_returns_1x_bps={"A": [17.0, 19.0] * 5},
        strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 5},
        strategy_clusters={"A": "C"},
        closed_roundtrips_by_strategy={"A": 10**5000},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert len(json.dumps(m_5000, allow_nan=False)) > 100
    m_mill = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=10),
        strategy_daily_returns_1x_bps={"A": [17.0, 19.0] * 5},
        strategy_daily_returns_2x_bps={"A": [9.0, 11.0] * 5},
        strategy_clusters={"A": "C"},
        closed_roundtrips_by_strategy={"A": 10**1_000_000},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_mill["promotion_eligible"] is False

    # Astra P2 #2: Manifest EB mean is permutation-invariant across `[2^-300, 2^-600, -2^-300] + [0]*7`!
    ord1 = [2.0**-300, 2.0**-600, -(2.0**-300)] + [0.0] * 7
    ord2 = [2.0**-300, -(2.0**-300), 2.0**-600] + [0.0] * 7
    m_o1 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=10),
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in ord1]},
        strategy_daily_returns_2x_bps={"A": ord1},
        strategy_clusters={"A": "C"},
        risk_unit_bps=2.0**-600,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    m_o2 = build_advanced_backtest_playback_manifest(
        audit_input=replace(clean, distinct_sessions=10),
        strategy_daily_returns_1x_bps={"A": [x + 8.0 for x in ord2]},
        strategy_daily_returns_2x_bps={"A": ord2},
        strategy_clusters={"A": "C"},
        risk_unit_bps=2.0**-600,
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_o1["hierarchical_eb_merit_pool"]["A"]["observed_mean_r"] == pytest.approx(0.1, abs=1e-4)
    assert m_o2["hierarchical_eb_merit_pool"]["A"]["observed_mean_r"] == pytest.approx(0.1, abs=1e-4)

    # Astra P2 #3: Whole-trip tolerance NEVER creates a free trip at `10**12 + 1` with exactly `8 * 10**12` bps drag!
    m_1e12 = build_advanced_backtest_playback_manifest(
        audit_input=clean,
        strategy_daily_returns_1x_bps={"A": [8.0 * 10**12 + 10.0] + [10.0, 12.0] * 199 + [10.0]},
        strategy_daily_returns_2x_bps={"A": [10.0] + [10.0, 12.0] * 199 + [10.0]},
        strategy_clusters={"A": "C"},
        session_dates=w1_dates,
        closed_roundtrips_by_strategy={"A": 10**12 + 1},
        cumulative_trials_n=105,
        cpcv_pbo=0.05,
        dsr_probability=0.98,
        cross_trial_sr_std=0.15,
    )
    assert m_1e12["promotion_eligible"] is False


def test_redteam_pass10_claude_counterexamples_all_closed() -> None:
    """Close Pass-10 Claude Opus 5.5 P1-A (CPCV dominated-control padding) & P2 #1-#3 permanently."""
    from sota_compounding.backtest_playback_manifest import _compute_empirical_cpcv_pbo

    # P1-A direct: ALL dominated padding variants preserve exact 11/15 HAC PBO!
    s_a = [12.0, -4.0] * 150
    s_b = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0] * 50
    s_l = [-1.0, 1.0] * 150
    pbo_base = _compute_empirical_cpcv_pbo({"A": s_a, "B": s_b, "L": s_l}, purge_sessions=20, selector="hac_t")
    assert pbo_base == pytest.approx(11.0 / 15.0, abs=1e-9)

    # Variant 1: 6 controls perturbed by 2^-40
    panel_eps = {"A": s_a, "B": s_b, "L": s_l}
    for idx in range(1, 7):
        pert = list(s_l)
        pert[0] += idx * (2.0**-40)
        panel_eps[f"LP{idx}"] = pert
    assert _compute_empirical_cpcv_pbo(panel_eps, purge_sessions=20, selector="hac_t") == pbo_base

    # Variant 2: strictly dominated shifted controls `L - 0.001`, `L - 0.002`
    panel_shift = {
        "A": s_a,
        "B": s_b,
        "L": s_l,
        "LM1": [x - 0.001 for x in s_l],
        "LM2": [x - 0.002 for x in s_l],
    }
    assert _compute_empirical_cpcv_pbo(panel_shift, purge_sessions=20, selector="hac_t") == pbo_base

    # Variant 3: 6 weak positive-drift controls (both IS > 0 and OOS > 0, strictly weaker HAC-t than A & B)
    panel_pos = {"A": s_a, "B": s_b, "L": s_l}
    for idx in range(1, 7):
        panel_pos[f"POS{idx}"] = [1.0 + 0.01 * idx, -1.0 + 0.02 * idx] * 150
    assert _compute_empirical_cpcv_pbo(panel_pos, purge_sessions=20, selector="hac_t") == pbo_base

    # End-to-end W1 manifest: dominated padding NEVER flips promotion_eligible to True!
    w1_400 = _nyse_dates("2024-01-02", 400)
    a_400 = [12.0, -4.0] * 200
    b_400 = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0] * 66 + [2.0, 2.0, 2.0, 0.0]
    l_400 = [-1.0, 1.0] * 200
    padded_2x: dict[str, list[float]] = {
        "A": a_400,
        "B": b_400,
        "L": l_400,
        "LM1": [x - 0.001 for x in l_400],
        "LM2": [x - 0.002 for x in l_400],
    }
    for idx in range(1, 7):
        pert4 = list(l_400)
        pert4[0] += idx * (2.0**-40)
        padded_2x[f"LP{idx}"] = pert4
    padded_1x = {k: [x + 8.0 for x in v] for k, v in padded_2x.items()}
    w1_pad = replace(
        _clean_audit_input(),
        sample_start_date=w1_400[0],
        sample_max_date=w1_400[-1],
        distinct_sessions=400,
        ar1_rho_for_min_trl=0.0,
        active_strategy_ids=tuple(sorted(padded_2x.keys())),
    )
    m_pad = build_advanced_backtest_playback_manifest(
        audit_input=w1_pad,
        strategy_daily_returns_1x_bps=padded_1x,
        strategy_daily_returns_2x_bps=padded_2x,
        strategy_clusters={k: "C" for k in padded_2x},
        session_dates=w1_400,
        closed_roundtrips_by_strategy={k: 400 for k in padded_2x},
        cumulative_trials_n=105,
        cpcv_pbo=0.05,
        dsr_probability=0.98,
        cross_trial_sr_std=0.15,
    )
    assert m_pad["effective_cpcv_pbo"] >= 0.50
    assert m_pad["promotion_eligible"] is False

    # P2 #1 & P2 #2: Exact backward max-heap rejects `(1,4,2,1)` for `rt=4, ts=2` & exposes `all_strategies_cost_ok`!
    w5_spec = SUITABLE_PLAYBACK_HORIZONS["W5_LIVE_PAPER_PARITY_REPLAY"]
    w5_10 = replace(
        _clean_audit_input(),
        horizon_id="W5_LIVE_PAPER_PARITY_REPLAY",
        sample_start_date=w5_spec.start_date,
        sample_max_date=w5_spec.end_date,
        distinct_sessions=10,
        cadence="paper_parity",
        active_strategy_ids=("A",),
    )
    m_bad_ts = build_advanced_backtest_playback_manifest(
        audit_input=w5_10,
        strategy_daily_returns_1x_bps={"A": [14.0, 26.0, 18.0, 14.0] + [10.0] * 6},
        strategy_daily_returns_2x_bps={"A": [10.0] * 10},
        strategy_clusters={"A": "C"},
        closed_roundtrips_by_strategy={"A": 4},
        closed_roundtrip_sessions_by_strategy={"A": 2},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_bad_ts["all_strategies_cost_ok"] is False
    m_ok_ts = build_advanced_backtest_playback_manifest(
        audit_input=w5_10,
        strategy_daily_returns_1x_bps={"A": [14.0, 26.0, 18.0, 14.0] + [10.0] * 6},
        strategy_daily_returns_2x_bps={"A": [10.0] * 10},
        strategy_clusters={"A": "C"},
        closed_roundtrips_by_strategy={"A": 4},
        closed_roundtrip_sessions_by_strategy={"A": 3},
        cumulative_trials_n=105,
        cpcv_pbo=0.10,
        dsr_probability=0.98,
    )
    assert m_ok_ts["all_strategies_cost_ok"] is True

    # P2 #3: Huge `10**5000` audit fields (incl. cadence/horizon_id/edgar) record clean violations without crash!
    big_audit = replace(
        _clean_audit_input(),
        distinct_sessions=10**5000,
        max_stop_pct=10**5000,  # type: ignore[arg-type]
        vol_max_leverage=10**5000,  # type: ignore[arg-type]
        cadence=10**5000,  # type: ignore[arg-type]
        horizon_id=10**5000,  # type: ignore[arg-type]
        edgar_timestamp_field=10**5000,  # type: ignore[arg-type]
    )
    res_big = audit_low_quality_practices(big_audit, observed_sessions=10)
    assert res_big["passed"] is False
    assert any("ADR0022_STOP_OVER_2PCT" in v for v in res_big["violations"])
    assert any("VOL_TIMING_LEVERAGE_ABOVE_1X" in v for v in res_big["violations"])
    assert any("UNKNOWN_HORIZON_ID" in v for v in res_big["violations"])
    assert any("invalid_cadence" in v for v in res_big["violations"])

    # Pass-11 Claude P1-A split-regime `{A, B, C}` vs `{A, B, C, CM1}` (6/15 = 0.40 preserved end-to-end!):
    a_400_r = [3.5 if i % 2 == 0 else 1.5 for i in range(400)]
    b_400_r = [3.5 if i % 2 == 0 else 2.5 for i in range(200)] + [1.0 if i % 2 == 0 else -1.0 for i in range(200)]
    c_400_r = [1.0 if i % 2 == 0 else -1.0 for i in range(200)] + [3.5 if i % 2 == 0 else 2.5 for i in range(200)]
    cm1_400_r = [x - 0.001 for x in c_400_r]
    for sel in ("sharpe", "hac_t"):
        p_abc = _compute_empirical_cpcv_pbo({"A": a_400_r, "B": b_400_r, "C": c_400_r}, purge_sessions=20, selector=sel)
        p_abcm = _compute_empirical_cpcv_pbo(
            {"A": a_400_r, "B": b_400_r, "C": c_400_r, "CM1": cm1_400_r}, purge_sessions=20, selector=sel
        )
        assert p_abc is not None and p_abc == pytest.approx(6.0 / 15.0, abs=1e-9)
        assert p_abcm is not None and p_abcm >= p_abc
    m_abcm = build_advanced_backtest_playback_manifest(
        audit_input=replace(w1_pad, active_strategy_ids=("A", "B", "C", "CM1")),
        strategy_daily_returns_1x_bps={
            "A": [x + 8.0 for x in a_400_r],
            "B": [x + 8.0 for x in b_400_r],
            "C": [x + 8.0 for x in c_400_r],
            "CM1": [x + 8.0 for x in cm1_400_r],
        },
        strategy_daily_returns_2x_bps={"A": a_400_r, "B": b_400_r, "C": c_400_r, "CM1": cm1_400_r},
        strategy_clusters={"A": "C", "B": "C", "C": "C", "CM1": "C"},
        session_dates=w1_400,
        closed_roundtrips_by_strategy={"A": 400, "B": 400, "C": 400, "CM1": 400},
        cumulative_trials_n=105,
        cpcv_pbo=0.05,
        dsr_probability=0.98,
        cross_trial_sr_std=0.25,
    )
    assert m_abcm["effective_cpcv_pbo"] >= 0.40
    assert m_abcm["promotion_eligible"] is False
