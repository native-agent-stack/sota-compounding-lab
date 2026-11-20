"""Tests for ADR-0029 Advanced Sim-to-Merit Convergence (SPA/MCS, Conformal PID, Hedge, Hierarchical EB)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sota_compounding.advanced_merit_sim_convergence import (
    NEXT_HLZ_T_STAR,
    NEXT_WAVE_STRATEGY_BLUEPRINTS,
    ConformalPidStopCalibrator,
    StrategyClusterObservation,
    build_advanced_sim_merit_convergence_manifest,
    compute_asymmetric_vol_crash_brake,
    compute_hierarchical_eb_merit_pool,
    evaluate_hansen_spa_and_mcs,
    update_fixed_share_hedge_weights,
)
from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT


def test_hansen_spa_and_mcs_separates_dominant_alpha_from_bleeders() -> None:
    # 80 dates: one strong positive alpha, one flat noise, one negative bleeder
    n = 80
    alpha_series = [18.0 + (2.0 if i % 2 == 0 else -1.5) for i in range(n)]
    flat_series = [0.2 if i % 2 == 0 else -0.2 for i in range(n)]
    bleeder_series = [-14.0 + (1.5 if i % 3 == 0 else -1.0) for i in range(n)]

    res = evaluate_hansen_spa_and_mcs(
        {"STRONG_ALPHA": alpha_series, "FLAT_NOISE": flat_series, "BLEEDER": bleeder_series},
        alpha=0.10,
        n_bootstrap=300,
        avg_block_len=4.0,
        seed=42,
    )
    assert res["best_strategy_id"] == "STRONG_ALPHA"
    assert res["spa_p_lower"] <= res["spa_p_consistent"] <= res["spa_p_upper"]
    assert res["spa_passes_alpha"] is True
    assert "STRONG_ALPHA" in res["mcs_surviving_strategies"]
    assert "BLEEDER" not in res["mcs_surviving_strategies"]
    assert res["mcs_p_values"]["STRONG_ALPHA"] == 1.0
    assert res["mcs_p_values"]["BLEEDER"] < 0.10


def test_hansen_spa_rejects_pure_zero_mean_noise() -> None:
    n = 60
    noise_a = [(-1.0 if i % 2 == 0 else 1.0) * 5.0 for i in range(n)]
    noise_b = [(-1.0 if i % 3 == 0 else 0.5) * 4.0 for i in range(n)]
    res = evaluate_hansen_spa_and_mcs({"N_A": noise_a, "N_B": noise_b}, alpha=0.05, n_bootstrap=250, seed=7)
    assert res["spa_passes_alpha"] is False


def test_conformal_pid_stop_calibrator_adapts_and_never_breaches_adr0022() -> None:
    cal = ConformalPidStopCalibrator(target_miscoverage_alpha=0.10)
    # Shock regime: large realized MAE should widen multiplier while staying strictly <= 2.00%
    prev_mult = cal.quantile_multiplier
    for _ in range(15):
        out = cal.update(realized_mae_pct=2.50, atr_pct=1.20)
        assert out["adr0022_compliant"] is True
        assert out["next_stop_pct"] <= ADR0022_MAX_STOP_PCT
    assert cal.quantile_multiplier > prev_mult

    # Calm regime: tiny MAE should tighten stop cleanly
    wide_mult = cal.quantile_multiplier
    for _ in range(25):
        out = cal.update(realized_mae_pct=0.05, atr_pct=0.50)
        assert 0.15 <= out["next_stop_pct"] <= ADR0022_MAX_STOP_PCT
    assert cal.quantile_multiplier < wide_mult


def test_fixed_share_hedge_adapts_quickly_to_regime_switch() -> None:
    w = {"A": 0.5, "B": 0.5}
    for _ in range(10):
        w = update_fixed_share_hedge_weights(w, {"A": 1.2, "B": -0.6}, eta=0.30, switching_alpha=0.08)
    assert w["A"] > 0.80
    assert w["B"] >= 0.04  # fixed-share floor keeps exploration alive for regime shifts!
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-4)

    # Regime flips sharply to B -> Fixed-Share Hedge pivots rapidly
    for _ in range(12):
        w = update_fixed_share_hedge_weights(w, {"A": -0.8, "B": 1.4}, eta=0.30, switching_alpha=0.08)
    assert w["B"] > 0.75


def test_hierarchical_eb_merit_pool_shrinks_small_sample_lucky_outlier() -> None:
    obs = [
        StrategyClusterObservation(
            "MOM_LUCKY_N3", "MOMENTUM", observed_mean_r=1.50, observed_se_r=0.90, closed_roundtrips=3
        ),
        StrategyClusterObservation(
            "MOM_STEADY_N60", "MOMENTUM", observed_mean_r=0.22, observed_se_r=0.06, closed_roundtrips=60
        ),
        StrategyClusterObservation(
            "REV_NEGATIVE", "MEAN_REVERSION", observed_mean_r=-0.25, observed_se_r=0.08, closed_roundtrips=40
        ),
    ]
    pool = compute_hierarchical_eb_merit_pool(obs)
    # Lucky N=3 outlier is strongly shrunk toward cluster posterior + zero
    assert pool["MOM_LUCKY_N3"]["hierarchical_eb_mean_r"] < 0.30
    assert pool["MOM_LUCKY_N3"]["shrinkage_weight_on_obs"] < 0.05
    # High-N steady strategy retains most of its empirical estimate
    assert pool["MOM_STEADY_N60"]["hierarchical_eb_mean_r"] > 0.14
    assert pool["MOM_STEADY_N60"]["posterior_prob_positive"] > 0.95
    assert pool["REV_NEGATIVE"]["hierarchical_eb_mean_r"] < 0.0


def test_asymmetric_vol_crash_brake_never_levers_above_one() -> None:
    assert compute_asymmetric_vol_crash_brake(8.0, target_vol_ann_pct=18.0) == 1.0
    assert compute_asymmetric_vol_crash_brake(36.0, target_vol_ann_pct=18.0) == 0.5
    assert compute_asymmetric_vol_crash_brake(120.0, target_vol_ann_pct=18.0, min_scale=0.25) == 0.25


def test_manifest_governance_invariants_and_zero_trial_consumption() -> None:
    mf = build_advanced_sim_merit_convergence_manifest()
    assert mf["adr"] == "ADR-0029"
    assert mf["study_wall_date"] == "2026-10-15"
    assert mf["promotion_track"] == "PROTECTIVE_FILTER_OR_GOVERNANCE"
    assert mf["standalone_alpha_claim"] is False
    assert mf["hlz_trial_consumed"] is False
    assert mf["next_trial_id"] == 105
    assert mf["next_hlz_t_star"] == NEXT_HLZ_T_STAR == pytest.approx(4.2084)
    assert mf["ready_now_blueprint_count"] == 5
    assert mf["future_l2_l3_blueprint_count"] == 1
    assert len(mf["manifest_sha256"]) == 64
    assert all(bp.max_stop_pct <= ADR0022_MAX_STOP_PCT for bp in NEXT_WAVE_STRATEGY_BLUEPRINTS)


@settings(max_examples=80, deadline=None)
@given(
    mae=st.floats(min_value=0.0, max_value=15.0, allow_nan=False, allow_infinity=False),
    atr=st.floats(min_value=0.05, max_value=10.0, allow_nan=False, allow_infinity=False),
    r1=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    r2=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    vol=st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_safety_and_simplex_invariants(mae: float, atr: float, r1: float, r2: float, vol: float) -> None:
    cal = ConformalPidStopCalibrator()
    out = cal.update(realized_mae_pct=mae, atr_pct=atr)
    assert 0.15 <= out["next_stop_pct"] <= ADR0022_MAX_STOP_PCT

    w = update_fixed_share_hedge_weights({"S1": 0.6, "S2": 0.4}, {"S1": r1, "S2": r2})
    assert all(0.0 <= v <= 1.0 for v in w.values())
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-4)

    brake = compute_asymmetric_vol_crash_brake(vol)
    assert 0.25 <= brake <= 1.0
