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


def test_redteam_spa_mcs_exact_invariants_zero_drift_and_scale_invariance() -> None:
    """Close Astra #9-#12 + P2 & sota-falsifier P0-3 across SPA/MCS."""
    # #9: Three identical strategies must NEVER eliminate one another, and MCS is benchmark-invariant!
    ident = evaluate_hansen_spa_and_mcs(
        {"A": [0.1] * 10, "B": [0.1] * 10, "C": [0.1] * 10},
        alpha=0.10,
        n_bootstrap=400,
        seed=20261015,
    )
    assert set(ident["mcs_surviving_strategies"]) == {"A", "B", "C"}
    assert ident["mcs_eliminated"] == []

    m_cash = evaluate_hansen_spa_and_mcs(
        {"A": [2.0, 3.0] * 5, "B": [0.0, 1.0] * 5}, alpha=0.10, n_bootstrap=200, seed=42
    )
    m_huge_b = evaluate_hansen_spa_and_mcs(
        {"A": [2.0, 3.0] * 5, "B": [0.0, 1.0] * 5},
        benchmark_returns_bps=[1e20] * 10,
        alpha=0.10,
        n_bootstrap=200,
        seed=42,
    )
    assert m_cash["mcs_surviving_strategies"] == m_huge_b["mcs_surviving_strategies"]
    assert m_cash["mcs_p_values"] == m_huge_b["mcs_p_values"]

    # #10 / P0-3: Non-finite benchmark or subtraction overflow fails closed!
    with pytest.raises(ValueError):
        evaluate_hansen_spa_and_mcs({"A": [0.0] * 10}, benchmark_returns_bps=[-float("inf")] * 10)
    with pytest.raises(ValueError):
        evaluate_hansen_spa_and_mcs({"A": [0.0] * 10}, benchmark_returns_bps=[float("nan")] * 10)
    with pytest.raises(ValueError):
        evaluate_hansen_spa_and_mcs({"A": [1e308] * 10}, benchmark_returns_bps=[-1e308] * 10)

    # #11: Positive rescaling by 2**-40 preserves exact studentized SPA p-value!
    x = [-1.0, 1.0] * 5
    a_base = [1.0 + v / 64.0 for v in x]
    b_base = [128.0 * v for v in x]
    s = 2.0**-40
    res_orig = evaluate_hansen_spa_and_mcs({"A": a_base, "B": b_base}, alpha=0.10, n_bootstrap=200, seed=7)
    res_scaled = evaluate_hansen_spa_and_mcs(
        {"A": [v * s for v in a_base], "B": [v * s for v in b_base]}, alpha=0.10, n_bootstrap=200, seed=7
    )
    assert res_orig["spa_p_consistent"] == res_scaled["spa_p_consistent"]

    # #12: Exact Hansen (2005) recentering threshold -sqrt(2*log(log(T))) at T=10
    a_h = [v - 17.0 / 128.0 for v in x]
    b_h = [-v + 1.0 / 8.0 for v in x]
    res_h = evaluate_hansen_spa_and_mcs({"A": a_h, "B": b_h}, alpha=0.25, n_bootstrap=400, seed=20261015)
    assert res_h["spa_p_consistent"] < 0.25
    assert res_h["spa_passes_alpha"] is True

    # P2: Disjoint surviving vs eliminated partition even on boundary alpha, and strict bootstrap validation
    res_part = evaluate_hansen_spa_and_mcs(
        {"A": [0.0] * 10, "B": [1.0] * 10}, alpha=0.009900995, n_bootstrap=100, seed=1
    )
    elim_ids = {e["strategy_id"] for e in res_part["mcs_eliminated"]}
    assert elim_ids.isdisjoint(set(res_part["mcs_surviving_strategies"]))
    assert elim_ids | set(res_part["mcs_surviving_strategies"]) == {"A", "B"}
    with pytest.raises(ValueError):
        evaluate_hansen_spa_and_mcs({"A": [0.0] * 10}, avg_block_len=0.0)
    with pytest.raises(ValueError):
        evaluate_hansen_spa_and_mcs({"A": [0.0] * 10}, n_bootstrap=-1)


def test_redteam_conformal_pid_hedge_eb_vol_brake_and_blueprint_feeds() -> None:
    """Close Astra #13-#15 + P2s & sota-falsifier P1-8 across PID, Hedge, EB, Vol Brake & Blueprints."""
    # #13: Reject non-representable/NaN stop bounds & post-construction ADR-0022 tampering
    with pytest.raises(ValueError):
        ConformalPidStopCalibrator(min_stop_pct=1e-6, max_stop_pct=1e-5)
    with pytest.raises(ValueError):
        ConformalPidStopCalibrator(max_stop_pct=float("nan"))
    cal_tamp = ConformalPidStopCalibrator()
    cal_tamp.max_stop_pct = 3.0
    with pytest.raises(ValueError):
        cal_tamp.recommended_stop_pct(1.0)

    # #14 / P1-8: Breach feedback matches actual issued stop (never opposite sign at upper/lower clamp)
    cal_hi = ConformalPidStopCalibrator(quantile_multiplier=1.50)
    out_hi = cal_hi.update(realized_mae_pct=2.20, atr_pct=2.00)
    assert out_hi["previous_stop_pct"] == 2.0
    assert out_hi["stop_breached"] is True
    assert out_hi["empirical_miscoverage_rate"] == 1.0
    assert out_hi["next_quantile_multiplier"] > 1.50

    cal_lo = ConformalPidStopCalibrator()
    out_lo = cal_lo.update(realized_mae_pct=0.12, atr_pct=0.05)
    assert out_lo["previous_stop_pct"] == 0.15
    assert out_lo["stop_breached"] is False

    # #15: Attainable coverage up to 2.00% even at tiny ATR=0.10% (no artificial 4.0 multiplier ceiling)
    cal_reach = ConformalPidStopCalibrator()
    max_reached = 0.0
    for _ in range(400):
        out_r = cal_reach.update(realized_mae_pct=0.50, atr_pct=0.10)
        max_reached = max(max_reached, out_r["next_stop_pct"])
    assert max_reached >= 0.50
    assert out_r["empirical_miscoverage_rate"] < 0.25
    assert out_r["next_stop_pct"] <= ADR0022_MAX_STOP_PCT

    # P2 Hedge: log-sum-exp stability at [-800, -801] and +800, exact sum==1.0, missing key fails closed
    w_neg = update_fixed_share_hedge_weights({"A": 0.5, "B": 0.5}, {"A": -800.0, "B": -801.0}, eta=1.0, r_clip=1000.0)
    assert math.isclose(w_neg["A"], 0.719506, abs_tol=1e-5)
    assert math.isclose(w_neg["B"], 0.280494, abs_tol=1e-5)
    w_pos = update_fixed_share_hedge_weights({"A": 0.5, "B": 0.5}, {"A": 800.0, "B": 0.0}, eta=1.0, r_clip=1000.0)
    assert sum(w_pos.values()) == pytest.approx(1.0, abs=1e-9)
    w_three = update_fixed_share_hedge_weights({"A": 1.0, "B": 1.0, "C": 1.0}, {"A": 0.0, "B": 0.0, "C": 0.0})
    assert sum(w_three.values()) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError):
        update_fixed_share_hedge_weights({"A": 0.5, "B": 0.5}, {"A": 1.0})

    # P2 EB: analytic tau_within_cluster=0, finite 1e308 handling, reject duplicate strategy IDs
    eb_zero = compute_hierarchical_eb_merit_pool(
        [
            StrategyClusterObservation("A", "C", 1.0, 0.01, 30),
            StrategyClusterObservation("B", "C", -1.0, 0.01, 30),
        ],
        tau_within_cluster=0.0,
    )
    assert eb_zero["A"]["hierarchical_eb_mean_r"] == 0.0
    assert eb_zero["B"]["hierarchical_eb_mean_r"] == 0.0
    eb_big = compute_hierarchical_eb_merit_pool([StrategyClusterObservation("BIG", "C", 1e308, 0.01, 30)])
    assert math.isfinite(eb_big["BIG"]["hierarchical_eb_mean_r"])
    with pytest.raises(ValueError):
        compute_hierarchical_eb_merit_pool(
            [
                StrategyClusterObservation("DUP", "C", 0.5, 0.01, 30),
                StrategyClusterObservation("DUP", "C", 0.5, 0.01, 30),
            ]
        )

    # P2 Vol crash brake: reject negative & bool inputs
    with pytest.raises(ValueError):
        compute_asymmetric_vol_crash_brake(-1.0)
    with pytest.raises(ValueError):
        compute_asymmetric_vol_crash_brake(True)  # type: ignore[arg-type]

    # P2 Blueprint feeds aligned with canonical registry (no unregistered DF10_VIX_REGIME / DF12_CAUSAL_TCA)
    for bp in NEXT_WAVE_STRATEGY_BLUEPRINTS:
        assert "DF10_VIX_REGIME" not in bp.required_feeds
        assert "DF12_CAUSAL_TCA" not in bp.required_feeds


def test_redteam_pass2_merit_counterexamples_all_closed() -> None:
    """Close every Pass-2 Astra & Gemini counterexample across SPA/MCS, PID, Hedge, EB, HAC & Vol Brake."""
    # P1 #9: Exact zero-variance ordinary decimal `[0.22]*10` vs `[0.0]*10` gives exact 1/101 SPA & MCS!
    res_dec = evaluate_hansen_spa_and_mcs({"A": [0.22] * 10, "B": [0.0] * 10}, alpha=0.10, n_bootstrap=100, seed=42)
    assert res_dec["spa_p_consistent"] == pytest.approx(1.0 / 101.0, abs=1e-4)
    assert res_dec["mcs_surviving_strategies"] == ["A"]
    assert res_dec["mcs_p_values"]["B"] == pytest.approx(1.0 / 101.0, abs=1e-4)

    # P1 #10: Configured max_stop_pct=0.15006% never rounds UP to 0.1501%
    cal_tight = ConformalPidStopCalibrator(min_stop_pct=0.15, max_stop_pct=0.15006)
    stop_tight = cal_tight.recommended_stop_pct(2.0)
    assert stop_tight <= 0.15006
    assert stop_tight == 0.15

    # P1 #11: Incompatible exact evidence (`se=0` with `+1` vs `-1` under `tau_within=0`) raises ValueError
    with pytest.raises(ValueError, match="Incompatible exact"):
        compute_hierarchical_eb_merit_pool(
            [
                StrategyClusterObservation("A", "C", 1.0, 0.0, 30),
                StrategyClusterObservation("B", "C", -1.0, 0.0, 30),
            ],
            tau_within_cluster=0.0,
        )
    with pytest.raises(ValueError, match="Incompatible exact"):
        compute_hierarchical_eb_merit_pool(
            [StrategyClusterObservation("A", "C", 1.0, 0.0, 30)],
            tau_between_clusters=0.0,
            tau_within_cluster=0.0,
        )

    # P2 Hedge: extra strategy key raises; 1e308 shifted before eta; switching_alpha=1e-8 rescues winner across steps
    with pytest.raises(ValueError, match="key mismatch"):
        update_fixed_share_hedge_weights({"A": 0.5, "B": 0.5}, {"A": 1.0, "B": 0.0, "EXTRA": 5.0})
    w_big = update_fixed_share_hedge_weights(
        {"A": 0.5, "B": 0.5}, {"A": 1e308, "B": 0.0}, eta=2.0, switching_alpha=0.05, r_clip=1e308
    )
    assert w_big["A"] == pytest.approx(0.975, abs=1e-6)
    w_tiny = {"A": 1.0, "B": 0.0}
    for _ in range(30):
        w_tiny = update_fixed_share_hedge_weights(w_tiny, {"A": 0.0, "B": 3.0}, eta=1.0, switching_alpha=1e-8)
    assert w_tiny["B"] > 0.99

    # P2 EB: zero artificial variance floor (1e-8 SE vs 1e-7 tau_b -> 0.9901), finite 1e100 & 1e200 scales
    eb_small = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("S", "C", 1.0, 1e-8, 30)],
        tau_between_clusters=1e-7,
        tau_within_cluster=0.0,
    )
    assert eb_small["S"]["hierarchical_eb_mean_r"] == pytest.approx(0.9901, abs=1e-4)

    eb_1e100 = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("H", "C", 1.0, 1e100, 30)],
        tau_between_clusters=0.0,
        tau_within_cluster=1e100,
    )
    assert math.isfinite(eb_1e100["H"]["hierarchical_eb_sd_r"])
    assert eb_1e100["H"]["hierarchical_eb_sd_r"] == pytest.approx(math.sqrt(5e199), rel=1e-6)

    eb_1e200 = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("H2", "C", 1.0, 1.0, 30)],
        tau_between_clusters=1e200,
        tau_within_cluster=1e200,
    )
    assert math.isfinite(eb_1e200["H2"]["hierarchical_eb_mean_r"])
    assert math.isfinite(eb_1e200["H2"]["hierarchical_eb_sd_r"])

    # P2 HAC 1.5e308 overflow invariance & small positive vol brake never rounds to 0.0
    res_huge = evaluate_hansen_spa_and_mcs({"HUGE": [1.5e308] * 9 + [-1.5e308]}, alpha=0.10, n_bootstrap=100, seed=42)
    assert res_huge["per_strategy_t_stats"]["HUGE"] == pytest.approx(4.5195, abs=0.05)
    brake_tiny = compute_asymmetric_vol_crash_brake(1e6, target_vol_ann_pct=1.0, min_scale=1e-5)
    assert brake_tiny == pytest.approx(1e-5, rel=1e-6)
    assert brake_tiny > 0.0


def test_redteam_pass3_merit_counterexamples_all_closed() -> None:
    """Close every Pass-3 Astra & Gemini counterexample across SPA/MCS, Hedge & Hierarchical EB."""
    # Astra P1-11 & Gemini P2-3: Bit-exact positive scale invariance down to subnormal `2**-1074` in SPA & MCS!
    base_s = [1.0] * 6 + [-1.0] * 4
    sub_s = [x * (2.0**-1074) for x in base_s]
    r_base = evaluate_hansen_spa_and_mcs({"S": base_s}, alpha=0.10, n_bootstrap=100, avg_block_len=5.0, seed=42)
    r_sub = evaluate_hansen_spa_and_mcs({"S": sub_s}, alpha=0.10, n_bootstrap=100, avg_block_len=5.0, seed=42)
    assert r_base["spa_passes_alpha"] is False
    assert r_sub["spa_p_consistent"] == r_base["spa_p_consistent"]
    assert r_sub["spa_passes_alpha"] is False

    r_mcs_sub = evaluate_hansen_spa_and_mcs(
        {"A_ZERO": [0.0] * 10, "B_SUB": [2.0**-1074] * 10}, alpha=0.10, n_bootstrap=100, avg_block_len=5.0, seed=42
    )
    assert r_mcs_sub["mcs_surviving_strategies"] == ["B_SUB"]
    assert r_mcs_sub["mcs_p_values"]["A_ZERO"] == pytest.approx(1.0 / 101.0, abs=1e-5)

    # Astra P2-4: Stationary bootstrap rejects degenerate avg_block_len > T/2 (e.g. 1000 on T=10)
    with pytest.raises(ValueError, match="avg_block_len"):
        evaluate_hansen_spa_and_mcs({"S": base_s}, alpha=0.10, n_bootstrap=100, avg_block_len=1000.0, seed=42)

    # Astra P2-1 & Gemini P2-2: Hedge 1e308 priors & ±1e308 rewards with eta=1e-308
    w_p = update_fixed_share_hedge_weights(
        {"A": 1e308, "B": 1e308}, {"A": 3.0, "B": 0.0}, eta=1.0, switching_alpha=0.05
    )
    assert w_p["A"] == pytest.approx(0.929945, abs=1e-5)
    w_2to1 = update_fixed_share_hedge_weights(
        {"A": 1.5e308, "B": 0.75e308}, {"A": 0.0, "B": 0.0}, eta=1.0, switching_alpha=0.0
    )
    assert w_2to1["A"] == pytest.approx(2.0 / 3.0, abs=1e-6)
    w_pm = update_fixed_share_hedge_weights(
        {"A": 0.5, "B": 0.5}, {"A": 1e308, "B": -1e308}, eta=1e-308, switching_alpha=0.05, r_clip=1e308
    )
    assert w_pm["A"] == pytest.approx(0.861757, abs=1e-5)

    # Astra P2-2, P2-3 & Gemini P2-3: EB 1.5e308 hypot safety, tiny positive SD preserved, exact zero P(R>0)==0.0
    eb_huge = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("X", "C", 2.0, 1.5e308, 30)],
        tau_between_clusters=0.0,
        tau_within_cluster=1.5e308,
    )
    assert eb_huge["X"]["hierarchical_eb_mean_r"] == pytest.approx(1.0, abs=1e-4)
    assert eb_huge["X"]["hierarchical_eb_sd_r"] == pytest.approx(1.5e308 / math.sqrt(2.0), rel=1e-6)

    eb_tiny_sd = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("T", "C", 1.0, 1e-8, 30)],
        tau_between_clusters=1e-7,
        tau_within_cluster=0.0,
    )
    assert eb_tiny_sd["T"]["hierarchical_eb_sd_r"] > 0.0

    eb_zero_pt = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("Z", "C", 0.0, 0.0, 30)],
        tau_between_clusters=0.0,
        tau_within_cluster=0.0,
    )
    assert eb_zero_pt["Z"]["posterior_prob_positive"] == 0.0


def test_redteam_pass4_merit_counterexamples_all_closed() -> None:
    """Close ALL Pass-4 Astra & Gemini counterexamples across SPA ties, PID saturation, Hedge & EB."""
    # Astra Additional P1-3: Exact bootstrap tie preservation on `[1]+[0]*19` -> 91/401 = 0.226933 (NOT 26/401)
    res_ties = evaluate_hansen_spa_and_mcs(
        {"S": [1.0] + [0.0] * 19}, alpha=0.10, n_bootstrap=400, avg_block_len=1.0, seed=42
    )
    assert res_ties["spa_p_consistent"] == pytest.approx(91.0 / 401.0, abs=1e-5)
    assert res_ties["spa_passes_alpha"] is False

    # Astra Additional P2-5: Exported `per_strategy_t_stats_raw` keeps constant positive > finite ~1e13
    res_order = evaluate_hansen_spa_and_mcs(
        {"CONST": [1.0] * 10, "NEAR": [1.0] * 9 + [1.000000000001]}, alpha=0.10, n_bootstrap=100, seed=42
    )
    assert res_order["best_strategy_id"] == "CONST"
    assert res_order["per_strategy_t_stats_raw"]["CONST"] > res_order["per_strategy_t_stats_raw"]["NEAR"]

    # Astra Additional P2-3 & P2-4: PID reaches exact 2.0000% at ATR=0.0503% (ki=0) & rejects _miscoverages > _steps
    cal_sat = ConformalPidStopCalibrator(kp_learning_rate=100.0, ki_integral_gain=0.0)
    out_sat = cal_sat.update(realized_mae_pct=2.0, atr_pct=0.0503)
    assert out_sat["next_stop_pct"] == 2.0
    out_sat2 = cal_sat.update(realized_mae_pct=2.0, atr_pct=0.0503)
    assert out_sat2["stop_breached"] is False

    with pytest.raises(ValueError, match="_miscoverages"):
        ConformalPidStopCalibrator(_steps=0, _miscoverages=1)

    # Astra P2-1 & Gemini P2-4: Hedge `(1e-300, 1e300)` & `(1e200, 1e-200)` log-priors + `(1e16, 1e16-2)` precision
    w_wide = update_fixed_share_hedge_weights(
        {"A": 1e-300, "B": 1e300}, {"A": 1000.0, "B": -1000.0}, eta=1.0, switching_alpha=0.0, r_clip=1000.0
    )
    assert w_wide["A"] == pytest.approx(1.0, abs=1e-9)
    w_wide2 = update_fixed_share_hedge_weights(
        {"A": 1e200, "B": 1e-200}, {"A": 0.0, "B": 1000.0}, eta=1.0, switching_alpha=0.0, r_clip=1000.0
    )
    assert w_wide2["B"] == pytest.approx(1.0, abs=1e-9)
    w_prec = update_fixed_share_hedge_weights(
        {"A": 0.5, "B": 0.5}, {"A": 1e16, "B": 1e16 - 2.0}, eta=0.1, switching_alpha=0.0, r_clip=1e16
    )
    assert w_prec["A"] == pytest.approx(0.549834, abs=1e-5)

    # Astra P2-2 + Additional P2-6 & Gemini P2-1: EB `1e-100/1e300`, `1e-170/1e170`, `1e-200/1/1e200`, DBL_MAX convex
    eb_small_tw = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("A", "C", 0.0, 1e300, 30)],
        tau_between_clusters=0.0,
        tau_within_cluster=1e-100,
    )
    assert eb_small_tw["A"]["hierarchical_eb_sd_r"] == pytest.approx(1e-100, rel=1e-6, abs=0.0)
    assert eb_small_tw["A"]["posterior_prob_positive"] == pytest.approx(0.5, abs=1e-4)

    eb_wide_ratio = compute_hierarchical_eb_merit_pool(
        [
            StrategyClusterObservation("A", "C", 1.0, 1e-170, 30),
            StrategyClusterObservation("B", "C", 2.0, 1e170, 30),
        ],
        tau_between_clusters=1.0,
        tau_within_cluster=0.0,
    )
    assert math.isfinite(eb_wide_ratio["A"]["hierarchical_eb_mean_r"])
    assert eb_wide_ratio["A"]["hierarchical_eb_mean_r"] == pytest.approx(1.0, abs=1e-4)

    eb_mixed = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("M", "C", 1.0, 1e200, 30)],
        tau_between_clusters=1e-200,
        tau_within_cluster=1.0,
    )
    assert math.isfinite(eb_mixed["M"]["hierarchical_eb_sd_r"])

    dbl_max = 1.7976931348623157e308
    eb_convex = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("MAX", "C", dbl_max, 4.0, 30)],
        tau_between_clusters=1e100,
        tau_within_cluster=3.0,
    )
    assert math.isfinite(eb_convex["MAX"]["hierarchical_eb_mean_r"])


def test_redteam_pass5_merit_counterexamples_all_closed() -> None:
    """Close ALL Pass-5 Astra (P1 #4 + P2 #13/#17/#20/#21) counterexamples in ADR-0029."""
    # Astra P1 #4: Non-dyadic `/5` rescaling drift eliminated via ldexp + sum-space comparison -> exact 44/401
    res_spa5 = evaluate_hansen_spa_and_mcs(
        {"S": [5.0, -3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]},
        alpha=0.10,
        n_bootstrap=400,
        avg_block_len=1.0,
        seed=42,
    )
    assert res_spa5["spa_p_consistent"] == pytest.approx(44.0 / 401.0, abs=1e-6)
    assert res_spa5["spa_passes_alpha"] is False

    # Astra P2 #13: Two DBL_MAX observations in one cluster NEVER overflow `math.fsum`
    dbl_max = 1.7976931348623157e308
    eb_two_max = compute_hierarchical_eb_merit_pool(
        [
            StrategyClusterObservation("A", "C", dbl_max, 1.0, 30),
            StrategyClusterObservation("B", "C", dbl_max, 14.0, 30),
        ],
        tau_between_clusters=1e100,
        tau_within_cluster=0.0,
    )
    assert eb_two_max["A"]["hierarchical_eb_mean_r"] == pytest.approx(dbl_max, rel=1e-12)
    assert eb_two_max["B"]["hierarchical_eb_mean_r"] == pytest.approx(dbl_max, rel=1e-12)

    # Astra P2 #17: EB posterior mean immune to `1e20 - 1e20` subtractive cancellation
    eb_cancel = compute_hierarchical_eb_merit_pool(
        [
            StrategyClusterObservation("A", "C", 0.0, 0.01, 30),
            StrategyClusterObservation("B", "C", 2e20, 0.0, 30),
        ],
        tau_between_clusters=1e100,
        tau_within_cluster=1e8,
    )
    assert eb_cancel["A"]["hierarchical_eb_mean_r"] == pytest.approx(1.0, abs=1e-3)
    assert eb_cancel["A"]["posterior_prob_positive"] == pytest.approx(1.0, abs=1e-4)

    # Astra P2 #20: Non-finite persisted `_integral_err=NaN` rejected at construction and update
    with pytest.raises(ValueError, match="_integral_err"):
        ConformalPidStopCalibrator(_integral_err=float("nan"))
    cal_mut = ConformalPidStopCalibrator()
    cal_mut._integral_err = float("nan")
    with pytest.raises(ValueError, match="_integral_err"):
        cal_mut.update(realized_mae_pct=0.0, atr_pct=1.0)

    # Astra P2 #21: All-zero Hedge priors `(0, 0)` raise ValueError instead of discarding rewards
    with pytest.raises(ValueError, match="strictly positive"):
        update_fixed_share_hedge_weights({"A": 0.0, "B": 0.0}, {"A": 3.0, "B": -3.0}, eta=1.0, switching_alpha=0.0)


def test_redteam_pass6_merit_counterexamples_all_closed() -> None:
    """Close ALL Pass-6 Astra (P1 #2/#3 + P2 #9/#10/#11) counterexamples in ADR-0029."""
    from dataclasses import asdict

    from sota_compounding.advanced_merit_sim_convergence import run_fixed_share_hedge_sequence

    # Astra P1 #2: Equal-SE complementary fractional-bps series preserve exact 42/401 (NO 1-ULP tie drop!)
    a_frac = [x / 8.0 for x in [0, -6, 8, 0, -1, 0, 4, -1, 7, -4]]
    b_frac = [0.375 - x for x in a_frac]
    res_p2 = evaluate_hansen_spa_and_mcs(
        {"A": a_frac, "B": b_frac}, alpha=0.10, n_bootstrap=400, avg_block_len=1.0, seed=1
    )
    assert res_p2["spa_p_consistent"] == pytest.approx(42.0 / 401.0, abs=1e-6)
    assert res_p2["spa_passes_alpha"] is False

    # Astra P1 #3: Exact dyadic integer sum preserves `2^-54` bit -> exact 40/401 = 0.099751 passes alpha=0.10!
    r_sub_bit = [5.0, -3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0**-54, 0.0]
    res_p3 = evaluate_hansen_spa_and_mcs({"S": r_sub_bit}, alpha=0.10, n_bootstrap=400, avg_block_len=1.0, seed=42)
    assert res_p3["spa_p_consistent"] == pytest.approx(40.0 / 401.0, abs=1e-6)
    assert res_p3["spa_passes_alpha"] is True

    # Astra P2 #9: Restoring PID state via `ConformalPidStopCalibrator(**asdict(pid))` preserves next stop & breach!
    pid_live = ConformalPidStopCalibrator()
    pid_live.update(realized_mae_pct=2.0, atr_pct=1.0)
    pid_restored = ConformalPidStopCalibrator(**asdict(pid_live))
    u_live_2 = pid_live.update(realized_mae_pct=0.0, atr_pct=1.0)
    u_rest_2 = pid_restored.update(realized_mae_pct=0.0, atr_pct=1.0)
    assert u_rest_2["next_stop_pct"] == u_live_2["next_stop_pct"] == pytest.approx(1.4666, abs=1e-4)
    u_live_3 = pid_live.update(realized_mae_pct=1.55, atr_pct=1.0)
    u_rest_3 = pid_restored.update(realized_mae_pct=1.55, atr_pct=1.0)
    assert u_rest_3["stop_breached"] is u_live_3["stop_breached"] is True

    # Astra P2 #10: Log-domain EB prevents `w * mean` underflow on `(mean=1e300, SE=1), tb=1e-170, tw=0`!
    eb_under = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("U", "C", 1e300, 1.0, 30)],
        tau_between_clusters=1e-170,
        tau_within_cluster=0.0,
    )
    assert eb_under["U"]["hierarchical_eb_mean_r"] == pytest.approx(1e-40, rel=1e-4, abs=0.0)
    assert eb_under["U"]["hierarchical_eb_sd_r"] == pytest.approx(1e-170, rel=1e-4, abs=0.0)
    assert eb_under["U"]["posterior_prob_positive"] == 1.0

    # Astra P2 #11: Log-space zero-share Hedge recovers `(0.5, 0.5)` after 125 `(+3,-3)` + 125 `(-3,+3)`!
    seq_250 = [{"A": 3.0, "B": -3.0}] * 125 + [{"A": -3.0, "B": 3.0}] * 125
    w_rec = run_fixed_share_hedge_sequence({"A": 0.5, "B": 0.5}, seq_250, eta=1.0, switching_alpha=0.0)
    assert w_rec["A"] == pytest.approx(0.5, abs=1e-9)
    assert w_rec["B"] == pytest.approx(0.5, abs=1e-9)


def test_redteam_pass7_merit_counterexamples_all_closed() -> None:
    """Close ALL Pass-7 Astra (P1 #1/#5 + P2 #4-#7) & Gemini (P1-1 + P2-1) counterexamples in ADR-0029."""
    from sota_compounding.advanced_merit_sim_convergence import (
        _newey_west_long_run_var,
        run_fixed_share_hedge_sequence,
    )

    # Astra P1 #1: Common dyadic integer subtraction preserves `-3 - 2^-54` -> exact 43/401 fails alpha=0.10!
    r_sub_bit = [5.0, -3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0**-54, 0.0]
    bench_bit = [0.0, 2.0**-54] + [0.0] * 8
    res_b1 = evaluate_hansen_spa_and_mcs(
        {"S": r_sub_bit},
        benchmark_returns_bps=bench_bit,
        alpha=0.10,
        n_bootstrap=400,
        avg_block_len=1.0,
        seed=42,
    )
    assert res_b1["spa_p_consistent"] == pytest.approx(43.0 / 401.0, abs=1e-6)
    assert res_b1["spa_passes_alpha"] is False

    # Astra P1 #5 & Gemini P1-1: 80-digit Decimal EB eliminates `log1p(-1)` crashes AND `[1e16,-1e16,3]` erasure!
    eb_tiny_diff = compute_hierarchical_eb_merit_pool(
        [
            StrategyClusterObservation("A", "C", 4.0, 1.0, 30),
            StrategyClusterObservation("B", "C", -4.0, 1.0, 30),
            StrategyClusterObservation("C", "C", 1e-16, 1.0, 30),
        ],
        tau_between_clusters=1.0,
        tau_within_cluster=0.0,
    )
    assert eb_tiny_diff["A"]["hierarchical_eb_mean_r"] == pytest.approx(2.5e-17, rel=1e-6, abs=0.0)

    for sign_c in (1e-18, -1e-18):
        eb_def = compute_hierarchical_eb_merit_pool(
            [
                StrategyClusterObservation("A", "C", 3.7, 0.06, 30),
                StrategyClusterObservation("B", "C", -3.7, 0.06, 30),
                StrategyClusterObservation("C", "C", sign_c, 0.06, 30),
            ]
        )
        assert math.isfinite(eb_def["A"]["hierarchical_eb_mean_r"])

    eb_cancel_3 = compute_hierarchical_eb_merit_pool(
        [
            StrategyClusterObservation("A", "C", 1e16, 1.0, 30),
            StrategyClusterObservation("B", "C", -1e16, 1.0, 30),
            StrategyClusterObservation("C", "C", 3.0, 1.0, 30),
        ],
        tau_between_clusters=1.0,
        tau_within_cluster=0.0,
    )
    assert eb_cancel_3["A"]["hierarchical_eb_mean_r"] == pytest.approx(0.75, abs=1e-4)
    assert eb_cancel_3["A"]["hierarchical_eb_sd_r"] == pytest.approx(0.5, abs=1e-4)
    assert eb_cancel_3["A"]["posterior_prob_positive"] == pytest.approx(0.9332, abs=1e-4)

    # Astra P2 #4 & Gemini P2-1: Symmetric `w_obs` underflow `(1e300, SE=1), tb=0, tw=1e-170` -> exact `1e-40` & `1.0`!
    eb_w_obs = compute_hierarchical_eb_merit_pool(
        [StrategyClusterObservation("U", "C", 1e300, 1.0, 30)],
        tau_between_clusters=0.0,
        tau_within_cluster=1e-170,
    )
    assert eb_w_obs["U"]["hierarchical_eb_mean_r"] == pytest.approx(1e-40, rel=1e-4, abs=0.0)
    assert eb_w_obs["U"]["posterior_prob_positive"] == 1.0

    # Astra P2 #5 & #6: Decimal Hedge preserves `[1e16, 1, -1e16]` AND handles subnormal `switching_alpha=2^-1074`!
    w_mid = run_fixed_share_hedge_sequence(
        {"A": 0.5, "B": 0.5},
        [{"A": 1e16, "B": 0.0}, {"A": 1.0, "B": 0.0}, {"A": -1e16, "B": 0.0}],
        eta=1.0,
        switching_alpha=0.0,
        r_clip=1e16,
    )
    assert w_mid["A"] == pytest.approx(0.731059, abs=1e-5)
    assert w_mid["B"] == pytest.approx(0.268941, abs=1e-5)

    w_sub_a = run_fixed_share_hedge_sequence(
        {"A": 0.5, "B": 0.5}, [{"A": 0.0, "B": 0.0}], eta=1.0, switching_alpha=2.0**-1074
    )
    assert w_sub_a["A"] == pytest.approx(0.5, abs=1e-9)

    # Astra P2 #7: `_newey_west_long_run_var([1.2e154, -1.2e154], 0)` returns finite `1.44e308` (not inf)!
    assert _newey_west_long_run_var([1.2e154, -1.2e154], 0) == pytest.approx(1.44e308, rel=1e-6)


def test_redteam_pass8_merit_counterexamples_all_closed() -> None:
    """Close ALL Pass-8 Astra (P1 #4/#5 + P2 #3-#5/#8) & Gemini (P2-1) counterexamples in ADR-0029."""
    from sota_compounding.advanced_merit_sim_convergence import (
        _exact_hac_int_stats,
        run_fixed_share_hedge_sequence,
    )

    # Astra P1 #4: Exact dyadic mean breaks +inf t-stat ties -> selects A (NOT Z eliminated by MCS!)
    u = 2.0**-52
    res_tie_mean = evaluate_hansen_spa_and_mcs(
        {"A": [1.0 + 3.0 * u] * 10, "Z": [1.0 + 2.0 * u] * 10}, alpha=0.10, n_bootstrap=400, seed=42
    )
    assert res_tie_mean["best_strategy_id"] == "A"
    assert res_tie_mean["mcs_surviving_strategies"] == ["A"]

    # Astra P1 #5 & Gemini P2-1: Permutation-invariant exact Fraction EB for extreme scales!
    for big_m in (float(2**400), 1e100):
        eb_a = compute_hierarchical_eb_merit_pool(
            [
                StrategyClusterObservation("A", "C", big_m, 1.0, 30),
                StrategyClusterObservation("C", "C", 3.0, 1.0, 30),
                StrategyClusterObservation("B", "C", -big_m, 1.0, 30),
            ],
            tau_between_clusters=1.0,
            tau_within_cluster=0.0,
        )
        eb_b = compute_hierarchical_eb_merit_pool(
            [
                StrategyClusterObservation("A", "C", big_m, 1.0, 30),
                StrategyClusterObservation("B", "C", -big_m, 1.0, 30),
                StrategyClusterObservation("C", "C", 3.0, 1.0, 30),
            ],
            tau_between_clusters=1.0,
            tau_within_cluster=0.0,
        )
        assert (
            eb_a["A"]["hierarchical_eb_mean_r"] == eb_b["A"]["hierarchical_eb_mean_r"] == pytest.approx(0.75, abs=1e-4)
        )
        assert (
            eb_a["A"]["posterior_prob_positive"]
            == eb_b["A"]["posterior_prob_positive"]
            == pytest.approx(0.9332, abs=1e-4)
        )

    # Astra P2 #3: Huge finite HAC t (`1.06e308`) never crashes `math.ceil(inf)`!
    res_huge_t = evaluate_hansen_spa_and_mcs(
        {"S": [1.0] * 10}, benchmark_returns_bps=[1e-307] + [0.0] * 9, alpha=0.10, n_bootstrap=100, avg_block_len=1.0
    )
    assert math.isfinite(res_huge_t["spa_t_stat"])

    # Astra P2 #4 & #5: Exact Fraction zero-share Hedge handles `[2**400, 1, -2**400]` AND preserves `2^-1074`!
    big_r = float(2**400)
    w_exact_r = run_fixed_share_hedge_sequence(
        {"A": 0.5, "B": 0.5},
        [{"A": big_r, "B": 0.0}, {"A": 1.0, "B": 0.0}, {"A": -big_r, "B": 0.0}],
        eta=1.0,
        switching_alpha=0.0,
        r_clip=big_r,
    )
    assert w_exact_r["A"] == pytest.approx(0.731059, abs=1e-5)
    assert w_exact_r["B"] == pytest.approx(0.268941, abs=1e-5)

    w_sub_keep = run_fixed_share_hedge_sequence(
        {"A": 0.5, "B": 0.5}, [{"A": -745.0625, "B": 0.0}], eta=1.0, switching_alpha=0.0, r_clip=746.0
    )
    assert w_sub_keep["A"] == 2.0**-1074

    # Astra P2 #8: Indexed HAC uses index span (`max - min = 5`) rather than retained count (`n - 1 = 1`)!
    _, _, _, _, _, t_sparse = _exact_hac_int_stats([3.0, 1.0], 5, indices=[0, 5])
    assert t_sparse == pytest.approx(3.098387, abs=1e-5)


def test_redteam_pass9_merit_counterexamples_all_closed() -> None:
    """Close Pass-9 3-arm subnormal Hedge rounding, overflow t-cap reporting tie & exact-Fraction EB inputs."""
    import sys
    from fractions import Fraction

    from sota_compounding.advanced_merit_sim_convergence import run_fixed_share_hedge_sequence

    # Astra P2 #4: 3-arm Hedge `[-744.4, 0, 0]` preserves representable `2^-1074` (single-rounded log-sum-exp!)
    w3 = run_fixed_share_hedge_sequence(
        {"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0},
        [{"A": -744.4, "B": 0.0, "C": 0.0}],
        eta=1.0,
        switching_alpha=0.0,
        r_clip=745.0,
    )
    assert w3["A"] == 2.0**-1074
    assert w3["B"] == pytest.approx(0.5, abs=1e-12)
    assert w3["C"] == pytest.approx(0.5, abs=1e-12)

    # Astra P2 #5: Coexisting `1.06e308` finite t and `+inf` zero-var t never tie in `per_strategy_t_stats_raw`!
    res_coexist = evaluate_hansen_spa_and_mcs(
        {"FIN": [1.0] * 10, "INF": [2e-307] + [1e-307] * 9},
        benchmark_returns_bps=[1e-307] + [0.0] * 9,
        alpha=0.10,
        n_bootstrap=100,
        avg_block_len=1.0,
    )
    assert res_coexist["per_strategy_t_stats_raw"]["INF"] == sys.float_info.max
    assert res_coexist["per_strategy_t_stats_raw"]["INF"] > res_coexist["per_strategy_t_stats_raw"]["FIN"]

    # Astra P1 #5: Exact Fraction EB inputs preserve `z = 19.97498 -> posterior_prob_positive = 1.0` even below 2^-1074!
    f_sub = Fraction(1, 1 << 1074)
    eb_sub = compute_hierarchical_eb_merit_pool(
        [
            StrategyClusterObservation(
                strategy_id="SUB",
                cluster_id="C",
                observed_mean_r=0.0,
                observed_se_r=0.0,
                closed_roundtrips=200,
                exact_mean_r=f_sub / 200,
                exact_se_sq_r=(f_sub * f_sub) / (40_000 * 399),
            )
        ]
    )
    assert eb_sub["SUB"]["posterior_prob_positive"] == 1.0
    # Falsifier Pass-10 P2-1: direct exact_mean_r overrides observed_mean_r clamping bounds!
    eb_dir = compute_hierarchical_eb_merit_pool(
        [
            StrategyClusterObservation(
                strategy_id="DIR",
                cluster_id="C",
                observed_mean_r=0.0,
                observed_se_r=0.0,
                closed_roundtrips=30,
                exact_mean_r=Fraction(1, 2),
                exact_se_sq_r=Fraction(1, 100),
            )
        ]
    )
    assert eb_dir["DIR"]["hierarchical_eb_mean_r"] > 0.1
    assert eb_dir["DIR"]["posterior_prob_positive"] == 1.0

    # Astra P1 #4 parity: Inexact integer exceeding binary64 precision fails closed at intake!
    with pytest.raises(ValueError, match="binary64"):
        evaluate_hansen_spa_and_mcs({"A": [2**56 + 9] * 10}, n_bootstrap=100)
