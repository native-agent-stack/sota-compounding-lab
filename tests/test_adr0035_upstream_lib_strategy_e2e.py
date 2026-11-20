"""Tests for ADR-0035: SOTA Upstream-Library Strategy E2E Sim & Regime-Conditioned Per-Symbol Factor Engine."""

from __future__ import annotations

import json
import pathlib
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sota_compounding.gpu_scoring_sidecar import build_sidecar_state, make_sidecar_handler
from sota_compounding.upstream_lib_strategy_e2e import (
    ADR0022_MAX_STOP_PCT,
    ALLOWED_PAPER_VENUE_LANES,
    FORBIDDEN_LIVE_LANES_EXTENDED,
    SOTA_UPSTREAM_BACKTEST_PLATFORMS,
    SOTA_UPSTREAM_STRATEGY_LIBRARY,
    LiveTargetObservation,
    MarketSituationInput,
    apply_regime_conditioned_per_symbol_factors,
    build_upstream_lib_e2e_manifest,
    classify_market_situation,
    compute_manifest_sha256,
    infer_peer_cluster,
    render_upstream_lib_e2e_markdown,
    simulate_queue_parity_e2e_bar_series,
)


def test_upstream_platforms_and_strategy_library_invariants() -> None:
    assert len(SOTA_UPSTREAM_BACKTEST_PLATFORMS) == 4
    sha_re = re.compile(r"^[0-9a-f]{40}$")
    total_libs = 0
    for items in SOTA_UPSTREAM_BACKTEST_PLATFORMS.values():
        for lib in items:
            total_libs += 1
            assert sha_re.fullmatch(lib["pin_sha"]), f"Unpinned SHA for {lib['name']}"
    assert total_libs >= 18

    assert len(SOTA_UPSTREAM_STRATEGY_LIBRARY) == 12
    for sid, spec in SOTA_UPSTREAM_STRATEGY_LIBRARY.items():
        assert spec["strategy_id"] == sid
        assert 0.0 < spec["base_stop_pct"] <= spec["max_stop_pct"] <= ADR0022_MAX_STOP_PCT
        assert spec["take_profit_cap_enabled"] is False


def test_market_situation_classifier_regimes_and_cederburg_cap() -> None:
    bull = classify_market_situation(
        MarketSituationInput(
            session_segment="RTH_CORE",
            rsp_spy_trend_z=0.9,
            hyg_lqd_trend_z=0.6,
            iwm_spy_trend_z=0.7,
            xly_xlp_trend_z=0.5,
            breadth_score_0_100=72.0,
            distribution_days_25d=1,
            vix_spot=13.5,
            vix_9d=12.8,
            vix_3m=16.5,
            uvxy_move_pct=-3.5,
            svix_move_pct=2.2,
            spy_move_pct=1.1,
            qqq_move_pct=1.4,
            realized_vol_ratio=0.75,
        )
    )
    assert bull.composite_market_situation == "BROADENING_BULL"
    assert bull.structural_macro_regime == "BROADENING"
    assert bull.tactical_vix_regime == "CALM_CONTANGO"
    assert bull.cederburg_crash_brake_mult <= 1.0  # Never levers above 1.0x in low vol
    assert bull.soft_vix_log_odds > 0.0

    # Real 2026-11-20 crisis backwardation telemetry
    crisis = classify_market_situation(
        MarketSituationInput(
            session_segment="RTH_CORE",
            rsp_spy_trend_z=-0.65,
            hyg_lqd_trend_z=-0.85,
            xly_xlp_trend_z=-0.60,
            breadth_score_0_100=22.0,
            distribution_days_25d=6,
            vix_spot=37.67,
            vix_9d=41.86,
            vix_3m=31.90,
            uvxy_move_pct=20.69,
            svix_move_pct=-13.96,
            spy_move_pct=-2.8,
            qqq_move_pct=-3.9,
            realized_vol_ratio=2.1,
        )
    )
    assert crisis.composite_market_situation == "CONTRACTION_RISK_OFF"
    assert crisis.tactical_vix_regime == "CRISIS_SPIKE"
    assert crisis.vix_term_ratio_9d_3m > 1.12
    assert 0.25 <= crisis.cederburg_crash_brake_mult < 0.70
    assert crisis.short_overlay_enabled is True
    assert "VIX_SPIKE_AND_CRUSH_REGIME_SURFER" in crisis.favored_strategies

    # Overnight BOATS must disable short overlay
    boats = classify_market_situation(MarketSituationInput(session_segment="OVERNIGHT_BOATS"))
    assert boats.composite_market_situation == "OVERNIGHT_BOATS"
    assert boats.short_overlay_enabled is False

    with pytest.raises(ValueError, match="finite"):
        classify_market_situation(MarketSituationInput(vix_spot=float("nan")))
    with pytest.raises(ValueError, match="finite"):
        classify_market_situation(MarketSituationInput(uvxy_move_pct=True))  # type: ignore[arg-type]


def test_queue_parity_simulator_stop_first_and_prewall_wall() -> None:
    # Intrabar collision: bar 1 drops below stop AND rallies +10% on same bar -> adverse stop MUST fire first
    collision_bars = [
        {"date": "2026-10-10", "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
        {"date": "2026-10-11", "open": 100.0, "high": 110.0, "low": 97.5, "close": 109.0},
        {"date": "2026-10-12", "open": 109.0, "high": 112.0, "low": 108.0, "close": 111.0},
    ]
    res = simulate_queue_parity_e2e_bar_series(
        symbol="NVDA",
        strategy_id="S04_MINERVINI_VCP_QULLAMAGGIE",
        bars=collision_bars,
        entry_indices=[0],
        direction="LONG",
        stop_pct=1.50,
        cadence="DAILY_SWING",
        max_hold_bars=5,
    )
    assert res["events_n"] == 1
    assert res["stop_check_before_mfe_update"] is True
    assert res["mean_gross_bps"] < -100.0  # Stopped out at ~-150 bps despite +10% bar high
    assert res["mean_2x_bps"] < res["mean_1x_bps"]
    assert res["take_profit_cap_enabled"] is False

    # Post-wall bar MUST fail closed
    postwall_bars = [
        {"date": "2026-10-15", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
        {"date": "2026-10-16", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5},
    ]
    with pytest.raises(ValueError, match="PREWALL_VIOLATION"):
        simulate_queue_parity_e2e_bar_series(
            symbol="SPY",
            strategy_id="S09_CONNORS_IBS_CONFORMAL_MR",
            bars=postwall_bars,
            entry_indices=[0],
        )

    # Quarantined strategy and >2.00% stop MUST fail closed
    with pytest.raises(ValueError, match="QUARANTINED_STRATEGY_BLOCKED"):
        simulate_queue_parity_e2e_bar_series(
            symbol="SPY",
            strategy_id="SOTA_DUAL_CLIMAX_STOP_AND_REVERSE",
            bars=collision_bars,
            entry_indices=[0],
        )
    with pytest.raises(ValueError, match="ADR-0022"):
        simulate_queue_parity_e2e_bar_series(
            symbol="SPY",
            strategy_id="S09_CONNORS_IBS_CONFORMAL_MR",
            bars=collision_bars,
            entry_indices=[0],
            stop_pct=2.50,
        )


def test_per_symbol_eb_regime_routing_safety_and_live_refusal() -> None:
    sit = classify_market_situation(
        MarketSituationInput(
            session_segment="RTH_CORE",
            vix_spot=24.0,
            vix_9d=25.5,
            vix_3m=22.0,
            uvxy_move_pct=7.5,
            breadth_score_0_100=38.0,
            distribution_days_25d=5,
        )
    )
    prewall_profiles = {
        "NVDA": {
            "tuned_1000rpm_near_lossless": {
                "initial_stop_pct": 1.15,
                "micro_be_arm_bps": 10.0,
                "micro_be_lock_bps": 4.0,
                "micro_trail_bps": 12.0,
                "passive_ttl_sec": 8.0,
                "max_spread_bps": 12.0,
            },
            "tuned_1000rpm_short_near_lossless": {
                "initial_stop_pct": 1.20,
                "micro_be_arm_bps": 10.0,
                "micro_be_lock_bps": 4.0,
                "micro_trail_bps": 12.0,
                "passive_ttl_sec": 8.0,
                "max_spread_bps": 15.0,
            },
        }
    }
    targets = [
        LiveTargetObservation(
            symbol="NVDA",
            price=135.50,
            move_pct=2.8,
            rvol_tod=3.4,
            ofi_pct=14.2,
            spread_bps=4.0,
            sip_inst_dark_pct=42.0,
            lit_iso_pct=28.0,
            zero_sell_score=74.0,
            early_potential_score=78.0,
            has_sec_8k_catalyst=True,
            is_etb_shortable=True,
            symbol_empirical_n=35,
            symbol_empirical_mean_2x_bps=12.5,
        ),
        LiveTargetObservation(
            symbol="DILU",
            price=8.40,
            move_pct=4.5,
            rvol_tod=5.1,
            ofi_pct=11.0,
            spread_bps=7.0,
            has_sec_90d_dilution=True,
        ),
        LiveTargetObservation(
            symbol="SHRT",
            price=42.00,
            move_pct=-6.5,
            rvol_tod=4.0,
            ofi_pct=-12.5,
            spread_bps=6.0,
            has_sec_90d_dilution=True,
            is_etb_shortable=True,
        ),
        LiveTargetObservation(
            symbol="GAPF",
            price=25.00,
            move_pct=-5.0,
            rvol_tod=3.0,
            ofi_pct=-10.0,
            spread_bps=6.0,
            is_etb_shortable=True,
            is_boats_gap_fade_window=True,
        ),
    ]

    plans = apply_regime_conditioned_per_symbol_factors(
        targets=targets,
        market_situation=sit,
        per_symbol_prewall_profiles=prewall_profiles,
        venue_lane=":8787",
    )
    by_sym = {p.symbol: p for p in plans}

    assert by_sym["NVDA"].eligible is True
    assert by_sym["NVDA"].prewall_calibrated is True
    assert by_sym["NVDA"].peer_cluster == "HIGH_BETA_SEMI"
    assert 0.0 < by_sym["NVDA"].eb_shrinkage_weight_ratio < 1.0
    assert 0.0 < by_sym["NVDA"].quantized_stop_pct <= ADR0022_MAX_STOP_PCT
    assert by_sym["NVDA"].take_profit_cap_enabled is False
    assert by_sym["NVDA"].live_capital_refused is True

    # 90d dilution MUST block Long
    assert by_sym["DILU"].eligible is False
    assert "SEC_90D_DILUTION_LONG_BLOCK" in by_sym["DILU"].rejection_reasons

    # 90d dilution + negative OFI + ETB MUST route cleanly to SHORT
    assert by_sym["SHRT"].eligible is True
    assert by_sym["SHRT"].direction == "SHORT"
    assert by_sym["SHRT"].selected_strategy_id == "SEC_DILUTION_OVERHANG_ETB_SHORT"

    # ADR-0032 Trial #105 blind BOATS gap-fade short MUST be blocked
    assert by_sym["GAPF"].eligible is False
    assert "ADR0032_TRIAL105_BLIND_BOATS_GAP_FADE_SHORT_BLOCK" in by_sym["GAPF"].rejection_reasons

    # Live-capital venues MUST be 100% hard-refused
    for bad_lane in (":8798", ":8796", ":4001", ":7496", "api.alpaca.markets", "live"):
        with pytest.raises(ValueError, match="FAIL_CLOSED_LIVE_CAPITAL_REFUSAL"):
            apply_regime_conditioned_per_symbol_factors(
                targets=targets[:1],
                market_situation=sit,
                venue_lane=bad_lane,
            )


@given(
    vix_spot=st.floats(min_value=9.0, max_value=75.0, allow_nan=False, allow_infinity=False),
    uvxy_mv=st.floats(min_value=-25.0, max_value=45.0, allow_nan=False, allow_infinity=False),
    price=st.floats(min_value=3.0, max_value=1500.0, allow_nan=False, allow_infinity=False),
    ofi=st.floats(min_value=-60.0, max_value=60.0, allow_nan=False, allow_infinity=False),
    spread=st.floats(min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    emp_n=st.integers(min_value=0, max_value=200),
    emp_mean=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=120, deadline=None)
def test_hypothesis_invariants_always_hold(
    vix_spot: float,
    uvxy_mv: float,
    price: float,
    ofi: float,
    spread: float,
    emp_n: int,
    emp_mean: float,
) -> None:
    sit = classify_market_situation(
        MarketSituationInput(
            session_segment="RTH_CORE",
            vix_spot=vix_spot,
            vix_9d=max(8.0, vix_spot * 1.02),
            vix_3m=max(8.0, vix_spot * 0.95),
            uvxy_move_pct=uvxy_mv,
        )
    )
    assert 0.15 <= sit.cederburg_crash_brake_mult <= 1.0
    t = LiveTargetObservation(
        symbol="AAPL",
        price=price,
        move_pct=2.0,
        rvol_tod=3.0,
        ofi_pct=ofi,
        spread_bps=spread,
        is_etb_shortable=True,
        symbol_empirical_n=emp_n,
        symbol_empirical_mean_2x_bps=emp_mean,
    )
    plans = apply_regime_conditioned_per_symbol_factors(targets=[t], market_situation=sit, venue_lane=":8787")
    assert len(plans) == 1
    p = plans[0]
    assert 0.0 < p.quantized_stop_pct <= ADR0022_MAX_STOP_PCT
    assert 0.15 <= p.position_size_scale_mult <= 1.0
    assert p.take_profit_cap_enabled is False
    assert p.live_capital_refused is True
    assert p.paper_venue_lane in ALLOWED_PAPER_VENUE_LANES
    assert p.paper_venue_lane not in FORBIDDEN_LIVE_LANES_EXTENDED
    if emp_n == 0:
        assert p.eb_shrinkage_weight_ratio == 0.0
        assert p.eb_posterior_bps == p.cluster_prior_bps


def test_manifest_sha_determinism_and_sidecar_http_routes(tmp_path: pathlib.Path) -> None:
    sit = classify_market_situation(MarketSituationInput(session_segment="RTH_CORE"))
    plans = apply_regime_conditioned_per_symbol_factors(
        targets=[
            LiveTargetObservation(
                symbol="SPY",
                price=590.0,
                move_pct=0.8,
                rvol_tod=2.5,
                ofi_pct=8.0,
                spread_bps=2.0,
                is_etb_shortable=True,
            )
        ],
        market_situation=sit,
        venue_lane=":8787",
    )
    m1 = build_upstream_lib_e2e_manifest(
        market_situation=sit,
        target_plans=plans,
        upstream_library_runs={"vectorbt": {"executed": True, "version": "1.2.0", "summary": "ok"}},
        prewall_sim_summary=[],
        generated_at_utc="2026-11-20T21:00:00Z",
    )
    m2 = build_upstream_lib_e2e_manifest(
        market_situation=sit,
        target_plans=plans,
        upstream_library_runs={"vectorbt": {"executed": True, "version": "1.2.0", "summary": "ok"}},
        prewall_sim_summary=[],
        generated_at_utc="2026-11-20T22:00:00Z",
    )
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    assert m1["manifest_sha256"] == compute_manifest_sha256(m1)
    md = render_upstream_lib_e2e_markdown(m1)
    assert m1["manifest_sha256"] in md
    assert "ADR-0035" in md
    assert infer_peer_cluster("NVDA") == "HIGH_BETA_SEMI"

    # Spin up local sidecar server on ephemeral port and verify ADR-0035 routes + 403 live refusal
    manifest_file = tmp_path / "manifest.json"
    state = build_sidecar_state(tmp_path)
    state["e2e_manifest_path"] = str(manifest_file)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_sidecar_handler(state))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        with urllib.request.urlopen(f"{base}/health", timeout=2.0) as r:
            assert json.loads(r.read())["ok"] is True

        # Missing manifest fails closed with 404
        with pytest.raises(urllib.error.HTTPError) as exc_nf:
            urllib.request.urlopen(f"{base}/api/upstream_lib_e2e/manifest", timeout=2.0)
        assert exc_nf.value.code == 404

        manifest_file.write_text(json.dumps(m1, allow_nan=False), encoding="utf-8")
        with urllib.request.urlopen(f"{base}/api/upstream_lib_e2e/manifest", timeout=2.0) as r:
            served = json.loads(r.read())
        assert served["manifest_sha256"] == m1["manifest_sha256"]

        # POST live target routing on sanctioned paper lane
        req_ok = urllib.request.Request(
            f"{base}/api/upstream_lib_e2e/route_targets",
            data=json.dumps(
                {
                    "venue_lane": ":8787",
                    "market_situation": {"session_segment": "RTH_CORE", "vix_spot": 19.0},
                    "targets": [
                        {
                            "symbol": "SPY",
                            "price": 590.0,
                            "move_pct": 0.9,
                            "rvol_tod": 2.4,
                            "ofi_pct": 9.5,
                            "spread_bps": 2.0,
                        }
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ok, timeout=2.0) as r:
            routed = json.loads(r.read())
        assert routed["ok"] is True
        assert routed["eligible_n"] == 1

        # POST to forbidden live lane MUST fail closed with HTTP 403
        req_live = urllib.request.Request(
            f"{base}/api/upstream_lib_e2e/route_targets",
            data=json.dumps(
                {
                    "venue_lane": ":8798",
                    "market_situation": {"session_segment": "RTH_CORE"},
                    "targets": [
                        {
                            "symbol": "SPY",
                            "price": 590.0,
                            "move_pct": 0.5,
                            "rvol_tod": 2.0,
                            "ofi_pct": 5.0,
                            "spread_bps": 2.0,
                        }
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_live:
            urllib.request.urlopen(req_live, timeout=2.0)
        assert exc_live.value.code == 403
    finally:
        srv.shutdown()
        srv.server_close()
