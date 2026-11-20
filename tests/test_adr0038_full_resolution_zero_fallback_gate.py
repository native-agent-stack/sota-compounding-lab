"""Tests for ADR-0038: Tier-0 Full-Resolution, Always-Full-Resolution, Zero-Residual, Zero-Fallback Rule."""

from __future__ import annotations

import json
import pathlib
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sota_compounding.gpu_scoring_sidecar import build_sidecar_state, make_sidecar_handler
from sota_compounding.upstream_lib_strategy_e2e import (
    FORBIDDEN_RESIDUAL_OR_FALLBACK_RUN_KEYS,
    REQUIRED_FULL_RESOLUTION_ENGINES,
    LiveTargetObservation,
    MarketSituationInput,
    apply_regime_conditioned_per_symbol_factors,
    audit_python_source_for_full_resolution,
    build_upstream_lib_e2e_manifest,
    classify_market_situation,
    enforce_full_resolution_zero_fallback_gate,
)


def _valid_full_resolution_runs() -> dict[str, Any]:
    return {
        "polars_duckdb_pandera_xnys": {
            "executed": True,
            "synthetic": False,
            "fallback_used": False,
            "version": "polars 2.1.0 / duckdb 2.0.0 / pandera 0.35.0 / xcals 4.13.2",
            "sessions_n": 700,
            "total_1m_bars_aggregated": 7_836_606,
            "non_xnys_n": 0,
            "post_wall_n": 0,
            "summary": "700 official XNYS sessions, 0 post-wall bytes",
        },
        "nautilus_trader": {
            "executed": True,
            "synthetic": False,
            "fallback_used": False,
            "version": "2.0.0rc9",
            "instruments_n": 15,
            "bars_or_ticks_loaded": 10_500,
            "strategies_n": 1,
            "iterations": 10_500,
            "orders_total": 420,
            "fills_total": 310,
            "positions_total": 155,
            "summary": "Full-resolution Rust BacktestEngine replay: 15 instruments, 10,500 iterations",
        },
        "hftbacktest": {
            "executed": True,
            "synthetic": False,
            "fallback_used": False,
            "engine_kind": "RUST_HFTBACKTEST_HASHMAP_DEPTH",
            "queue_model": "RiskAdverseQueueModel",
            "version": "2.4.2",
            "slices_replayed": 12,
            "events_replayed": 185_000,
            "passive_orders_submitted": 12,
            "passive_fills": 9,
            "post_wall_n": 0,
            "summary": "Rust hftbacktest v2.4.2 HashMapMarketDepthBacktest + RiskAdverseQueueModel",
        },
        "vectorbt": {
            "executed": True,
            "synthetic": False,
            "fallback_used": False,
            "version": "1.2.0",
            "plateau_results": [{"config": "VBT_20D_BREAKOUT", "median_sharpe_2x": 0.81}],
            "summary": "4 plateau sweeps over 15 symbols x 700 sessions",
        },
        "statsmodels": {
            "executed": True,
            "synthetic": False,
            "fallback_used": False,
            "version": "0.15.0",
            "hac_t_by_strategy": {"VBT_20D_BREAKOUT": 2.06},
            "summary": "OLS Newey-West HAC max_t=+2.0600",
        },
        "skfolio": {
            "executed": True,
            "synthetic": False,
            "fallback_used": False,
            "version": "1.12.7",
            "cpcv_splits_n": 15,
            "summary": "CombinatorialPurgedCV C(6,2)=15 purged+embargoed splits",
        },
        "ml4t_diagnostic": {
            "executed": True,
            "synthetic": False,
            "fallback_used": False,
            "version": "0.2.0",
            "dsr_probability": 0.42,
            "whites_rc_p": 0.12,
            "summary": "DSR=0.4200, WRC p=0.1200",
        },
        "bashtage_arch": {
            "executed": True,
            "synthetic": False,
            "fallback_used": False,
            "version": "9.0.0",
            "spa_p_consistent": 0.08,
            "mcs_surviving_set": ["VBT_20D_BREAKOUT"],
            "summary": "Hansen SPA p=0.0800; MCS M*=['VBT_20D_BREAKOUT']",
        },
        "rtx4090_gpu_cuda": {
            "executed": True,
            "synthetic": False,
            "fallback_used": False,
            "version": "torch-cuda / NVIDIA GeForce RTX 4090",
            "device": "cuda:0",
            "n_boot": 10_000,
            "summary": "RTX 4090 (cuda:0) 10,000-rep circular block bootstrap",
        },
    }


def test_full_resolution_gate_passes_authentic_9_engine_runs() -> None:
    runs = _valid_full_resolution_runs()
    audit = enforce_full_resolution_zero_fallback_gate(runs, require_all_engines=True)
    assert audit["full_resolution_verified"] is True
    assert audit["allow_silent_fallback"] is False
    assert audit["residual_fallbacks_count"] == 0
    assert audit["engines_verified_n"] == len(REQUIRED_FULL_RESOLUTION_ENGINES)
    assert audit["violations"] == []


def test_full_resolution_gate_rejects_empty_nautilus_boot_stub() -> None:
    runs = _valid_full_resolution_runs()
    runs["nautilus_trader"] = {
        "executed": True,
        "version": "2.0.0rc9",
        "iterations": 0,
        "summary": "BacktestEngine Rust-core XNAS NETTING/MARGIN verified (run_ok=True, iterations=0)",
    }
    audit = enforce_full_resolution_zero_fallback_gate(runs, require_all_engines=True)
    assert audit["full_resolution_verified"] is False
    assert audit["residual_fallbacks_count"] >= 1
    assert "NAUTILUS_ZERO_ITERATIONS_BOOT_STUB" in audit["violations"]


def test_full_resolution_gate_rejects_python_residual_hftbacktest_and_cpu_fallback() -> None:
    runs = _valid_full_resolution_runs()
    runs["hftbacktest_queue_parity_ref"] = {
        "executed": True,
        "version": "STOP_CHECK_BEFORE_MFE_UPDATE",
        "summary": "30 bar loops",
    }
    runs["rtx4090_gpu_cuda"]["device"] = "python-cpu"
    audit = enforce_full_resolution_zero_fallback_gate(runs, require_all_engines=True)
    assert audit["full_resolution_verified"] is False
    assert "FORBIDDEN_RESIDUAL_OR_FALLBACK_ENGINE:hftbacktest_queue_parity_ref" in audit["violations"]
    assert any(v.startswith("GPU_CUDA_REQUIRED_NO_CPU_FALLBACK") for v in audit["violations"])


def test_strict_manifest_builder_raises_on_any_residual_or_missing_engine() -> None:
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
    ok_manifest = build_upstream_lib_e2e_manifest(
        market_situation=sit,
        target_plans=plans,
        upstream_library_runs=_valid_full_resolution_runs(),
        prewall_sim_summary=[],
        generated_at_utc="2026-11-20T22:00:00Z",
        strict_full_resolution=True,
    )
    assert ok_manifest["full_resolution_audit"]["full_resolution_verified"] is True

    bad_runs = _valid_full_resolution_runs()
    del bad_runs["hftbacktest"]
    with pytest.raises(ValueError, match="FAIL_CLOSED_FULL_RESOLUTION_VIOLATION"):
        build_upstream_lib_e2e_manifest(
            market_situation=sit,
            target_plans=plans,
            upstream_library_runs=bad_runs,
            prewall_sim_summary=[],
            strict_full_resolution=True,
        )


def test_ast_audit_catches_empty_backtest_engine_boot_stub_including_inline() -> None:
    bad_src = '''
def boot():
    code = """
from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
e = BacktestEngine(BacktestEngineConfig())
e.add_venue(venue="XNAS")
e.run()
"""
    return code
'''
    issues = audit_python_source_for_full_resolution(bad_src, "scripts/bad_boot.py")
    assert len(issues) == 1
    assert "EMPTY_NAUTILUS_BACKTEST_ENGINE_STUB" in issues[0]

    good_src = """
def real_replay():
    e = BacktestEngine(BacktestEngineConfig())
    e.add_venue(venue="XNAS")
    e.add_instrument(eq)
    e.add_data(bars)
    e.add_strategy(strat)
    e.run()
"""
    assert audit_python_source_for_full_resolution(good_src, "scripts/real_replay.py") == []


def test_sidecar_full_resolution_routes_fail_closed_on_degraded_manifest(tmp_path: pathlib.Path) -> None:
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
    bad_manifest = build_upstream_lib_e2e_manifest(
        market_situation=sit,
        target_plans=plans,
        upstream_library_runs={"vectorbt": {"executed": True, "version": "1.2.0", "summary": "only 1 lib"}},
        prewall_sim_summary=[],
        strict_full_resolution=False,
    )
    good_manifest = build_upstream_lib_e2e_manifest(
        market_situation=sit,
        target_plans=plans,
        upstream_library_runs=_valid_full_resolution_runs(),
        prewall_sim_summary=[],
        strict_full_resolution=True,
    )

    manifest_file = tmp_path / "manifest.json"
    state = build_sidecar_state(tmp_path)
    state["e2e_manifest_path"] = str(manifest_file)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_sidecar_handler(state))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        manifest_file.write_text(json.dumps(bad_manifest, allow_nan=False), encoding="utf-8")
        with pytest.raises(urllib.error.HTTPError) as exc_bad:
            urllib.request.urlopen(f"{base}/api/full_resolution/manifest", timeout=2.0)
        assert exc_bad.value.code == 503

        manifest_file.write_text(json.dumps(good_manifest, allow_nan=False), encoding="utf-8")
        with urllib.request.urlopen(f"{base}/api/full_resolution/status", timeout=2.0) as r:
            st_obj = json.loads(r.read())
        assert st_obj["ok"] is True
        assert st_obj["full_resolution_verified"] is True
        assert st_obj["allow_silent_fallback"] is False
    finally:
        srv.shutdown()
        srv.server_close()


@given(
    forbidden_key=st.sampled_from(sorted(FORBIDDEN_RESIDUAL_OR_FALLBACK_RUN_KEYS)),
    bad_iters=st.integers(min_value=-10, max_value=0),
    post_wall=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=80, deadline=None)
def test_hypothesis_no_residual_stub_or_postwall_ever_passes(
    forbidden_key: str,
    bad_iters: int,
    post_wall: int,
) -> None:
    r1 = _valid_full_resolution_runs()
    r1[forbidden_key] = {"executed": True, "version": "1.0.0", "summary": "sneak"}
    assert enforce_full_resolution_zero_fallback_gate(r1)["full_resolution_verified"] is False

    r2 = _valid_full_resolution_runs()
    r2["nautilus_trader"]["iterations"] = bad_iters
    assert enforce_full_resolution_zero_fallback_gate(r2)["full_resolution_verified"] is False

    r3 = _valid_full_resolution_runs()
    r3["polars_duckdb_pandera_xnys"]["post_wall_n"] = post_wall
    assert enforce_full_resolution_zero_fallback_gate(r3)["full_resolution_verified"] is False
