"""Property-Based (`Hypothesis`) & 5-Stage Lifecycle Tests for `sota-compounding-lab`."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sota_compounding import (
    ADR0022_MAX_STOP_PCT,
    PaperExecutionMetrics,
    PrewallBacktestMetrics,
    SimStageMetrics,
    evaluate_full_lifecycle,
    export_broker_strategy_manifest,
    hlz_dynamic_t_hurdle,
    quantize_nms_stop,
    tune_empirical_bayes_ucb,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@settings(max_examples=150)
@given(
    entry_px=st.floats(min_value=1.00, max_value=2500.0, allow_nan=False, allow_infinity=False),
    stop_pct=st.floats(min_value=0.50, max_value=2.00, allow_nan=False, allow_infinity=False),
    side=st.sampled_from(["long", "short"]),
)
def test_property_nms_stop_never_exceeds_adr0022(
    entry_px: float,
    stop_pct: float,
    side: Literal["long", "short"],
) -> None:
    q_entry, q_stop, realized_pct = quantize_nms_stop(entry_px, stop_pct, side=side)
    assert 0.0 < realized_pct <= ADR0022_MAX_STOP_PCT + 1e-9
    if side == "long":
        assert q_stop < q_entry
    else:
        assert q_stop > q_entry


def test_hlz_hurdle_monotonically_increases() -> None:
    h1 = hlz_dynamic_t_hurdle(1)
    h10 = hlz_dynamic_t_hurdle(10)
    h80 = hlz_dynamic_t_hurdle(80)
    assert h1 == 3.0
    assert h1 < h10 < h80
    assert h80 >= 4.14


def test_five_stage_lifecycle_and_broker_firewall() -> None:
    sim = SimStageMetrics(causal_no_lookahead=True, gross_expectancy_bps=32.5, simulated_events=400)
    prewall = PrewallBacktestMetrics(
        sample_max_date="2026-10-15",
        cumulative_trials_n=80,
        events_count=340,
        distinct_dates=115,
        t_stat_clustered=4.45,
        dsr_probability=0.98,
        cpcv_n_paths=5,
        cpcv_pbo=0.10,
        net_return_1x_bps=24.0,
        net_return_2x_bps=13.2,
    )
    paper = PaperExecutionMetrics(
        attribution_source="REAL_BROKER_ONLY",
        closed_roundtrips=42,
        distinct_sessions=14,
        closed_mean_r=0.38,
        combined_expectancy_r=0.40,
        open_mtm_pct=0.25,
        mean_slippage_bps=-3.2,
        max_stop_pct_observed=1.88,
        alpaca_order_rpm_cap=200,
        ibkr_port=4002,
        eval_date="2026-11-17",
    )

    ev = evaluate_full_lifecycle("H07_PASSIVE_MID_MFE_TRAIL", sim, prewall, paper)
    assert ev.highest_stage_passed == "STAGE_4_EMPIRICAL_BAYES_PROMOTED"
    assert ev.cutover_regime == "pre_2026_12_06_24x5_boats"
    assert ev.tuning.tuning_regime == "PROMOTED_ALPHA_LEADER"

    manifest = export_broker_strategy_manifest([ev])
    assert len(manifest.manifest_sha256) == 64
    assert manifest.alpaca_paper_lane["max_order_rpm"] == 200
    assert manifest.ibkr_paper_lane["port"] == 4002

    # Post-wall leakage must hard-fail
    leaked = replace(prewall, sample_max_date="2026-10-16")
    with pytest.raises(ValueError, match="Study-Wall Violation"):
        evaluate_full_lifecycle("BAD_WALL", sim, leaked, paper)

    # Synthetic paper_sim splicing must block Stage 3 promotion
    spliced = replace(paper, attribution_source="PAPER_SIM_FALLBACK")
    ev_spliced = evaluate_full_lifecycle("SPLICED", sim, prewall, spliced)
    assert ev_spliced.paper_passed is False

    # Live broker hosts and live IBKR ports (4001/7496) must hard-fail
    live_host = "api." + "alpaca.markets"
    with pytest.raises(ValueError, match="Live-Capital Firewall"):
        export_broker_strategy_manifest([ev], alpaca_endpoint_host=live_host)
    with pytest.raises(ValueError, match="Live-Capital Firewall"):
        export_broker_strategy_manifest([ev], ibkr_port=4001)


def test_realized_loss_never_promoted_via_solo_mtm() -> None:
    losing = tune_empirical_bayes_ucb(
        "LOSER_LANE",
        closed_roundtrips=19,
        closed_mean_r=-0.05,
        combined_samples=20,
        combined_expectancy_r=0.18,
        open_mtm_pct=0.45,
        mean_slippage_bps=-4.0,
    )
    assert losing.tuning_regime != "PROMOTED_ALPHA_LEADER"

    bleeder = tune_empirical_bayes_ucb(
        "POST_VIX_REVERSAL_MAJOR_GROWTH_THRUST",
        closed_roundtrips=12,
        closed_mean_r=-0.68,
        combined_samples=12,
        combined_expectancy_r=-0.68,
        open_mtm_pct=0.0,
        mean_slippage_bps=-8.0,
        is_major_core=True,
    )
    assert bleeder.tuning_regime == "PROBATION_NEGATIVE_EB_THROTTLE"
    assert bleeder.alloc_multiplier == 0.15
    assert bleeder.ucb_priority_weight <= 0.45
    assert bleeder.min_ofi_pct >= 25.0


def test_adr_catalog_and_offline_org_firewall_scripts() -> None:
    p_adr = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "architecture_catalog.py"), "--check"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ADR-GATE PASS" in p_adr.stdout

    p_org = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "org_visibility_doctor.py"), "--offline"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"status": "PASS"' in p_org.stdout
