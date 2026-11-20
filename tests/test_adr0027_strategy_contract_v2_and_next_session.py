"""Unit & invariant tests for `ADR-0027` (`sota.strategy-contract.v2`, `next_session_date`, & TCA calibration)."""

from __future__ import annotations

import pytest

from sota_compounding.strategy_contract_v2 import (
    QUARANTINED_PAPER_STRATEGIES,
    STRATEGY_CONTRACT_SCHEMA_V2,
    build_client_order_id_v2,
    calibrate_sim_cost_from_paper_tca,
    contract_v2_summary,
    parse_client_order_id_v2,
    validate_next_session_date_alignment,
)


def test_strategy_contract_v2_client_ids_and_quarantine() -> None:
    coid = build_client_order_id_v2("BJZZ_SUBPENNY_INST_DARK_SWEEP", "p2", 123)
    assert coid == "sota-bjzz_dark_sweep-p2-000123"
    assert len(coid) <= 48
    parsed = parse_client_order_id_v2(coid)
    assert parsed is not None
    assert parsed["strategy_id"] == "BJZZ_SUBPENNY_INST_DARK_SWEEP"
    assert parsed["lane"] == "p2"
    assert parsed["seq"] == "000123"

    coid_obb = build_client_order_id_v2("BOATS_24X5_OPENBB_STAGED_LIMIT", "p2", 7)
    assert coid_obb == "sota-boats_obb_stage-p2-000007"
    p_obb = parse_client_order_id_v2(coid_obb)
    assert p_obb is not None and p_obb["strategy_id"] == "BOATS_24X5_OPENBB_STAGED_LIMIT"
    with pytest.raises(ValueError, match="Invalid seq"):
        build_client_order_id_v2("BJZZ_SUBPENNY_INST_DARK_SWEEP", "p2", 1_000_000_000_000)
    with pytest.raises(ValueError, match="Invalid seq"):
        build_client_order_id_v2("BJZZ_SUBPENNY_INST_DARK_SWEEP", "p2", 999_999_999_999.5)  # type: ignore[arg-type]

    assert "SOTA_DUAL_CLIMAX_STOP_AND_REVERSE" in QUARANTINED_PAPER_STRATEGIES
    assert "POST_VIX_REVERSAL" in QUARANTINED_PAPER_STRATEGIES
    assert "FLOW_VELOCITY_CAPITAL_RECYCLE" in QUARANTINED_PAPER_STRATEGIES
    summ = contract_v2_summary()
    assert summ["schema_version"] == STRATEGY_CONTRACT_SCHEMA_V2


def test_next_session_date_alignment_blocks_gap_jump_and_postwall() -> None:
    sessions = ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]
    res = validate_next_session_date_alignment(
        ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"],
        ["2026-10-13", "2026-10-14", "2026-10-15", None],
        sessions,
    )
    assert res["ok"] is True and res["verified_1step_rows"] == 3 and res["wall_boundary_nulls"] == 1

    # Gap jump (skipping 2026-10-13 -> 2026-10-14 causes CPCV overlap) MUST fail closed
    with pytest.raises(ValueError, match="NEXT_SESSION_DATE_MISMATCH"):
        validate_next_session_date_alignment(
            ["2026-10-12"],
            ["2026-10-14"],
            sessions,
        )

    # Unknown non-XNYS pre-wall date with null successor MUST fail closed (never treated as wall boundary!)
    with pytest.raises(ValueError, match="NEXT_SESSION_DATE_MISMATCH"):
        validate_next_session_date_alignment(
            ["2026-10-10"],
            [None],
            sessions,
        )

    # Post-wall next_session_date MUST fail closed
    with pytest.raises(ValueError, match="POSTWALL_BREACH"):
        validate_next_session_date_alignment(
            ["2026-10-15"],
            ["2026-10-16"],
            sessions,
        )


def test_tca_cost_calibration_feedback_is_conservative() -> None:
    # Even if noisy paper sample has 0 slippage, never reduces below prior_half_spread_bps
    cal_low = calibrate_sim_cost_from_paper_tca(
        strategy_id="ZARATTINI_2024_5M_ORB",
        paper_fills_count=5,
        observed_mean_slippage_bps=0.5,
        observed_adverse_markout_5s_bps=1.0,
        prior_half_spread_bps=4.0,
    )
    assert cal_low.calibrated_1x_half_spread_bps >= 4.0
    assert cal_low.calibrated_2x_stress_bps == cal_low.calibrated_1x_half_spread_bps * 2.0

    # High real slippage + negative markout ratchets cost model upward conservatively
    cal_hi = calibrate_sim_cost_from_paper_tca(
        strategy_id="ZARATTINI_2024_5M_ORB",
        paper_fills_count=40,
        observed_mean_slippage_bps=9.0,
        observed_adverse_markout_5s_bps=-8.0,
        prior_half_spread_bps=4.0,
    )
    assert cal_hi.calibrated_1x_half_spread_bps > 4.0
    assert cal_hi.feedback_applied is True
