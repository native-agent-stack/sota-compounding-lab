"""Deterministic & Hypothesis property tests for `ADR-0021` Converged Strategy Factory & Reflection Engine."""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sota_compounding import (
    ADR0022_MAX_STOP_PCT,
    CONVERGED_FOUNDATION_CATALOG,
    RETIRED_TRAP_DENYLIST,
    STARRED_FRAMEWORK_ADAPTERS,
    STUDY_WALL_DATE,
    TradeReflectionRecord,
    build_noesis_prewall_factory_plan,
    check_retired_trap_firewall,
    reflect_and_finetune_batch,
)


def test_converged_catalog_respects_adr0022_and_includes_six_archetypes() -> None:
    assert len(CONVERGED_FOUNDATION_CATALOG) == 6
    assert "IPO_DAY1_CROSS_STABILIZATION_AND_VWAP_WAVE2_V1" in CONVERGED_FOUNDATION_CATALOG
    assert "SOTA_SLOW_COIL_VDU_ACCUMULATION_V1" in CONVERGED_FOUNDATION_CATALOG
    for sid, spec in CONVERGED_FOUNDATION_CATALOG.items():
        assert 0.50 <= spec.default_stop_pct <= ADR0022_MAX_STOP_PCT, sid
        assert spec.max_spread_bps > 0.0
        assert spec.tier1_floor_gain_pct > 0.0
        assert spec.tier2_harvest_mfe_pct > spec.tier1_arm_mfe_pct


def test_retired_trap_firewall_blocks_all_known_negative_ev_traps() -> None:
    ok, v = check_retired_trap_firewall(tags=["EARLY_IGNITION"])
    assert ok is True
    assert v == ()

    for trap_key in RETIRED_TRAP_DENYLIST:
        ok_t, v_t = check_retired_trap_firewall(tags=[trap_key])
        assert ok_t is False
        assert f"RETIRED_TRAP:{trap_key}" in v_t

    ok_flags, v_flags = check_retired_trap_firewall(
        tags=[],
        is_post_incentive=True,
        holds_overnight_moc_to_moo=True,
        unshielded_lit_passive=True,
        in_opening_0930_0935_taker=True,
        ofi_only_without_joint_sweep=True,
    )
    assert ok_flags is False
    assert len(v_flags) == 5


def test_reflection_engine_detects_slippage_premature_stopouts_and_probation() -> None:
    # 1. Empty batch -> collect more samples
    d0 = reflect_and_finetune_batch("IPO_DAY1_CROSS_STABILIZATION_AND_VWAP_WAVE2_V1", [])
    assert d0.sample_count == 0
    assert d0.recommended_stop_pct <= ADR0022_MAX_STOP_PCT
    assert len(d0.digest_sha256) == 64

    # 2. High slippage + premature stopouts -> tighten collar & earlier Tier-1 lock while respecting <= 2.00%
    noisy_batch = [
        TradeReflectionRecord(
            strategy_id="IPO_DAY1_CROSS_STABILIZATION_AND_VWAP_WAVE2_V1",
            regime="RTH_CORE",
            realized_r=-0.8,
            mfe_r=0.9,
            mae_r=1.0,
            slippage_bps=8.5,
            spread_bps=24.0,
            entry_ofi=0.25,
            stop_pct=1.85,
            exit_reason="STOP_HIT",
        )
        for _ in range(12)
    ]
    d1 = reflect_and_finetune_batch("IPO_DAY1_CROSS_STABILIZATION_AND_VWAP_WAVE2_V1", noisy_batch)
    assert d1.recommended_stop_pct <= ADR0022_MAX_STOP_PCT
    assert d1.premature_stopout_rate_pct == 100.0
    assert "TIGHTEN_SPREAD_COLLAR_AND_REQUIRE_PASSIVE_MICROPRICE_PEG" in d1.reflection_actions
    assert "ARM_TIER1_LOSSLESS_FLOOR_EARLIER_AND_CALIBRATE_GK_VOL_STOP" in d1.reflection_actions
    assert "ENFORCE_PROBATION_015X_THROTTLE_AND_AUDIT_REGIME_DRIFT" in d1.reflection_actions
    assert d1.eb_tuning.alloc_multiplier == 0.15

    # 3. Rejects ADR-0022 > 2.00% or non-finite inputs fail-closed
    bad_rec = TradeReflectionRecord(
        strategy_id="IPO_DAY1_CROSS_STABILIZATION_AND_VWAP_WAVE2_V1",
        regime="RTH_CORE",
        realized_r=0.5,
        mfe_r=1.0,
        mae_r=0.2,
        slippage_bps=1.5,
        spread_bps=8.0,
        entry_ofi=0.35,
        stop_pct=2.25,
        exit_reason="HARVEST",
    )
    with pytest.raises(ValueError, match="ADR-0022"):
        reflect_and_finetune_batch("IPO_DAY1_CROSS_STABILIZATION_AND_VWAP_WAVE2_V1", [bad_rec])


def test_noesis_prewall_factory_plan_enforces_study_wall_and_hlz_hurdle() -> None:
    plan = build_noesis_prewall_factory_plan(cumulative_trials_n=101)
    assert plan["study_wall_date"] == STUDY_WALL_DATE.isoformat()
    assert plan["next_trial_number"] == 102
    assert plan["next_hlz_t_hurdle"] >= 4.20
    assert len(plan["plan_sha256"]) == 64
    assert len(plan["stages"]) == 4
    assert set(STARRED_FRAMEWORK_ADAPTERS.keys()) >= {"qlib", "kronos", "timesfm", "hftbacktest", "skfolio"}

    with pytest.raises(ValueError, match="Pre-wall breach"):
        build_noesis_prewall_factory_plan(cumulative_trials_n=101, study_wall_cutoff=date(2026, 10, 16))


@settings(max_examples=40, deadline=None)
@given(
    realized_r=st.floats(min_value=-3.0, max_value=8.0, allow_nan=False, allow_infinity=False),
    mfe_r=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    slip_bps=st.floats(min_value=-5.0, max_value=40.0, allow_nan=False, allow_infinity=False),
    spread_bps=st.floats(min_value=0.5, max_value=60.0, allow_nan=False, allow_infinity=False),
    stop_pct=st.floats(min_value=0.25, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_reflection_invariants_property(
    realized_r: float,
    mfe_r: float,
    slip_bps: float,
    spread_bps: float,
    stop_pct: float,
) -> None:
    rec = TradeReflectionRecord(
        strategy_id="SOTA_MEGA_CAP_LOSSLESS_ORB_PRECURSOR_V1",
        regime="RTH_CORE",
        realized_r=realized_r,
        mfe_r=mfe_r,
        mae_r=abs(min(0.0, realized_r)),
        slippage_bps=slip_bps,
        spread_bps=spread_bps,
        entry_ofi=0.30,
        stop_pct=stop_pct,
        exit_reason="TEST",
    )
    digest = reflect_and_finetune_batch("SOTA_MEGA_CAP_LOSSLESS_ORB_PRECURSOR_V1", [rec])
    assert 0.50 <= digest.recommended_stop_pct <= ADR0022_MAX_STOP_PCT
    assert 0.0 <= digest.capture_efficiency_ratio <= 1.0
    assert 0.0 <= digest.premature_stopout_rate_pct <= 100.0
    assert len(digest.digest_sha256) == 64
