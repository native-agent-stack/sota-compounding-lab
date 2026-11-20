"""Tests for `v3.11.1` Post-ANT whole-market strategy factory & quarantine consolidation (`ADR-0028`)."""

from __future__ import annotations

from datetime import date

import pytest

from sota_compounding.converged_strategy_factory import (
    POST_ANT_ERA_ID,
    RETIRED_TRAP_DENYLIST,
    TradeReflectionRecord,
    check_retired_trap_firewall,
    evaluate_post_ant_symbol_merit_contract,
    reflect_and_finetune_batch,
)


def test_post_ant_merit_contract_firewall_reflection_and_failclosed_defaults() -> None:
    assert "ADR0064_QUARANTINED_NOISY_PAPER_STRATEGIES" in RETIRED_TRAP_DENYLIST
    assert "SINGLE_TICKER_IPO_DAY1_OVERFIT" in RETIRED_TRAP_DENYLIST

    # Quarantined IDs & family variants blocked in firewall (incl. scalar str!) AND refused in reflection!
    fw_ok, fw_v = check_retired_trap_firewall(tags=["POST_VIX_REVERSAL", "POST_VIX_REVERSAL_CUSTOM"])
    assert fw_ok is False and any("QUARANTINED_ADR0064" in v for v in fw_v)
    fw_str_ok, fw_str_v = check_retired_trap_firewall(tags="POST_VIX_REVERSAL")
    assert fw_str_ok is False and any("QUARANTINED_ADR0064" in v for v in fw_str_v)
    with pytest.raises(ValueError, match="QUARANTINED_ADR0064"):
        reflect_and_finetune_batch("POST_VIX_REVERSAL", [])

    # Pass-2 Finding 8: quarantined or mismatched inner record MUST NOT be credited to an active batch!
    quarantined_rec = TradeReflectionRecord(
        strategy_id="POST_VIX_REVERSAL",
        regime="RTH_CORE",
        realized_r=1.2,
        mfe_r=1.5,
        mae_r=-0.2,
        slippage_bps=1.5,
        spread_bps=4.0,
        entry_ofi=0.35,
        stop_pct=1.25,
        exit_reason="CHANDELIER",
    )
    with pytest.raises(ValueError, match="QUARANTINED_ADR0064"):
        reflect_and_finetune_batch("BJZZ_SUBPENNY_INST_DARK_SWEEP", [quarantined_rec])

    mismatched_rec = TradeReflectionRecord(
        strategy_id="H01_PRE_TRAVEL_POCKET_PIVOT",
        regime="RTH_CORE",
        realized_r=0.8,
        mfe_r=1.1,
        mae_r=-0.2,
        slippage_bps=1.5,
        spread_bps=4.0,
        entry_ofi=0.35,
        stop_pct=1.25,
        exit_reason="CHANDELIER",
    )
    with pytest.raises(ValueError, match="STRATEGY_ID_MISMATCH"):
        reflect_and_finetune_batch("BJZZ_SUBPENNY_INST_DARK_SWEEP", [mismatched_rec])

    # Pass-3 Findings 7 & 8:
    # - Case-mismatched record rejected; lowercase catalog batch gets exact 0.85% catalog baseline (not 1.50%)!
    # - Noise-VWAP reflection enforces published 0.50% stop ceiling!
    case_rec = TradeReflectionRecord(
        strategy_id="SOTA_MEGA_CAP_LOSSLESS_ORB_PRECURSOR_V1",
        regime="RTH_CORE",
        realized_r=0.5,
        mfe_r=1.0,
        mae_r=-0.1,
        slippage_bps=1.0,
        spread_bps=3.0,
        entry_ofi=0.35,
        stop_pct=0.80,
        exit_reason="CHANDELIER",
    )
    with pytest.raises(ValueError, match="STRATEGY_ID_MISMATCH"):
        reflect_and_finetune_batch("sota_mega_cap_lossless_orb_precursor_v1", [case_rec])
    d_lower = reflect_and_finetune_batch("sota_mega_cap_lossless_orb_precursor_v1", [])
    assert d_lower.strategy_id == "SOTA_MEGA_CAP_LOSSLESS_ORB_PRECURSOR_V1"
    assert d_lower.recommended_stop_pct == 0.85

    d_vwap = reflect_and_finetune_batch("ZARATTINI_2025_NOISE_VWAP", [])
    assert d_vwap.recommended_stop_pct <= 0.50

    # Pass-4 P1s: trailing whitespace identity rejected; VIX reflection enforces symbol allowlist;
    # Noise-VWAP tightening MUST tighten (<3.0 bps), NEVER widen to 6.0 bps!
    ws_rec = TradeReflectionRecord(
        strategy_id="BJZZ_SUBPENNY_INST_DARK_SWEEP ",
        regime="RTH_CORE",
        realized_r=0.5,
        mfe_r=1.0,
        mae_r=-0.1,
        slippage_bps=1.0,
        spread_bps=4.0,
        entry_ofi=0.35,
        stop_pct=1.25,
        exit_reason="CHANDELIER",
    )
    with pytest.raises(ValueError, match="STRATEGY_ID_MISMATCH"):
        reflect_and_finetune_batch("BJZZ_SUBPENNY_INST_DARK_SWEEP", [ws_rec])

    vix_bad_sym = TradeReflectionRecord(
        strategy_id="VIX_SPIKE_AND_CRUSH_REGIME_SURFER",
        regime="RTH_CORE",
        realized_r=0.6,
        mfe_r=1.0,
        mae_r=-0.1,
        slippage_bps=1.0,
        spread_bps=5.0,
        entry_ofi=0.35,
        stop_pct=1.00,
        exit_reason="CHANDELIER",
        symbol="NVDA",
    )
    with pytest.raises(ValueError, match="STRATEGY_SYMBOL_NOT_ALLOWED"):
        reflect_and_finetune_batch("VIX_SPIKE_AND_CRUSH_REGIME_SURFER", [vix_bad_sym])

    vwap_rec = TradeReflectionRecord(
        strategy_id="ZARATTINI_2025_NOISE_VWAP",
        regime="RTH_CORE",
        realized_r=0.4,
        mfe_r=0.8,
        mae_r=-0.1,
        slippage_bps=1.0,
        spread_bps=2.8,
        entry_ofi=0.30,
        stop_pct=0.50,
        exit_reason="EOD",
        symbol="SPY",
    )
    d_vwap_tight = reflect_and_finetune_batch("ZARATTINI_2025_NOISE_VWAP", [vwap_rec])
    assert d_vwap_tight.recommended_max_spread_bps < 3.0
    assert d_vwap_tight.recommended_stop_pct <= 0.50

    # Pass-2/Pass-3 Finding 7: per-strategy ADR-0064 contract parity (H090 30bps & VIX non-ETF rejected!)
    h090_wide = evaluate_post_ant_symbol_merit_contract(
        symbol="PLTR",
        strategy_id="H090_TIGHT_SPREAD_HIGH_OFI_PASSIVE_BID",
        merit_score=70.0,
        df_pillars_manifested=5,
        spread_bps=30.0,
        requested_stop_pct=1.25,
        sec_dilution_clean=True,
        session_mode="RTH_CORE",
    )
    assert h090_wide["admitted"] is False
    assert any("STRATEGY_SPREAD_EXCEEDS_CONTRACT" in r for r in h090_wide["rejection_reasons"])

    ok = evaluate_post_ant_symbol_merit_contract(
        symbol="PLTR",
        strategy_id="BJZZ_SUBPENNY_INST_DARK_SWEEP",
        merit_score=68.0,
        df_pillars_manifested=5,
        spread_bps=7.5,
        requested_stop_pct=1.50,
        sec_dilution_clean=True,
    )
    assert ok["era"] == POST_ANT_ERA_ID
    assert ok["admitted"] is True
    assert ok["merit_tier"] == "MERIT_TIER_1_CORE"
    assert ok["take_profit_cap_enabled"] is False
    assert ok["standalone_alpha_claim"] is False

    # Omitted sec_dilution_clean defaults fail-closed; impossible pillars > 6 rejected
    bad_q = evaluate_post_ant_symbol_merit_contract(
        symbol="PLTR",
        strategy_id="POST_VIX_REVERSAL_V2",
        merit_score=85.0,
        df_pillars_manifested=9,
        spread_bps=5.0,
    )
    assert bad_q["admitted"] is False
    assert any("QUARANTINED_ADR0064" in r for r in bad_q["rejection_reasons"])
    assert "SEC_90D_DILUTION_UNVERIFIED_OR_BLOCKED" in bad_q["rejection_reasons"]
    assert any("INVALID_DF_PILLARS_COUNT" in r for r in bad_q["rejection_reasons"])

    with pytest.raises(ValueError, match="Pre-wall breach"):
        evaluate_post_ant_symbol_merit_contract(
            symbol="PLTR",
            strategy_id="BJZZ_SUBPENNY_INST_DARK_SWEEP",
            merit_score=68.0,
            df_pillars_manifested=5,
            spread_bps=7.5,
            sec_dilution_clean=True,
            study_wall_cutoff=date(2026, 10, 16),
        )
