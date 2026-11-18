"""Unit & Hypothesis property tests for `sota-ranking-and-datafeeds.v2` (`ADR-0010`)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sota_compounding.ranking_and_datafeeds import (
    CANONICAL_7_PIT_FEEDS,
    RANKING_AND_DATAFEEDS_SCHEMA,
    classify_bjzz_dark_pool_print,
    classify_execution_quality_tier,
    compute_sota_mover_ranking,
    convert_sip_round_lots_to_shares,
    evaluate_first_hand_incentive,
    evaluate_stockbee_momentum_burst,
    validate_pit_datafeed_observation,
)


def test_pit_datafeed_validator_catches_lookahead_and_stale_boats() -> None:
    ok = validate_pit_datafeed_observation(
        feeds_present=CANONICAL_7_PIT_FEEDS,
        sec_acceptance_delta_sec=45.0,
        finra_short_vol_lag_days=1,
        finra_short_interest_lag_bdays=8,
        boats_quote_age_sec=120.0,
        bar_age_sessions=0,
    )
    assert ok.valid and ok.violations == ()

    bad = validate_pit_datafeed_observation(
        feeds_present=CANONICAL_7_PIT_FEEDS[:-1],
        sec_acceptance_delta_sec=-5.0,
        finra_short_vol_lag_days=0,
        finra_short_interest_lag_bdays=5,
        boats_quote_age_sec=950.0,
        bar_age_sessions=-1,
    )
    assert not bad.valid
    assert "MISSING_CANONICAL_FEEDS:ibkr_paper_4002_l1" in bad.violations
    assert "LOOKAHEAD_SEC_ACCEPTANCE_DATETIME" in bad.violations
    assert "LOOKAHEAD_FINRA_CNMSSHVOL_REQUIRES_T_MINUS_1" in bad.violations
    assert "LOOKAHEAD_FINRA_RULE_4560_REQUIRES_T_PLUS_8_BD" in bad.violations
    assert "STALE_OR_FUTURE_BOATS_QUOTE_GT_900S" in bad.violations
    assert "UNCOMPLETED_FUTURE_EXCHANGE_BAR" in bad.violations


def test_bjzz_dark_pool_subpenny_vs_institutional_and_round_lot_scaling() -> None:
    assert convert_sip_round_lots_to_shares(12.0) == 1200.0
    assert classify_bjzz_dark_pool_print(25.10, "Q") == "LIT_EXCHANGE"
    assert classify_bjzz_dark_pool_print(25.10, "D") == "DARK_INSTITUTIONAL_ROUND_OR_MIDPOINT"
    assert classify_bjzz_dark_pool_print(25.1050, "D") == "DARK_INSTITUTIONAL_ROUND_OR_MIDPOINT"
    assert classify_bjzz_dark_pool_print(25.1001, "D") == "DARK_RETAIL_SUBPENNY_INTERNALIZATION"
    assert classify_bjzz_dark_pool_print(25.1099, "D") == "DARK_RETAIL_SUBPENNY_INTERNALIZATION"


def test_intraday_88min_headline_never_misclassifies_as_prenews_ignition() -> None:
    # Regression test for MWH 88-min-old (5285.8s) intraday headline: must NOT be TAPE_PRE_NEWS_IGNITION!
    mwh = evaluate_first_hand_incentive(
        catalyst_age_seconds=5285.8,
        news_count_36h=1,
        is_sec_edgar_primary=False,
        is_tier1_breaking_catalyst=False,
        has_dilution_or_adverse_filing=False,
        move_pct=3.12,
        vwap_dist_pct=0.22,
        room_left_pct=65.0,
        vol_zscore=3.2,
        rvol_tod=3.1,
        burst_5m_mult=2.2,
        ofi_pct=28.0,
    )
    assert mwh.source_tier == "TIER2_INTRADAY_POST_NEWS_DRIFT"
    assert mwh.reaction_stage == "ACTIVE_FOLLOW_THROUGH"
    assert mwh.is_first_hand_early is False

    # True pre-news volume shock (no headline, strong OFI >= 15%, vol_z >= 2.6) -> FIRST_HAND_EARLY
    true_pre = evaluate_first_hand_incentive(
        catalyst_age_seconds=None,
        news_count_36h=0,
        is_sec_edgar_primary=False,
        is_tier1_breaking_catalyst=False,
        has_dilution_or_adverse_filing=False,
        move_pct=2.4,
        vwap_dist_pct=0.45,
        room_left_pct=72.0,
        vol_zscore=3.4,
        rvol_tod=3.2,
        burst_5m_mult=2.4,
        ofi_pct=32.0,
    )
    assert true_pre.source_tier == "TAPE_PRE_NEWS_IGNITION"
    assert true_pre.is_first_hand_early is True

    # Fresh SEC EDGAR 8-K acceptance (120s old) -> TIER1_SEC_EDGAR_PRIMARY & FIRST_HAND_EARLY
    sec_ep = evaluate_first_hand_incentive(
        catalyst_age_seconds=120.0,
        news_count_36h=1,
        is_sec_edgar_primary=True,
        is_tier1_breaking_catalyst=True,
        has_dilution_or_adverse_filing=False,
        move_pct=4.5,
        vwap_dist_pct=0.80,
        room_left_pct=68.0,
        vol_zscore=4.1,
        rvol_tod=4.5,
        burst_5m_mult=3.0,
        ofi_pct=40.0,
    )
    assert sec_ep.source_tier == "TIER1_SEC_EDGAR_PRIMARY"
    assert sec_ep.is_first_hand_early is True
    assert sec_ep.incentive_score > 90.0


def test_h090_execution_tier_and_stockbee_momentum_burst() -> None:
    assert classify_execution_quality_tier(spread_bps=8.5, ofi_pct=30.0) == "TIER1_TIGHT_SPREAD_HIGH_OFI_PASSIVE"
    assert classify_execution_quality_tier(spread_bps=22.0, ofi_pct=10.0) == "TIER2_TWO_STAGE_HYBRID_QUEUE"
    assert classify_execution_quality_tier(spread_bps=42.0, ofi_pct=30.0) == "TIER3_WIDE_SPREAD_OR_ADVERSE_OFI_BLOCKED"
    assert classify_execution_quality_tier(spread_bps=8.0, ofi_pct=-5.0) == "TIER3_WIDE_SPREAD_OR_ADVERSE_OFI_BLOCKED"

    assert evaluate_stockbee_momentum_burst(
        move_pct=4.8,
        dollar_change=1.25,
        rvol_tod=2.8,
        range_expansion_mult=1.9,
        close_location=0.84,
        stop_distance_pct=1.65,
        bar_age_sessions=0,
    )
    # Stop > ADR-0022 2.00% or stale bar -> rejected
    assert not evaluate_stockbee_momentum_burst(
        move_pct=5.5,
        dollar_change=2.0,
        rvol_tod=3.5,
        range_expansion_mult=2.2,
        close_location=0.90,
        stop_distance_pct=2.25,
    )
    assert not evaluate_stockbee_momentum_burst(
        move_pct=5.5,
        dollar_change=2.0,
        rvol_tod=3.5,
        range_expansion_mult=2.2,
        close_location=0.90,
        stop_distance_pct=1.50,
        bar_age_sessions=1,
    )


@given(
    p_move=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    p_vol=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    p_dark=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    p_iso=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    p_ovn=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    p_cat=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    boats_active=st.booleans(),
    spread_bps=st.floats(min_value=0.5, max_value=80.0, allow_nan=False, allow_infinity=False),
    ofi_pct=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_property_6pillar_sota_score_bounded_and_monotonic(
    p_move: float,
    p_vol: float,
    p_dark: float,
    p_iso: float,
    p_ovn: float,
    p_cat: float,
    boats_active: bool,
    spread_bps: float,
    ofi_pct: float,
) -> None:
    fhi = evaluate_first_hand_incentive(
        catalyst_age_seconds=300.0,
        news_count_36h=1,
        is_sec_edgar_primary=True,
        is_tier1_breaking_catalyst=True,
        has_dilution_or_adverse_filing=False,
        move_pct=3.5,
        vwap_dist_pct=0.5,
        room_left_pct=70.0,
        vol_zscore=3.0,
        rvol_tod=2.8,
        burst_5m_mult=2.0,
        ofi_pct=ofi_pct,
    )
    for seg in ("REGULAR_RTH", "OVERNIGHT_BOATS"):
        ev = compute_sota_mover_ranking(
            pillars={
                "move": p_move,
                "unusual_vol": p_vol,
                "inst_dark": p_dark,
                "iso_urgency": p_iso,
                "overnight": p_ovn,
                "catalyst": p_cat,
            },
            session_segment=seg,
            boats_active=boats_active,
            early_potential_score=75.0,
            no_sell_pressure_score=80.0,
            sota_vol_rank_score=70.0,
            inst_dark_pct=42.0,
            session_dvol_usd=35_000_000.0,
            live_price=24.50,
            move_pct=4.5,
            dollar_change=1.05,
            rvol_tod=2.9,
            range_expansion_mult=1.8,
            close_location=0.82,
            spread_bps=spread_bps,
            ofi_pct=ofi_pct,
            stop_distance_pct=1.60,
            strategy_confidence_0_1=0.85,
            ucb_priority_weight=1.35,
            first_hand=fhi,
        )
        assert ev.schema_version == RANKING_AND_DATAFEEDS_SCHEMA
        assert 0.0 <= ev.sota_score <= 100.0
        assert 0.0 <= ev.early_composite_rank <= 100.0
        assert ev.ucb_arbitrated_rank_score == pytest.approx(114.75, abs=1e-6)
