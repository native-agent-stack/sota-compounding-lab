"""Pure-stdlib unit & Hypothesis property tests for Option B Pre-Wall DF10/DF6 & H103/H104 Factory scripts."""

from __future__ import annotations

import datetime as dt
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from materialize_prewall_df10_df6 import (
    STUDY_WALL_DATE,
    assert_prewall_date,
    compute_regsho_published_ts_utc,
    parse_regsho_pipe_text,
)
from run_optb_prewall_factory import (
    EFFECTIVE_HLZ_FLOOR,
    compute_almgren_chriss_cost_bps,
    evaluate_strict_prewall_gate,
)


def test_prewall_date_guard_enforces_2026_10_15_cutoff() -> None:
    assert assert_prewall_date(STUDY_WALL_DATE) == dt.date(2026, 10, 15)
    assert assert_prewall_date(dt.date(2024, 1, 2)) == dt.date(2024, 1, 2)
    with pytest.raises(ValueError, match="PREWALL_CUTOFF_BREACH"):
        assert_prewall_date(dt.date(2026, 10, 16), "trade_date")


def test_parse_regsho_pipe_text_pit_causality_and_schema() -> None:
    sample_nq = (
        "Symbol|Security Name|Market Category|Reg SHO Threshold Flag|Rule 3210|Filler\n"
        "GME|GameStop Corp|Q|Y|N|\n"
        "SKIP|Not Threshold|Q|N|N|\n"
        "202610142300\n"
    )
    rows = parse_regsho_pipe_text(
        sample_nq,
        list_date=dt.date(2026, 10, 14),
        effective_trade_date=dt.date(2026, 10, 15),
        exchange_source="NASDAQ",
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "GME"
    assert rows[0]["list_date"] == "2026-10-14"
    assert rows[0]["effective_trade_date"] == "2026-10-15"
    assert rows[0]["published_ts_utc"] == compute_regsho_published_ts_utc(dt.date(2026, 10, 14))
    assert rows[0]["published_ts_utc"].endswith("Z")

    # Same-day or backward causality must fail closed
    with pytest.raises(ValueError, match="DF6_CAUSALITY_ERROR"):
        parse_regsho_pipe_text(
            sample_nq,
            list_date=dt.date(2026, 10, 14),
            effective_trade_date=dt.date(2026, 10, 14),
            exchange_source="NASDAQ",
        )

    # Post-wall effective date must fail closed
    with pytest.raises(ValueError, match="PREWALL_CUTOFF_BREACH"):
        parse_regsho_pipe_text(
            sample_nq,
            list_date=dt.date(2026, 10, 15),
            effective_trade_date=dt.date(2026, 10, 16),
            exchange_source="NASDAQ",
        )


@given(
    half_spr=st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    fee_bps=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
    sigma=st.floats(min_value=0.005, max_value=0.20, allow_nan=False, allow_infinity=False),
    notional=st.floats(min_value=1_000.0, max_value=500_000.0, allow_nan=False, allow_infinity=False),
    adv=st.floats(min_value=5_000_000.0, max_value=5_000_000_000.0, allow_nan=False, allow_infinity=False),
)
def test_property_almgren_chriss_cost_monotonicity_and_finiteness(
    half_spr: float, fee_bps: float, sigma: float, notional: float, adv: float
) -> None:
    c1 = compute_almgren_chriss_cost_bps(
        base_half_spread_bps=half_spr,
        fee_and_slip_bps_per_side=fee_bps,
        daily_sigma_frac=sigma,
        order_notional_usd=notional,
        adv_usd=adv,
        stress_multiplier=1.0,
    )
    c2 = compute_almgren_chriss_cost_bps(
        base_half_spread_bps=half_spr,
        fee_and_slip_bps_per_side=fee_bps,
        daily_sigma_frac=sigma,
        order_notional_usd=notional,
        adv_usd=adv,
        stress_multiplier=2.0,
    )
    assert math.isfinite(c1) and math.isfinite(c2)
    assert c1 > 0.0
    assert c2 == pytest.approx(2.0 * c1, abs=1e-3)


def test_almgren_chriss_rejects_non_finite_and_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_almgren_chriss_cost_bps(
            base_half_spread_bps=float("nan"),
            fee_and_slip_bps_per_side=2.0,
            daily_sigma_frac=0.02,
            order_notional_usd=50_000.0,
            adv_usd=10_000_000.0,
        )
    with pytest.raises(ValueError, match="> 0"):
        compute_almgren_chriss_cost_bps(
            base_half_spread_bps=2.0,
            fee_and_slip_bps_per_side=2.0,
            daily_sigma_frac=0.02,
            order_notional_usd=50_000.0,
            adv_usd=0.0,
        )


def test_strict_prewall_gate_enforces_hlz_42019_dsr_cpcv_pbo_and_2x_net() -> None:
    ok = evaluate_strict_prewall_gate(
        trial_number=103,
        t_stat_clustered_2x=4.55,
        dsr_probability=0.97,
        cpcv_positive_share=0.9333,
        pbo=0.0667,
        mean_net_2x_bps=18.5,
        distinct_dates=250,
        total_events=1200,
    )
    assert ok["passed"] is True
    assert ok["verdict"] == "PASSED"
    assert ok["effective_hlz_t_hurdle"] >= EFFECTIVE_HLZ_FLOOR
    assert ok["fail_reasons"] == []

    # Fails if t_stat < 4.2019 even if dynamic trial_n=99 hurdle is 4.1951
    sub_hlz = evaluate_strict_prewall_gate(
        trial_number=99,
        t_stat_clustered_2x=4.199,
        dsr_probability=0.99,
        cpcv_positive_share=1.0,
        pbo=0.0,
        mean_net_2x_bps=12.0,
        distinct_dates=250,
        total_events=1200,
    )
    assert sub_hlz["passed"] is False
    assert sub_hlz["verdict"] == "FAILED"
    assert any("HLZ_T_FAIL" in r for r in sub_hlz["fail_reasons"])

    # Fails on DSR < 0.95 or PBO > 0.30 or non-positive 2x net
    multi_fail = evaluate_strict_prewall_gate(
        trial_number=100,
        t_stat_clustered_2x=1.25,
        dsr_probability=0.40,
        cpcv_positive_share=0.53,
        pbo=0.47,
        mean_net_2x_bps=-4.2,
        distinct_dates=45,
        total_events=120,
    )
    assert multi_fail["passed"] is False
    assert len(multi_fail["fail_reasons"]) == 7
