"""Pure-stdlib unit & Hypothesis property tests for Option B Pre-Wall DF10/DF6 & H103/H104 Factory scripts."""

from __future__ import annotations

import datetime as dt
import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st
from materialize_prewall_df10_df6 import (
    STUDY_WALL_DATE,
    assert_prewall_date,
    compute_regsho_published_ts_utc,
    parse_regsho_pipe_text,
)
from run_h106_df8_short_crowding_xsec import (
    compute_intraday_leg_arrays,
    compute_intraday_leg_gross_bps,
    cross_sectional_rank_matrix,
)
from run_optb_prewall_factory import (
    EFFECTIVE_HLZ_FLOOR,
    _replay_one_orb_slice_hftbacktest,
    compute_almgren_chriss_cost_bps,
    evaluate_strict_prewall_gate,
    normalize_alpaca_quote_size_to_shares,
)


def test_historical_quote_sizes_change_units_at_utc_midnight():
    boundary = int(dt.datetime(2025, 11, 3, tzinfo=dt.UTC).timestamp()) * 10**9
    for ts, bid, ask in (
        (boundary - 1, 300, 700),
        (boundary, 3, 7),
        (boundary + 1, 3, 7),
        (boundary + 5 * 3600 * 10**9, 3, 7),
    ):
        assert normalize_alpaca_quote_size_to_shares(3, ts) == bid
        assert normalize_alpaca_quote_size_to_shares(7, ts) == ask


def test_queue_replay_emits_bid_and_ask_shares_using_each_exchange_timestamp(monkeypatch):
    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    boundary = int(dt.datetime(2025, 11, 3, tzinfo=dt.UTC).timestamp()) * 10**9
    times = boundary + np.arange(-80, 40, dtype=np.int64) * 10**9
    times[79] = boundary - 1  # feed latency puts this quote's local timestamp after the cutover
    n = len(times)
    tables = {
        "quotes": pa.table({"t_ns": times, "bp": [10.0] * n, "ap": [10.01] * n, "bs": [3] * n, "as": [7] * n}),
        "trades": pa.table({"t_ns": times, "p": [10.01] * n, "s": [13] * n, "x": ["X"] * n}),
    }
    monkeypatch.setattr(parquet, "read_table", lambda path, columns: tables[path].select(columns))
    event_dtype = np.dtype(
        [
            (name, kind)
            for name, kind in zip(
                ("flags", "exch_ts", "local_ts", "px", "qty", "id", "aux", "value"),
                ("i8", "i8", "i8", "f8", "f8", "i8", "i8", "f8"),
                strict=True,
            )
        ]
    )
    asset = Mock()
    asset.data.side_effect = StopIteration("captured replay input")
    engine = SimpleNamespace(
        EXCH_EVENT=1,
        LOCAL_EVENT=2,
        DEPTH_EVENT=4,
        TRADE_EVENT=8,
        BUY_EVENT=16,
        SELL_EVENT=32,
        GTC=0,
        LIMIT=0,
        event_dtype=event_dtype,
        BacktestAsset=lambda: asset,
        HashMapMarketDepthBacktest=None,
    )
    monkeypatch.setitem(sys.modules, "hftbacktest", engine)
    _replay_one_orb_slice_hftbacktest(("trades", "quotes", "fixture", "2025-11-03", 5e6, False))
    asset.data.assert_called_once()
    events = asset.data.call_args.args[0]
    for ts, bid, ask in ((boundary - 1, 300, 700), (boundary, 3, 7), (boundary + 10**9, 3, 7)):
        quantities = {row["flags"]: row["qty"] for row in events[events["exch_ts"] == ts]}
        assert quantities == {1 | 2 | 4 | 16: bid, 1 | 2 | 4 | 32: ask, 1 | 2 | 8 | 16: 13}


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


def test_h106_intraday_leg_gross_bps_adverse_stop_and_ohlc_validation() -> None:
    # LONG adverse stop triggers strictly at -200.0 bps even if close recovers above entry
    assert compute_intraday_leg_gross_bps(100.0, 104.0, 97.99, 103.5, side="LONG") == -200.0
    # LONG unstopped exit at close
    assert compute_intraday_leg_gross_bps(100.0, 103.0, 99.0, 102.0, side="LONG") == 200.0
    # SHORT adverse stop triggers strictly at -200.0 bps even if close drops below entry
    assert compute_intraday_leg_gross_bps(100.0, 102.01, 96.0, 97.0, side="SHORT") == -200.0
    # SHORT unstopped exit at close
    assert compute_intraday_leg_gross_bps(100.0, 101.0, 96.5, 98.0, side="SHORT") == 200.0

    # Inconsistent OHLC bounds and ADR-0022 stop cap > 2.00% must fail closed
    with pytest.raises(ValueError, match="Inconsistent OHLC"):
        compute_intraday_leg_gross_bps(105.0, 104.0, 98.0, 101.0, side="LONG")
    with pytest.raises(ValueError, match="stop_frac"):
        compute_intraday_leg_gross_bps(100.0, 102.0, 99.0, 101.0, side="LONG", stop_frac=0.025)


@given(
    op=st.floats(min_value=5.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    up_pct=st.floats(min_value=0.0, max_value=0.15, allow_nan=False, allow_infinity=False),
    dn_pct=st.floats(min_value=0.0, max_value=0.15, allow_nan=False, allow_infinity=False),
    cl_frac=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property_h106_intraday_stop_floor_never_breached(
    op: float, up_pct: float, dn_pct: float, cl_frac: float
) -> None:
    hi = op * (1.0 + up_pct)
    lo = max(0.01, op * (1.0 - dn_pct))
    if lo > op:
        lo = op
    cl = lo + cl_frac * (hi - lo)
    long_bps = compute_intraday_leg_gross_bps(op, hi, lo, cl, side="LONG")
    short_bps = compute_intraday_leg_gross_bps(op, hi, lo, cl, side="SHORT")
    assert long_bps >= -200.0 and short_bps >= -200.0
    assert math.isfinite(long_bps) and math.isfinite(short_bps)


def test_h106_vectorized_arrays_and_date_isolated_ranking() -> None:
    np = pytest.importorskip("numpy")
    op = np.array([100.0, 100.0])
    hi = np.array([101.0, 103.0])
    lo = np.array([97.5, 99.0])
    cl = np.array([100.5, 98.0])
    l_g, s_g, l_st, s_st = compute_intraday_leg_arrays(op, hi, lo, cl)
    assert bool(l_st[0]) is True and float(l_g[0]) == -200.0
    assert bool(s_st[1]) is True and float(s_g[1]) == -200.0
    assert float(l_g[1]) == pytest.approx(-200.0, abs=1e-6)
    assert float(s_g[0]) == pytest.approx(-50.0, abs=1e-6)

    # Rankings on date 0 must remain invariant when date 1 shifts by +1000.0 (zero cross-date leakage)
    m_base = np.array([[10.0], [20.0], [30.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    m_shift = np.array([[10.0], [20.0], [30.0], [1001.0], [2002.0], [3003.0]], dtype=np.float32)
    dc = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    r_base = cross_sectional_rank_matrix(m_base, dc)
    r_shift = cross_sectional_rank_matrix(m_shift, dc)
    assert np.allclose(r_base[:3], r_shift[:3])
    assert float(r_base[0, 0]) == pytest.approx(-1.0)
    assert float(r_base[1, 0]) == pytest.approx(0.0)
    assert float(r_base[2, 0]) == pytest.approx(1.0)
