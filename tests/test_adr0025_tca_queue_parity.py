"""ADR-0025 Live-Paper TCA vs. Nautilus/hftbacktest queue-parity reconciler & EB decay gate tests."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sota_compounding.lifecycle import hlz_dynamic_t_hurdle
from sota_compounding.tca_queue_parity import (
    DRIFT_WARN,
    PARITY_GREEN,
    PARITY_STATUS_SEVERITY,
    TCA_QUEUE_PARITY_SCHEMA,
    THROTTLE_RED,
    LivePaperTcaObservation,
    QueueParityBaseline,
    compute_hlz_t_hurdle,
    reconcile_fleet_parity,
    reconcile_live_vs_sim_parity,
)

BPS = st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False)
RATIO = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
COUNT = st.integers(min_value=0, max_value=100_000)
SHRINK_K = st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False)


def _baseline(**overrides: object) -> QueueParityBaseline:
    fields: dict[str, object] = {
        "strategy_id": "H_TEST_QUEUE",
        "prewall_cutoff_date": "2026-10-15",
        "sim_fill_rate_ratio": 0.50,
        "sim_expected_slippage_bps": 3.0,
        "sim_gross_edge_bps": 15.0,
        "sim_sample_n": 400,
        "hlz_trials_n": 103,
    }
    fields.update(overrides)
    return QueueParityBaseline(**fields)  # type: ignore[arg-type]


def _obs(**overrides: object) -> LivePaperTcaObservation:
    fields: dict[str, object] = {
        "strategy_id": "H_TEST_QUEUE",
        "account_lane": ":8787",
        "live_sample_n": 40,
        "live_fill_rate_ratio": 0.55,
        "live_implementation_shortfall_bps": 5.0,
        "live_markout_1s_bps": 0.5,
        "live_markout_10s_bps": 0.2,
        "live_markout_60s_bps": -1.0,
        "live_markout_300s_bps": -1.5,
        "live_gross_edge_bps": 14.0,
    }
    fields.update(overrides)
    return LivePaperTcaObservation(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------- HLZ hurdle


def test_hlz_hurdle_matches_repo_formula_next_trial_convention() -> None:
    assert compute_hlz_t_hurdle(0) == 3.0
    assert compute_hlz_t_hurdle(102) == 4.2041
    assert compute_hlz_t_hurdle(103) == 4.2063
    assert compute_hlz_t_hurdle(103) == hlz_dynamic_t_hurdle(104)
    for n in (0, 1, 10, 50, 103, 500):
        assert compute_hlz_t_hurdle(n) == hlz_dynamic_t_hurdle(n + 1)


@pytest.mark.parametrize("bad", [-1, -103, 1.5, True, "103", None])
def test_hlz_hurdle_rejects_negative_or_non_integer_trials(bad: object) -> None:
    with pytest.raises(ValueError):
        compute_hlz_t_hurdle(bad)  # type: ignore[arg-type]


@given(a=st.integers(min_value=0, max_value=1_000_000), b=st.integers(min_value=0, max_value=1_000_000))
def test_hlz_hurdle_non_decreasing_property(a: int, b: int) -> None:
    lo, hi = min(a, b), max(a, b)
    assert compute_hlz_t_hurdle(lo) <= compute_hlz_t_hurdle(hi)
    if hi >= 103:
        assert compute_hlz_t_hurdle(hi) >= 4.2063


# --------------------------------------------------------------------------------------------- status


def test_parity_green_baseline_case() -> None:
    rec = reconcile_live_vs_sim_parity(_baseline(), _obs(), hlz_t_stat=4.5)
    assert rec["schema_version"] == TCA_QUEUE_PARITY_SCHEMA
    assert rec["parity_status"] == PARITY_GREEN
    assert rec["recommended_action"] == "MAINTAIN_COLLAR"
    assert rec["breach_reasons"] == []
    assert rec["slippage_drift_bps"] == 2.0
    assert rec["fill_rate_gap_ratio"] == 0.05
    assert rec["prior_edge_bps"] == 12.0
    assert rec["live_net_edge_bps"] == 9.0
    # w = 40 / (40 + 20) = 2/3 -> 2/3 * 9 + 1/3 * 12 = 10.0
    assert rec["eb_shrunk_net_edge_bps"] == pytest.approx(10.0, abs=1e-6)
    assert rec["hlz_t_hurdle"] == 4.2063
    assert rec["hlz_t_stat"] == 4.5
    assert rec["hlz_hurdle_cleared"] is True
    assert rec["halts_paper_farm"] is False
    assert len(rec["receipt_sha256"]) == 64
    json.dumps(rec, allow_nan=False)


def test_exact_float_boundaries_stay_green() -> None:
    # 0.65 - 0.5 == 0.15000000000000002 in IEEE-754; rounding keeps the boundary inclusive.
    rec = reconcile_live_vs_sim_parity(
        _baseline(sim_fill_rate_ratio=0.5, sim_expected_slippage_bps=3.0),
        _obs(live_fill_rate_ratio=0.65, live_implementation_shortfall_bps=7.0, live_markout_60s_bps=-3.0),
    )
    assert rec["slippage_drift_bps"] == 4.0
    assert rec["fill_rate_gap_ratio"] == 0.15
    assert rec["parity_status"] == PARITY_GREEN


@pytest.mark.parametrize(
    ("overrides", "reason_prefix"),
    [
        ({"live_implementation_shortfall_bps": 7.5}, "SLIPPAGE_DRIFT_MODERATE"),
        ({"live_implementation_shortfall_bps": 13.0, "live_gross_edge_bps": 30.0}, "SLIPPAGE_DRIFT_MODERATE"),
        ({"live_fill_rate_ratio": 0.30}, "FILL_RATE_GAP_MODERATE"),
        ({"live_fill_rate_ratio": 0.80}, "FILL_RATE_GAP_MODERATE"),
        ({"live_markout_60s_bps": -5.0}, "ADVERSE_SELECTION_MILD"),
        ({"live_markout_60s_bps": -8.0}, "ADVERSE_SELECTION_MILD"),
    ],
)
def test_drift_warn_cases(overrides: dict[str, object], reason_prefix: str) -> None:
    rec = reconcile_live_vs_sim_parity(_baseline(), _obs(**overrides))
    assert rec["parity_status"] == DRIFT_WARN
    assert rec["recommended_action"] == "MONITOR_AND_TIGHTEN_COLLAR"
    assert any(r.startswith(reason_prefix) for r in rec["breach_reasons"])


def test_non_positive_eb_with_small_live_sample_is_warn_not_red() -> None:
    rec = reconcile_live_vs_sim_parity(
        _baseline(sim_gross_edge_bps=3.0), _obs(live_sample_n=5, live_gross_edge_bps=4.0)
    )
    assert rec["eb_shrunk_net_edge_bps"] <= 0.0
    assert rec["parity_status"] == DRIFT_WARN
    assert any(r.startswith("EB_NET_EDGE_NON_POSITIVE_LOW_N") for r in rec["breach_reasons"])


@pytest.mark.parametrize(
    ("overrides", "reason_prefix"),
    [
        ({"live_implementation_shortfall_bps": 13.01, "live_gross_edge_bps": 40.0}, "SLIPPAGE_DRIFT_SEVERE"),
        ({"live_fill_rate_ratio": 0.19}, "FILL_RATE_GAP_SEVERE"),
        ({"live_fill_rate_ratio": 0.81}, "FILL_RATE_GAP_SEVERE"),
        ({"live_markout_60s_bps": -8.01}, "ADVERSE_SELECTION_SEVERE"),
        ({"live_sample_n": 200, "live_gross_edge_bps": 2.0}, "EB_NET_EDGE_NON_POSITIVE"),
    ],
)
def test_throttle_red_cases_are_advisory(overrides: dict[str, object], reason_prefix: str) -> None:
    rec = reconcile_live_vs_sim_parity(_baseline(), _obs(**overrides))
    assert rec["parity_status"] == THROTTLE_RED
    assert rec["recommended_action"] == "THROTTLE_OR_RECALIBRATE_COLLAR"
    assert rec["halts_paper_farm"] is False
    assert rec["paper_policy"] == "NEVER_GATED_ADVISORY"
    assert any(r.startswith(reason_prefix) for r in rec["breach_reasons"])


def test_zero_live_sample_returns_prior() -> None:
    rec = reconcile_live_vs_sim_parity(_baseline(), _obs(live_sample_n=0, live_gross_edge_bps=-50.0))
    assert rec["eb_live_weight_ratio"] == 0.0
    assert rec["eb_shrunk_net_edge_bps"] == rec["prior_edge_bps"] == 12.0


# --------------------------------------------------------------------------------------------- validation


@pytest.mark.parametrize("bad_date", ["2026-10-16", "2026-11-19", date(2026, 12, 6)])
def test_rejects_post_wall_baseline_cutoff(bad_date: object) -> None:
    with pytest.raises(ValueError, match="Study-Wall"):
        _baseline(prewall_cutoff_date=bad_date)


def test_accepts_wall_date_and_normalizes_date_objects() -> None:
    assert _baseline(prewall_cutoff_date=date(2026, 10, 15)).prewall_cutoff_date == "2026-10-15"
    assert _baseline(prewall_cutoff_date="2026-09-30").prewall_cutoff_date == "2026-09-30"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["sim_expected_slippage_bps", "sim_gross_edge_bps", "sim_fill_rate_ratio"])
def test_baseline_rejects_nan_inf(field: str, bad: float) -> None:
    with pytest.raises(ValueError):
        _baseline(**{field: bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field",
    [
        "live_fill_rate_ratio",
        "live_implementation_shortfall_bps",
        "live_markout_1s_bps",
        "live_markout_10s_bps",
        "live_markout_60s_bps",
        "live_markout_300s_bps",
        "live_gross_edge_bps",
    ],
)
def test_observation_rejects_nan_inf(field: str, bad: float) -> None:
    with pytest.raises(ValueError):
        _obs(**{field: bad})


@pytest.mark.parametrize("bad", [-0.01, 1.01, True])
def test_fill_rates_must_be_unit_interval_ratios(bad: object) -> None:
    with pytest.raises(ValueError):
        _baseline(sim_fill_rate_ratio=bad)
    with pytest.raises(ValueError):
        _obs(live_fill_rate_ratio=bad)


@pytest.mark.parametrize("field", ["sim_sample_n", "hlz_trials_n"])
def test_baseline_rejects_negative_counts(field: str) -> None:
    with pytest.raises(ValueError):
        _baseline(**{field: -1})


def test_observation_rejects_negative_live_sample_and_live_lanes() -> None:
    with pytest.raises(ValueError):
        _obs(live_sample_n=-1)
    for live_lane in (":8798", ":4001", ":7496"):
        with pytest.raises(ValueError, match="Live-Capital Firewall"):
            _obs(account_lane=live_lane)
    with pytest.raises(ValueError):
        _obs(account_lane=":9999")


@pytest.mark.parametrize("bad_k", [0.0, -1.0, float("nan"), float("inf")])
def test_shrinkage_k_must_be_finite_positive(bad_k: float) -> None:
    with pytest.raises(ValueError):
        reconcile_live_vs_sim_parity(_baseline(), _obs(), shrinkage_k=bad_k)


def test_rejects_nan_t_stat_and_strategy_mismatch() -> None:
    with pytest.raises(ValueError):
        reconcile_live_vs_sim_parity(_baseline(), _obs(), hlz_t_stat=float("nan"))
    with pytest.raises(ValueError, match="mismatch"):
        reconcile_live_vs_sim_parity(_baseline(), _obs(strategy_id="OTHER"))


# --------------------------------------------------------------------------------------------- determinism


def test_sha256_receipt_is_deterministic_and_sensitive() -> None:
    a = reconcile_live_vs_sim_parity(_baseline(), _obs())
    b = reconcile_live_vs_sim_parity(_baseline(), _obs())
    assert a == b
    assert a["receipt_sha256"] == b["receipt_sha256"]
    c = reconcile_live_vs_sim_parity(_baseline(), _obs(live_markout_300s_bps=-1.6))
    assert c["receipt_sha256"] != a["receipt_sha256"]


# --------------------------------------------------------------------------------------------- fleet


def test_fleet_aggregation_worst_status_counts_and_unmatched() -> None:
    b1 = _baseline(strategy_id="H_A")
    b2 = _baseline(strategy_id="H_B")
    b3 = _baseline(strategy_id="H_C")
    observations = [
        _obs(strategy_id="H_A", account_lane=":8787"),
        _obs(strategy_id="H_A", account_lane=":4002", live_markout_60s_bps=-5.0, live_sample_n=60),
        _obs(strategy_id="H_B", account_lane=":8799", live_fill_rate_ratio=0.05, live_sample_n=100),
        _obs(strategy_id="H_ORPHAN", account_lane=":8797"),
    ]
    fleet = reconcile_fleet_parity([b1, b2, b3], observations, hlz_t_stats={"H_A": 3.9})
    assert fleet["schema_version"] == TCA_QUEUE_PARITY_SCHEMA
    assert fleet["record_type"] == "fleet_summary"
    assert fleet["cockpit_endpoint"] == ":8787/api/unified_cockpit_snapshot"
    assert fleet["fleet_status"] == THROTTLE_RED
    assert fleet["fleet_recommended_action"] == "THROTTLE_OR_RECALIBRATE_COLLAR"
    assert fleet["status_counts"] == {PARITY_GREEN: 1, DRIFT_WARN: 1, THROTTLE_RED: 1}
    assert fleet["lanes_n"] == 3
    assert fleet["strategies_n"] == 2
    assert fleet["total_live_sample_n"] == 200
    assert fleet["throttle_red_lanes"] == ["H_B@:8799"]
    assert fleet["unmatched_observations"] == [{"strategy_id": "H_ORPHAN", "account_lane": ":8797"}]
    assert fleet["baselines_without_observations"] == ["H_C"]
    assert [(r["strategy_id"], r["account_lane"]) for r in fleet["rows"]] == [
        ("H_A", ":4002"),
        ("H_A", ":8787"),
        ("H_B", ":8799"),
    ]
    assert all(r["hlz_hurdle_cleared"] is False for r in fleet["rows"] if r["strategy_id"] == "H_A")
    assert fleet["rows"][2]["hlz_t_stat"] is None
    expected_drift = sum(r["slippage_drift_bps"] * r["live_sample_n"] for r in fleet["rows"]) / 200
    assert fleet["weighted_slippage_drift_bps"] == pytest.approx(expected_drift, abs=1e-6)
    assert fleet["halts_paper_farm"] is False
    json.dumps(fleet, allow_nan=False)

    permuted = reconcile_fleet_parity([b3, b1, b2], list(reversed(observations)), hlz_t_stats={"H_A": 3.9})
    assert permuted == fleet
    assert permuted["receipt_sha256"] == fleet["receipt_sha256"]


def test_fleet_empty_and_duplicates() -> None:
    empty = reconcile_fleet_parity([], [])
    assert empty["fleet_status"] == PARITY_GREEN
    assert empty["lanes_n"] == 0
    assert empty["weighted_eb_shrunk_net_edge_bps"] is None
    with pytest.raises(ValueError, match="duplicate baseline"):
        reconcile_fleet_parity([_baseline(), _baseline()], [])
    with pytest.raises(ValueError, match="duplicate observation"):
        reconcile_fleet_parity([_baseline()], [_obs(), _obs()])


# --------------------------------------------------------------------------------------------- properties


@settings(max_examples=300)
@given(
    sim_gross=BPS,
    sim_slip=BPS,
    live_gross=BPS,
    live_shortfall=BPS,
    live_n=COUNT,
    k=SHRINK_K,
)
def test_eb_shrinkage_bounded_between_prior_and_live_net(
    sim_gross: float, sim_slip: float, live_gross: float, live_shortfall: float, live_n: int, k: float
) -> None:
    rec = reconcile_live_vs_sim_parity(
        _baseline(sim_gross_edge_bps=sim_gross, sim_expected_slippage_bps=sim_slip),
        _obs(live_gross_edge_bps=live_gross, live_implementation_shortfall_bps=live_shortfall, live_sample_n=live_n),
        shrinkage_k=k,
    )
    prior, live, eb = rec["prior_edge_bps"], rec["live_net_edge_bps"], rec["eb_shrunk_net_edge_bps"]
    assert math.isfinite(eb)
    assert min(prior, live) <= eb <= max(prior, live)
    assert 0.0 <= rec["eb_live_weight_ratio"] <= 1.0
    if live_n == 0:
        assert eb == prior


@settings(max_examples=300)
@given(
    base=st.tuples(RATIO, BPS, BPS),
    live=st.tuples(RATIO, BPS, BPS, COUNT),
    shortfalls=st.tuples(BPS, BPS),
)
def test_status_monotonically_degrades_with_slippage_drift(
    base: tuple[float, float, float],
    live: tuple[float, float, float, int],
    shortfalls: tuple[float, float],
) -> None:
    sim_fill, sim_slip, sim_gross = base
    live_fill, markout_60s, live_gross, live_n = live
    lo, hi = sorted(shortfalls)
    b = _baseline(sim_fill_rate_ratio=sim_fill, sim_expected_slippage_bps=sim_slip, sim_gross_edge_bps=sim_gross)
    template = _obs(
        live_fill_rate_ratio=live_fill,
        live_markout_60s_bps=markout_60s,
        live_gross_edge_bps=live_gross,
        live_sample_n=live_n,
    )
    better = reconcile_live_vs_sim_parity(b, dataclasses.replace(template, live_implementation_shortfall_bps=lo))
    worse = reconcile_live_vs_sim_parity(b, dataclasses.replace(template, live_implementation_shortfall_bps=hi))
    assert better["slippage_drift_bps"] <= worse["slippage_drift_bps"]
    assert better["eb_shrunk_net_edge_bps"] >= worse["eb_shrunk_net_edge_bps"]
    assert PARITY_STATUS_SEVERITY[better["parity_status"]] <= PARITY_STATUS_SEVERITY[worse["parity_status"]]
