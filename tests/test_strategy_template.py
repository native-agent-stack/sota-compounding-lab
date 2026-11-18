"""Unit & Hypothesis Property Tests for Research-Converged Strategy Templates (`ADR-0006`)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sota_compounding import (
    ARCHETYPES,
    hlz_dynamic_t_hurdle,
    scaffold_converged_strategy,
    validate_converged_strategy_template,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_INPUT_SHA = "a" * 64
VALID_SPEC_SHA = "b" * 64
VALID_COMMITS = {
    "alpaca-py": "c42add7c0a7026e2ead3ed64e76f4def2b99fda4",
    "nautilus_trader": "da6b57aefdd62b44f675e434ffcfefca7f4025dd",
}


def _make_valid_payload(archetype: str = "PRE_TRAVEL_EARLY_IGNITION", trials_n: int = 82) -> dict[str, Any]:
    return scaffold_converged_strategy(
        "H_TEST_CONVERGED",
        archetype,
        input_package_sha256=VALID_INPUT_SHA,
        pinned_upstream_commits=VALID_COMMITS,
        spec_sha256=VALID_SPEC_SHA,
        cumulative_trials_n=trials_n,
    )


def test_archetypes_exact_tuple() -> None:
    assert ARCHETYPES == (
        "PRE_TRAVEL_EARLY_IGNITION",
        "FIRST_HAND_SEC_EPISODIC_PIVOT",
        "MEGA_CAP_NEAR_LOSSLESS_ORB",
        "CLIMAX_VWAP_MEAN_REVERSION",
    )


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_scaffold_all_archetypes_valid(archetype: str) -> None:
    payload = _make_valid_payload(archetype=archetype, trials_n=84)
    assert validate_converged_strategy_template(payload) == []
    assert payload["statistical_and_queue_gates"]["hlz_t_hurdle"] == hlz_dynamic_t_hurdle(84)


def test_committed_json_and_pine_v6_templates() -> None:
    json_path = REPO_ROOT / "templates" / "sota_converged_strategy.template.json"
    pine_path = REPO_ROOT / "templates" / "sota_pine_v6_nonrepainting.template.pine"
    assert json_path.is_file()
    assert pine_path.is_file()

    template_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert validate_converged_strategy_template(template_payload) == []

    pine_src = pine_path.read_text(encoding="utf-8")
    assert pine_src.startswith("//@version=6\n")
    assert "calc_on_every_tick=false" in pine_src
    assert "barstate.isconfirmed" in pine_src
    assert ":8799/api/tv-webhook" in pine_src
    assert "ADR0022_MAX_STOP_PCT = 2.00" in pine_src
    assert "alert.freq_once_per_bar_close" in pine_src
    assert "lookahead_on" not in pine_src


@pytest.mark.parametrize(
    ("section", "field", "bad_value", "expected_substr"),
    [
        # (a) ADR-0004 3-tier Research Convergence
        ("research_convergence", "input_package_sha256", "abc123", "input_package_sha256"),
        ("research_convergence", "input_package_sha256", "A" * 64, "input_package_sha256"),
        ("research_convergence", "pinned_upstream_commits", {}, "pinned_upstream_commits"),
        ("research_convergence", "pinned_upstream_commits", {"alpaca-py": "short"}, "40-character"),
        ("research_convergence", "reproducible_command", "   ", "reproducible_command"),
        ("research_convergence", "evidence_grade", "E5", "evidence_grade"),
        ("research_convergence", "other_repos_authority", "DIRECT_IMPORT", "PHILOSOPHY_ONLY"),
        # (b) 7-Feed PIT DataFeed routing
        ("datafeed_pit_contract", "rth_extended_feed", "iex", "rth_extended_feed"),
        ("datafeed_pit_contract", "overnight_feed", "sip", "overnight_feed"),
        ("datafeed_pit_contract", "dec6_2026_23x5_cutover_enabled", 1, "explicit bool"),
        ("datafeed_pit_contract", "raw_filter_min_price_usd", 1.99, "raw_filter_min_price_usd"),
        ("datafeed_pit_contract", "raw_filter_min_dollar_vol_usd", 1_500_000.0, "raw_filter_min_dollar_vol_usd"),
        ("datafeed_pit_contract", "return_adjustment", "raw", "split_dividend_all"),
        ("datafeed_pit_contract", "split_exclusion_window_sessions", [0, 2], "split_exclusion_window_sessions"),
        ("datafeed_pit_contract", "sec_edgar_timestamp_field", "filingDate", "acceptanceDateTime"),
        ("datafeed_pit_contract", "sec_freshness_half_life_sec", 600, "sec_freshness_half_life_sec"),
        ("datafeed_pit_contract", "dilution_block_forms", ["S-3"], "dilution_block_forms"),
        ("datafeed_pit_contract", "dilution_block_forms", ["424B5"], "dilution_block_forms"),
        ("datafeed_pit_contract", "finra_short_vol_lag_sessions", 0, "finra_short_vol_lag_sessions"),
        ("datafeed_pit_contract", "finra_short_vol_lag_sessions", True, "finra_short_vol_lag_sessions"),
        # (c) Statistical & Queue Gates
        ("statistical_and_queue_gates", "study_wall_date", "2026-10-16", "2026-10-15"),
        ("statistical_and_queue_gates", "spec_sha256", "z" * 64, "spec_sha256"),
        ("statistical_and_queue_gates", "hlz_t_hurdle", 2.95, "hlz_t_hurdle"),
        ("statistical_and_queue_gates", "cpcv_paths", 4, "cpcv_paths"),
        ("statistical_and_queue_gates", "cpcv_purge_sessions", 2, "cpcv_purge_sessions"),
        ("statistical_and_queue_gates", "cpcv_embargo_sessions", 1, "cpcv_embargo_sessions"),
        ("statistical_and_queue_gates", "max_pbo", 0.31, "max_pbo"),
        ("statistical_and_queue_gates", "min_dsr", 0.94, "min_dsr"),
        ("statistical_and_queue_gates", "intrabar_tie_rule", "TARGET_FIRST", "ADVERSE_STOP_FIRST"),
        ("statistical_and_queue_gates", "cost_stress_multipliers", [1.0], "cost_stress_multipliers"),
        # (d) Paper Execution & ADR-0022
        ("paper_execution_and_adr0022", "live_execution_enabled", True, "live_execution_enabled"),
        ("paper_execution_and_adr0022", "max_initial_stop_pct", 2.01, "max_initial_stop_pct"),
        ("paper_execution_and_adr0022", "max_initial_stop_pct", float("nan"), "max_initial_stop_pct"),
        ("paper_execution_and_adr0022", "alpaca_paper_max_rpm", 201, "alpaca_paper_max_rpm"),
        ("paper_execution_and_adr0022", "ibkr_paper_port", 4001, "ibkr_paper_port"),
        ("paper_execution_and_adr0022", "ibkr_paper_port", 7497, "ibkr_paper_port"),
        ("paper_execution_and_adr0022", "ibkr_paper_max_msg_per_min", 3001, "ibkr_paper_max_msg_per_min"),
    ],
)
def test_validate_fails_closed_on_each_contract_violation(
    section: str,
    field: str,
    bad_value: Any,
    expected_substr: str,
) -> None:
    mutated = copy.deepcopy(_make_valid_payload())
    mutated[section][field] = bad_value
    errs = validate_converged_strategy_template(mutated)
    assert errs, f"Expected failure for {section}.{field}={bad_value!r}"
    assert any(expected_substr in e for e in errs), errs


def test_scaffold_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unsupported archetype"):
        scaffold_converged_strategy(
            "BAD",
            "UNKNOWN_ARCHETYPE",
            input_package_sha256=VALID_INPUT_SHA,
            pinned_upstream_commits=VALID_COMMITS,
            spec_sha256=VALID_SPEC_SHA,
        )
    with pytest.raises(ValueError, match="input_package_sha256"):
        scaffold_converged_strategy(
            "BAD_SHA",
            "PRE_TRAVEL_EARLY_IGNITION",
            input_package_sha256="not-a-sha",
            pinned_upstream_commits=VALID_COMMITS,
            spec_sha256=VALID_SPEC_SHA,
        )


@settings(max_examples=100)
@given(
    archetype=st.sampled_from(list(ARCHETYPES)),
    trials_n=st.integers(min_value=1, max_value=500),
    stop_pct=st.floats(min_value=0.05, max_value=2.00, allow_nan=False, allow_infinity=False),
    min_price=st.floats(min_value=2.00, max_value=50.0, allow_nan=False, allow_infinity=False),
    min_dvol=st.floats(min_value=2_000_000.0, max_value=50_000_000.0, allow_nan=False, allow_infinity=False),
    cutover=st.booleans(),
)
def test_property_valid_bounds_always_pass(
    archetype: str,
    trials_n: int,
    stop_pct: float,
    min_price: float,
    min_dvol: float,
    cutover: bool,
) -> None:
    payload = _make_valid_payload(archetype=archetype, trials_n=trials_n)
    payload["paper_execution_and_adr0022"]["max_initial_stop_pct"] = stop_pct
    payload["datafeed_pit_contract"]["raw_filter_min_price_usd"] = min_price
    payload["datafeed_pit_contract"]["raw_filter_min_dollar_vol_usd"] = min_dvol
    payload["datafeed_pit_contract"]["dec6_2026_23x5_cutover_enabled"] = cutover
    assert validate_converged_strategy_template(payload) == []


@settings(max_examples=100)
@given(
    bad_stop=st.one_of(
        st.floats(min_value=2.0001, max_value=50.0, allow_nan=False, allow_infinity=False),
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
        st.just(float("inf")),
    )
)
def test_property_adr0022_violations_always_rejected(bad_stop: float) -> None:
    payload = _make_valid_payload()
    payload["paper_execution_and_adr0022"]["max_initial_stop_pct"] = bad_stop
    errs = validate_converged_strategy_template(payload)
    assert any("max_initial_stop_pct" in e for e in errs)
