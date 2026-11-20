"""Regression & fail-closed governance tests for ADR-0034 (Trial #107 H110 + Unified Sim->Paper->Live Merit)."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

from sota_compounding.sim_to_paper_to_live_merit_manifest import (
    PaperBrokerEvidence,
    PrewallSimEvidence,
    evaluate_sim_to_paper_to_live_merit,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRIAL107_PATH = REPO_ROOT / "docs" / "TRIAL107_H110_PREWALL_RECEIPT.json"
MERIT_JSON_PATH = REPO_ROOT / "docs" / "SOTA_SIM_TO_PAPER_TO_LIVE_MERIT_MANIFEST.json"
MERIT_MD_PATH = REPO_ROOT / "docs" / "SOTA_SIM_TO_PAPER_TO_LIVE_MERIT_MANIFEST.md"


def test_trial107_h110_receipt_integrity_and_honest_outcome() -> None:
    assert TRIAL107_PATH.is_file()
    data = json.loads(TRIAL107_PATH.read_text(encoding="utf-8"))
    assert data["schema"] == "sota.prewall-trial107-h110-receipt.v1"
    assert data["trial_number"] == 107
    assert data["study_wall_date"] == "2026-10-15"
    assert data["postwall_bytes_read"] == 0
    assert data["prewall_sip_sessions_scanned"] == 700
    assert data["overall_passed"] is False
    assert data["overall_promotion_track"] == "PROTECTIVE_FILTER_OR_GOVERNANCE"

    stored_sha = data.pop("receipt_sha256")
    computed = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert computed == stored_sha

    p = data["primary_beta_neutral_directed_leadlag"]
    assert p["events"] == 6740
    assert p["distinct_dates"] == 337
    assert p["hygiene_14pt_passed"] is True
    assert p["mean_gross_day_bps"] > 0.0
    assert p["mean_net_2x_day_bps"] < 0.0
    assert p["passed"] is False


def test_fail_closed_merit_gate_hard_refuses_zero_and_one_trade_seed_canaries() -> None:
    sim_fail = PrewallSimEvidence(
        strategy_id="SOTA_ZERO_SELL_VWAP_BRACKET",
        family="INTRADAY_ABSORPTION",
        cadence="INTRADAY_5M",
        trial_number=None,
        events=400,
        distinct_dates=200,
        gross_bps_day=2.0,
        net_1x_bps_day=-8.0,
        net_2x_bps_day=-18.0,
        t_stat_2x=-4.5,
        hlz_t_hurdle=4.2127,
        dsr_2x=0.0,
        cpcv_pos_share=0.0,
        cpcv_pbo=0.80,
        spa_p_2x=1.0,
        hygiene_14pt_passed=True,
    )
    paper_one_trade = PaperBrokerEvidence(
        strategy_id="SOTA_ZERO_SELL_VWAP_BRACKET",
        closed_rts=1,
        distinct_sessions=1,
        win_rate_pct=100.0,
        profit_factor=99.0,
        mean_bps=22.0,
        expectancy_r=0.22,
        payoff_ratio=2.0,
        slippage_bps=-3.2,
    )
    res_zero = evaluate_sim_to_paper_to_live_merit(sim_fail, None, cumulative_trials_n=107)
    res_one = evaluate_sim_to_paper_to_live_merit(sim_fail, paper_one_trade, cumulative_trials_n=107)
    assert res_zero["live_autonomous_allowed"] is False
    assert res_zero["live_notional_scale"] == 0.0
    assert res_one["live_autonomous_allowed"] is False
    assert res_one["live_notional_scale"] == 0.0
    assert res_one["owner_order_lossless_adoption"] is True


def test_merit_gate_only_promotes_when_both_sim_and_paper_w5_pass_and_breaker_clear() -> None:
    sim_pass = PrewallSimEvidence(
        strategy_id="SYNTHETIC_GATE_PROOF_STRATEGY",
        family="MULTIDAY_SWING",
        cadence="DAILY_SWING",
        trial_number=108,
        events=350,
        distinct_dates=220,
        gross_bps_day=45.0,
        net_1x_bps_day=32.0,
        net_2x_bps_day=19.0,
        t_stat_2x=4.65,
        hlz_t_hurdle=4.2148,
        dsr_2x=0.985,
        cpcv_pos_share=0.80,
        cpcv_pbo=0.12,
        spa_p_2x=0.012,
        hygiene_14pt_passed=True,
    )
    paper_pass = PaperBrokerEvidence(
        strategy_id="SYNTHETIC_GATE_PROOF_STRATEGY",
        closed_rts=36,
        distinct_sessions=14,
        win_rate_pct=58.3,
        profit_factor=1.62,
        mean_bps=18.5,
        expectancy_r=0.31,
        payoff_ratio=1.35,
        slippage_bps=-4.1,
        tail_breaches_over_205bps=0,
    )
    ok_res = evaluate_sim_to_paper_to_live_merit(
        sim_pass, paper_pass, cumulative_trials_n=108, live_circuit_breaker_clear=True
    )
    assert ok_res["sim_stage_passed"] is True
    assert ok_res["paper_w5_stage_passed"] is True
    assert ok_res["live_autonomous_allowed"] is True
    assert ok_res["merit_tier"] == "STAGE5_LIVE_8798_AUTONOMOUS_ELIGIBLE_OWNER_GATED"

    # Circuit breaker tripped MUST pause autonomous entries even when Sim + Paper pass!
    halted_res = evaluate_sim_to_paper_to_live_merit(
        sim_pass, paper_pass, cumulative_trials_n=108, live_circuit_breaker_clear=False
    )
    assert halted_res["live_autonomous_allowed"] is False
    assert halted_res["merit_tier"] == "STAGE5_MERIT_CLEARED_PAUSED_BY_BREAKER"


def test_sealed_sim_paper_live_merit_manifest_sha_and_zero_leaks() -> None:
    assert MERIT_JSON_PATH.is_file()
    assert MERIT_MD_PATH.is_file()
    raw_txt = MERIT_JSON_PATH.read_text(encoding="utf-8")
    md_txt = MERIT_MD_PATH.read_text(encoding="utf-8")

    # Zero local home paths, account IDs, or raw dollar balances
    for blob in (raw_txt, md_txt):
        assert "/Users/" not in blob
        assert "C:\\Users\\" not in blob
        assert not re.search(r"\bPA[0-9A-Z]{6,}\b", blob)
        assert not re.search(r"\bDU[0-9A-Z]{5,}\b", blob)

    data = json.loads(raw_txt)
    assert data["schema"] == "sota.sim-to-paper-to-live-merit-manifest.v1"
    assert data["completed_hlz_trials_n"] == 107
    assert data["summary_counts"]["total_evaluated_strategies"] == 19
    assert data["summary_counts"]["live_autonomous_eligible_count"] == 0
    assert data["standalone_alpha_claim"] is False

    stored_sha = data.pop("manifest_sha256")
    computed = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert computed == stored_sha
