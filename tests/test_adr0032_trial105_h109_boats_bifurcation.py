"""Tests for ADR-0032: Trial #105 (H109) Pre-Wall BOATS Bifurcation Receipt & Bar-Path Lossless Simulator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_h109_trial105_boats_bifurcation.py"
RECEIPT_PATH = REPO_ROOT / "docs" / "TRIAL105_H109_PREWALL_RECEIPT.json"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_h109_trial105_boats_bifurcation", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trial105_receipt_integrity_prewall_and_honest_disposition() -> None:
    assert RECEIPT_PATH.is_file()
    r = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert r["schema"] == "sota.prewall-trial105-h109-receipt.v1"
    assert r["trial_number"] == 105
    assert r["study_wall_date"] == "2026-10-15"
    assert r["sample_end_date"] <= "2026-10-15"
    assert r["prewall_overlap_sessions"] >= 497
    assert len(r["receipt_sha256"]) == 64

    leg_a = r["primary_leg_a_boats_fade_short"]
    assert leg_a["events"] >= 200
    assert leg_a["distinct_dates"] >= 60
    assert leg_a["hygiene_14pt_passed"] is True
    assert leg_a["hygiene_violations"] == []
    assert leg_a["passed"] is False
    assert leg_a["promotion_track"] == "PROTECTIVE_FILTER_OR_GOVERNANCE"

    leg_b = r["diagnostic_leg_b_8k_vwap_hold_long"]
    assert leg_b["events"] >= 200
    assert leg_b["hygiene_14pt_passed"] is True
    assert leg_b["conformal_pid_calibrated_stop_pct"] <= 2.0
    assert leg_b["passed"] is False
    assert leg_b["promotion_track"] == "PROTECTIVE_FILTER_OR_GOVERNANCE"


def test_uncapped_lossless_bar_path_respects_adr0022_and_uncapped_upside() -> None:
    mod = _load_script_module()

    # Adverse short path must stop out strictly at -200 bps (ADR-0022 <= 2.00%)
    o = [100.0, 101.0, 105.0]
    h = [101.5, 103.5, 106.0]
    lo = [99.8, 100.5, 104.0]
    c = [101.0, 103.0, 105.5]
    res_stop = mod._simulate_uncapped_lossless_bar_path(o, h, lo, c, side="short", stop_pct=2.0)
    assert res_stop["exit_reason"] == "ADR0022_HARD_STOP"
    assert abs(res_stop["gross_bps"] - (-200.0)) < 1e-6

    # Favorable long runner (+12% run without fixed TP cap) harvests 50% at +1.25% and retains runner convexity
    o2 = [100.0, 101.5, 106.0, 112.0]
    h2 = [101.8, 106.5, 112.5, 112.5]
    lo2 = [99.9, 101.4, 105.8, 111.5]
    c2 = [101.5, 106.0, 112.0, 112.0]
    res_run = mod._simulate_uncapped_lossless_bar_path(o2, h2, lo2, c2, side="long", stop_pct=2.0)
    assert res_run["harvested_50"] is True
    assert res_run["gross_bps"] > 500.0
