"""Tests for ADR-0033: Trial #106 (H112) Pre-Wall 424B5 Dilution Short Receipt & Anti-Truncation Guard."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_h112_trial106_424b5_dilution_short.py"
RECEIPT_PATH = REPO_ROOT / "docs" / "TRIAL106_H112_PREWALL_RECEIPT.json"


def _load_mod():
    spec = importlib.util.spec_from_file_location("run_h112_trial106_424b5_dilution_short", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trial106_receipt_integrity_prewall_unclamped_and_honest_verdict() -> None:
    assert RECEIPT_PATH.is_file()
    r = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert r["schema"] == "sota.prewall-trial106-h112-receipt.v1"
    assert r["trial_number"] == 106
    assert r["study_wall_date"] == "2026-10-15"
    assert r["sample_end_date"] <= "2026-10-15"
    assert r["prewall_sip_sessions_scanned"] == 700
    assert len(r["receipt_sha256"]) == 64

    p = r["primary_424b5_rth_short"]
    assert p["events"] >= 2000
    assert p["distinct_dates"] >= 500
    assert p["hygiene_14pt_passed"] is True
    assert p["hygiene_violations"] == []
    assert p["passed"] is False
    assert p["promotion_track"] == "PROTECTIVE_FILTER_OR_GOVERNANCE"

    # Verify anti-truncation law: 3d MTM diagnostic must NOT contain artificial max(-200.0) daily clamping
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "max(-200.0," not in src
    d3 = r["diagnostic_424b5_3d_nonoverlapping_mtm_short"]
    assert d3["passed"] is False
    assert d3["calendar_day_stats_2x"]["max_drawdown_pct"] > 10.0

    sh = r["diagnostic_s3_effect_shelf_short"]
    assert sh["hygiene_14pt_passed"] is True
    assert sh["passed"] is False


def test_short_uncapped_lossless_bar_path_clamps_loss_at_adr0022() -> None:
    mod = _load_mod()
    res = mod._simulate_short_uncapped_lossless([100.0, 103.0], [103.5, 104.0], [99.8, 102.0], [103.0, 103.5])
    assert res["exit_reason"] == "ADR0022_HARD_STOP"
    assert abs(res["gross_bps"] - (-200.0)) < 1e-6
