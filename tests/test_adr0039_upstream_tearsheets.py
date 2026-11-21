from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "export_upstream_tearsheets.py"


def _load_exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_upstream_tearsheets", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exporter_enforces_prewall_study_wall_and_adr0022_stop_bounds() -> None:
    mod = _load_exporter()
    assert mod.STUDY_WALL_ISO == "2026-10-15"
    assert mod.ADR0022_MAX_STOP_PCT == 2.0

    with pytest.raises(ValueError, match="PREWALL_STUDY_WALL_VIOLATION"):
        mod.export_all_upstream_tearsheets(end_date="2026-10-16")

    with pytest.raises(ValueError, match="ADR0022_STOP_VIOLATION"):
        mod.export_all_upstream_tearsheets(stop_pct=2.01)

    with pytest.raises(ValueError, match="ADR0022_STOP_VIOLATION"):
        mod.export_all_upstream_tearsheets(stop_pct=0.01)


def test_exporter_contains_zero_custom_html_js_and_uses_official_upstream_apis() -> None:
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "create_tearsheet" in src
    assert "TearsheetConfig" in src
    assert "vbt.Portfolio.from_signals" in src
    assert "qs.reports.html" in src
    assert "FAIL_CLOSED_NAUTILUS" in src
    assert "BANNED_FEED_BLOCKED" in src
    # Zero self-built HTML/JS template literals
    assert "<!DOCTYPE html>" not in src
    assert "<html" not in src.lower()
    assert "<script" not in src.lower()
