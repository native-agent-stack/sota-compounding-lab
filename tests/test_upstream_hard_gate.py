"""Fail-closed unit tests for sota-compounding-lab upstream-map.toml & merge_gate.py."""

from __future__ import annotations

import pathlib
import subprocess

import merge_gate

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_compounding_lab_passes_hard_upstream_map_gate() -> None:
    assert merge_gate.verify_upstream_map(REPO_ROOT) == []


def test_rejects_unmapped_or_placeholder_engine_entry(tmp_path: pathlib.Path) -> None:
    (tmp_path / "UPSTREAM.md").write_text(
        "- `nautechsystems/nautilus_trader` @ `da6b57aefdd62b44f675e434ffcfefca7f4025dd`\n",
        encoding="utf-8",
    )
    (tmp_path / "upstream-map.toml").write_text(
        '[areas]\n"src/bad.py" = { upstream = "cpython-stdlib", how = "build-on", feature = "f", why = "w" }\n',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "bad.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "orphan.py").write_text("y = 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    errs = merge_gate.verify_upstream_map(tmp_path, shelf_dir=tmp_path / "absent")
    assert any("non-library placeholder" in e for e in errs)
    assert any("orphan.py" in e for e in errs)
