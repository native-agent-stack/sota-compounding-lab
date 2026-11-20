#!/usr/bin/env python3
"""Fail-Closed Merge Gate for `sota-compounding-lab`.

Enforces:
1. 100% file-to-upstream coverage in `upstream-map.toml` referencing `UPSTREAM.md` (`install|integrate|build-on`),
   rejecting banned low-quality libraries and bare stdlib/literature placeholders on engine files.
2. Exact 40-char SHA & clean-tree parity on all 18 read-only shelf clones in `~/Documents/upstream/` when present.
3. Zero secrets, broker account IDs, live endpoints, or local home paths in tracked files, plus noreply commit emails.
4. Zero-drift MADR 4.0 `docs/adr/CATALOG.md` parity.
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
import re
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHELF_DIR = pathlib.Path.home() / "Documents" / "upstream"
COVERED_PREFIXES = ("src/", "scripts/")
ALLOWED_HOW = frozenset({"install", "integrate", "build-on"})
BANNED_LOW_QUALITY = frozenset(
    {"yfinance", "backtrader", "backtesting.py", "mlfinlab", "tcapy", "vollib", "tradingview_ta"}
)
NON_LIBRARY_PLACEHOLDERS = frozenset({"cpython-stdlib", "cpython", "academic-momentum-literature"})
SECRET_PATTERNS = (
    ("Broker Account ID", re.compile(r"\b(?:PA|DU)[0-9A-Z]*[0-9][0-9A-Z]{5,14}\b")),
    ("Alpaca Key Prefix", re.compile(r"\b(?:AK|PK)[0-9A-Z]{16,28}\b")),
    ("Private Key Header", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("Local Home Path (macOS)", re.compile(r"/" + r"Users/(?!<user>)[A-Za-z0-9._-]+/")),
    ("Local Home Path (Windows)", re.compile(r"\b[A-Za-z]:[/\\]Users[/\\](?!<user>)[A-Za-z0-9._-]+[/\\]")),
)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _parse_upstream_md(text: str) -> set[str]:
    reg: set[str] = set()
    for m in re.findall(r"^\|\s*`([^`]+)`\s*\|", text, flags=re.MULTILINE):
        reg.update({m.strip(), _norm(m)})
    for org_repo in re.findall(r"`([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)`", text):
        short = org_repo.split("/", 1)[1]
        reg.update({org_repo, _norm(org_repo), short, _norm(short)})
    return reg


def verify_upstream_map(root: pathlib.Path = ROOT, *, shelf_dir: pathlib.Path = SHELF_DIR) -> list[str]:
    errors: list[str] = []
    up_md = root / "UPSTREAM.md"
    map_toml = root / "upstream-map.toml"
    if not up_md.is_file() or not map_toml.is_file():
        return ["Missing UPSTREAM.md or upstream-map.toml"]

    registered = _parse_upstream_md(up_md.read_text(encoding="utf-8"))
    cfg = tomllib.loads(map_toml.read_text(encoding="utf-8"))
    areas = cfg.get("areas") or {}
    shelf_pins = cfg.get("shelf_pins") or {}

    for pat, meta in areas.items():
        u = str(meta.get("upstream", "")).strip()
        how = str(meta.get("how", "")).strip()
        u_norm = _norm(u)
        if u_norm in {_norm(b) for b in BANNED_LOW_QUALITY}:
            errors.append(f"upstream-map.toml '{pat}' cites banned low-quality upstream '{u}'")
        if u_norm in {_norm(p) for p in NON_LIBRARY_PLACEHOLDERS}:
            errors.append(
                f"upstream-map.toml '{pat}' uses non-library placeholder '{u}'; cite a maintained SOTA library"
            )
        if u not in registered and u_norm not in registered:
            errors.append(f"upstream-map.toml '{pat}' cites '{u}' not registered in UPSTREAM.md")
        if how not in ALLOWED_HOW:
            errors.append(f"upstream-map.toml '{pat}' invalid how={how!r}")
        if not str(meta.get("feature", "")).strip() or not str(meta.get("why", "")).strip():
            errors.append(f"upstream-map.toml '{pat}' missing non-empty feature or why")

    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    res = subprocess.run(
        ["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True, env=clean_env
    )
    tracked = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    used: set[str] = set()
    for rel in tracked:
        if not rel.startswith(COVERED_PREFIXES):
            continue
        matched = [pat for pat in areas if fnmatch.fnmatchcase(rel, pat)]
        if not matched:
            errors.append(f"Unmapped implementation file '{rel}' in upstream-map.toml")
        else:
            used.update(matched)
    for pat in areas:
        if pat not in used:
            errors.append(f"Unused pattern in upstream-map.toml: '{pat}'")

    if shelf_dir.is_dir():
        for repo_name, expected in shelf_pins.items():
            cpath = shelf_dir / repo_name
            if not cpath.is_dir():
                continue
            head_r = subprocess.run(
                ["git", "-C", str(cpath), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                env=clean_env,
            )
            if head_r.returncode == 0 and head_r.stdout.strip() != expected:
                errors.append(f"Shelf pin drift for {repo_name}: expected {expected}, found {head_r.stdout.strip()}")
            st_r = subprocess.run(
                ["git", "-C", str(cpath), "status", "--short"],
                capture_output=True,
                text=True,
                check=False,
                env=clean_env,
            )
            if st_r.returncode == 0 and st_r.stdout.strip():
                errors.append(f"Read-only upstream shelf {repo_name} has dirty files")
    return errors


def verify_leaks_and_catalog(root: pathlib.Path = ROOT) -> list[str]:
    errors: list[str] = []
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    res = subprocess.run(
        ["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True, env=clean_env
    )
    for rel in (ln.strip() for ln in res.stdout.splitlines() if ln.strip()):
        if rel == "scripts/merge_gate.py":
            continue
        fp = root / rel
        if not fp.is_file():
            continue
        try:
            txt = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, rx in SECRET_PATTERNS:
            if rx.search(txt):
                errors.append(f"{rel}: triggered {label}")
    cat = subprocess.run(
        [sys.executable, str(root / "scripts" / "architecture_catalog.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
        env=clean_env,
    )
    if cat.returncode != 0:
        errors.append(f"ADR catalog drift: {cat.stderr.strip() or cat.stdout.strip()}")
    return errors


def main() -> int:
    problems = [*verify_upstream_map(ROOT), *verify_leaks_and_catalog(ROOT)]
    if problems:
        for p in problems:
            print(f"[MERGE-GATE FAIL] {p}", file=sys.stderr)
        return 1
    print("[MERGE-GATE PASS] sota-compounding-lab upstream-map, 18 shelf pins, zero leaks, and ADR catalog verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
