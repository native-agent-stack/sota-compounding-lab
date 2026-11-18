#!/usr/bin/env python3
"""v3.0.0 Organization Visibility, Cold-Archive, Zero-Burn Actions & MADR 4.0 Parity Doctor."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

ORG_NAME = "native-agent-stack"

MUST_BE_PRIVATE: frozenset[str] = frozenset(
    {
        "canon",
        "cc-sota-harness",
        "dsh-plugin-landscape",
        "ecosystem-records",
        "legacy-agent-stack-202610",
        "meridian",
        "sota-dynamic-trail-c",
        "sota-mover-ranker",
        "sota-private-vault",
        "sota-strategy-vault",
        "trading-sota",
        "tsl",
    }
)

PUBLIC_OPEN_CORE_ALLOWED: frozenset[str] = frozenset(
    {
        ".github",
        "agy-sota-harness",
        "astrolabe",
        "dsh-sota-harness",
        "native-agent-stack",
        "sota-compounding-lab",
        "sota-harness-foundation",
        "sota-multi-model-workflows",
        "sota-upstream-shelf-lock",
    }
)


def is_cold_archive_repo(name: str) -> bool:
    return name == "ecosystem-records" or name.startswith("legacy-")


def evaluate_org_repos(repos: list[dict[str, object]]) -> tuple[bool, list[str]]:
    """Verify Public/Private firewall and Cold-Archive supersession invariants."""
    violations: list[str] = []
    for row in repos:
        name = str(row.get("name") or "")
        is_priv = bool(row.get("isPrivate"))
        is_arch = bool(row.get("isArchived"))
        if (name in MUST_BE_PRIVATE or is_cold_archive_repo(name)) and not is_priv:
            violations.append(f"CRITICAL VISIBILITY BREACH: '{ORG_NAME}/{name}' MUST be PRIVATE (found PUBLIC)!")
        if not is_priv and name not in PUBLIC_OPEN_CORE_ALLOWED:
            violations.append(f"UNAPPROVED PUBLIC REPO: '{ORG_NAME}/{name}' is not in PUBLIC_OPEN_CORE_ALLOWED!")
        if is_cold_archive_repo(name) and not is_arch:
            violations.append(f"COLD-ARCHIVE DRIFT: Superseded repo '{ORG_NAME}/{name}' must have isArchived=true!")
    return (len(violations) == 0), violations


def _gh_api_json(endpoint: str) -> object | None:
    proc = subprocess.run(
        ["gh", "api", endpoint],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _gh_api_raw(endpoint: str) -> str | None:
    proc = subprocess.run(
        ["gh", "api", endpoint, "-H", "Accept: application/vnd.github.raw"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def audit_live_deep_invariants(repos: list[dict[str, object]]) -> list[str]:
    """Enforce zero private Actions burn, active MADR 4.0 CATALOG.md, and 40-char SHA workflow pins."""
    violations: list[str] = []
    for row in sorted(repos, key=lambda r: str(r.get("name") or "")):
        name = str(row.get("name") or "")
        is_priv = bool(row.get("isPrivate"))
        is_arch = bool(row.get("isArchived"))

        if is_priv:
            perms = _gh_api_json(f"/repos/{ORG_NAME}/{name}/actions/permissions")
            if isinstance(perms, dict) and bool(perms.get("enabled")):
                violations.append(
                    f"PRIVATE ACTIONS BURN: '{ORG_NAME}/{name}' has actions_enabled=true (must be false)!"
                )

        if not is_arch:
            cat = _gh_api_json(f"/repos/{ORG_NAME}/{name}/contents/docs/adr/CATALOG.md")
            if not isinstance(cat, dict) or cat.get("name") != "CATALOG.md":
                violations.append(
                    f"MISSING MADR 4.0 CATALOG: Active repo '{ORG_NAME}/{name}' lacks docs/adr/CATALOG.md!"
                )

            wfs = _gh_api_json(f"/repos/{ORG_NAME}/{name}/contents/.github/workflows")
            if isinstance(wfs, list):
                for item in wfs:
                    if not isinstance(item, dict):
                        continue
                    wf_name = str(item.get("name") or "")
                    if not wf_name.endswith((".yml", ".yaml")):
                        continue
                    raw = _gh_api_raw(f"/repos/{ORG_NAME}/{name}/contents/.github/workflows/{wf_name}")
                    if not raw:
                        continue
                    for m in re.finditer(r"^\s*-?\s*uses:\s*([^\s#]+)", raw, flags=re.MULTILINE):
                        ref = m.group(1)
                        if ref.startswith("./"):
                            continue
                        if "@" not in ref or not re.fullmatch(r"[0-9a-f]{40}", ref.split("@", 1)[1]):
                            violations.append(
                                f"UNPINNED GITHUB ACTION in '{ORG_NAME}/{name}/.github/workflows/{wf_name}': {ref}"
                            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit native-agent-stack v3.0.0 4-invariant org firewall")
    parser.add_argument("--offline", action="store_true", help="Run deterministic offline policy self-test")
    parser.add_argument("--live", action="store_true", help="Query live GitHub org via `gh` API")
    args = parser.parse_args()

    if args.offline or not args.live:
        ok_good, v_good = evaluate_org_repos(
            [
                {"name": "native-agent-stack", "isPrivate": False, "isArchived": False},
                {"name": "sota-compounding-lab", "isPrivate": False, "isArchived": False},
                {"name": "sota-mover-ranker", "isPrivate": True, "isArchived": False},
                {"name": "legacy-agent-stack-202610", "isPrivate": True, "isArchived": True},
                {"name": "ecosystem-records", "isPrivate": True, "isArchived": True},
            ]
        )
        ok_bad_vis, _ = evaluate_org_repos([{"name": "sota-mover-ranker", "isPrivate": False, "isArchived": False}])
        ok_bad_arch, _ = evaluate_org_repos(
            [{"name": "legacy-agent-stack-202610", "isPrivate": True, "isArchived": False}]
        )
        if not ok_good or ok_bad_vis or ok_bad_arch:
            print(f"[ORG-FIREWALL FAIL] Offline policy self-test failed: {v_good}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "offline",
                    "org": ORG_NAME,
                    "protected_private_tier_count": len(MUST_BE_PRIVATE),
                    "allowed_public_open_core_count": len(PUBLIC_OPEN_CORE_ALLOWED),
                },
                indent=2,
            )
        )
        return 0

    proc = subprocess.run(
        ["gh", "repo", "list", ORG_NAME, "--limit", "50", "--json", "name,isPrivate,isArchived,visibility,url"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"[ORG-FIREWALL FAIL] gh repo list failed: {proc.stderr.strip()}", file=sys.stderr)
        return 1

    live_repos: list[dict[str, object]] = json.loads(proc.stdout)
    _, base_violations = evaluate_org_repos(live_repos)
    deep_violations = audit_live_deep_invariants(live_repos)
    violations = base_violations + deep_violations
    ok = len(violations) == 0
    summary = {
        "status": "PASS" if ok else "FAIL",
        "mode": "live",
        "org": ORG_NAME,
        "repos_total": len(live_repos),
        "public_open_core_repos": sorted(str(r["name"]) for r in live_repos if not r.get("isPrivate")),
        "active_private_repos": sorted(
            str(r["name"]) for r in live_repos if r.get("isPrivate") and not r.get("isArchived")
        ),
        "cold_archived_private_repos": sorted(
            str(r["name"]) for r in live_repos if r.get("isPrivate") and r.get("isArchived")
        ),
        "invariants_checked": [
            "1_public_vs_private_visibility_firewall",
            "2_superseded_legacy_cold_archived",
            "3_zero_private_github_actions_burn",
            "4_active_madr40_catalog_and_40char_sha_actions",
        ],
        "violations": violations,
    }
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
