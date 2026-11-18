#!/usr/bin/env python3
"""Organization Public/Private Visibility Firewall Verifier (`github.com/native-agent-stack`)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

ORG_NAME = "native-agent-stack"

MUST_BE_PRIVATE: frozenset[str] = frozenset(
    {
        "sota-mover-ranker",
        "sota-dynamic-trail-c",
        "cc-sota-harness",
        "sota-private-vault",
        "sota-strategy-vault",
        "ecosystem-records",
        "trading-sota",
    }
)

PUBLIC_OPEN_CORE_ALLOWED: frozenset[str] = frozenset(
    {
        ".github",
        "native-agent-stack",
        "sota-compounding-lab",
        "sota-multi-model-workflows",
        "sota-harness-foundation",
        "dsh-sota-harness",
        "agy-sota-harness",
        "sota-upstream-shelf-lock",
    }
)


def evaluate_org_repos(repos: list[dict[str, object]]) -> tuple[bool, list[str]]:
    """Verify no proprietary execution/vault repo in `MUST_BE_PRIVATE` is exposed publicly."""
    violations: list[str] = []
    for row in repos:
        name = str(row.get("name") or "")
        is_priv = bool(row.get("isPrivate"))
        if name in MUST_BE_PRIVATE and not is_priv:
            violations.append(f"CRITICAL VISIBILITY BREACH: '{ORG_NAME}/{name}' MUST be PRIVATE (found PUBLIC)!")
    return (len(violations) == 0), violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit native-agent-stack Public vs. Private firewall")
    parser.add_argument("--offline", action="store_true", help="Run deterministic offline policy self-test")
    parser.add_argument("--live", action="store_true", help="Query live GitHub org via `gh repo list`")
    args = parser.parse_args()

    if args.offline or not args.live:
        ok_good, v_good = evaluate_org_repos(
            [
                {"name": "sota-compounding-lab", "isPrivate": False},
                {"name": "dsh-sota-harness", "isPrivate": False},
                {"name": "sota-mover-ranker", "isPrivate": True},
                {"name": "sota-dynamic-trail-c", "isPrivate": True},
                {"name": "ecosystem-records", "isPrivate": True},
            ]
        )
        ok_bad, _ = evaluate_org_repos([{"name": "sota-mover-ranker", "isPrivate": False}])
        if not ok_good or ok_bad:
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
        ["gh", "repo", "list", ORG_NAME, "--limit", "50", "--json", "name,isPrivate,visibility,url"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"[ORG-FIREWALL FAIL] gh repo list failed: {proc.stderr.strip()}", file=sys.stderr)
        return 1

    live_repos: list[dict[str, object]] = json.loads(proc.stdout)
    ok, violations = evaluate_org_repos(live_repos)
    summary = {
        "status": "PASS" if ok else "FAIL",
        "mode": "live",
        "org": ORG_NAME,
        "repos_total": len(live_repos),
        "public_repos": sorted(str(r["name"]) for r in live_repos if not r.get("isPrivate")),
        "private_repos": sorted(str(r["name"]) for r in live_repos if r.get("isPrivate")),
        "violations": violations,
    }
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
