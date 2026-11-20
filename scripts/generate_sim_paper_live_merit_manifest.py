#!/usr/bin/env python3
"""Generate deterministic SHA-256 sealed Sim -> Paper -> Live Real Performance & Merit Manifest (`ADR-0034`)."""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sota_compounding.sim_to_paper_to_live_merit_manifest import (  # noqa: E402
    PaperBrokerEvidence,
    load_real_receipts_and_emit_manifest,
)

# Live-verified dimensionless telemetry snapshots captured 2026-11-20 11:11:37 ET (16:11:37Z) from Sovereign Noesis
# (`REAL_PAPER_PERFORMANCE_MANIFEST.md` on P2 :8787 + read-only :8798/:8796 — strictly zero USD balances / account IDs).
W5_PAPER_COHORT_SNAPSHOT_20261120 = {
    "observed_at_utc": "2026-11-20T16:11:37Z",
    "paper_accounts_online": 4,
    "active_concentrated_strategies": 10,
    "quarantined_legacy_bleeder_rts_isolated": 25,
    "closed_rts": 34,
    "w5_ready": True,
    "win_rate_pct": 52.9,
    "profit_factor": 0.59,
    "mean_bps": -11.9,
    "payoff_ratio": 0.46,
    "fill_latency_ms": 3981.9,
    "p2_overall_fill_latency_ms": 2955.3,
    "p2_overall_slippage_bps": -6.43,
    "mean_stop_pct": -0.76,
    "stop_bps_saved_per_stop": 124.5,
    "tail_breaches_over_205bps": 0,
    "ibkr_p4_online": True,
    "ibkr_p4_closed_rts": 0,
    "merit_verdict": "NOT_YET_PROMOTABLE_PF_0.59_BELOW_1.25_AND_PAYOFF_0.46X_BELOW_1.10X",
}

LIVE_8798_READONLY_SNAPSHOT_20261120 = {
    "observed_at_utc": "2026-11-20T16:11:37Z",
    "mode": "HALTED_DAILY_DRAWDOWN_BREAKER",
    "enabled": True,
    "draft_only": False,
    "auto_adopt_owner_entries": True,
    "circuit_breaker_clear": False,
    "breaker_halted": True,
    "daily_drawdown_limit_pct": -1.5,
    "daily_pnl_pct": -3.577,
    "ws_microstructure_symbols_fresh": 71,
    "last_loop_error": None,
    "guardian_ok": True,
    "guardian_open_positions": 6,
    "guardian_tracked_stops": 1,
    "guardian_untracked_positions": 0,
    "guardian_exits_fired": 13,
    "guardian_last_error": None,
}

# Real broker per-strategy dimensionless fills reconciled on P2 :8787 (`2026-11-20T16:11:37Z`) mapped to canonical IDs
PAPER_EVIDENCE_20261120: dict[str, PaperBrokerEvidence] = {
    "S01_ZARATTINI_2024_ORB": PaperBrokerEvidence(
        strategy_id="S01_ZARATTINI_2024_ORB",
        closed_rts=3,
        distinct_sessions=2,
        win_rate_pct=100.0,
        profit_factor=2.0,
        mean_bps=12.0,
        expectancy_r=0.12,
        payoff_ratio=1.0,
        slippage_bps=-0.1,
    ),
    "S02_ZARATTINI_2025_NOISE_VWAP": PaperBrokerEvidence(
        strategy_id="S02_ZARATTINI_2025_NOISE_VWAP",
        closed_rts=0,
        distinct_sessions=0,
        win_rate_pct=0.0,
        profit_factor=0.0,
        mean_bps=0.0,
        expectancy_r=0.0,
        payoff_ratio=0.0,
        slippage_bps=0.0,
    ),
    "S05_BJZZ_INST_DARK_OFI_PASSIVE": PaperBrokerEvidence(
        strategy_id="S05_BJZZ_INST_DARK_OFI_PASSIVE",
        closed_rts=20,
        distinct_sessions=3,
        win_rate_pct=65.0,
        profit_factor=0.88,
        mean_bps=-6.0,
        expectancy_r=-0.06,
        payoff_ratio=0.47,
        slippage_bps=-15.8,
    ),
    "S06_VIX_TERM_STRUCTURE_SURFER": PaperBrokerEvidence(
        strategy_id="S06_VIX_TERM_STRUCTURE_SURFER",
        closed_rts=1,
        distinct_sessions=1,
        win_rate_pct=0.0,
        profit_factor=0.0,
        mean_bps=-52.0,
        expectancy_r=-0.52,
        payoff_ratio=0.0,
        slippage_bps=0.0,
    ),
}


def main() -> int:
    mover_root = REPO_ROOT.parent / "sota-mover-ranker"
    manifest = load_real_receipts_and_emit_manifest(
        compounding_repo_root=REPO_ROOT,
        mover_ranker_repo_root=mover_root,
        w5_paper_cohort_snapshot=W5_PAPER_COHORT_SNAPSHOT_20261120,
        paper_by_strategy=PAPER_EVIDENCE_20261120,
        live_8798_readonly_snapshot=LIVE_8798_READONLY_SNAPSHOT_20261120,
    )
    counts = manifest["summary_counts"]
    print(f"MERIT MANIFEST SHA-256: {manifest['manifest_sha256'][:16]}...")
    pos_ids = manifest["positive_1x_net_candidates"]
    print(
        f"  evaluated={counts['total_evaluated_strategies']} | "
        f"live_auto_eligible={counts['live_autonomous_eligible_count']} | "
        f"positive_1x_candidates={counts['positive_1x_net_candidates_count']} ({pos_ids}) | "
        f"cost_killed_intraday={counts['cost_killed_intraday_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
