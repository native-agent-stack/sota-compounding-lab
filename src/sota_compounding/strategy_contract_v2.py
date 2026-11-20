"""Canonical `sota.strategy-contract.v2` ("Write Once, Run Everywhere") & `ADR-0059` TCA Feedback (`ADR-0027`).

Standardizes:
1. Shared `StrategyContractV2` schema across `S1` Pre-Wall Factory, `S2` Nautilus/hftbacktest Queue Parity,
   and `P1-P4` Paper execution (`sota-<strategy_slug>-<lane>-<seq>` client IDs).
2. Closed-loop `ADR-0059` Causal Markout TCA -> `S1`/`S2` cost-model calibration (`calibrate_sim_cost_from_paper_tca`).
3. Strictly causal calendar-aligned `next_session_date` validation ensuring ZERO multi-day gap jump and
   ZERO CPCV fold span overlap across `<= 2026-10-15` pre-wall sweeps.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

STRATEGY_CONTRACT_SCHEMA_V2 = "sota.strategy-contract.v2"
STUDY_WALL_ISO = "2026-10-15"
VALID_LANES = frozenset({"p1", "p2", "p3", "p4", "sim"})

QUARANTINED_PAPER_STRATEGIES: frozenset[str] = frozenset(
    {
        "SOTA_DUAL_CLIMAX_STOP_AND_REVERSE",
        "POST_VIX_REVERSAL",
        "POST_VIX_REVERSAL_MAJOR_GROWTH_THRUST",
        "SOTA_DUAL_EXTREME_BALANCE_BREAK_EARLY",
        "BLIND_WAVE2_REENTRY",
        "RAPID_WAVE2_REENTRY_STRESS",
        "FLOW_VELOCITY_CAPITAL_RECYCLE",
    }
)

CANONICAL_SLUGS: dict[str, str] = {
    "BJZZ_SUBPENNY_INST_DARK_SWEEP": "bjzz_dark_sweep",
    "VIX_SPIKE_AND_CRUSH_REGIME_SURFER": "vix_regime_surf",
    "ZARATTINI_2024_5M_ORB": "zar24_5m_orb",
    "ZARATTINI_2025_NOISE_VWAP": "zar25_noise_vwap",
    "H090_TIGHT_SPREAD_HIGH_OFI_PASSIVE_BID": "h090_ofi_bid",
    "H01_PRE_TRAVEL_POCKET_PIVOT": "h01_pocket_vdu",
    "H08_FIRST_HAND_CATALYST_IGNITION": "h08_sec_8k_ep",
    "BOATS_24X5_OPENBB_STAGED_LIMIT": "boats_obb_stage",
    "QLIB_ALPHA158_LGBM_XSEC_BENCHMARK": "qlib_a158_lgbm",
}
_SLUG_TO_ID = {v: k for k, v in CANONICAL_SLUGS.items()}
_COID_RE = re.compile(r"^sota-(?P<slug>[a-z0-9_]{2,28})-(?P<lane>p1|p2|p3|p4|sim)-(?P<seq>[a-z0-9]{1,12})$")


def build_client_order_id_v2(strategy_id: str, lane: str, seq: int) -> str:
    sid = str(strategy_id or "").strip().upper()
    slug = CANONICAL_SLUGS.get(sid) or re.sub(r"[^a-z0-9_]+", "_", sid.lower()).strip("_")[:24]
    ln = str(lane or "").strip().lower()
    if ln not in VALID_LANES:
        raise ValueError(f"Unsupported lane {lane!r}")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0 or seq > 999_999_999_999:
        raise ValueError(f"Invalid seq (must be int 0..999_999_999_999): {seq!r}")
    coid = f"sota-{slug}-{ln}-{int(seq):06d}"
    if len(coid) > 48:
        raise ValueError(f"client_order_id exceeds 48 chars: {coid}")
    return coid


def parse_client_order_id_v2(client_order_id: str) -> dict[str, str] | None:
    m = _COID_RE.match(str(client_order_id or "").strip().lower())
    if not m:
        return None
    slug = m.group("slug")
    return {
        "client_order_id": m.group(0),
        "strategy_slug": slug,
        "strategy_id": _SLUG_TO_ID.get(slug, slug.upper()),
        "lane": m.group("lane"),
        "seq": m.group("seq"),
    }


@dataclass(frozen=True, slots=True)
class TcaCostCalibration:
    strategy_id: str
    paper_fills_count: int
    observed_mean_slippage_bps: float
    observed_adverse_markout_5s_bps: float
    calibrated_1x_half_spread_bps: float
    calibrated_2x_stress_bps: float
    feedback_applied: bool


def calibrate_sim_cost_from_paper_tca(
    *,
    strategy_id: str,
    paper_fills_count: int,
    observed_mean_slippage_bps: float,
    observed_adverse_markout_5s_bps: float,
    prior_half_spread_bps: float = 4.0,
) -> TcaCostCalibration:
    """Conservatively ratchet `S1`/`S2` cost assumptions upward (never below prior) from `ADR-0059` Paper TCA."""
    for val, nm in (
        (observed_mean_slippage_bps, "observed_mean_slippage_bps"),
        (observed_adverse_markout_5s_bps, "observed_adverse_markout_5s_bps"),
        (prior_half_spread_bps, "prior_half_spread_bps"),
    ):
        if isinstance(val, bool) or not math.isfinite(float(val)):
            raise ValueError(f"{nm} must be finite")
    n_fills = max(0, int(paper_fills_count))
    prior = max(0.5, float(prior_half_spread_bps))
    # Empirical-Bayes shrinkage (kappa=20) so noisy small paper samples never artificially under-estimate sim costs
    kappa = 20.0
    obs_cost = max(0.0, float(observed_mean_slippage_bps)) + max(0.0, -float(observed_adverse_markout_5s_bps) * 0.25)
    shrunk = (prior * kappa + obs_cost * n_fills) / (kappa + n_fills)
    eff_1x = round(max(prior, shrunk), 4)
    return TcaCostCalibration(
        strategy_id=str(strategy_id).strip().upper(),
        paper_fills_count=n_fills,
        observed_mean_slippage_bps=round(float(observed_mean_slippage_bps), 4),
        observed_adverse_markout_5s_bps=round(float(observed_adverse_markout_5s_bps), 4),
        calibrated_1x_half_spread_bps=eff_1x,
        calibrated_2x_stress_bps=round(eff_1x * 2.0, 4),
        feedback_applied=(n_fills > 0),
    )


def validate_next_session_date_alignment(
    trade_dates: list[str],
    next_session_dates: list[str | None],
    xnys_ordered_sessions: list[str],
    *,
    study_wall_iso: str = STUDY_WALL_ISO,
) -> dict[str, Any]:
    """Fail-closed verifier proving every `(trade_date, next_session_date)` is exact 1-step XNYS & `<= 2026-10-15`."""
    if len(trade_dates) != len(next_session_dates):
        raise ValueError("trade_dates and next_session_dates length mismatch")
    if not xnys_ordered_sessions:
        raise ValueError("xnys_ordered_sessions must be non-empty")
    if any(d > study_wall_iso for d in xnys_ordered_sessions):
        raise ValueError(f"XNYS session list crosses pre-wall {study_wall_iso}")
    xnys_set = set(xnys_ordered_sessions)
    last_session = xnys_ordered_sessions[-1]
    next_map: dict[str, str] = {}
    for i in range(len(xnys_ordered_sessions) - 1):
        d_cur = xnys_ordered_sessions[i]
        d_nxt = xnys_ordered_sessions[i + 1]
        next_map[d_cur] = d_nxt

    checked = 0
    wall_boundary_nulls = 0
    mismatches = 0
    postwall_breaches = 0
    for td, nsd in zip(trade_dates, next_session_dates, strict=True):
        if td > study_wall_iso:
            postwall_breaches += 1
            continue
        if td not in xnys_set:
            # Unknown non-XNYS date MUST fail closed (never silently treated as wall boundary!)
            mismatches += 1
            continue
        expected = next_map.get(td)
        if expected is None:
            # Only the exact final pre-wall XNYS session may have next_session_date is None
            if td == last_session and nsd is None:
                wall_boundary_nulls += 1
            else:
                postwall_breaches += 1
            continue
        if nsd is None:
            mismatches += 1
        elif nsd > study_wall_iso:
            postwall_breaches += 1
        elif nsd != expected:
            mismatches += 1
        else:
            checked += 1

    if postwall_breaches > 0:
        raise ValueError(f"POSTWALL_BREACH in next_session_date alignment: {postwall_breaches} rows")
    if mismatches > 0:
        raise ValueError(f"NEXT_SESSION_DATE_MISMATCH (multi-day gap / CPCV span overlap): {mismatches} rows")
    return {
        "ok": True,
        "schema": STRATEGY_CONTRACT_SCHEMA_V2,
        "study_wall_date": study_wall_iso,
        "verified_1step_rows": checked,
        "wall_boundary_nulls": wall_boundary_nulls,
        "mismatches": 0,
        "postwall_breaches": 0,
        "verified_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
    }


def contract_v2_summary() -> dict[str, Any]:
    return {
        "schema_version": STRATEGY_CONTRACT_SCHEMA_V2,
        "study_wall_date": STUDY_WALL_ISO,
        "quarantined_strategies": sorted(QUARANTINED_PAPER_STRATEGIES),
        "active_canonical_slugs": dict(CANONICAL_SLUGS),
        "tca_cost_feedback": asdict(
            calibrate_sim_cost_from_paper_tca(
                strategy_id="BJZZ_SUBPENNY_INST_DARK_SWEEP",
                paper_fills_count=0,
                observed_mean_slippage_bps=0.0,
                observed_adverse_markout_5s_bps=0.0,
            )
        ),
    }
