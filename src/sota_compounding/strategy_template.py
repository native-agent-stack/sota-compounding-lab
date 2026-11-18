"""Research-Converged Quantitative Strategy Templates & 7-Feed Point-in-Time Contract Gate.

Enforces `ADR-0006` across all scaffolded strategy manifests in `sota-compounding-lab`:
1. **3-Tier Research Convergence (`ADR-0004`):** Immutable 64-hex `input_package_sha256`, `>= 1`
   40-hex `pinned_upstream_commits`, non-empty `reproducible_command`, `E0..E4` `evidence_grade`,
   and `other_repos_authority == "PHILOSOPHY_ONLY"`.
2. **7-Feed Point-in-Time (`PIT`) Routing:** Consolidated `sip` (`04:00–20:00 ET`) + `boats`
   (`20:00–04:00 ET`), explicit `dec6_2026_23x5_cutover_enabled` flag, raw universe hygiene
   (`>= $2.00`, `>= $2,000,000` ADV) paired with `split_dividend_all` total-return series and
   `[-1, 3]` split-exclusion windows, SEC EDGAR `acceptanceDateTime` with `900s` (`15m`) half-life
   decay, `S-3`/`424B5` dilution shields, and `>= 1` session FINRA RegSHO short-volume lag.
3. **Pre-Wall Statistical & Queue Gates:** `study_wall_date <= "2026-10-15"`, 64-hex `spec_sha256`,
   Harvey-Liu-Zhu `hlz_t_hurdle >= 3.0`, `>= 5` CPCV paths (`purge >= 3`, `embargo >= 2`,
   `max_pbo <= 0.30`), `min_dsr >= 0.95`, `ADVERSE_STOP_FIRST` intrabar tie-breaking, and
   `[1.0, 2.0]` transaction + square-root impact cost stress multipliers.
4. **Paper-Only Execution & `ADR-0022` Stops:** `live_execution_enabled is False`,
   `max_initial_stop_pct <= 2.0`, `alpaca_paper_max_rpm <= 200`, and `ibkr_paper_port == 4002`
   (`ibkr_paper_max_msg_per_min <= 3000`).
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

from sota_compounding.lifecycle import (
    ADR0022_MAX_STOP_PCT,
    ALPACA_PAPER_MAX_RPM,
    IBKR_PAPER_MAX_MSG_PER_MIN,
    IBKR_PAPER_PORT,
    STUDY_WALL_DATE,
    hlz_dynamic_t_hurdle,
)

ARCHETYPES: tuple[str, ...] = (
    "PRE_TRAVEL_EARLY_IGNITION",
    "FIRST_HAND_SEC_EPISODIC_PIVOT",
    "MEGA_CAP_NEAR_LOSSLESS_ORB",
    "CLIMAX_VWAP_MEAN_REVERSION",
)

VALID_EVIDENCE_GRADES: frozenset[str] = frozenset({"E0", "E1", "E2", "E3", "E4"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_40_RE = re.compile(r"^[0-9a-f]{40}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_ARCHETYPE_DEFAULTS: dict[str, dict[str, object]] = {
    "PRE_TRAVEL_EARLY_IGNITION": {
        "signal_family": "early_volume_velocity_pocket_pivot",
        "min_rvol_mult": 2.2,
        "min_vol_z": 2.3,
        "max_spread_bps": 35.0,
        "default_initial_stop_pct": 1.80,
    },
    "FIRST_HAND_SEC_EPISODIC_PIVOT": {
        "signal_family": "sec_edgar_8k_day1_episodic_pivot",
        "primary_forms": ["8-K", "SC 13D"],
        "max_spread_bps": 35.0,
        "default_initial_stop_pct": 1.90,
    },
    "MEGA_CAP_NEAR_LOSSLESS_ORB": {
        "signal_family": "opening_range_breakout_passive_mid_mfe_trail",
        "orb_window_minutes": 5,
        "max_spread_bps": 18.0,
        "default_initial_stop_pct": 1.25,
    },
    "CLIMAX_VWAP_MEAN_REVERSION": {
        "signal_family": "parabolic_exhaustion_vwap_reversion",
        "require_locate_and_ssr_check": True,
        "max_spread_bps": 28.0,
        "default_initial_stop_pct": 1.75,
    },
}


def _is_int_not_bool(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _lookup_field(
    payload: dict[str, Any],
    section_name: str,
    field_name: str,
    errors: list[str],
) -> Any:
    has_top = field_name in payload
    sec_raw = payload.get(section_name)
    if section_name in payload and not isinstance(sec_raw, dict):
        return None
    sec = sec_raw if isinstance(sec_raw, dict) else {}
    has_sec = field_name in sec
    if has_top and has_sec and payload[field_name] != sec[field_name]:
        errors.append(f"Conflicting top-level and {section_name}.{field_name} values")
        return payload[field_name]
    if has_top:
        return payload[field_name]
    return sec.get(field_name)


def validate_converged_strategy_template(payload: dict[str, Any]) -> list[str]:
    """Validate a converged SOTA strategy template fail-closed; returns `[]` iff all gates pass."""
    if not isinstance(payload, dict):
        return ["payload must be a dict"]

    errors: list[str] = []

    for section_name in (
        "research_convergence",
        "datafeed_pit_contract",
        "statistical_and_queue_gates",
        "paper_execution_and_adr0022",
    ):
        if section_name in payload and not isinstance(payload[section_name], dict):
            errors.append(f"{section_name} must be a dict when present")

    strategy_id = payload.get("strategy_id")
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        errors.append("strategy_id must be a non-empty string")

    archetype = payload.get("archetype")
    if not isinstance(archetype, str) or archetype not in ARCHETYPES:
        errors.append(f"archetype must be one of {ARCHETYPES} (got {archetype!r})")

    # (a) ADR-0004 3-tier Research Convergence
    input_sha = _lookup_field(payload, "research_convergence", "input_package_sha256", errors)
    if not isinstance(input_sha, str) or not _SHA256_RE.fullmatch(input_sha):
        errors.append("input_package_sha256 must be a 64-character lowercase hex SHA-256")

    commits = _lookup_field(payload, "research_convergence", "pinned_upstream_commits", errors)
    if not isinstance(commits, dict) or len(commits) < 1:
        errors.append("pinned_upstream_commits must be a non-empty dict of {repo: 40-hex commit}")
    else:
        for repo_key, commit_sha in commits.items():
            if not isinstance(repo_key, str) or not repo_key.strip():
                errors.append("pinned_upstream_commits keys must be non-empty repo names")
            if not isinstance(commit_sha, str) or not _SHA1_40_RE.fullmatch(commit_sha):
                errors.append(f"pinned_upstream_commits[{repo_key!r}] must be a 40-character lowercase hex SHA")

    repro_cmd = _lookup_field(payload, "research_convergence", "reproducible_command", errors)
    if not isinstance(repro_cmd, str) or not repro_cmd.strip():
        errors.append("reproducible_command must be a non-empty string")

    ev_grade = _lookup_field(payload, "research_convergence", "evidence_grade", errors)
    if not isinstance(ev_grade, str) or ev_grade not in VALID_EVIDENCE_GRADES:
        errors.append(f"evidence_grade must be in E0..E4 (got {ev_grade!r})")

    authority = _lookup_field(payload, "research_convergence", "other_repos_authority", errors)
    if authority != "PHILOSOPHY_ONLY":
        errors.append(f"other_repos_authority must equal 'PHILOSOPHY_ONLY' (got {authority!r})")

    # (b) 7-Feed PIT DataFeed routing
    rth_feed = _lookup_field(payload, "datafeed_pit_contract", "rth_extended_feed", errors)
    if rth_feed != "sip":
        errors.append(f"rth_extended_feed must equal 'sip' (got {rth_feed!r})")

    overnight_feed = _lookup_field(payload, "datafeed_pit_contract", "overnight_feed", errors)
    if overnight_feed != "boats":
        errors.append(f"overnight_feed must equal 'boats' (got {overnight_feed!r})")

    cutover_flag = _lookup_field(payload, "datafeed_pit_contract", "dec6_2026_23x5_cutover_enabled", errors)
    if not isinstance(cutover_flag, bool):
        errors.append("dec6_2026_23x5_cutover_enabled must be an explicit bool")

    min_px = _lookup_field(payload, "datafeed_pit_contract", "raw_filter_min_price_usd", errors)
    if not _is_finite_num(min_px) or float(min_px) < 2.0:
        errors.append(f"raw_filter_min_price_usd must be finite and >= 2.0 (got {min_px!r})")

    min_dvol = _lookup_field(payload, "datafeed_pit_contract", "raw_filter_min_dollar_vol_usd", errors)
    if not _is_finite_num(min_dvol) or float(min_dvol) < 2_000_000.0:
        errors.append(f"raw_filter_min_dollar_vol_usd must be finite and >= 2000000.0 (got {min_dvol!r})")

    ret_adj = _lookup_field(payload, "datafeed_pit_contract", "return_adjustment", errors)
    if ret_adj != "split_dividend_all":
        errors.append(f"return_adjustment must equal 'split_dividend_all' (got {ret_adj!r})")

    split_win = _lookup_field(payload, "datafeed_pit_contract", "split_exclusion_window_sessions", errors)
    if (
        not isinstance(split_win, list)
        or len(split_win) != 2
        or not all(_is_int_not_bool(x) for x in split_win)
        or split_win != [-1, 3]
    ):
        errors.append(f"split_exclusion_window_sessions must equal [-1, 3] (got {split_win!r})")

    sec_ts = _lookup_field(payload, "datafeed_pit_contract", "sec_edgar_timestamp_field", errors)
    if sec_ts != "acceptanceDateTime":
        errors.append(f"sec_edgar_timestamp_field must equal 'acceptanceDateTime' (got {sec_ts!r})")

    sec_hl = _lookup_field(payload, "datafeed_pit_contract", "sec_freshness_half_life_sec", errors)
    if not _is_finite_num(sec_hl) or float(sec_hl) != 900.0:
        errors.append(f"sec_freshness_half_life_sec must equal 900 (got {sec_hl!r})")

    dil_forms = _lookup_field(payload, "datafeed_pit_contract", "dilution_block_forms", errors)
    if (
        not isinstance(dil_forms, list)
        or not all(isinstance(f, str) for f in dil_forms)
        or "424B5" not in dil_forms
        or "S-3" not in dil_forms
    ):
        errors.append("dilution_block_forms must be a list of strings containing '424B5' and 'S-3'")

    finra_lag = _lookup_field(payload, "datafeed_pit_contract", "finra_short_vol_lag_sessions", errors)
    if not _is_int_not_bool(finra_lag) or finra_lag < 1:
        errors.append(f"finra_short_vol_lag_sessions must be an int >= 1 (got {finra_lag!r})")

    # (c) Statistical & Queue Gates
    wall_dt_raw = _lookup_field(payload, "statistical_and_queue_gates", "study_wall_date", errors)
    if not isinstance(wall_dt_raw, str) or not _ISO_DATE_RE.fullmatch(wall_dt_raw):
        errors.append(f"study_wall_date must be an ISO YYYY-MM-DD string (got {wall_dt_raw!r})")
    else:
        try:
            wall_dt = date.fromisoformat(wall_dt_raw)
            if wall_dt > STUDY_WALL_DATE:
                errors.append(
                    f"study_wall_date {wall_dt.isoformat()} exceeds pre-wall cutoff {STUDY_WALL_DATE.isoformat()}"
                )
        except ValueError:
            errors.append(f"study_wall_date is not a valid calendar date: {wall_dt_raw!r}")

    spec_sha = _lookup_field(payload, "statistical_and_queue_gates", "spec_sha256", errors)
    if not isinstance(spec_sha, str) or not _SHA256_RE.fullmatch(spec_sha):
        errors.append("spec_sha256 must be a 64-character lowercase hex SHA-256")

    trials_n = _lookup_field(payload, "statistical_and_queue_gates", "cumulative_trials_n", errors)
    hlz_t = _lookup_field(payload, "statistical_and_queue_gates", "hlz_t_hurdle", errors)
    if not _is_finite_num(hlz_t) or float(hlz_t) < 3.0:
        errors.append(f"hlz_t_hurdle must be finite and >= 3.0 (got {hlz_t!r})")
    elif trials_n is not None:
        if not _is_int_not_bool(trials_n) or trials_n < 1:
            errors.append(f"cumulative_trials_n must be an int >= 1 when set (got {trials_n!r})")
        elif float(hlz_t) + 1e-6 < hlz_dynamic_t_hurdle(trials_n):
            errors.append(
                f"hlz_t_hurdle={hlz_t} below dynamic hurdle {hlz_dynamic_t_hurdle(trials_n)} for N={trials_n}"
            )

    cpcv_paths = _lookup_field(payload, "statistical_and_queue_gates", "cpcv_paths", errors)
    if not _is_int_not_bool(cpcv_paths) or cpcv_paths < 5:
        errors.append(f"cpcv_paths must be an int >= 5 (got {cpcv_paths!r})")

    cpcv_purge = _lookup_field(payload, "statistical_and_queue_gates", "cpcv_purge_sessions", errors)
    if not _is_int_not_bool(cpcv_purge) or cpcv_purge < 3:
        errors.append(f"cpcv_purge_sessions must be an int >= 3 (got {cpcv_purge!r})")

    cpcv_embargo = _lookup_field(payload, "statistical_and_queue_gates", "cpcv_embargo_sessions", errors)
    if not _is_int_not_bool(cpcv_embargo) or cpcv_embargo < 2:
        errors.append(f"cpcv_embargo_sessions must be an int >= 2 (got {cpcv_embargo!r})")

    max_pbo = _lookup_field(payload, "statistical_and_queue_gates", "max_pbo", errors)
    if not _is_finite_num(max_pbo) or float(max_pbo) < 0.0 or float(max_pbo) > 0.30:
        errors.append(f"max_pbo must be in [0.0, 0.30] (got {max_pbo!r})")

    min_dsr = _lookup_field(payload, "statistical_and_queue_gates", "min_dsr", errors)
    if not _is_finite_num(min_dsr) or float(min_dsr) < 0.95 or float(min_dsr) > 1.0:
        errors.append(f"min_dsr must be in [0.95, 1.0] (got {min_dsr!r})")

    tie_rule = _lookup_field(payload, "statistical_and_queue_gates", "intrabar_tie_rule", errors)
    if tie_rule != "ADVERSE_STOP_FIRST":
        errors.append(f"intrabar_tie_rule must equal 'ADVERSE_STOP_FIRST' (got {tie_rule!r})")

    cost_mults = _lookup_field(payload, "statistical_and_queue_gates", "cost_stress_multipliers", errors)
    if (
        not isinstance(cost_mults, list)
        or len(cost_mults) != 2
        or not all(_is_finite_num(m) for m in cost_mults)
        or [float(m) for m in cost_mults] != [1.0, 2.0]
    ):
        errors.append(f"cost_stress_multipliers must equal [1.0, 2.0] (got {cost_mults!r})")

    # (d) Paper Execution & ADR-0022
    live_enabled = _lookup_field(payload, "paper_execution_and_adr0022", "live_execution_enabled", errors)
    if live_enabled is not False:
        errors.append(f"live_execution_enabled must be False (got {live_enabled!r})")

    max_stop = _lookup_field(payload, "paper_execution_and_adr0022", "max_initial_stop_pct", errors)
    if not _is_finite_num(max_stop) or float(max_stop) <= 0.0 or float(max_stop) > ADR0022_MAX_STOP_PCT:
        errors.append(f"max_initial_stop_pct must be in (0.0, {ADR0022_MAX_STOP_PCT}] (got {max_stop!r})")

    alpaca_rpm = _lookup_field(payload, "paper_execution_and_adr0022", "alpaca_paper_max_rpm", errors)
    if not _is_int_not_bool(alpaca_rpm) or alpaca_rpm <= 0 or alpaca_rpm > ALPACA_PAPER_MAX_RPM:
        errors.append(f"alpaca_paper_max_rpm must be in [1, {ALPACA_PAPER_MAX_RPM}] (got {alpaca_rpm!r})")

    ibkr_port = _lookup_field(payload, "paper_execution_and_adr0022", "ibkr_paper_port", errors)
    if not _is_int_not_bool(ibkr_port) or ibkr_port != IBKR_PAPER_PORT:
        errors.append(f"ibkr_paper_port must equal {IBKR_PAPER_PORT} (got {ibkr_port!r})")

    ibkr_msg = _lookup_field(payload, "paper_execution_and_adr0022", "ibkr_paper_max_msg_per_min", errors)
    if not _is_int_not_bool(ibkr_msg) or ibkr_msg <= 0 or ibkr_msg > IBKR_PAPER_MAX_MSG_PER_MIN:
        errors.append(f"ibkr_paper_max_msg_per_min must be in [1, {IBKR_PAPER_MAX_MSG_PER_MIN}] (got {ibkr_msg!r})")

    return errors


def scaffold_converged_strategy(
    strategy_id: str,
    archetype: str,
    *,
    input_package_sha256: str,
    pinned_upstream_commits: dict[str, str],
    spec_sha256: str,
    cumulative_trials_n: int = 82,
) -> dict[str, Any]:
    """Scaffold a canonical research-converged strategy template enforcing all ADR-0006 gates."""
    if not _is_int_not_bool(cumulative_trials_n) or cumulative_trials_n < 1:
        raise ValueError(f"cumulative_trials_n must be an int >= 1 (got {cumulative_trials_n!r})")
    if archetype not in ARCHETYPES:
        raise ValueError(f"Unsupported archetype {archetype!r}; expected one of {ARCHETYPES}")

    preset = dict(_ARCHETYPE_DEFAULTS[archetype])
    default_stop = float(str(preset.get("default_initial_stop_pct", 1.80)))

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "strategy_id": strategy_id,
        "archetype": archetype,
        "archetype_profile": preset,
        "research_convergence": {
            "input_package_sha256": input_package_sha256,
            "pinned_upstream_commits": dict(pinned_upstream_commits),
            "reproducible_command": f"uv run pytest -q && python -m sota_compounding --strategy {strategy_id}",
            "evidence_grade": "E3",
            "other_repos_authority": "PHILOSOPHY_ONLY",
        },
        "datafeed_pit_contract": {
            "feeds_enabled": [
                "alpaca_sip_equities",
                "alpaca_boats_overnight",
                "ibkr_outside_rth_4002",
                "sec_edgar_pit_filings",
                "alpaca_news_pit",
                "finra_regsho_short_volume",
                "exchange_calendars_xnys",
            ],
            "rth_extended_feed": "sip",
            "overnight_feed": "boats",
            "dec6_2026_23x5_cutover_enabled": False,
            "raw_filter_min_price_usd": 2.0,
            "raw_filter_min_dollar_vol_usd": 2_000_000.0,
            "return_adjustment": "split_dividend_all",
            "split_exclusion_window_sessions": [-1, 3],
            "sec_edgar_timestamp_field": "acceptanceDateTime",
            "sec_freshness_half_life_sec": 900,
            "dilution_block_forms": ["S-3", "424B2", "424B3", "424B4", "424B5"],
            "finra_short_vol_lag_sessions": 1,
        },
        "statistical_and_queue_gates": {
            "study_wall_date": STUDY_WALL_DATE.isoformat(),
            "spec_sha256": spec_sha256,
            "cumulative_trials_n": cumulative_trials_n,
            "hlz_t_hurdle": hlz_dynamic_t_hurdle(cumulative_trials_n),
            "cpcv_paths": 5,
            "cpcv_purge_sessions": 3,
            "cpcv_embargo_sessions": 2,
            "max_pbo": 0.30,
            "min_dsr": 0.95,
            "intrabar_tie_rule": "ADVERSE_STOP_FIRST",
            "cost_stress_multipliers": [1.0, 2.0],
        },
        "paper_execution_and_adr0022": {
            "live_execution_enabled": False,
            "max_initial_stop_pct": min(ADR0022_MAX_STOP_PCT, default_stop),
            "alpaca_paper_max_rpm": ALPACA_PAPER_MAX_RPM,
            "ibkr_paper_port": IBKR_PAPER_PORT,
            "ibkr_paper_max_msg_per_min": IBKR_PAPER_MAX_MSG_PER_MIN,
            "tv_webhook_endpoint": "http://127.0.0.1:8799/api/tv-webhook",
        },
    }

    violations = validate_converged_strategy_template(payload)
    if violations:
        raise ValueError("Invalid converged strategy template: " + "; ".join(violations))
    return payload
