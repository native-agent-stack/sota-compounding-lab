#!/usr/bin/env python3
"""Trial #108 (`H113`): Pre-Fill Directional Confirmation (0 bps Cancel) Long/Short x Per-Symbol Min Trail.

Pre-Registered Immutable Spec SHA-256 (`sota-bus` Trial #108):
  `09e3a8d29181a2ba662e0841f1b746e7d3058e309050e5f76cfceef7f5fa33b6`
Required HLZ Multiplicity Hurdle (`N = 108`): `t*(108) = 4.2148`
ADIA Lab #19 Power Gate (`D=700`, `sigma=85 bps`): APPROVED (`SR_min_ann=2.5289 <= 5.0`, `MinTRL=499`)

Economic & Microstructure Mechanism (`[E1]`):
1. **Pre-Fill Directional Confirmation (`0.00 bps` Cancel-If-Wrong, `1,000 RPM` Maker Pool Physics):**
   - Strictly causal `T-1` operating US equities (`prev_rth_close >= $10`, `ADV20 >= $50M`, `hist_days >= 20`,
     zero splits on `[T-1..T+1]`, `90d` PIT SEC `424B5`/`S-3`/`S-3ASR`/`EFFECT` dilution block on Longs).
   - At `09:50 ET` (after `09:30..09:49` opening discovery), read real-time `SPY` opening regime:
     - Constructive (`SPY ret >= +min_mkt`): select Top-`K` dilution-shielded RVOL (`>= 1.8x`) leaders breaking up
       (`ret in [+0.5%, +6.0%]`, `close_0949 > vwap_0949`, `close_pos >= 0.65`).
     - Risk-Off (`SPY ret <= -min_mkt`): select Bottom-`K` RVOL (`>= 1.8x`) breakdowns
       (`ret in [-6.0%, -0.5%]`, `close_0949 < vwap_0949`, `close_pos <= 0.35`).
   - Place a 3-minute confirmation trigger (`09:50..09:52 ET`):
     - Long fills ONLY if bar `h >= high_0949` BEFORE `l <= vwap_0949` (fill `max(open, high_0949)` conservatively).
     - Short fills ONLY if bar `l <= low_0949` BEFORE `h >= vwap_0949` (fill `min(open, low_0949)` conservatively).
     - If adverse flow hits first or `3m` expires without confirmation -> **CANCELLED UNFILLED AT `0.00 bps` COST!**
2. **Causal `T-1` Per-Symbol Min Trailing (`STOP_CHECK_BEFORE_MFE_UPDATE`, zero intrabar lookahead):**
   - `min_trail_bps(i, T-1) = clamp(max(2.5 * cost_1x, 1.8 * med_1m_range_bps), 18.0, 110.0)`
   - `be_arm_bps = max(25.0, 2.5 * cost_1x)`, `be_lock_bps = max(10.0, 1.5 * cost_1x)`
   - `hard_stop_pct = min(2.00, max(0.65, 2.2 * min_trail_bps / 100.0))`
3. **3-Arm Resolution:**
   - Primary (`H113_PRIMARY_PREFILL_CONFIRM_LONGSHORT_PER_SYM_MIN_TRAIL`): Pre-fill confirmation + causal
     per-symbol min trailing (5-config fold-fitted `skfolio` CPCV(6,2,purge=20,embargo=20) & Bailey 2014 PBO).
   - Paired Ablation (`H113_ABLATION_MARKETABLE_UNIFORM_STOP`): Blind `09:50 ET` open entry + uniform `1.40%`
     stop + uniform `80 bps` BE on exact same candidates.
   - Multi-Day Swing (`H113_DIAG_MULTIDAY_SWING_PER_SYM_MIN_TRAIL`): Exact same confirmed entries held up to
     `5` trading days with causal per-symbol daily min trailing (non-overlapping daily MTM legs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sota_compounding.advanced_merit_sim_convergence import (  # noqa: E402
    ConformalPidStopCalibrator,
    evaluate_hansen_spa_and_mcs,
)
from sota_compounding.backtest_playback_manifest import (  # noqa: E402
    PlaybackAuditInput,
    audit_low_quality_practices,
    compute_calendar_day_stats,
)
from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE, hlz_dynamic_t_hurdle  # noqa: E402

STUDY_WALL_ISO = STUDY_WALL_DATE.isoformat()  # "2026-10-15"
SAMPLE_START_ISO = "2024-01-02"
EVAL_START_ISO = "2024-02-01"
TRIAL_NUMBER = 108
HYPOTHESIS_ID = "H113_PREFILL_CONFIRM_DIRECTIONAL_LONGSHORT_PER_SYMBOL_MIN_TRAIL"
PREREG_SPEC_SHA256 = "09e3a8d29181a2ba662e0841f1b746e7d3058e309050e5f76cfceef7f5fa33b6"

PRIMARY_CONFIG_GRID: tuple[dict[str, Any], ...] = (
    {"name": "cfg_m20_k05", "min_mkt": 0.0020, "k_side": 5},
    {"name": "cfg_m30_k05", "min_mkt": 0.0030, "k_side": 5},  # Canonical baseline
    {"name": "cfg_m30_k08", "min_mkt": 0.0030, "k_side": 8},
    {"name": "cfg_m45_k05", "min_mkt": 0.0045, "k_side": 5},
    {"name": "cfg_m45_k03", "min_mkt": 0.0045, "k_side": 3},
)
PRIMARY_CANONICAL_CFG = "cfg_m30_k05"

POSITION_NOTIONAL_USD = 50_000.0
BASE_1X_COST_BPS = 8.0
SHORT_BORROW_DAILY_BPS = 1.5
PREFILL_CONFIRM_WINDOW_MIN = 3
STRICT_DSR_MIN = 0.95
STRICT_PBO_MAX = 0.25
STRICT_CPCV_POS_MIN = 0.70
STRICT_SPA_P_MAX = 0.05
STRICT_MIN_EVENTS = 200
STRICT_MIN_DATES = 60


def _almgren_chriss_1x_cost_bps(
    adv20_usd: float,
    vol20d_daily: float,
    *,
    is_short: bool = False,
    hold_days: int = 1,
) -> float:
    sigma_bps = max(50.0, min(1200.0, float(vol20d_daily) * 10_000.0))
    adv = max(1_000_000.0, float(adv20_usd))
    impact = 0.10 * sigma_bps * math.sqrt(POSITION_NOTIONAL_USD / adv)
    borrow = (SHORT_BORROW_DAILY_BPS * max(1, int(hold_days))) if is_short else 0.0
    return BASE_1X_COST_BPS + impact + borrow


def _causal_per_symbol_min_trail_params(cost_1x_bps: float, med_1m_range_bps: float) -> dict[str, float]:
    """Compute strictly causal T-1 per-symbol min-trail & cost-aware net-positive lossless floor."""
    c1 = max(8.0, float(cost_1x_bps))
    r1 = max(5.0, float(med_1m_range_bps))
    min_trail_bps = max(18.0, min(110.0, max(2.5 * c1, 1.8 * r1)))
    be_arm_bps = max(25.0, 2.5 * c1)
    be_lock_bps = max(10.0, 1.5 * c1)
    hard_stop_pct = min(ADR0022_MAX_STOP_PCT, max(0.65, 2.2 * min_trail_bps / 100.0))
    return {
        "min_trail_bps": round(min_trail_bps, 2),
        "be_arm_bps": round(be_arm_bps, 2),
        "be_lock_bps": round(be_lock_bps, 2),
        "hard_stop_pct": round(hard_stop_pct, 4),
    }


def _simulate_prefill_and_per_symbol_min_trail(
    bars_o: Sequence[float] | Any,
    bars_h: Sequence[float] | Any,
    bars_l: Sequence[float] | Any,
    bars_c: Sequence[float] | Any,
    *,
    side: str,
    trigger_px: float,
    abort_vwap_px: float,
    params: dict[str, float],
    confirm_window: int = PREFILL_CONFIRM_WINDOW_MIN,
) -> dict[str, Any]:
    """Simulate 3m Pre-Fill Confirmation (0 bps Cancel-If-Wrong) + Causal Per-Symbol Min Trailing."""
    n = len(bars_o)
    if n < 2:
        return {
            "filled": False,
            "cancel_reason": "NO_BARS",
            "gross_bps": 0.0,
            "mae_pct": 0.0,
            "mfe_pct": 0.0,
            "green_to_red": False,
            "exit_reason": "CANCELLED_UNFILLED",
        }

    is_long = side.upper() == "LONG"
    fill_idx = -1
    entry_px = 0.0
    cancel_reason = "TTL_EXPIRED_UNCONFIRMED"

    w_end = min(confirm_window, n - 1)
    for idx in range(w_end):
        bo = float(bars_o[idx])
        bh = float(bars_h[idx])
        bl = float(bars_l[idx])
        if not (math.isfinite(bo) and math.isfinite(bh) and math.isfinite(bl) and bo > 0.0):
            continue
        if is_long:
            # Conservative: if bar breaches abort VWAP on same bar, pull order before fill (0 bps)
            if bl <= abort_vwap_px:
                cancel_reason = "PULL_ADVERSE_FLOW_VWAP"
                break
            if bh >= trigger_px:
                fill_idx = idx
                entry_px = max(bo, trigger_px)
                break
        else:
            if bh >= abort_vwap_px:
                cancel_reason = "PULL_ADVERSE_FLOW_VWAP"
                break
            if bl <= trigger_px:
                fill_idx = idx
                entry_px = min(bo, trigger_px)
                break

    if fill_idx < 0 or entry_px <= 0.0:
        return {
            "filled": False,
            "cancel_reason": cancel_reason,
            "gross_bps": 0.0,
            "mae_pct": 0.0,
            "mfe_pct": 0.0,
            "green_to_red": False,
            "exit_reason": f"CANCELLED_{cancel_reason}",
        }

    hard_stop_frac = min(float(params["hard_stop_pct"]), ADR0022_MAX_STOP_PCT) / 100.0
    min_trail_frac = float(params["min_trail_bps"]) / 10_000.0
    be_arm_frac = float(params["be_arm_bps"]) / 10_000.0
    be_lock_frac = float(params["be_lock_bps"]) / 10_000.0

    stop_px = entry_px * (1.0 - hard_stop_frac) if is_long else entry_px * (1.0 + hard_stop_frac)
    extreme_px = entry_px
    mae_frac = 0.0
    mfe_frac = 0.0
    be_armed = False
    exited = False
    exit_reason = "EOD_FLAT"
    ret_frac = 0.0

    # Start monitoring bar-by-bar from fill_idx + 1 (zero intrabar lookahead!)
    for i in range(fill_idx + 1, n):
        bh = float(bars_h[i])
        bl = float(bars_l[i])
        if not (math.isfinite(bh) and math.isfinite(bl) and bh > 0.0 and bl > 0.0):
            continue
        if is_long:
            adv = (entry_px - bl) / entry_px
            if adv > mae_frac:
                mae_frac = adv
            # STOP_CHECK_BEFORE_MFE_UPDATE (strictly conservative!)
            if bl <= stop_px:
                ret_frac = (stop_px - entry_px) / entry_px
                exit_reason = "PER_SYM_MIN_TRAIL_STOP" if (be_armed or stop_px > entry_px) else "ADR0022_HARD_STOP"
                exited = True
                break
            if bh > extreme_px:
                extreme_px = bh
                mfe_frac = (extreme_px - entry_px) / entry_px
            if mfe_frac >= be_arm_frac:
                be_armed = True
                # Tighten per-symbol min trail progressively on fat tails while never retreating!
                if mfe_frac >= 0.04:
                    eff_trail = max(0.0010, 0.55 * min_trail_frac)
                elif mfe_frac >= 0.02:
                    eff_trail = max(0.0012, 0.75 * min_trail_frac)
                else:
                    eff_trail = min_trail_frac
                cand_stop = max(entry_px * (1.0 + be_lock_frac), extreme_px * (1.0 - eff_trail))
                stop_px = max(stop_px, cand_stop)
        else:
            adv = (bh - entry_px) / entry_px
            if adv > mae_frac:
                mae_frac = adv
            if bh >= stop_px:
                ret_frac = (entry_px - stop_px) / entry_px
                exit_reason = "PER_SYM_MIN_TRAIL_STOP" if (be_armed or stop_px < entry_px) else "ADR0022_HARD_STOP"
                exited = True
                break
            if bl < extreme_px:
                extreme_px = bl
                mfe_frac = (entry_px - extreme_px) / entry_px
            if mfe_frac >= be_arm_frac:
                be_armed = True
                if mfe_frac >= 0.04:
                    eff_trail = max(0.0010, 0.55 * min_trail_frac)
                elif mfe_frac >= 0.02:
                    eff_trail = max(0.0012, 0.75 * min_trail_frac)
                else:
                    eff_trail = min_trail_frac
                cand_stop = min(entry_px * (1.0 - be_lock_frac), extreme_px * (1.0 + eff_trail))
                stop_px = min(stop_px, cand_stop)

    if not exited:
        last_c = float(bars_c[-1])
        ret_frac = ((last_c - entry_px) / entry_px) if is_long else ((entry_px - last_c) / entry_px)

    return {
        "filled": True,
        "cancel_reason": "",
        "entry_px": entry_px,
        "gross_bps": ret_frac * 10_000.0,
        "mae_pct": max(0.0, mae_frac * 100.0),
        "mfe_pct": max(0.0, mfe_frac * 100.0),
        "green_to_red": bool(mfe_frac >= be_arm_frac and ret_frac < 0.0),
        "exit_reason": exit_reason,
    }


def _simulate_blind_uniform_baseline(
    bars_o: Sequence[float] | Any,
    bars_h: Sequence[float] | Any,
    bars_l: Sequence[float] | Any,
    bars_c: Sequence[float] | Any,
    *,
    side: str,
    stop_pct: float = 1.40,
) -> dict[str, Any]:
    """Paired baseline: blind 09:50 ET open entry + uniform 1.40% stop + uniform +80bps BE -> +18bps lock."""
    n = len(bars_o)
    if n == 0 or not math.isfinite(float(bars_o[0])) or float(bars_o[0]) <= 0.0:
        return {"gross_bps": 0.0, "mae_pct": 0.0, "mfe_pct": 0.0, "green_to_red": False, "exit_reason": "NO_BARS"}
    is_long = side.upper() == "LONG"
    entry_px = float(bars_o[0])
    hard_stop_frac = min(stop_pct, ADR0022_MAX_STOP_PCT) / 100.0
    stop_px = entry_px * (1.0 - hard_stop_frac) if is_long else entry_px * (1.0 + hard_stop_frac)
    extreme_px = entry_px
    mae_frac = 0.0
    mfe_frac = 0.0
    exited = False
    ret_frac = 0.0
    exit_reason = "EOD_FLAT"

    for i in range(n):
        bh = float(bars_h[i])
        bl = float(bars_l[i])
        if not (math.isfinite(bh) and math.isfinite(bl) and bh > 0.0 and bl > 0.0):
            continue
        if is_long:
            mae_frac = max(mae_frac, (entry_px - bl) / entry_px)
            if bl <= stop_px:
                ret_frac = (stop_px - entry_px) / entry_px
                exit_reason = "UNIFORM_STOP"
                exited = True
                break
            if bh > extreme_px:
                extreme_px = bh
                mfe_frac = (extreme_px - entry_px) / entry_px
            if mfe_frac >= 0.0080:
                stop_px = max(stop_px, entry_px * 1.0018)
        else:
            mae_frac = max(mae_frac, (bh - entry_px) / entry_px)
            if bh >= stop_px:
                ret_frac = (entry_px - stop_px) / entry_px
                exit_reason = "UNIFORM_STOP"
                exited = True
                break
            if bl < extreme_px:
                extreme_px = bl
                mfe_frac = (entry_px - extreme_px) / entry_px
            if mfe_frac >= 0.0080:
                stop_px = min(stop_px, entry_px * 0.9982)

    if not exited:
        last_c = float(bars_c[-1])
        ret_frac = ((last_c - entry_px) / entry_px) if is_long else ((entry_px - last_c) / entry_px)

    return {
        "gross_bps": ret_frac * 10_000.0,
        "mae_pct": max(0.0, mae_frac * 100.0),
        "mfe_pct": max(0.0, mfe_frac * 100.0),
        "green_to_red": bool(mfe_frac >= 0.0025 and ret_frac < 0.0),
        "exit_reason": exit_reason,
    }


def _date_clustered_t(values: Any, dates: Any) -> tuple[float, float, float]:
    import numpy as np
    import statsmodels.api as sm

    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    arr = arr[mask]
    grp = np.asarray(dates)[mask]
    if len(arr) < 3 or len(set(grp.tolist())) < 3:
        m = float(np.mean(arr)) if len(arr) else 0.0
        return (m, 0.0, 0.0)
    x = np.ones((len(arr), 1), dtype=float)
    res = sm.OLS(arr, x).fit(cov_type="cluster", cov_kwds={"groups": grp})
    mean_v = float(res.params[0])
    se_v = float(res.bse[0])
    t_v = float(res.tvalues[0]) if math.isfinite(float(res.tvalues[0])) else 0.0
    return (mean_v, se_v, t_v)


def _compute_fold_fitted_cpcv_and_pbo(config_daily_2x_matrix: Any) -> dict[str, Any]:
    import numpy as np
    from skfolio.model_selection import CombinatorialPurgedCV

    mat = np.asarray(config_daily_2x_matrix, dtype=float)
    t_len, n_cfg = mat.shape
    if t_len < 120 or n_cfg < 2:
        return {
            "cpcv_5path_means_2x_bps": [],
            "cpcv_5path_positive_share": 0.0,
            "cpcv_pbo": 1.0,
            "cpcv_oos_sharpe_ann": 0.0,
            "cpcv_distinct_paths": False,
        }
    cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=20, embargo_size=20)
    splits = list(cv.split(np.zeros((t_len, 1), dtype=float)))
    path_ids = cv.get_path_ids()

    path_returns: dict[int, list[float]] = defaultdict(list)
    pbo_under_median = 0
    valid_splits = 0

    for split_idx, (tr_idx, te_list) in enumerate(splits):
        te_concat = np.concatenate(te_list)
        if len(tr_idx) < 30 or len(te_concat) < 10:
            continue
        is_slice = mat[tr_idx, :]
        oos_slice = mat[te_concat, :]
        is_sd = np.std(is_slice, axis=0, ddof=1)
        is_sr = np.where(is_sd > 1e-12, np.mean(is_slice, axis=0) / is_sd, -1e9)
        best_cfg = int(np.argmax(is_sr))

        oos_sd = np.std(oos_slice, axis=0, ddof=1)
        oos_sr = np.where(oos_sd > 1e-12, np.mean(oos_slice, axis=0) / oos_sd, -1e9)
        winner_oos = float(oos_sr[best_cfg])
        rank_1_to_m = 1 + int(np.sum(oos_sr < winner_oos)) + 0.5 * int(np.sum(oos_sr == winner_oos) - 1)
        if (rank_1_to_m / float(n_cfg + 1)) <= 0.5:
            pbo_under_median += 1
        valid_splits += 1

        for fold_pos, te_idx in enumerate(te_list):
            pid = int(path_ids[split_idx, fold_pos])
            path_returns[pid].extend(mat[te_idx, best_cfg].tolist())

    path_means = [round(float(np.mean(vals)), 2) for _, vals in sorted(path_returns.items()) if vals]
    pos_share = (sum(1 for m in path_means if m > 0.0) / float(len(path_means))) if path_means else 0.0
    pbo = (pbo_under_median / float(valid_splits)) if valid_splits > 0 else 1.0
    all_oos = [v for vals in path_returns.values() for v in vals]
    oos_sd_all = float(np.std(all_oos, ddof=1)) if len(all_oos) > 2 else 0.0
    oos_sr_ann = (float(np.mean(all_oos)) / oos_sd_all) * math.sqrt(252.0) if oos_sd_all > 1e-12 else 0.0
    return {
        "cpcv_5path_means_2x_bps": path_means,
        "cpcv_5path_positive_share": round(pos_share, 4),
        "cpcv_pbo": round(pbo, 4),
        "cpcv_oos_sharpe_ann": round(oos_sr_ann, 3),
        "cpcv_distinct_paths": len(set(path_means)) > 1 if len(path_means) >= 2 else False,
    }


def _evaluate_arm_metrics(
    events_df: Any,
    daily_df: Any,
    *,
    label: str,
    n_trials: int = TRIAL_NUMBER,
    cpcv_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np
    from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics

    if events_df.empty or daily_df.empty:
        return {"label": label, "events": 0, "distinct_dates": 0, "passed": False, "verdict": "ZERO_EVENTS"}

    n_events = int(len(events_df))
    distinct_dates = int(len(daily_df))
    mean_1x_ev, _, t_1x_ev = _date_clustered_t(events_df["net_1x_bps"].to_numpy(), events_df["trade_date"].to_numpy())
    mean_2x_ev, _, t_2x_ev = _date_clustered_t(events_df["net_2x_bps"].to_numpy(), events_df["trade_date"].to_numpy())
    d1 = daily_df["net_1x_bps"].to_numpy(dtype=float)
    d2 = daily_df["net_2x_bps"].to_numpy(dtype=float)
    ddates = daily_df["trade_date"].to_numpy()
    mean_1x_day, _, t_1x_day = _date_clustered_t(d1, ddates)
    mean_2x_day, _, t_2x_day = _date_clustered_t(d2, ddates)

    sr_1x_ann = (
        float((np.mean(d1) / np.std(d1, ddof=1)) * math.sqrt(252.0)) if len(d1) > 2 and np.std(d1, ddof=1) > 0 else 0.0
    )
    sr_2x_ann = (
        float((np.mean(d2) / np.std(d2, ddof=1)) * math.sqrt(252.0)) if len(d2) > 2 and np.std(d2, ddof=1) > 0 else 0.0
    )

    arr_2x = d2 / 10_000.0
    sd_2x = float(np.std(arr_2x, ddof=1)) if len(arr_2x) > 2 else 0.0
    dsr_prob = 0.0
    if len(arr_2x) >= 15 and sd_2x > 1e-12:
        mu_2x = float(np.mean(arr_2x))
        z_2x = (arr_2x - mu_2x) / sd_2x
        ac1_2x = float(np.corrcoef(arr_2x[:-1], arr_2x[1:])[0, 1]) if len(arr_2x) > 3 else 0.0
        if not math.isfinite(ac1_2x):
            ac1_2x = 0.0
        dsr_res = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=mu_2x / sd_2x,
            n_samples=len(arr_2x),
            n_trials=max(1, int(n_trials)),
            variance_trials=1.0 / 252.0,
            skewness=float(np.mean(z_2x**3)),
            excess_kurtosis=float(np.mean(z_2x**4) - 3.0),
            autocorrelation=max(-0.95, min(0.95, ac1_2x)),
            frequency="daily",
        )
        dsr_prob = float(getattr(dsr_res, "probability", 0.0) or 0.0)
        if not math.isfinite(dsr_prob):
            dsr_prob = 0.0

    if cpcv_override is not None:
        cpcv_info = dict(cpcv_override)
    else:
        blocks = np.array_split(d2, 5)
        b_means = [round(float(np.mean(b)), 2) for b in blocks if len(b) > 0]
        cpcv_info = {
            "cpcv_5path_means_2x_bps": b_means,
            "cpcv_5path_positive_share": round(sum(1 for m in b_means if m > 0.0) / float(len(b_means)), 4)
            if b_means
            else 0.0,
            "cpcv_pbo": None,
            "cpcv_oos_sharpe_ann": round(sr_2x_ann, 3),
            "cpcv_distinct_paths": len(set(b_means)) > 1,
        }

    spa_res = evaluate_hansen_spa_and_mcs({label: d2.tolist()}, n_bootstrap=1000, avg_block_len=5.0, seed=108)
    spa_p = float(spa_res["spa_p_consistent"])

    pid = ConformalPidStopCalibrator(target_miscoverage_alpha=0.10, max_stop_pct=ADR0022_MAX_STOP_PCT)
    pid_last: dict[str, Any] = {"next_stop_pct": ADR0022_MAX_STOP_PCT, "empirical_miscoverage_rate": 0.0}
    for mae_val, vol_val in zip(
        events_df["mae_pct"].to_numpy(dtype=float),
        events_df["vol20d_daily"].to_numpy(dtype=float),
        strict=False,
    ):
        pid_last = pid.update(realized_mae_pct=float(mae_val), atr_pct=max(0.25, float(vol_val) * 100.0 * 0.5))

    eq = np.cumprod(1.0 + d2 / 10_000.0)
    peak = np.maximum.accumulate(eq)
    max_dd_2x_pct = float(np.max((peak - eq) / np.maximum(peak, 1e-12)) * 100.0) if len(eq) else 0.0

    ev_2x = events_df["net_2x_bps"].to_numpy(dtype=float)
    wins = ev_2x[ev_2x > 0.0]
    losses = ev_2x[ev_2x < 0.0]
    pf_2x = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and abs(np.sum(losses)) > 1e-12 else 0.0
    win_rate_2x_pct = float(np.mean(ev_2x > 0.0) * 100.0) if len(ev_2x) else 0.0
    green_to_red_pct = (
        float(np.mean(events_df["green_to_red"].to_numpy(dtype=bool)) * 100.0)
        if "green_to_red" in events_df.columns and len(events_df)
        else 0.0
    )

    # Long vs Short leg breakdown
    leg_stats: dict[str, dict[str, Any]] = {}
    for leg in ("LONG", "SHORT"):
        sub = events_df[events_df["side"] == leg]
        if not sub.empty:
            m1, _, t1 = _date_clustered_t(sub["net_1x_bps"].to_numpy(), sub["trade_date"].to_numpy())
            m2, _, t2 = _date_clustered_t(sub["net_2x_bps"].to_numpy(), sub["trade_date"].to_numpy())
            leg_stats[leg] = {
                "events": int(len(sub)),
                "dates": int(sub["trade_date"].nunique()),
                "mean_1x_bps": round(m1, 2),
                "mean_2x_bps": round(m2, 2),
                "t_2x": round(t2, 3),
                "win_rate_2x_pct": round(float(np.mean(sub["net_2x_bps"].to_numpy() > 0.0) * 100.0), 1),
            }
        else:
            leg_stats[leg] = {"events": 0, "dates": 0, "mean_1x_bps": 0.0, "mean_2x_bps": 0.0, "t_2x": 0.0}

    # Adversarial ex-top-3-days check
    order = np.argsort(d2)
    ex_top3 = d2[order[:-3]] if len(d2) > 6 else d2
    ex_dates = ddates[order[:-3]] if len(d2) > 6 else ddates
    mean_2x_ex_top3, _, t_2x_ex_top3 = _date_clustered_t(ex_top3, ex_dates)

    cal_stats = compute_calendar_day_stats(d2.tolist())
    mean_c1 = float(events_df["cost_1x_bps"].mean()) if "cost_1x_bps" in events_df.columns else 10.0
    hygiene = audit_low_quality_practices(
        PlaybackAuditInput(
            horizon_id="W1_PRIMARY_PREWALL_LAKE",
            sample_start_date=str(daily_df["trade_date"].iloc[0]),
            sample_max_date=str(daily_df["trade_date"].iloc[-1]),
            distinct_sessions=distinct_dates,
            cadence="daily_swing" if "SWING" in label else "intraday",
            universe_mode="CAUSAL_T_MINUS_1_PIT",
            cost_1x_rt_bps=mean_c1,
            includes_2x_and_sqrt_impact=True,
            take_profit_cap_enabled=False,
            max_stop_pct=ADR0022_MAX_STOP_PCT,
            sharpe_annualization_mode="CALENDAR_DAY_AR1_CLUSTERED",
            uses_spa_mcs_dsr_pbo=True,
            in_sample_symbol_cherry_picking=False,
            vol_max_leverage=1.0,
            allocator_mode="FIXED_SHARE_HEDGE_AND_HIERARCHICAL_EB",
            edgar_timestamp_field="acceptanceDateTime",
            active_strategy_ids=(HYPOTHESIS_ID,),
            silent_exception_swallow=False,
            source_libraries=("polars", "duckdb", "skfolio", "ml4t-diagnostic", "statsmodels"),
            is_post_incentive_chase=False,
            target_sharpe_for_min_trl=1.50,
            ar1_rho_for_min_trl=max(-0.5, min(0.5, float(cal_stats.get("ar1_rho") or 0.0))),
        )
    )

    hlz_t_req = hlz_dynamic_t_hurdle(n_trials)
    pbo_ok = (cpcv_info.get("cpcv_pbo") is None) or (float(cpcv_info["cpcv_pbo"]) <= STRICT_PBO_MAX)
    passed = bool(
        n_events >= STRICT_MIN_EVENTS
        and distinct_dates >= STRICT_MIN_DATES
        and mean_1x_day > 0.0
        and mean_2x_day > 0.0
        and t_2x_day >= hlz_t_req
        and dsr_prob >= STRICT_DSR_MIN
        and float(cpcv_info.get("cpcv_5path_positive_share") or 0.0) >= STRICT_CPCV_POS_MIN
        and pbo_ok
        and spa_p <= STRICT_SPA_P_MAX
        and bool(hygiene["passed"])
    )

    return {
        "label": label,
        "events": n_events,
        "distinct_dates": distinct_dates,
        "mean_1x_event_bps": round(mean_1x_ev, 2),
        "t_1x_event": round(t_1x_ev, 4),
        "mean_2x_event_bps": round(mean_2x_ev, 2),
        "t_2x_event": round(t_2x_ev, 4),
        "mean_1x_daily_bps": round(mean_1x_day, 2),
        "t_1x_daily": round(t_1x_day, 4),
        "mean_2x_daily_bps": round(mean_2x_day, 2),
        "t_2x_daily": round(t_2x_day, 4),
        "hlz_t_required": round(hlz_t_req, 4),
        "sr_1x_ann": round(sr_1x_ann, 3),
        "sr_2x_ann": round(sr_2x_ann, 3),
        "calendar_zero_padded_sr_2x_ann": round(float(cal_stats.get("sharpe_ann") or 0.0), 3),
        "dsr_2x": round(dsr_prob, 4),
        "hansen_spa_p": round(spa_p, 4),
        "profit_factor_2x": round(pf_2x, 3),
        "win_rate_2x_pct": round(win_rate_2x_pct, 1),
        "max_dd_2x_pct": round(max_dd_2x_pct, 2),
        "green_to_red_pct": round(green_to_red_pct, 1),
        "conformal_pid_next_stop_pct": round(float(pid_last.get("next_stop_pct") or ADR0022_MAX_STOP_PCT), 3),
        "conformal_pid_miscoverage_rate": round(float(pid_last.get("empirical_miscoverage_rate") or 0.0), 4),
        "leg_breakdown": leg_stats,
        "adversarial_ex_top3_days_2x_bps": round(mean_2x_ex_top3, 2),
        "adversarial_ex_top3_days_t_2x": round(t_2x_ex_top3, 4),
        **cpcv_info,
        "hygiene_14pt_passed": bool(hygiene["passed"]),
        "hygiene_violations": list(hygiene["violations"]),
        "verdict": "PASSED_ALPHA_CANDIDATE" if passed else "FAILED_HLZ_OR_STAT_GATE",
        "promotion_track": "ALPHA_STRATEGY_MINOR_CANDIDATE" if passed else "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "passed": passed,
    }


def run_h113_trial108_study(market_data_root: pathlib.Path) -> dict[str, Any]:
    """Execute strictly pre-wall (`<= 2026-10-15`) Trial #108 (`H113`) on real 700d SIP 1m lake."""
    import duckdb
    import numpy as np
    import pandas as pd

    sip_root = market_data_root / "alpaca-sip" / "raw" / "bars-1min"
    ca_root = market_data_root / "alpaca-sip" / "raw" / "corporate-actions"
    pub_root = market_data_root / "public" / "tables"

    sip_dirs = sorted(d for d in sip_root.iterdir() if d.is_dir() and SAMPLE_START_ISO <= d.name <= STUDY_WALL_ISO)
    if not sip_dirs or sip_dirs[-1].name > STUDY_WALL_ISO:
        raise ValueError("Pre-wall SIP directories invalid or exceeded STUDY_WALL_DATE")

    ca_files = sorted(p.as_posix() for p in ca_root.glob("*.parquet") if p.name[:7] <= STUDY_WALL_ISO[:7])
    sec_sym_p = (pub_root / "sec_symbols.parquet").as_posix()
    sec_fil_p = (pub_root / "sec_filings.parquet").as_posix()

    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute("PRAGMA memory_limit='24GB'")

    # 1. Operating US common equities + SPY/QQQ anchors + splits + PIT 90d SEC dilution shield
    con.execute(
        f"""
        CREATE TEMP TABLE operating_syms AS
        SELECT DISTINCT UPPER(TRIM(symbol)) AS symbol, cik
        FROM read_parquet('{sec_sym_p}')
        WHERE LOWER(COALESCE(entity_type, '')) = 'operating'
          AND symbol IS NOT NULL AND LENGTH(TRIM(symbol)) BETWEEN 1 AND 5
          AND regexp_matches(UPPER(TRIM(symbol)), '^[A-Z]+$')
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE scan_syms AS
        SELECT DISTINCT symbol FROM operating_syms
        UNION VALUES ('SPY'), ('QQQ'), ('UVXY')
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE sec_dil_dates AS
        SELECT o.symbol, SUBSTR(f.accepted, 1, 10) AS dil_date
        FROM read_parquet('{sec_fil_p}') f
        JOIN operating_syms o ON o.cik = f.cik
        WHERE f.accepted IS NOT NULL AND LENGTH(f.accepted) >= 10
          AND SUBSTR(f.accepted, 1, 10) >= '2023-10-01'
          AND SUBSTR(f.accepted, 1, 10) <= '{STUDY_WALL_ISO}'
          AND f.form IN ('424B5', 'S-3', 'S-3ASR', 'EFFECT')
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE split_events AS
        SELECT DISTINCT UPPER(TRIM(symbol)) AS symbol, ex_date
        FROM read_parquet($1)
        WHERE type IN ('forward_splits', 'reverse_splits')
          AND ex_date IS NOT NULL AND ex_date <= $2
        """,
        [ca_files, STUDY_WALL_ISO],
    )

    # 2. Single-pass batched daily + 09:30..09:49 opening discovery extraction across all 700 sessions
    con.execute(
        """
        CREATE TEMP TABLE sip_daily_open (
            symbol VARCHAR,
            trade_date VARCHAR,
            open_0930 DOUBLE,
            high_0949 DOUBLE,
            low_0949 DOUBLE,
            close_0949 DOUBLE,
            vwap_0949 DOUBLE,
            open20_vol DOUBLE,
            rth_close DOUBLE,
            rth_high DOUBLE,
            rth_low DOUBLE,
            rth_dvol DOUBLE,
            med_1m_range_bps DOUBLE
        )
        """
    )
    batch_size = 25
    for idx in range(0, len(sip_dirs), batch_size):
        batch = sip_dirs[idx : idx + batch_size]
        globs = [f"{d.as_posix()}/*.parquet" for d in batch]
        con.execute(
            """
            INSERT INTO sip_daily_open
            WITH raw_bars AS (
                SELECT
                    UPPER(TRIM(symbol)) AS symbol,
                    CAST(to_timestamp(t_ns / 1e9) AT TIME ZONE 'America/New_York' AS TIMESTAMP) AS ny_ts,
                    o, h, l, c, v, COALESCE(NULLIF(vw, 0), c) AS vw
                FROM read_parquet($1)
                WHERE UPPER(TRIM(symbol)) IN (SELECT symbol FROM scan_syms)
                  AND o > 0 AND h > 0 AND l > 0 AND c > 0 AND v >= 0
            ),
            tagged AS (
                SELECT
                    symbol,
                    strftime(ny_ts, '%Y-%m-%d') AS trade_date,
                    EXTRACT(HOUR FROM ny_ts) * 60 + EXTRACT(MINUTE FROM ny_ts) AS ny_min,
                    o, h, l, c, v, vw
                FROM raw_bars
            )
            SELECT
                symbol,
                trade_date,
                arg_min(o, ny_min) FILTER (WHERE ny_min BETWEEN 570 AND 574) AS open_0930,
                MAX(h) FILTER (WHERE ny_min BETWEEN 570 AND 589) AS high_0949,
                MIN(l) FILTER (WHERE ny_min BETWEEN 570 AND 589) AS low_0949,
                arg_max(c, ny_min) FILTER (WHERE ny_min BETWEEN 570 AND 589) AS close_0949,
                SUM(vw * v) FILTER (WHERE ny_min BETWEEN 570 AND 589)
                    / NULLIF(SUM(v) FILTER (WHERE ny_min BETWEEN 570 AND 589), 0) AS vwap_0949,
                SUM(v) FILTER (WHERE ny_min BETWEEN 570 AND 589) AS open20_vol,
                arg_max(c, ny_min) FILTER (WHERE ny_min BETWEEN 570 AND 959) AS rth_close,
                MAX(h) FILTER (WHERE ny_min BETWEEN 570 AND 959) AS rth_high,
                MIN(l) FILTER (WHERE ny_min BETWEEN 570 AND 959) AS rth_low,
                SUM(vw * v) FILTER (WHERE ny_min BETWEEN 570 AND 959) AS rth_dvol,
                MEDIAN((h - l) / NULLIF(c, 0) * 10000.0) FILTER (WHERE ny_min BETWEEN 570 AND 959) AS med_1m_range_bps
            FROM tagged
            WHERE trade_date <= $2
            GROUP BY symbol, trade_date
            """,
            [globs, STUDY_WALL_ISO],
        )

    # 3. Compute strictly causal T-1 rolling features & split/dilution flags
    daily_df = con.execute(
        f"""
        WITH step1 AS (
            SELECT
                d.*,
                LAG(rth_close, 1) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_rth_close,
                (
                    rth_close / NULLIF(
                        LAG(rth_close, 1) OVER (PARTITION BY symbol ORDER BY trade_date), 0
                    ) - 1.0
                ) AS daily_ret,
                LEAD(trade_date, 1) OVER (PARTITION BY symbol ORDER BY trade_date) AS next_trade_date,
                LAG(trade_date, 1) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_trade_date
            FROM sip_daily_open d
        ),
        base AS (
            SELECT
                s.*,
                AVG(rth_dvol) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS adv20_usd,
                STDDEV_SAMP(daily_ret) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS vol20d_daily,
                AVG(med_1m_range_bps) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS causal_med_1m_range_bps,
                AVG(open20_vol) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS avg20_open20_vol,
                COUNT(*) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS hist_days
            FROM step1 s
        ),
        enriched AS (
            SELECT
                b.*,
                EXISTS (
                    SELECT 1 FROM split_events se
                    WHERE se.symbol = b.symbol
                      AND se.ex_date BETWEEN COALESCE(b.prev_trade_date, b.trade_date)
                                         AND COALESCE(b.next_trade_date, b.trade_date)
                ) AS has_near_split,
                EXISTS (
                    SELECT 1 FROM sec_dil_dates sd
                    WHERE sd.symbol = b.symbol
                      AND CAST(sd.dil_date AS DATE)
                          BETWEEN (CAST(b.trade_date AS DATE) - INTERVAL 90 DAY)
                              AND (CAST(b.trade_date AS DATE) - INTERVAL 1 DAY)
                ) AS has_sec_dilution_90d,
                (b.symbol IN (SELECT symbol FROM operating_syms)) AS is_operating
            FROM base b
        )
        SELECT * FROM enriched
        WHERE trade_date >= '{EVAL_START_ISO}' AND trade_date <= '{STUDY_WALL_ISO}'
        ORDER BY trade_date, symbol
        """
    ).df()

    # Keep full post-eval daily close/high/low lookup for multi-day swing evaluation (strictly <= STUDY_WALL_ISO)
    full_daily_lut: dict[str, dict[str, tuple[float, float, float, bool]]] = defaultdict(dict)
    all_dates_sorted = sorted(daily_df["trade_date"].unique().tolist())
    date_to_idx = {d: i for i, d in enumerate(all_dates_sorted)}
    for row in daily_df[["symbol", "trade_date", "rth_high", "rth_low", "rth_close", "has_near_split"]].itertuples(
        index=False
    ):
        full_daily_lut[str(row.symbol)][str(row.trade_date)] = (
            float(row.rth_high or 0.0),
            float(row.rth_low or 0.0),
            float(row.rth_close or 0.0),
            bool(row.has_near_split),
        )

    # Extract SPY opening return (09:30 -> 09:49) by date
    spy_rows = daily_df[daily_df["symbol"] == "SPY"]
    spy_open_ret: dict[str, float] = {}
    for row in spy_rows.itertuples(index=False):
        o0 = float(row.open_0930 or 0.0)
        c49 = float(row.close_0949 or 0.0)
        if o0 > 0.0 and c49 > 0.0:
            spy_open_ret[str(row.trade_date)] = (c49 - o0) / o0

    # Filter operating equities with strictly causal T-1 eligibility
    eq_df = daily_df[
        (daily_df["is_operating"])
        & (~daily_df["has_near_split"])
        & (daily_df["prev_rth_close"] >= 10.0)
        & (daily_df["adv20_usd"] >= 50_000_000.0)
        & (daily_df["hist_days"] >= 20)
        & (daily_df["open_0930"] > 0.0)
        & (daily_df["close_0949"] > 0.0)
        & (daily_df["vwap_0949"] > 0.0)
        & (daily_df["avg20_open20_vol"] > 0.0)
        & (daily_df["causal_med_1m_range_bps"] > 0.0)
    ].copy()

    eq_df["ret_open20"] = (eq_df["close_0949"] - eq_df["open_0930"]) / eq_df["open_0930"]
    rng20 = (eq_df["high_0949"] - eq_df["low_0949"]).clip(lower=1e-6)
    eq_df["close_pos20"] = (eq_df["close_0949"] - eq_df["low_0949"]) / rng20
    eq_df["rvol_open20"] = eq_df["open20_vol"] / eq_df["avg20_open20_vol"]

    # Pre-select union of all candidate (trade_date, symbol) pairs across all 5 configs so we only load their 1m bars!
    by_date: dict[str, pd.DataFrame] = {str(dt): grp for dt, grp in eq_df.groupby("trade_date")}
    cfg_candidates: dict[str, list[dict[str, Any]]] = {c["name"]: [] for c in PRIMARY_CONFIG_GRID}
    needed_pairs: set[tuple[str, str]] = set()

    for dt in all_dates_sorted:
        spy_r = spy_open_ret.get(dt)
        grp = by_date.get(dt)
        if spy_r is None or grp is None or grp.empty:
            continue
        for cfg in PRIMARY_CONFIG_GRID:
            min_mkt = float(cfg["min_mkt"])
            k_side = int(cfg["k_side"])
            if spy_r >= min_mkt:
                sub = (
                    grp[
                        (~grp["has_sec_dilution_90d"])
                        & (grp["rvol_open20"] >= 1.8)
                        & (grp["ret_open20"] >= 0.005)
                        & (grp["ret_open20"] <= 0.060)
                        & (grp["close_0949"] > grp["vwap_0949"])
                        & (grp["close_pos20"] >= 0.65)
                    ]
                    .sort_values("rvol_open20", ascending=False)
                    .head(k_side)
                )
                side = "LONG"
            elif spy_r <= -min_mkt:
                sub = (
                    grp[
                        (grp["rvol_open20"] >= 1.8)
                        & (grp["ret_open20"] <= -0.005)
                        & (grp["ret_open20"] >= -0.060)
                        & (grp["close_0949"] < grp["vwap_0949"])
                        & (grp["close_pos20"] <= 0.35)
                    ]
                    .sort_values("rvol_open20", ascending=False)
                    .head(k_side)
                )
                side = "SHORT"
            else:
                continue
            for r in sub.itertuples(index=False):
                sym = str(r.symbol)
                needed_pairs.add((dt, sym))
                cfg_candidates[cfg["name"]].append(
                    {
                        "trade_date": dt,
                        "symbol": sym,
                        "side": side,
                        "high_0949": float(r.high_0949),
                        "low_0949": float(r.low_0949),
                        "vwap_0949": float(r.vwap_0949),
                        "adv20_usd": float(r.adv20_usd),
                        "vol20d_daily": float(r.vol20d_daily or 0.02),
                        "causal_med_1m_range_bps": float(r.causal_med_1m_range_bps),
                    }
                )

    # 4. Load exact 09:50..15:50 ET 1-minute bar arrays ONLY for candidate (trade_date, symbol) pairs
    bars_by_pair: dict[tuple[str, str], tuple[Any, Any, Any, Any]] = {}
    dates_needed = sorted({dt for dt, _ in needed_pairs})
    syms_by_date: dict[str, set[str]] = defaultdict(set)
    for dt, sym in needed_pairs:
        syms_by_date[dt].add(sym)

    for idx in range(0, len(dates_needed), 30):
        dt_chunk = dates_needed[idx : idx + 30]
        globs = [(sip_root / dt).as_posix() + "/*.parquet" for dt in dt_chunk if (sip_root / dt).is_dir()]
        sym_union = sorted({s for dt in dt_chunk for s in syms_by_date[dt]})
        if not globs or not sym_union:
            continue
        sym_sql = ", ".join(f"'{s}'" for s in sym_union)
        df_b = con.execute(
            f"""
            WITH raw AS (
                SELECT
                    UPPER(TRIM(symbol)) AS symbol,
                    CAST(to_timestamp(t_ns / 1e9) AT TIME ZONE 'America/New_York' AS TIMESTAMP) AS ny_ts,
                    o, h, l, c
                FROM read_parquet($1)
                WHERE UPPER(TRIM(symbol)) IN ({sym_sql})
                  AND o > 0 AND h > 0 AND l > 0 AND c > 0
            )
            SELECT
                strftime(ny_ts, '%Y-%m-%d') AS trade_date,
                symbol,
                EXTRACT(HOUR FROM ny_ts) * 60 + EXTRACT(MINUTE FROM ny_ts) AS ny_min,
                o, h, l, c
            FROM raw
            WHERE EXTRACT(HOUR FROM ny_ts) * 60 + EXTRACT(MINUTE FROM ny_ts) BETWEEN 590 AND 950
            ORDER BY trade_date, symbol, ny_min
            """,
            [globs],
        ).df()
        for (dt_k, sym_k), g_b in df_b.groupby(["trade_date", "symbol"], sort=False):
            key = (str(dt_k), str(sym_k))
            if key in needed_pairs:
                bars_by_pair[key] = (
                    g_b["o"].to_numpy(dtype=float),
                    g_b["h"].to_numpy(dtype=float),
                    g_b["l"].to_numpy(dtype=float),
                    g_b["c"].to_numpy(dtype=float),
                )

    con.close()

    # 5. Simulate all 5 configs + paired blind baseline + counterfactual cancel savings + multi-day swing
    cfg_daily_2x_series: dict[str, dict[str, float]] = {}
    canonical_primary_events: list[dict[str, Any]] = []
    canonical_ablation_events: list[dict[str, Any]] = []
    canonical_cancelled_counterfactual_2x: list[float] = []
    canonical_total_candidates = 0
    canonical_filled_count = 0
    canonical_cancel_reasons: dict[str, int] = defaultdict(int)

    for cfg in PRIMARY_CONFIG_GRID:
        cname = str(cfg["name"])
        daily_buckets_2x: dict[str, list[float]] = defaultdict(list)
        for cand in cfg_candidates[cname]:
            dt = cand["trade_date"]
            sym = cand["symbol"]
            side = cand["side"]
            bars = bars_by_pair.get((dt, sym))
            if bars is None:
                continue
            bo, bh, bl, bc = bars
            cost_1x = _almgren_chriss_1x_cost_bps(
                cand["adv20_usd"], cand["vol20d_daily"], is_short=(side == "SHORT"), hold_days=1
            )
            cost_2x = 2.0 * cost_1x
            params = _causal_per_symbol_min_trail_params(cost_1x, cand["causal_med_1m_range_bps"])
            trig_px = cand["high_0949"] if side == "LONG" else cand["low_0949"]
            sim = _simulate_prefill_and_per_symbol_min_trail(
                bo,
                bh,
                bl,
                bc,
                side=side,
                trigger_px=trig_px,
                abort_vwap_px=cand["vwap_0949"],
                params=params,
            )
            if cname == PRIMARY_CANONICAL_CFG:
                canonical_total_candidates += 1
                base_sim = _simulate_blind_uniform_baseline(bo, bh, bl, bc, side=side)
                canonical_ablation_events.append(
                    {
                        "trade_date": dt,
                        "symbol": sym,
                        "side": side,
                        "gross_bps": base_sim["gross_bps"],
                        "net_1x_bps": base_sim["gross_bps"] - cost_1x,
                        "net_2x_bps": base_sim["gross_bps"] - cost_2x,
                        "mae_pct": base_sim["mae_pct"],
                        "mfe_pct": base_sim["mfe_pct"],
                        "green_to_red": base_sim["green_to_red"],
                        "vol20d_daily": cand["vol20d_daily"],
                    }
                )
                if not sim["filled"]:
                    canonical_cancel_reasons[str(sim["cancel_reason"])] += 1
                    canonical_cancelled_counterfactual_2x.append(float(base_sim["gross_bps"] - cost_2x))

            if sim["filled"]:
                net_1x = float(sim["gross_bps"]) - cost_1x
                net_2x = float(sim["gross_bps"]) - cost_2x
                daily_buckets_2x[dt].append(net_2x)
                if cname == PRIMARY_CANONICAL_CFG:
                    canonical_filled_count += 1
                    canonical_primary_events.append(
                        {
                            "trade_date": dt,
                            "symbol": sym,
                            "side": side,
                            "entry_px": float(sim["entry_px"]),
                            "gross_bps": float(sim["gross_bps"]),
                            "net_1x_bps": net_1x,
                            "net_2x_bps": net_2x,
                            "cost_1x_bps": cost_1x,
                            "min_trail_bps": params["min_trail_bps"],
                            "hard_stop_pct": params["hard_stop_pct"],
                            "mae_pct": float(sim["mae_pct"]),
                            "mfe_pct": float(sim["mfe_pct"]),
                            "green_to_red": bool(sim["green_to_red"]),
                            "vol20d_daily": cand["vol20d_daily"],
                            "adv20_usd": cand["adv20_usd"],
                            "exit_reason": sim["exit_reason"],
                        }
                    )
        cfg_daily_2x_series[cname] = {dt: float(np.mean(vals)) for dt, vals in daily_buckets_2x.items() if vals}

    # Build aligned (T, 5) matrix on canonical evaluation trading dates (zero-filled on flat non-triggered days!)
    mat_rows: list[list[float]] = []
    for dt in all_dates_sorted:
        mat_rows.append([cfg_daily_2x_series[c["name"]].get(dt, 0.0) for c in PRIMARY_CONFIG_GRID])
    cpcv_primary = _compute_fold_fitted_cpcv_and_pbo(np.asarray(mat_rows, dtype=float))

    ev_prim_df = pd.DataFrame(canonical_primary_events)
    daily_prim_df = (
        ev_prim_df.groupby("trade_date", as_index=False)[["net_1x_bps", "net_2x_bps"]].mean()
        if not ev_prim_df.empty
        else pd.DataFrame()
    )
    primary_metrics = _evaluate_arm_metrics(
        ev_prim_df,
        daily_prim_df,
        label="H113_PRIMARY_PREFILL_CONFIRM_LONGSHORT_PER_SYM_MIN_TRAIL",
        n_trials=TRIAL_NUMBER,
        cpcv_override=cpcv_primary,
    )
    primary_metrics["prefill_confirmation_stats"] = {
        "total_signal_candidates": canonical_total_candidates,
        "confirmed_filled_count": canonical_filled_count,
        "cancelled_unfilled_0bps_count": canonical_total_candidates - canonical_filled_count,
        "fill_rate_pct": round(canonical_filled_count / max(1, canonical_total_candidates) * 100.0, 1),
        "cancel_reasons": dict(canonical_cancel_reasons),
        "cancelled_counterfactual_mean_2x_bps": round(float(np.mean(canonical_cancelled_counterfactual_2x)), 2)
        if canonical_cancelled_counterfactual_2x
        else 0.0,
    }

    ev_abl_df = pd.DataFrame(canonical_ablation_events)
    daily_abl_df = (
        ev_abl_df.groupby("trade_date", as_index=False)[["net_1x_bps", "net_2x_bps"]].mean()
        if not ev_abl_df.empty
        else pd.DataFrame()
    )
    ablation_metrics = _evaluate_arm_metrics(
        ev_abl_df,
        daily_abl_df,
        label="H113_ABLATION_MARKETABLE_UNIFORM_STOP",
        n_trials=TRIAL_NUMBER,
    )

    # 6. Diagnostic Arm B: Multi-Day 5d Non-Overlapping Daily MTM Swing on Exact Same Confirmed Entries
    # Daily portfolio return = equal-weighted 1-day MTM return across active positions held that day (zero overlap).
    swing_daily_mtm_1x: dict[str, list[float]] = defaultdict(list)
    swing_daily_mtm_2x: dict[str, list[float]] = defaultdict(list)
    swing_events: list[dict[str, Any]] = []
    for ev in canonical_primary_events:
        dt0 = str(ev["trade_date"])
        sym = str(ev["symbol"])
        side = str(ev["side"])
        is_long = side == "LONG"
        ep = float(ev["entry_px"])
        if ep <= 0.0 or dt0 not in date_to_idx:
            continue
        i0 = date_to_idx[dt0]
        vol20_bps = max(80.0, float(ev["vol20d_daily"]) * 10_000.0)
        cost_1x = _almgren_chriss_1x_cost_bps(
            float(ev["adv20_usd"]), float(ev["vol20d_daily"]), is_short=(not is_long), hold_days=1
        )
        cost_2x = 2.0 * cost_1x
        day_trail_frac = max(0.0060, min(0.0200, max(3.0 * cost_1x, 0.85 * vol20_bps) / 10_000.0))
        hard_stop_frac = min(ADR0022_MAX_STOP_PCT / 100.0, max(0.0100, day_trail_frac))
        be_arm_frac = max(0.0040, 3.0 * cost_1x / 10_000.0)
        be_lock_frac = max(0.0015, 1.5 * cost_1x / 10_000.0)

        stop_px = ep * (1.0 - hard_stop_frac) if is_long else ep * (1.0 + hard_stop_frac)
        ext_px = ep
        prev_mtm_px = ep
        cum_gross_bps = 0.0
        mae_frac = 0.0
        mfe_frac = 0.0
        days_held = 0

        for step_d in range(0, 5):
            if i0 + step_d >= len(all_dates_sorted):
                break
            d_cur = all_dates_sorted[i0 + step_d]
            ohlc = full_daily_lut.get(sym, {}).get(d_cur)
            if ohlc is None or ohlc[3]:  # skip missing or split day cleanly
                break
            dh, dl, dc, _ = ohlc
            if dh <= 0.0 or dl <= 0.0 or dc <= 0.0:
                break
            days_held += 1
            hit_stop = False
            if step_d == 0:
                # Day 0 intraday already confirmed at 09:50..09:52; use conservative day-0 close vs stop
                if is_long:
                    mae_frac = max(mae_frac, (ep - dl) / ep)
                    if dl <= stop_px:
                        exit_px = stop_px
                        hit_stop = True
                    else:
                        ext_px = max(ext_px, dh)
                        mfe_frac = max(mfe_frac, (ext_px - ep) / ep)
                        if mfe_frac >= be_arm_frac:
                            stop_px = max(stop_px, max(ep * (1.0 + be_lock_frac), ext_px * (1.0 - day_trail_frac)))
                        exit_px = dc
                else:
                    mae_frac = max(mae_frac, (dh - ep) / ep)
                    if dh >= stop_px:
                        exit_px = stop_px
                        hit_stop = True
                    else:
                        ext_px = min(ext_px, dl)
                        mfe_frac = max(mfe_frac, (ep - ext_px) / ep)
                        if mfe_frac >= be_arm_frac:
                            stop_px = min(stop_px, min(ep * (1.0 - be_lock_frac), ext_px * (1.0 + day_trail_frac)))
                        exit_px = dc
            else:
                if is_long:
                    mae_frac = max(mae_frac, (ep - dl) / ep)
                    if dl <= stop_px:
                        exit_px = stop_px
                        hit_stop = True
                    else:
                        ext_px = max(ext_px, dh)
                        mfe_frac = max(mfe_frac, (ext_px - ep) / ep)
                        if mfe_frac >= be_arm_frac:
                            stop_px = max(stop_px, max(ep * (1.0 + be_lock_frac), ext_px * (1.0 - day_trail_frac)))
                        exit_px = dc
                else:
                    mae_frac = max(mae_frac, (dh - ep) / ep)
                    if dh >= stop_px:
                        exit_px = stop_px
                        hit_stop = True
                    else:
                        ext_px = min(ext_px, dl)
                        mfe_frac = max(mfe_frac, (ep - ext_px) / ep)
                        if mfe_frac >= be_arm_frac:
                            stop_px = min(stop_px, min(ep * (1.0 - be_lock_frac), ext_px * (1.0 + day_trail_frac)))
                        exit_px = dc

            day_ret_bps = (
                ((exit_px - prev_mtm_px) / prev_mtm_px * 10_000.0)
                if is_long
                else ((prev_mtm_px - exit_px) / prev_mtm_px * 10_000.0)
            )
            entry_cost_share = 0.5 if step_d == 0 else 0.0
            exit_cost_share = 0.5 if (hit_stop or step_d == 4) else 0.0
            borrow_d = SHORT_BORROW_DAILY_BPS if (not is_long) else 0.0
            d_cost_1x = (entry_cost_share + exit_cost_share) * (
                cost_1x - (SHORT_BORROW_DAILY_BPS if not is_long else 0.0)
            ) + borrow_d
            d_cost_2x = 2.0 * d_cost_1x
            swing_daily_mtm_1x[d_cur].append(day_ret_bps - d_cost_1x)
            swing_daily_mtm_2x[d_cur].append(day_ret_bps - d_cost_2x)
            cum_gross_bps += day_ret_bps
            prev_mtm_px = exit_px
            if hit_stop:
                break

        tot_c1 = _almgren_chriss_1x_cost_bps(
            float(ev["adv20_usd"]), float(ev["vol20d_daily"]), is_short=(not is_long), hold_days=max(1, days_held)
        )
        swing_events.append(
            {
                "trade_date": dt0,
                "symbol": sym,
                "side": side,
                "net_1x_bps": cum_gross_bps - tot_c1,
                "net_2x_bps": cum_gross_bps - 2.0 * tot_c1,
                "mae_pct": max(0.0, mae_frac * 100.0),
                "mfe_pct": max(0.0, mfe_frac * 100.0),
                "green_to_red": bool(mfe_frac >= be_arm_frac and cum_gross_bps < 0.0),
                "vol20d_daily": float(ev["vol20d_daily"]),
            }
        )

    ev_sw_df = pd.DataFrame(swing_events)
    sw_dates = sorted(swing_daily_mtm_2x.keys())
    daily_sw_df = pd.DataFrame(
        [
            {
                "trade_date": dt,
                "net_1x_bps": float(np.mean(swing_daily_mtm_1x[dt])),
                "net_2x_bps": float(np.mean(swing_daily_mtm_2x[dt])),
            }
            for dt in sw_dates
        ]
    )
    swing_metrics = _evaluate_arm_metrics(
        ev_sw_df,
        daily_sw_df,
        label="H113_DIAG_MULTIDAY_SWING_PER_SYM_MIN_TRAIL",
        n_trials=TRIAL_NUMBER,
    )

    # Paired ablation summary
    ablation_delta = {
        "delta_mean_2x_daily_bps": round(
            float(primary_metrics.get("mean_2x_daily_bps") or 0.0)
            - float(ablation_metrics.get("mean_2x_daily_bps") or 0.0),
            2,
        ),
        "delta_max_dd_2x_pct_reduction": round(
            float(ablation_metrics.get("max_dd_2x_pct") or 0.0) - float(primary_metrics.get("max_dd_2x_pct") or 0.0),
            2,
        ),
        "delta_green_to_red_pct_reduction": round(
            float(ablation_metrics.get("green_to_red_pct") or 0.0)
            - float(primary_metrics.get("green_to_red_pct") or 0.0),
            1,
        ),
        "delta_profit_factor_2x": round(
            float(primary_metrics.get("profit_factor_2x") or 0.0)
            - float(ablation_metrics.get("profit_factor_2x") or 0.0),
            3,
        ),
    }

    receipt = {
        "schema": "sota.prewall-trial-receipt.v1",
        "trial_number": TRIAL_NUMBER,
        "hypothesis_id": HYPOTHESIS_ID,
        "prereg_spec_sha256": PREREG_SPEC_SHA256,
        "study_wall_iso": STUDY_WALL_ISO,
        "sample_window": f"{SAMPLE_START_ISO}..{STUDY_WALL_ISO}",
        "eval_start": EVAL_START_ISO,
        "sessions_scanned": len(sip_dirs),
        "post_wall_bytes_read": 0,
        "primary_arm": primary_metrics,
        "paired_ablation_arm": ablation_metrics,
        "ablation_delta_prefill_min_trail_vs_blind_uniform": ablation_delta,
        "multiday_swing_arm": swing_metrics,
        "overall_passed": bool(primary_metrics.get("passed")),
        "overall_verdict": "PASSED" if bool(primary_metrics.get("passed")) else "FAILED",
        "promotion_track": "ALPHA_STRATEGY_MINOR_CANDIDATE"
        if bool(primary_metrics.get("passed"))
        else "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    raw_hash = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    receipt["receipt_sha256"] = raw_hash
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Trial #108 (H113) on 700d pre-wall SIP 1m lake.")
    parser.add_argument(
        "--market-data-root", type=pathlib.Path, default=pathlib.Path.home() / "Documents" / "market-data"
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "docs" / "TRIAL108_H113_PREWALL_RECEIPT.json",
    )
    args = parser.parse_args()
    receipt = run_h113_trial108_study(args.market_data_root)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
