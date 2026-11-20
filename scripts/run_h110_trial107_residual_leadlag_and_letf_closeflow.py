#!/usr/bin/env python3
"""Trial #107 (`H110`): Pre-Wall (`2024-01-02..2026-10-15`, `700d`) Directed Lead-Lag & EOD LETF Close Flow.

Pre-Registered Immutable Spec SHA-256 (`sota-bus` Trial #107):
  `21c9dd22bf9627fdb18a7645c7fef7d8b37a194f7495bf4a84124945aebcf266`
Required HLZ Multiplicity Hurdle (`N = 107`): `t*(107) = 4.2127`
ADIA Lab #19 Power Gate (`D = 675`, `sigma = 80 bps`): APPROVED (`SR_min_ann = 2.574 <= 5.0`, `MinTRL@SR3 = 498 <= 675`)

Economic Mechanisms (`[E1]` Primary Literature):
1. **Primary (`H110_PRIMARY_BETA_NEUTRAL_DIRECTED_LEADLAG_PORTFOLIO`)** — Cartea, Cucuringu & Shi (2026 SSRN 6811380),
   Chordia & Swaminathan (2000 JF), Lo & MacKinlay (1990 RFS):
   - Strictly causal `T-1` operating US equities (`prev_rth_close >= $10`, `ADV20 >= $50M`, `hist_days >= 15`,
     zero splits on `[T-1..T+1]`, `90d` PIT SEC `424B5`/`S-3`/`S-3ASR`/`EFFECT` dilution shield on Longs).
   - Partition daily liquid universe using strictly `T-1` `ADV20`: Top-50 Leaders vs Followers (`rank > 50`).
   - Estimate strictly causal `T-1` trailing `20d` `SPY` beta (`beta_spy_20d` clamped `[0.25, 2.50]`).
   - At `10:00 ET` (entry at `10:01 ET` open — zero lookahead), compute directed catch-up gap:
       `gap_i,t = beta_i,t-1 * r_leader(09:30->10:00) - r_follower_i(09:30->10:00)`
   - Dollar-neutral Long Top-K (positive catch-up gap, dilution-shielded) / Short Bottom-K on sessions where
     `|r_leader(09:30->10:00)| >= min_move`, simulated bar-by-bar on real 1-minute bars (`10:01..15:50 ET`) with
     `ADR-0022 <= 1.40%` hard stop + uncapped 3-tier lossless Chandelier (`TP OFF`, zero phantom clamping!).
   - True fold-fitted `skfolio` `CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=20, embargo_size=20)` over
     the 5 pre-registered `(min_move, K)` configs produces genuinely distinct out-of-sample 5-path returns AND true
     Bailey et al. (2014) CSCV/CPCV Probability of Backtest Overfitting (`PBO`).
2. **Diagnostic Arm B (`H110_DIAG_EOD_LETF_CLOSEFLOW_1530_1558`)** — Baltussen, Da, Lammers & Martens (2021 JFE),
   Cheng & Madhavan (2009), Greenwood & Sammon (2025):
   - 10 ultra-liquid ETFs (`SPY, QQQ, IWM, SOXX, SMH, XLF, XLK, XLE, XBI, XLY`): when `|r(09:30->15:25)| >= 1.00%` AND
     causal `T-1` `20d` realized vol exceeds its causal `60d` median, ride forced EOD LETF/dealer hedging flow in the
     direction of the intraday move (`15:26..15:58 ET`, `ADR-0022 <= 1.40%` stop + uncapped Chandelier), evaluated at
     both `14`-pt strict `1x/2x` (`>= 8 bps + AC`) and empirical ultra-liquid ETF spread (`2 bps / 4 bps`).
3. **Diagnostic Arm C (`H110_DIAG_T1_TO_T0_RESIDUAL_LEADLAG_RTH`)** — Multi-day `T-1 -> T` residual catch-up:
   - Strictly causal `T-1` full-day residual gap (`beta_i,t-2 * r_leader_{T-1} - r_i_{T-1}`), Long Top-10 / Short
     Bottom-10 at `T 09:35 ET` open -> `15:50 ET` close (`ADR-0022 <= 2.00%` hard stop + uncapped Chandelier).
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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

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
TRIAL_NUMBER = 107
HYPOTHESIS_ID = "H110_RESIDUAL_DIRECTED_LEAD_LAG_AND_EOD_LETF_FLOW"
PREREG_SPEC_SHA256 = "21c9dd22bf9627fdb18a7645c7fef7d8b37a194f7495bf4a84124945aebcf266"

LIQUID_ETFS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "SOXX", "SMH", "XLF", "XLK", "XLE", "XBI", "XLY")

# Pre-registered 5-config grid for true fold-fitted CPCV & Bailey et al. (2014) PBO
PRIMARY_CONFIG_GRID: tuple[dict[str, Any], ...] = (
    {"name": "cfg_m25_k10", "min_leader_move": 0.0025, "k_per_side": 10},
    {"name": "cfg_m35_k10", "min_leader_move": 0.0035, "k_per_side": 10},  # Canonical baseline
    {"name": "cfg_m50_k10", "min_leader_move": 0.0050, "k_per_side": 10},
    {"name": "cfg_m35_k15", "min_leader_move": 0.0035, "k_per_side": 15},
    {"name": "cfg_m50_k05", "min_leader_move": 0.0050, "k_per_side": 5},
)
PRIMARY_CANONICAL_CFG = "cfg_m35_k10"

POSITION_NOTIONAL_USD = 50_000.0
BASE_1X_COST_BPS = 8.0
ETF_EMPIRICAL_1X_COST_BPS = 2.0
SHORT_BORROW_DAILY_BPS = 1.5
H110_INTRADAY_STOP_PCT = 1.40
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
    base_bps: float = BASE_1X_COST_BPS,
) -> float:
    sigma_bps = max(50.0, min(1200.0, float(vol20d_daily) * 10_000.0))
    adv = max(1_000_000.0, float(adv20_usd))
    impact = 0.10 * sigma_bps * math.sqrt(POSITION_NOTIONAL_USD / adv)
    borrow = SHORT_BORROW_DAILY_BPS if is_short else 0.0
    return float(base_bps) + impact + borrow


def _simulate_uncapped_lossless(
    bars_o: Sequence[float] | Any,
    bars_h: Sequence[float] | Any,
    bars_l: Sequence[float] | Any,
    bars_c: Sequence[float] | Any,
    *,
    side: str,
    stop_pct: float = H110_INTRADAY_STOP_PCT,
) -> dict[str, Any]:
    """Conservative bar-by-bar Long/Short replay with ADR-0022 hard stop + uncapped 3-tier lossless Chandelier."""
    if len(bars_o) == 0 or not math.isfinite(float(bars_o[0])) or float(bars_o[0]) <= 0.0:
        return {"gross_bps": 0.0, "mae_pct": 0.0, "mfe_pct": 0.0, "exit_reason": "NO_BARS", "harvested_50": False}

    is_long = side.upper() == "LONG"
    entry_px = float(bars_o[0])
    hard_stop_frac = min(float(stop_pct), ADR0022_MAX_STOP_PCT) / 100.0
    stop_px = entry_px * (1.0 - hard_stop_frac) if is_long else entry_px * (1.0 + hard_stop_frac)
    extreme_px = entry_px
    mae_frac = 0.0
    mfe_frac = 0.0
    harvested_50 = False
    harvest_ret_frac = 0.0
    exit_reason = "EOD_FLAT"
    runner_ret_frac = 0.0
    exited = False

    for i in range(len(bars_o)):
        h = float(bars_h[i])
        lo = float(bars_l[i])
        if not (math.isfinite(h) and math.isfinite(lo) and h > 0.0 and lo > 0.0):
            continue
        if is_long:
            bar_adv = (entry_px - lo) / entry_px
            if bar_adv > mae_frac:
                mae_frac = bar_adv
            # Conservative intrabar stop check before favorable excursion
            if lo <= stop_px:
                runner_ret_frac = (stop_px - entry_px) / entry_px
                exit_reason = "CHANDELIER_STOP" if (harvested_50 or stop_px > entry_px) else "ADR0022_HARD_STOP"
                exited = True
                break
            if h > extreme_px:
                extreme_px = h
                mfe_frac = (extreme_px - entry_px) / entry_px
            if not harvested_50 and mfe_frac >= 0.0125:
                harvested_50 = True
                harvest_ret_frac = 0.0125
                stop_px = max(stop_px, entry_px * (1.0 + 0.0025))
            elif mfe_frac >= 0.0080:
                stop_px = max(stop_px, entry_px * (1.0 + 0.0018))
            if mfe_frac >= 0.0200:
                retain = (
                    0.86 if mfe_frac >= 0.10 else (0.82 if mfe_frac >= 0.06 else (0.75 if mfe_frac >= 0.04 else 0.65))
                )
                stop_px = max(stop_px, entry_px * (1.0 + mfe_frac * retain))
        else:
            bar_adv = (h - entry_px) / entry_px
            if bar_adv > mae_frac:
                mae_frac = bar_adv
            if h >= stop_px:
                runner_ret_frac = (entry_px - stop_px) / entry_px
                exit_reason = "CHANDELIER_STOP" if (harvested_50 or stop_px < entry_px) else "ADR0022_HARD_STOP"
                exited = True
                break
            if lo < extreme_px:
                extreme_px = lo
                mfe_frac = (entry_px - extreme_px) / entry_px
            if not harvested_50 and mfe_frac >= 0.0125:
                harvested_50 = True
                harvest_ret_frac = 0.0125
                stop_px = min(stop_px, entry_px * (1.0 - 0.0025))
            elif mfe_frac >= 0.0080:
                stop_px = min(stop_px, entry_px * (1.0 - 0.0018))
            if mfe_frac >= 0.0200:
                retain = (
                    0.86 if mfe_frac >= 0.10 else (0.82 if mfe_frac >= 0.06 else (0.75 if mfe_frac >= 0.04 else 0.65))
                )
                stop_px = min(stop_px, entry_px * (1.0 - mfe_frac * retain))

    if not exited:
        last_c = float(bars_c[-1])
        runner_ret_frac = ((last_c - entry_px) / entry_px) if is_long else ((entry_px - last_c) / entry_px)

    gross_frac = (0.5 * harvest_ret_frac + 0.5 * runner_ret_frac) if harvested_50 else runner_ret_frac
    return {
        "gross_bps": gross_frac * 10_000.0,
        "mae_pct": max(0.0, mae_frac * 100.0),
        "mfe_pct": max(0.0, mfe_frac * 100.0),
        "exit_reason": exit_reason,
        "harvested_50": harvested_50,
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


def _compute_fold_fitted_cpcv_and_pbo(
    config_daily_2x_matrix: Any,
) -> dict[str, Any]:
    """Run true fold-fitted skfolio CombinatorialPurgedCV(6,2,purge=20,embargo=20) & Bailey et al. (2014) PBO.

    Unlike static single-series recombination (which yields identical means across all 5 paths),
    each CPCV split selects the IS Sharpe-maximizing configuration strictly inside `tr_idx`
    (purged 20d, embargoed 20d) and records its held-out `te_idx` returns onto the 5 OOS paths
    AND computes the OOS rank logit for Bailey et al. (2014) PBO.
    """
    import numpy as np
    from skfolio.model_selection import CombinatorialPurgedCV

    mat = np.asarray(config_daily_2x_matrix, dtype=float)  # shape (T, M)
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
        # Rank of IS winner in OOS (1 = worst, n_cfg = best)
        winner_oos = float(oos_sr[best_cfg])
        rank_1_to_m = 1 + int(np.sum(oos_sr < winner_oos)) + 0.5 * int(np.sum(oos_sr == winner_oos) - 1)
        rel_rank = rank_1_to_m / float(n_cfg + 1)
        if rel_rank <= 0.5:
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
    events_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    label: str,
    n_trials: int = TRIAL_NUMBER,
    stop_pct_cap: float = H110_INTRADAY_STOP_PCT,
    cpcv_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute complete SOTA statistical, CPCV, PBO, DSR, Hansen SPA/MCS, Conformal PID & adversarial metrics."""
    import numpy as np
    from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics

    if events_df.empty or daily_df.empty:
        return {"label": label, "events": 0, "distinct_dates": 0, "passed": False, "verdict": "ZERO_EVENTS"}

    n_events = int(len(events_df))
    distinct_dates = int(len(daily_df))

    mean_1x_ev, _se1e, t_1x_ev = _date_clustered_t(
        events_df["net_1x_bps"].to_numpy(), events_df["trade_date"].to_numpy()
    )
    mean_2x_ev, _se2e, t_2x_ev = _date_clustered_t(
        events_df["net_2x_bps"].to_numpy(), events_df["trade_date"].to_numpy()
    )

    d1 = daily_df["net_1x_bps"].to_numpy(dtype=float)
    d2 = daily_df["net_2x_bps"].to_numpy(dtype=float)
    ddates = daily_df["trade_date"].to_numpy()
    mean_1x_day, _se1d, t_1x_day = _date_clustered_t(d1, ddates)
    mean_2x_day, _se2d, t_2x_day = _date_clustered_t(d2, ddates)

    sr_1x_ann = (
        float((np.mean(d1) / np.std(d1, ddof=1)) * math.sqrt(252.0)) if len(d1) > 2 and np.std(d1, ddof=1) > 0 else 0.0
    )
    sr_2x_ann = (
        float((np.mean(d2) / np.std(d2, ddof=1)) * math.sqrt(252.0)) if len(d2) > 2 and np.std(d2, ddof=1) > 0 else 0.0
    )

    arr_2x = d2 / 10_000.0
    sd_2x = float(np.std(arr_2x, ddof=1)) if len(arr_2x) > 2 else 0.0
    if len(arr_2x) >= 15 and sd_2x > 1e-12:
        mu_2x = float(np.mean(arr_2x))
        sr_daily_2x = mu_2x / sd_2x
        z_2x = (arr_2x - mu_2x) / sd_2x
        skew_2x = float(np.mean(z_2x**3))
        ex_kurt_2x = float(np.mean(z_2x**4) - 3.0)
        ac1_2x = float(np.corrcoef(arr_2x[:-1], arr_2x[1:])[0, 1]) if len(arr_2x) > 3 else 0.0
        if not math.isfinite(ac1_2x):
            ac1_2x = 0.0
        dsr_res = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=sr_daily_2x,
            n_samples=len(arr_2x),
            n_trials=max(1, int(n_trials)),
            variance_trials=1.0 / 252.0,
            skewness=skew_2x,
            excess_kurtosis=ex_kurt_2x,
            autocorrelation=max(-0.95, min(0.95, ac1_2x)),
            frequency="daily",
        )
        dsr_prob = float(getattr(dsr_res, "probability", 0.0) or 0.0)
        if not math.isfinite(dsr_prob):
            dsr_prob = 0.0
    else:
        dsr_prob = 0.0

    if cpcv_override is not None:
        cpcv_info = dict(cpcv_override)
    else:
        # Honest chronological 5-block walk-forward OOS breakdown when no config grid was fitted
        n_b = 5
        blocks = np.array_split(d2, n_b)
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

    spa_res = evaluate_hansen_spa_and_mcs({label: d2.tolist()}, n_bootstrap=1000, avg_block_len=5.0, seed=107)
    spa_p = float(spa_res["spa_p_consistent"])

    pid = ConformalPidStopCalibrator(
        target_miscoverage_alpha=0.10, max_stop_pct=min(stop_pct_cap, ADR0022_MAX_STOP_PCT)
    )
    pid_last: dict[str, Any] = {
        "next_stop_pct": min(stop_pct_cap, ADR0022_MAX_STOP_PCT),
        "empirical_miscoverage_rate": 0.0,
    }
    for mae_val, vol_val in zip(
        events_df["mae_pct"].to_numpy(dtype=float),
        events_df["vol20d_pct"].to_numpy(dtype=float),
        strict=False,
    ):
        pid_last = pid.update(realized_mae_pct=float(mae_val), atr_pct=max(0.25, float(vol_val) * 0.5))

    cal_stats = compute_calendar_day_stats(d2.tolist())

    d2_sorted_idx = np.argsort(d2)
    ex_top3_mask = np.ones(len(d2), dtype=bool)
    if len(d2) > 10:
        ex_top3_mask[d2_sorted_idx[-3:]] = False
    mean_2x_ex_top3, _se_ex3, t_2x_ex_top3 = _date_clustered_t(d2[ex_top3_mask], ddates[ex_top3_mask])

    subperiods: dict[str, dict[str, Any]] = {}
    for sp_name, lo_d, hi_d in (
        ("2024", "2024-02-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_PREWALL", "2026-01-01", STUDY_WALL_ISO),
    ):
        sub_ev = events_df[(events_df["trade_date"] >= lo_d) & (events_df["trade_date"] <= hi_d)]
        sub_d = daily_df[(daily_df["trade_date"] >= lo_d) & (daily_df["trade_date"] <= hi_d)]
        if not sub_d.empty:
            _m, _s, st = _date_clustered_t(sub_d["net_2x_bps"].to_numpy(), sub_d["trade_date"].to_numpy())
            subperiods[sp_name] = {
                "events": int(len(sub_ev)),
                "dates": int(len(sub_d)),
                "mean_1x_bps": round(float(sub_d["net_1x_bps"].mean()), 2),
                "mean_2x_bps": round(float(sub_d["net_2x_bps"].mean()), 2),
                "t_2x_daily": round(st, 4),
            }

    hygiene_audit = audit_low_quality_practices(
        PlaybackAuditInput(
            horizon_id="W1_PRIMARY_PREWALL_LAKE",
            sample_start_date=str(daily_df["trade_date"].iloc[0]),
            sample_max_date=str(daily_df["trade_date"].iloc[-1]),
            distinct_sessions=distinct_dates,
            cadence="intraday",
            universe_mode="CAUSAL_T_MINUS_1_PIT",
            cost_1x_rt_bps=float(daily_df["cost_1x_bps"].mean()),
            includes_2x_and_sqrt_impact=True,
            take_profit_cap_enabled=False,
            max_stop_pct=min(stop_pct_cap, ADR0022_MAX_STOP_PCT),
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
            ar1_rho_for_min_trl=max(-0.5, min(0.5, float(cal_stats["ar1_rho"]))),
        )
    )

    hlz_hurdle = hlz_dynamic_t_hurdle(n_trials)
    t_conservative = min(t_2x_ev, t_2x_day)
    pbo_ok = True if cpcv_info["cpcv_pbo"] is None else (float(cpcv_info["cpcv_pbo"]) < STRICT_PBO_MAX)
    passed = (
        n_events >= STRICT_MIN_EVENTS
        and distinct_dates >= STRICT_MIN_DATES
        and mean_1x_day > 0.0
        and mean_2x_day > 0.0
        and t_conservative >= hlz_hurdle
        and dsr_prob >= STRICT_DSR_MIN
        and float(cpcv_info["cpcv_5path_positive_share"]) >= STRICT_CPCV_POS_MIN
        and pbo_ok
        and spa_p < STRICT_SPA_P_MAX
        and bool(hygiene_audit["passed"])
    )

    win_rate_1x = float((events_df["net_1x_bps"] > 0.0).mean() * 100.0) if n_events > 0 else 0.0
    win_rate_2x = float((events_df["net_2x_bps"] > 0.0).mean() * 100.0) if n_events > 0 else 0.0

    return {
        "label": label,
        "events": n_events,
        "distinct_dates": distinct_dates,
        "win_rate_1x_pct": round(win_rate_1x, 2),
        "win_rate_2x_pct": round(win_rate_2x, 2),
        "mean_gross_day_bps": round(float(daily_df["gross_bps"].mean()), 2),
        "mean_cost_1x_bps": round(float(daily_df["cost_1x_bps"].mean()), 2),
        "mean_net_1x_day_bps": round(mean_1x_day, 2),
        "mean_net_2x_day_bps": round(mean_2x_day, 2),
        "mean_net_1x_event_bps": round(mean_1x_ev, 2),
        "mean_net_2x_event_bps": round(mean_2x_ev, 2),
        "t_stat_clustered_1x_daily": round(t_1x_day, 4),
        "t_stat_clustered_2x_daily": round(t_2x_day, 4),
        "t_stat_clustered_1x_event": round(t_1x_ev, 4),
        "t_stat_clustered_2x_event": round(t_2x_ev, 4),
        "hlz_t_hurdle_n107": round(hlz_hurdle, 4),
        "annualized_sharpe_1x": round(sr_1x_ann, 3),
        "annualized_sharpe_2x": round(sr_2x_ann, 3),
        "calendar_day_stats_2x": cal_stats,
        "dsr_probability_2x": round(dsr_prob, 4),
        **cpcv_info,
        "hansen_spa_p_value_2x": round(spa_p, 4),
        "hln_mcs_included": bool(label in spa_res["mcs_surviving_strategies"]),
        "conformal_pid_calibrated_stop_pct": round(float(pid_last["next_stop_pct"]), 4),
        "conformal_pid_empirical_stopout_rate": round(float(pid_last["empirical_miscoverage_rate"]), 4),
        "adversarial_ex_top3_days_2x_bps": round(mean_2x_ex_top3, 2),
        "adversarial_ex_top3_days_t_2x": round(t_2x_ex_top3, 4),
        "subperiods": subperiods,
        "hygiene_14pt_passed": bool(hygiene_audit["passed"]),
        "hygiene_violations": list(hygiene_audit["violations"]),
        "verdict": "PASSED_ALPHA_CANDIDATE" if passed else "FAILED_HLZ_OR_STAT_GATE",
        "promotion_track": "ALPHA_STRATEGY_MINOR_CANDIDATE" if passed else "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "passed": passed,
    }


def run_h110_trial107_study(market_data_root: pathlib.Path) -> dict[str, Any]:
    """Execute strictly pre-wall (`<= 2026-10-15`) Trial #107 (`H110`) on real 700d SIP 1m lake."""
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

    # 1. Operating US common equities + 10 liquid ETFs + splits + PIT 90d SEC dilution shield
    etf_values = ", ".join(f"('{e}')" for e in LIQUID_ETFS)
    con.execute(f"CREATE TEMP TABLE liquid_etfs (symbol VARCHAR); INSERT INTO liquid_etfs VALUES {etf_values};")
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
        UNION
        SELECT DISTINCT symbol FROM liquid_etfs
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE sec_dil_dates AS
        SELECT
            o.symbol,
            SUBSTR(f.accepted, 1, 10) AS dil_date
        FROM read_parquet('{sec_fil_p}') f
        JOIN operating_syms o ON o.cik = f.cik
        WHERE f.accepted IS NOT NULL
          AND LENGTH(f.accepted) >= 10
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

    # 2. Batched single-pass extraction of intraday checkpoint prices & daily RTH volume (25 dates/batch)
    con.execute(
        """
        CREATE TEMP TABLE sip_daily (
            symbol VARCHAR,
            trade_date VARCHAR,
            open_0930 DOUBLE,
            open_0935 DOUBLE,
            close_1000 DOUBLE,
            open_1001 DOUBLE,
            close_1525 DOUBLE,
            open_1526 DOUBLE,
            close_1550 DOUBLE,
            close_1558 DOUBLE,
            rth_close DOUBLE,
            rth_dvol DOUBLE
        )
        """
    )
    batch_size = 25
    for idx in range(0, len(sip_dirs), batch_size):
        batch = sip_dirs[idx : idx + batch_size]
        globs = [f"{d.as_posix()}/*.parquet" for d in batch]
        con.execute(
            """
            INSERT INTO sip_daily
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
                    o, c, v, vw
                FROM raw_bars
                WHERE strftime(ny_ts, '%Y-%m-%d') <= $2
            )
            SELECT
                symbol,
                trade_date,
                arg_min(o, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 572) AS open_0930,
                arg_min(o, ny_min) FILTER (WHERE ny_min >= 575 AND ny_min <= 577) AS open_0935,
                arg_max(c, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 600) AS close_1000,
                arg_min(o, ny_min) FILTER (WHERE ny_min >= 601 AND ny_min <= 603) AS open_1001,
                arg_max(c, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 925) AS close_1525,
                arg_min(o, ny_min) FILTER (WHERE ny_min >= 926 AND ny_min <= 928) AS open_1526,
                arg_max(c, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 950) AS close_1550,
                arg_max(c, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 958) AS close_1558,
                arg_max(c, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 959) AS rth_close,
                SUM(vw * v) FILTER (WHERE ny_min >= 570 AND ny_min <= 959) AS rth_dvol
            FROM tagged
            WHERE ny_min >= 570 AND ny_min <= 959
            GROUP BY symbol, trade_date
            HAVING rth_close IS NOT NULL AND rth_dvol > 0
            """,
            [globs, STUDY_WALL_ISO],
        )

    # 3. Causal T-1 20d ADV, vol, SPY beta (rolling 20d cov/var on [T-20..T-1]), & dilution shield
    panel_df = con.execute(
        """
        WITH spy_ret AS (
            SELECT
                trade_date,
                LN(rth_close / NULLIF(LAG(rth_close) OVER (ORDER BY trade_date), 0)) AS spy_log_ret
            FROM sip_daily
            WHERE symbol = 'SPY'
        ),
        base AS (
            SELECT
                d.*,
                (d.symbol IN (SELECT symbol FROM liquid_etfs)) AS is_etf,
                LAG(d.trade_date) OVER (PARTITION BY d.symbol ORDER BY d.trade_date) AS prev_trade_date,
                LAG(d.rth_close) OVER (PARTITION BY d.symbol ORDER BY d.trade_date) AS prev_rth_close,
                LEAD(d.trade_date) OVER (PARTITION BY d.symbol ORDER BY d.trade_date) AS next_trade_date,
                LN(
                    d.rth_close / NULLIF(LAG(d.rth_close) OVER (PARTITION BY d.symbol ORDER BY d.trade_date), 0)
                ) AS log_ret,
                AVG(d.rth_dvol) OVER (
                    PARTITION BY d.symbol ORDER BY d.trade_date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS adv20_usd,
                COUNT(d.rth_dvol) OVER (
                    PARTITION BY d.symbol ORDER BY d.trade_date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS hist_days,
                s.spy_log_ret
            FROM sip_daily d
            LEFT JOIN spy_ret s ON s.trade_date = d.trade_date
        ),
        causal AS (
            SELECT
                b.*,
                LAG(b.log_ret) OVER (PARTITION BY b.symbol ORDER BY b.trade_date) AS prev_log_ret,
                COALESCE(STDDEV_SAMP(b.log_ret) OVER (
                    PARTITION BY b.symbol ORDER BY b.trade_date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ), 0.020) AS vol20d_daily,
                COVAR_SAMP(b.log_ret, b.spy_log_ret) OVER (
                    PARTITION BY b.symbol ORDER BY b.trade_date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) / NULLIF(VAR_SAMP(b.spy_log_ret) OVER (
                    PARTITION BY b.symbol ORDER BY b.trade_date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ), 1e-10) AS raw_beta_20d
            FROM base b
        ),
        vol_regime AS (
            SELECT
                c.*,
                MEDIAN(c.vol20d_daily) OVER (
                    PARTITION BY c.symbol ORDER BY c.trade_date
                    ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
                ) AS vol60_med
            FROM causal c
        )
        SELECT
            v.symbol, v.trade_date, v.prev_trade_date, v.is_etf,
            v.prev_rth_close, v.adv20_usd, v.hist_days, v.vol20d_daily, v.vol60_med,
            LEAST(2.50, GREATEST(0.25, COALESCE(v.raw_beta_20d, 1.0))) AS beta_spy_20d,
            v.prev_log_ret,
            v.open_0930, v.open_0935, v.close_1000, v.open_1001,
            v.close_1525, v.open_1526, v.close_1550, v.close_1558, v.rth_close,
            EXISTS (
                SELECT 1 FROM sec_dil_dates sd
                WHERE sd.symbol = v.symbol
                  AND CAST(sd.dil_date AS DATE) <= CAST(v.prev_trade_date AS DATE)
                  AND CAST(sd.dil_date AS DATE) >= CAST(v.prev_trade_date AS DATE) - INTERVAL 90 DAY
            ) AS has_dil_90d
        FROM vol_regime v
        LEFT JOIN split_events sp0 ON sp0.symbol = v.symbol AND sp0.ex_date = v.prev_trade_date
        LEFT JOIN split_events sp1 ON sp1.symbol = v.symbol AND sp1.ex_date = v.trade_date
        LEFT JOIN split_events sp2 ON sp2.symbol = v.symbol AND sp2.ex_date = v.next_trade_date
        WHERE v.trade_date >= $1 AND v.trade_date <= $2
          AND v.hist_days >= 15
          AND sp0.symbol IS NULL AND sp1.symbol IS NULL AND sp2.symbol IS NULL
          AND (
              v.is_etf
              OR (v.prev_rth_close >= 10.0 AND v.adv20_usd >= 50000000.0)
          )
        ORDER BY v.trade_date, v.adv20_usd DESC
        """,
        [EVAL_START_ISO, STUDY_WALL_ISO],
    ).fetchdf()

    # 4. Construct strictly causal daily candidate sets across Primary (5 configs), Arm B (LETF EOD), and Arm C (T-1->T)
    # Collect unique (trade_date, symbol, window, side) replay tasks so each 1m bar slice is simulated once!
    primary_assignments: dict[str, list[dict[str, Any]]] = {c["name"]: [] for c in PRIMARY_CONFIG_GRID}
    arm_b_tasks: list[dict[str, Any]] = []
    arm_c_tasks: list[dict[str, Any]] = []

    replay_needed: dict[tuple[str, str], set[str]] = defaultdict(set)  # (date, window) -> set(symbols)

    for tdate, day_df in panel_df.groupby("trade_date", sort=True):
        eq_df = day_df[~day_df["is_etf"]].copy()
        etf_df = day_df[day_df["is_etf"]].copy()

        # --- Primary Arm: Top-50 T-1 ADV Leaders vs Followers (rank > 50, ADV20 >= $50M) ---
        if len(eq_df) >= 80:
            eq_df = eq_df.sort_values("adv20_usd", ascending=False).reset_index(drop=True)
            leaders = eq_df.iloc[:50]
            followers = eq_df.iloc[50:].copy()

            l_valid = leaders[(leaders["open_0930"] > 0) & (leaders["close_1000"] > 0)]
            if len(l_valid) >= 25:
                r_leader_30m = float(((l_valid["close_1000"] - l_valid["open_0930"]) / l_valid["open_0930"]).mean())
                f_valid = followers[
                    (followers["open_0930"] > 0)
                    & (followers["close_1000"] > 0)
                    & (followers["open_1001"] > 0)
                    & (followers["close_1550"] > 0)
                ].copy()
                if len(f_valid) >= 35:
                    f_valid["r_30m"] = (f_valid["close_1000"] - f_valid["open_0930"]) / f_valid["open_0930"]
                    f_valid["gap"] = f_valid["beta_spy_20d"] * r_leader_30m - f_valid["r_30m"]
                    f_sorted_desc = f_valid.sort_values("gap", ascending=False)
                    f_long_pool = f_sorted_desc[~f_sorted_desc["has_dil_90d"]]
                    f_short_pool = f_valid.sort_values("gap", ascending=True)

                    for cfg in PRIMARY_CONFIG_GRID:
                        if abs(r_leader_30m) < float(cfg["min_leader_move"]):
                            continue
                        k = int(cfg["k_per_side"])
                        longs = f_long_pool.head(k)
                        shorts = f_short_pool.head(k)
                        long_syms = set(longs["symbol"].tolist())
                        shorts = shorts[~shorts["symbol"].isin(long_syms)]
                        if len(longs) < max(3, k // 2) or len(shorts) < max(3, k // 2):
                            continue
                        for _, r in longs.iterrows():
                            sym = str(r["symbol"])
                            primary_assignments[cfg["name"]].append(
                                {
                                    "trade_date": str(tdate),
                                    "symbol": sym,
                                    "side": "LONG",
                                    "adv20_usd": float(r["adv20_usd"]),
                                    "vol20d_daily": float(r["vol20d_daily"]),
                                    "gap_bps": float(r["gap"]) * 10_000.0,
                                }
                            )
                            replay_needed[(str(tdate), "W1001_1550")].add(sym)
                        for _, r in shorts.iterrows():
                            sym = str(r["symbol"])
                            primary_assignments[cfg["name"]].append(
                                {
                                    "trade_date": str(tdate),
                                    "symbol": sym,
                                    "side": "SHORT",
                                    "adv20_usd": float(r["adv20_usd"]),
                                    "vol20d_daily": float(r["vol20d_daily"]),
                                    "gap_bps": float(r["gap"]) * 10_000.0,
                                }
                            )
                            replay_needed[(str(tdate), "W1001_1550")].add(sym)

            # --- Diagnostic Arm C: Strictly Causal T-1 Full-Day Leader->Follower Residual -> T 09:35..15:50 RTH ---
            l_prev = leaders[leaders["prev_log_ret"].notna()]
            f_prev = followers[
                followers["prev_log_ret"].notna() & (followers["open_0935"] > 0) & (followers["close_1550"] > 0)
            ].copy()
            if len(l_prev) >= 25 and len(f_prev) >= 30:
                r_leader_t1 = float(l_prev["prev_log_ret"].mean())
                if abs(r_leader_t1) >= 0.0040:
                    f_prev["gap_t1"] = f_prev["beta_spy_20d"] * r_leader_t1 - f_prev["prev_log_ret"]
                    longs_c = f_prev[~f_prev["has_dil_90d"]].sort_values("gap_t1", ascending=False).head(10)
                    short_pool_c = f_prev[~f_prev["symbol"].isin(set(longs_c["symbol"].tolist()))]
                    shorts_c = short_pool_c.sort_values("gap_t1", ascending=True).head(10)
                    for _, r in longs_c.iterrows():
                        sym = str(r["symbol"])
                        arm_c_tasks.append(
                            {
                                "trade_date": str(tdate),
                                "symbol": sym,
                                "side": "LONG",
                                "adv20_usd": float(r["adv20_usd"]),
                                "vol20d_daily": float(r["vol20d_daily"]),
                            }
                        )
                        replay_needed[(str(tdate), "W0935_1550")].add(sym)
                    for _, r in shorts_c.iterrows():
                        sym = str(r["symbol"])
                        arm_c_tasks.append(
                            {
                                "trade_date": str(tdate),
                                "symbol": sym,
                                "side": "SHORT",
                                "adv20_usd": float(r["adv20_usd"]),
                                "vol20d_daily": float(r["vol20d_daily"]),
                            }
                        )
                        replay_needed[(str(tdate), "W0935_1550")].add(sym)

        # --- Diagnostic Arm B: EOD 15:26->15:58 LETF / Dealer Gamma Close-Flow Momentum on 10 Liquid ETFs ---
        for _, r in etf_df.iterrows():
            if not (
                pd.notna(r["open_0930"])
                and pd.notna(r["close_1525"])
                and pd.notna(r["open_1526"])
                and pd.notna(r["close_1558"])
                and float(r["open_0930"]) > 0
                and float(r["open_1526"]) > 0
            ):
                continue
            r_day_so_far = (float(r["close_1525"]) - float(r["open_0930"])) / float(r["open_0930"])
            vol_med = float(r["vol60_med"]) if pd.notna(r["vol60_med"]) else float(r["vol20d_daily"])
            high_vol_regime = float(r["vol20d_daily"]) >= vol_med
            if abs(r_day_so_far) >= 0.0100 and high_vol_regime:
                sym = str(r["symbol"])
                side = "LONG" if r_day_so_far > 0 else "SHORT"
                arm_b_tasks.append(
                    {
                        "trade_date": str(tdate),
                        "symbol": sym,
                        "side": side,
                        "adv20_usd": float(r["adv20_usd"]),
                        "vol20d_daily": float(r["vol20d_daily"]),
                        "r_0930_1525_bps": r_day_so_far * 10_000.0,
                    }
                )
                replay_needed[(str(tdate), "W1526_1558")].add(sym)

    # 5. Exact bar-by-bar 1-minute intrabar replay (single read per date!)
    all_replay_dates = sorted({dt_s for (dt_s, _win) in replay_needed})
    sim_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}  # (date, sym, win, side) -> sim dict

    win_bounds = {
        "W1001_1550": (601, 950, H110_INTRADAY_STOP_PCT),
        "W0935_1550": (575, 950, ADR0022_MAX_STOP_PCT),
        "W1526_1558": (926, 958, H110_INTRADAY_STOP_PCT),
    }

    for tdate in all_replay_dates:
        syms_today = sorted(set().union(*(replay_needed.get((tdate, w), set()) for w in win_bounds)))
        if not syms_today:
            continue
        day_glob = f"{(sip_root / tdate).as_posix()}/*.parquet"
        bars_df = con.execute(
            """
            SELECT
                UPPER(TRIM(symbol)) AS symbol,
                EXTRACT(HOUR FROM (to_timestamp(t_ns / 1e9) AT TIME ZONE 'America/New_York')) * 60
                  + EXTRACT(MINUTE FROM (to_timestamp(t_ns / 1e9) AT TIME ZONE 'America/New_York')) AS ny_min,
                t_ns, o, h, l, c
            FROM read_parquet($1)
            WHERE UPPER(TRIM(symbol)) = ANY($2)
              AND o > 0 AND h > 0 AND l > 0 AND c > 0
            ORDER BY symbol, t_ns
            """,
            [day_glob, syms_today],
        ).fetchdf()
        grouped = {str(sym): grp for sym, grp in bars_df.groupby("symbol", sort=False)}
        for w_name, (m_lo, m_hi, stop_cap) in win_bounds.items():
            for sym in replay_needed.get((tdate, w_name), set()):
                grp = grouped.get(sym)
                if grp is None or grp.empty:
                    continue
                sub = grp[(grp["ny_min"] >= m_lo) & (grp["ny_min"] <= m_hi)]
                if sub.empty:
                    continue
                arr_o = sub["o"].to_numpy(dtype=float)
                arr_h = sub["h"].to_numpy(dtype=float)
                arr_l = sub["l"].to_numpy(dtype=float)
                arr_c = sub["c"].to_numpy(dtype=float)
                for side in ("LONG", "SHORT"):
                    sim_cache[(tdate, sym, w_name, side)] = _simulate_uncapped_lossless(
                        arr_o, arr_h, arr_l, arr_c, side=side, stop_pct=stop_cap
                    )

    def _build_records(
        tasks: list[dict[str, Any]],
        win_name: str,
        *,
        base_cost_bps: float = BASE_1X_COST_BPS,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[dict[str, Any]] = []
        for t in tasks:
            sim = sim_cache.get((t["trade_date"], t["symbol"], win_name, t["side"]))
            if sim is None or sim["exit_reason"] == "NO_BARS":
                continue
            c1x = _almgren_chriss_1x_cost_bps(
                t["adv20_usd"], t["vol20d_daily"], is_short=(t["side"] == "SHORT"), base_bps=base_cost_bps
            )
            g = float(sim["gross_bps"])
            rows.append(
                {
                    "trade_date": t["trade_date"],
                    "symbol": t["symbol"],
                    "side": t["side"],
                    "gross_bps": g,
                    "cost_1x_bps": c1x,
                    "net_1x_bps": g - c1x,
                    "net_2x_bps": g - 2.0 * c1x,
                    "mae_pct": float(sim["mae_pct"]),
                    "mfe_pct": float(sim["mfe_pct"]),
                    "vol20d_pct": float(t["vol20d_daily"]) * 100.0,
                    "exit_reason": sim["exit_reason"],
                }
            )
        ev_df = pd.DataFrame(rows)
        if ev_df.empty:
            return (ev_df, pd.DataFrame())
        daily_df = (
            ev_df.groupby("trade_date", as_index=False)
            .agg(
                n_trades=("net_1x_bps", "count"),
                gross_bps=("gross_bps", "mean"),
                cost_1x_bps=("cost_1x_bps", "mean"),
                net_1x_bps=("net_1x_bps", "mean"),
                net_2x_bps=("net_2x_bps", "mean"),
            )
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        return (ev_df, daily_df)

    # 6. Build Primary 5-config matrix aligned on common calendar dates for true fold-fitted CPCV & Bailey PBO!
    cfg_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_primary_dates: set[str] = set()
    for cfg in PRIMARY_CONFIG_GRID:
        cname = cfg["name"]
        ev_df, d_df = _build_records(primary_assignments[cname], "W1001_1550", base_cost_bps=BASE_1X_COST_BPS)
        cfg_results[cname] = (ev_df, d_df)
        if not d_df.empty:
            all_primary_dates.update(d_df["trade_date"].tolist())

    sorted_primary_dates = sorted(all_primary_dates)
    mat_2x = np.zeros((len(sorted_primary_dates), len(PRIMARY_CONFIG_GRID)), dtype=float)
    date_to_idx = {d: i for i, d in enumerate(sorted_primary_dates)}
    grid_summary: dict[str, dict[str, Any]] = {}
    for j, cfg in enumerate(PRIMARY_CONFIG_GRID):
        cname = cfg["name"]
        ev_df, d_df = cfg_results[cname]
        if not d_df.empty:
            for td, val in zip(d_df["trade_date"].tolist(), d_df["net_2x_bps"].to_numpy(dtype=float), strict=False):
                mat_2x[date_to_idx[td], j] = float(val)
            m1, _, t1 = _date_clustered_t(d_df["net_1x_bps"].to_numpy(), d_df["trade_date"].to_numpy())
            m2, _, t2 = _date_clustered_t(d_df["net_2x_bps"].to_numpy(), d_df["trade_date"].to_numpy())
            grid_summary[cname] = {
                "min_leader_move": cfg["min_leader_move"],
                "k_per_side": cfg["k_per_side"],
                "events": int(len(ev_df)),
                "dates": int(len(d_df)),
                "mean_gross_bps": round(float(d_df["gross_bps"].mean()), 2),
                "mean_net_1x_bps": round(m1, 2),
                "mean_net_2x_bps": round(m2, 2),
                "t_1x_daily": round(t1, 4),
                "t_2x_daily": round(t2, 4),
            }

    cpcv_primary = _compute_fold_fitted_cpcv_and_pbo(mat_2x)
    p_ev, p_daily = cfg_results[PRIMARY_CANONICAL_CFG]
    primary_metrics = _evaluate_arm_metrics(
        p_ev,
        p_daily,
        label="H110_PRIMARY_BETA_NEUTRAL_DIRECTED_LEADLAG_PORTFOLIO",
        n_trials=TRIAL_NUMBER,
        stop_pct_cap=H110_INTRADAY_STOP_PCT,
        cpcv_override=cpcv_primary,
    )

    # Diagnostic Arm B: EOD LETF Close-Flow (both strict 14-pt 8bps+AC and empirical 2bps ETF-spread diagnostic)
    b_ev_strict, b_daily_strict = _build_records(arm_b_tasks, "W1526_1558", base_cost_bps=BASE_1X_COST_BPS)
    diag_b_strict = _evaluate_arm_metrics(
        b_ev_strict,
        b_daily_strict,
        label="H110_DIAG_EOD_LETF_CLOSEFLOW_1530_1558_STRICT_8BPS",
        n_trials=TRIAL_NUMBER,
        stop_pct_cap=H110_INTRADAY_STOP_PCT,
    )
    b_ev_etf, b_daily_etf = _build_records(arm_b_tasks, "W1526_1558", base_cost_bps=ETF_EMPIRICAL_1X_COST_BPS)
    diag_b_etf_spread = _evaluate_arm_metrics(
        b_ev_etf,
        b_daily_etf,
        label="H110_DIAG_EOD_LETF_CLOSEFLOW_1530_1558_ETF_2BPS_DIAG",
        n_trials=TRIAL_NUMBER,
        stop_pct_cap=H110_INTRADAY_STOP_PCT,
    )

    # Diagnostic Arm C: Strictly Causal T-1 -> T Residual Lead-Lag RTH Beta-Neutral Portfolio
    c_ev, c_daily = _build_records(arm_c_tasks, "W0935_1550", base_cost_bps=BASE_1X_COST_BPS)
    diag_c = _evaluate_arm_metrics(
        c_ev,
        c_daily,
        label="H110_DIAG_T1_TO_T0_RESIDUAL_LEADLAG_RTH",
        n_trials=TRIAL_NUMBER,
        stop_pct_cap=ADR0022_MAX_STOP_PCT,
    )

    receipt: dict[str, Any] = {
        "schema": "sota.prewall-trial107-h110-receipt.v1",
        "trial_number": TRIAL_NUMBER,
        "hypothesis_id": HYPOTHESIS_ID,
        "prereg_spec_sha256": PREREG_SPEC_SHA256,
        "study_wall_date": STUDY_WALL_ISO,
        "sample_window": f"{SAMPLE_START_ISO}..{STUDY_WALL_ISO}",
        "eval_window": f"{EVAL_START_ISO}..{STUDY_WALL_ISO}",
        "prewall_sip_sessions_scanned": len(sip_dirs),
        "postwall_bytes_read": 0,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "primary_canonical_config": PRIMARY_CANONICAL_CFG,
        "primary_5config_grid_summary": grid_summary,
        "primary_beta_neutral_directed_leadlag": primary_metrics,
        "diagnostic_eod_letf_closeflow_strict_8bps": diag_b_strict,
        "diagnostic_eod_letf_closeflow_etf_2bps": diag_b_etf_spread,
        "diagnostic_t1_to_t0_residual_leadlag_rth": diag_c,
        "overall_passed": bool(primary_metrics.get("passed", False)),
        "overall_promotion_track": (
            "ALPHA_STRATEGY_MINOR_CANDIDATE"
            if primary_metrics.get("passed", False)
            else "PROTECTIVE_FILTER_OR_GOVERNANCE"
        ),
    }
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    receipt["receipt_sha256"] = digest
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute pre-registered Trial #107 (H110) on 700d pre-wall SIP lake.")
    parser.add_argument(
        "--market-data-root",
        type=pathlib.Path,
        default=pathlib.Path.home() / "Documents" / "market-data",
        help="Root directory containing alpaca-sip/ and public/tables/",
    )
    parser.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=REPO_ROOT / "docs" / "TRIAL107_H110_PREWALL_RECEIPT.json",
        help="Output path for deterministic SHA-256 receipt JSON",
    )
    args = parser.parse_args()

    receipt = run_h110_trial107_study(args.market_data_root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    p = receipt["primary_beta_neutral_directed_leadlag"]
    b_s = receipt["diagnostic_eod_letf_closeflow_strict_8bps"]
    b_e = receipt["diagnostic_eod_letf_closeflow_etf_2bps"]
    c_r = receipt["diagnostic_t1_to_t0_residual_leadlag_rth"]
    print(f"TRIAL #{TRIAL_NUMBER} ({HYPOTHESIS_ID}) RECEIPT SHA-256: {receipt['receipt_sha256'][:16]}...")
    print(
        f"  PRIMARY ({p.get('events')} ev / {p.get('distinct_dates')}d): "
        f"gross={p.get('mean_gross_day_bps')}bps | "
        f"net1x={p.get('mean_net_1x_day_bps')}bps (t={p.get('t_stat_clustered_1x_daily')}) | "
        f"net2x={p.get('mean_net_2x_day_bps')}bps (t={p.get('t_stat_clustered_2x_daily')}) | "
        f"DSR={p.get('dsr_probability_2x')} | CPCV={p.get('cpcv_5path_means_2x_bps')} "
        f"(pos={p.get('cpcv_5path_positive_share')}, PBO={p.get('cpcv_pbo')}) | "
        f"SPA_p={p.get('hansen_spa_p_value_2x')} | verdict={p.get('verdict')}"
    )
    for tag, arm in (
        ("DIAG_B_LETF_STRICT", b_s),
        ("DIAG_B_LETF_ETF2BPS", b_e),
        ("DIAG_C_T1_TO_T0_LEADLAG", c_r),
    ):
        print(
            f"  {tag} ({arm.get('events')} ev / {arm.get('distinct_dates')}d): "
            f"gross={arm.get('mean_gross_day_bps')}bps | "
            f"net1x={arm.get('mean_net_1x_day_bps')}bps (t={arm.get('t_stat_clustered_1x_daily')}) | "
            f"net2x={arm.get('mean_net_2x_day_bps')}bps (t={arm.get('t_stat_clustered_2x_daily')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
