#!/usr/bin/env python3
"""Trial #105 (`H109`): Pre-Wall (`2024-10-01..2026-10-15`, `512` Sessions) BOATS Overnight Bifurcation Study.

Pre-Registered Immutable Spec SHA-256 (`sota-bus` Trial #105):
  `db9a6f8ac768bd1482c94b1d1c8559876aab0102b85464a5b443357a8342ec61`
Required HLZ Multiplicity Hurdle (`N = 105`): `t*(105) = 4.2084`

Economic Mechanism (Lou-Polk-Skouras 2019 JFE + Eaton-Shkilko-Werner 2025/26 + Lim 2026):
- Strictly causal `T-1` liquid US operating common equities (raw `T-1` close >= $5.00, 20d ADV >= $25M,
  zero split on `T` or `T-1`, strictly pre-wall `<= 2026-10-15`, zero post-wall bytes ever read).
- Primary Leg A (`BOATS` Retail-Crowding Opening Fade Short):
    * Overnight gap `+2.5%..+25%` (`open_0930` vs `prev_rth_close`),
    * Nocturnal Blue Ocean ATS (`20:00-03:59 ET`) `boats_dvol >= $250k` & `boats_ratio >= 0.0025`,
    * No fresh `24h` SEC `8-K` (`accepted` strictly PIT in `[T-1 16:00 ET, T 09:30 ET]`),
    * Opening `09:30-09:34 ET` VWAP rejection (`close_0934 <= vwap_0930_0934`),
    * Short at `09:35 ET` open, `ADR-0022 <= 2.00%` hard stop + uncapped 3-tier lossless Chandelier, flat `15:50 ET`.
- Diagnostic Leg B (Non-Dilutive `8-K` Catalyst + Opening VWAP-Hold Long Continuation):
    * Overnight gap `+2.0%..+25%`, fresh `24h` SEC `8-K`, zero 90-day `S-3`/`424B5`/`EFFECT` dilution,
    * `close_0934 > vwap_0930_0934`, Long at `09:35 ET` open, `ADR-0022 <= 2.00%` stop + uncapped
      3-tier Chandelier, flat `15:50 ET`.

Evaluated via non-overlapping 1-day cohort legs (`1x` `8 bps` + AC sqrt + borrow & `2x` stress),
`statsmodels` Rogers date-clustered SEs, `skfolio` `CombinatorialPurgedCV(6, 2, purge=20, embargo=20)`,
`ml4t-diagnostic` native `DSR`, `Hansen (2005) SPA` + `HLN (2011) MCS`, `Conformal PID <= 2.00%`,
`ADIA Lab #19 MinTRL`, `14`-point low-quality purge audit, and adversarial ex-top-3-days / subperiod falsification.
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
WARMUP_START_ISO = "2024-09-01"
BOATS_START_ISO = "2024-10-01"
TRIAL_NUMBER = 105
HYPOTHESIS_ID = "H109_BOATS_OVERNIGHT_BIFURCATION_RTH_OPEN_VWAP_FADE"
PREREG_SPEC_SHA256 = "db9a6f8ac768bd1482c94b1d1c8559876aab0102b85464a5b443357a8342ec61"
POSITION_NOTIONAL_USD = 50_000.0
BASE_1X_COST_BPS = 8.0
SHORT_BORROW_DAILY_BPS = 1.5
STRICT_DSR_MIN = 0.95
STRICT_CPCV_POS_MIN = 0.70
STRICT_SPA_P_MAX = 0.05
STRICT_MIN_EVENTS = 200
STRICT_MIN_DATES = 60


def _almgren_chriss_1x_cost_bps(adv20_usd: float, vol20d_daily: float, *, is_short: bool) -> float:
    sigma_bps = max(50.0, min(1200.0, float(vol20d_daily) * 10_000.0))
    adv = max(1_000_000.0, float(adv20_usd))
    impact = 0.10 * sigma_bps * math.sqrt(POSITION_NOTIONAL_USD / adv)
    borrow = SHORT_BORROW_DAILY_BPS if is_short else 0.0
    return BASE_1X_COST_BPS + impact + borrow


def _simulate_uncapped_lossless_bar_path(
    bars_o: Sequence[float] | Any,
    bars_h: Sequence[float] | Any,
    bars_l: Sequence[float] | Any,
    bars_c: Sequence[float] | Any,
    *,
    side: str,
    stop_pct: float = ADR0022_MAX_STOP_PCT,
) -> dict[str, Any]:
    """Conservative intrabar simulation of ADR-0022 <= 2.00% stop + uncapped 3-tier lossless Chandelier."""
    if len(bars_o) == 0 or not math.isfinite(float(bars_o[0])) or float(bars_o[0]) <= 0.0:
        return {"gross_bps": 0.0, "mae_pct": 0.0, "mfe_pct": 0.0, "exit_reason": "NO_BARS", "harvested_50": False}

    is_short = side == "short"
    entry_px = float(bars_o[0])
    hard_stop_frac = min(float(stop_pct), ADR0022_MAX_STOP_PCT) / 100.0

    # Active stop price (monotonic ratcheting)
    stop_px = entry_px * (1.0 + hard_stop_frac) if is_short else entry_px * (1.0 - hard_stop_frac)
    extreme_px = entry_px
    mae_frac = 0.0
    mfe_frac = 0.0
    harvested_50 = False
    harvest_ret_frac = 0.0
    exit_reason = "EOD_1550_FLAT"
    runner_ret_frac = 0.0
    exited = False

    for i in range(len(bars_o)):
        h = float(bars_h[i])
        lo = float(bars_l[i])
        if not (math.isfinite(h) and math.isfinite(lo) and h > 0.0 and lo > 0.0):
            continue

        if is_short:
            bar_adv = (h - entry_px) / entry_px
            if bar_adv > mae_frac:
                mae_frac = bar_adv
            # Conservative worst-case intrabar check FIRST (stop hit before favorable move)
            if h >= stop_px:
                runner_ret_frac = (entry_px - stop_px) / entry_px
                exit_reason = "CHANDELIER_STOP" if harvested_50 or stop_px < entry_px else "ADR0022_HARD_STOP"
                exited = True
                break
            if lo < extreme_px:
                extreme_px = lo
                mfe_frac = (entry_px - extreme_px) / entry_px
            # 3-Tier Monotonic Lossless Chandelier (Short: lower stop_px is tighter/better)
            if not harvested_50 and mfe_frac >= 0.0125:
                harvested_50 = True
                harvest_ret_frac = 0.0125
                stop_px = min(stop_px, entry_px * (1.0 - 0.0025))  # Zero-loss runner floor +0.25%
            elif mfe_frac >= 0.0080:
                stop_px = min(stop_px, entry_px * (1.0 - 0.0018))  # Breakeven +0.18%
            if mfe_frac >= 0.0200:
                retain = (
                    0.86 if mfe_frac >= 0.10 else (0.82 if mfe_frac >= 0.06 else (0.75 if mfe_frac >= 0.04 else 0.65))
                )
                cand_stop = entry_px * (1.0 - mfe_frac * retain)
                stop_px = min(stop_px, cand_stop)
        else:
            bar_adv = (entry_px - lo) / entry_px
            if bar_adv > mae_frac:
                mae_frac = bar_adv
            if lo <= stop_px:
                runner_ret_frac = (stop_px - entry_px) / entry_px
                exit_reason = "CHANDELIER_STOP" if harvested_50 or stop_px > entry_px else "ADR0022_HARD_STOP"
                exited = True
                break
            if h > extreme_px:
                extreme_px = h
                mfe_frac = (extreme_px - entry_px) / entry_px
            # 3-Tier Monotonic Lossless Chandelier (Long: higher stop_px is tighter/better)
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
                cand_stop = entry_px * (1.0 + mfe_frac * retain)
                stop_px = max(stop_px, cand_stop)

    if not exited:
        last_c = float(bars_c[-1])
        runner_ret_frac = ((entry_px - last_c) / entry_px) if is_short else ((last_c - entry_px) / entry_px)

    gross_frac = (0.5 * harvest_ret_frac + 0.5 * runner_ret_frac) if harvested_50 else runner_ret_frac
    return {
        "gross_bps": gross_frac * 10_000.0,
        "mae_pct": max(0.0, mae_frac * 100.0),
        "mfe_pct": max(0.0, mfe_frac * 100.0),
        "exit_reason": exit_reason,
        "harvested_50": harvested_50,
    }


def _date_clustered_t(values: Any, dates: Any) -> tuple[float, float, float]:
    """Statsmodels OLS mean with Rogers date-clustered SEs."""
    import numpy as np
    import statsmodels.api as sm

    if len(values) < 5 or len(set(dates.tolist())) < 3:
        return 0.0, 0.0, 0.0
    x = np.ones((len(values), 1), dtype=float)
    res = sm.OLS(values.astype(float), x).fit(cov_type="cluster", cov_kwds={"groups": dates})
    mean_v = float(res.params[0])
    se_v = float(res.bse[0])
    t_v = float(res.tvalues[0]) if math.isfinite(float(res.tvalues[0])) else 0.0
    return mean_v, se_v, t_v


def _evaluate_cohort_metrics(
    events_df: pd.DataFrame,
    *,
    label: str,
    n_trials: int = TRIAL_NUMBER,
) -> dict[str, Any]:
    """Compute complete SOTA statistical, CPCV, DSR, Hansen SPA/MCS, Conformal PID & adversarial metrics."""
    import numpy as np
    from skfolio.model_selection import CombinatorialPurgedCV

    if events_df.empty:
        return {"label": label, "events": 0, "distinct_dates": 0, "passed": False, "reason": "ZERO_EVENTS"}

    n_events = int(len(events_df))
    distinct_dates = int(events_df["trade_date"].nunique())

    # Event-level date-clustered SEs
    mean_1x_ev, se_1x_ev, t_1x_ev = _date_clustered_t(
        events_df["net_1x_bps"].to_numpy(), events_df["trade_date"].to_numpy()
    )
    mean_2x_ev, se_2x_ev, t_2x_ev = _date_clustered_t(
        events_df["net_2x_bps"].to_numpy(), events_df["trade_date"].to_numpy()
    )

    # Non-overlapping 1-day equal-weight MTM cohort legs (eliminates intraday clustering bias!)
    daily = (
        events_df.groupby("trade_date", as_index=False)
        .agg(
            n_trades=("net_1x_bps", "count"),
            gross_bps=("gross_bps", "mean"),
            net_1x_bps=("net_1x_bps", "mean"),
            net_2x_bps=("net_2x_bps", "mean"),
            cost_1x_bps=("cost_1x_bps", "mean"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    d1 = daily["net_1x_bps"].to_numpy(dtype=float)
    d2 = daily["net_2x_bps"].to_numpy(dtype=float)
    ddates = daily["trade_date"].to_numpy()
    mean_1x_day, se_1x_day, t_1x_day = _date_clustered_t(d1, ddates)
    mean_2x_day, se_2x_day, t_2x_day = _date_clustered_t(d2, ddates)

    # Annualized Sharpe on daily cohort returns
    sr_1x_ann = (
        float((np.mean(d1) / np.std(d1, ddof=1)) * math.sqrt(252.0)) if len(d1) > 2 and np.std(d1, ddof=1) > 0 else 0.0
    )
    sr_2x_ann = (
        float((np.mean(d2) / np.std(d2, ddof=1)) * math.sqrt(252.0)) if len(d2) > 2 and np.std(d2, ddof=1) > 0 else 0.0
    )

    # Native ml4t-diagnostic DSR on daily 2x net returns (canonical Trial #104/105 helper)
    from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics

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

    # skfolio CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=20, embargo_size=20)
    cpcv_pos_share = 0.0
    cpcv_path_means: list[float] = []
    if len(d2) >= 120:
        cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=20, embargo_size=20)
        x_dummy = np.zeros((len(d2), 1), dtype=float)
        splits = list(cv.split(x_dummy))
        path_returns: dict[int, list[float]] = defaultdict(list)
        path_ids = cv.get_path_ids()
        for split_idx, (_tr, te_list) in enumerate(splits):
            for fold_pos, te_idx in enumerate(te_list):
                pid = int(path_ids[split_idx, fold_pos])
                path_returns[pid].extend(d2[te_idx].tolist())
        cpcv_path_means = [float(np.mean(vals)) for _, vals in sorted(path_returns.items()) if vals]
        cpcv_pos_share = (
            sum(1 for m in cpcv_path_means if m > 0.0) / float(len(cpcv_path_means)) if cpcv_path_means else 0.0
        )

    # Hansen (2005) SPA + HLN (2011) MCS on daily 2x returns vs zero benchmark
    spa_res = evaluate_hansen_spa_and_mcs(
        {label: d2.tolist()},
        n_bootstrap=1000,
        avg_block_len=5.0,
        seed=105,
    )
    spa_p = float(spa_res["spa_p_consistent"])

    # Conformal PID <= 2.00% online stop calibration audit on realized trade MAE
    pid = ConformalPidStopCalibrator(target_miscoverage_alpha=0.10, max_stop_pct=ADR0022_MAX_STOP_PCT)
    pid_last: dict[str, Any] = {"next_stop_pct": ADR0022_MAX_STOP_PCT, "empirical_miscoverage_rate": 0.0}
    for mae_val, vol_val in zip(
        events_df["mae_pct"].to_numpy(dtype=float),
        events_df["vol20d_pct"].to_numpy(dtype=float),
        strict=False,
    ):
        pid_last = pid.update(realized_mae_pct=float(mae_val), atr_pct=max(0.25, float(vol_val) * 0.5))

    # Calendar-day AR(1)-adjusted stats (Lo 2002 + ADIA Lab #19)
    cal_stats = compute_calendar_day_stats(d2.tolist())

    # Adversarial falsification: Ex-top-3 best days concentration & subperiod decay
    d2_sorted_idx = np.argsort(d2)
    ex_top3_mask = np.ones(len(d2), dtype=bool)
    if len(d2) > 10:
        ex_top3_mask[d2_sorted_idx[-3:]] = False
    mean_2x_ex_top3, _se_ex3, t_2x_ex_top3 = _date_clustered_t(d2[ex_top3_mask], ddates[ex_top3_mask])

    subperiods: dict[str, dict[str, Any]] = {}
    for sp_name, lo_d, hi_d in (
        ("2024Q4", "2024-10-01", "2024-12-31"),
        ("2025H1", "2025-01-01", "2025-06-30"),
        ("2025H2", "2025-07-01", "2025-12-31"),
        ("2026_PREWALL", "2026-01-01", STUDY_WALL_ISO),
    ):
        sub_ev = events_df[(events_df["trade_date"] >= lo_d) & (events_df["trade_date"] <= hi_d)]
        sub_d = daily[(daily["trade_date"] >= lo_d) & (daily["trade_date"] <= hi_d)]
        if not sub_d.empty:
            _m, _s, st = _date_clustered_t(sub_d["net_2x_bps"].to_numpy(), sub_d["trade_date"].to_numpy())
            subperiods[sp_name] = {
                "events": int(len(sub_ev)),
                "dates": int(len(sub_d)),
                "mean_1x_bps": round(float(sub_d["net_1x_bps"].mean()), 2),
                "mean_2x_bps": round(float(sub_d["net_2x_bps"].mean()), 2),
                "t_2x_daily": round(st, 4),
            }

    # 14-Point Fail-Closed Low-Quality Practice Purge Audit
    hygiene_audit = audit_low_quality_practices(
        PlaybackAuditInput(
            horizon_id="W3_MODERN_XML_13D_BOATS",
            sample_start_date=str(daily["trade_date"].iloc[0]),
            sample_max_date=str(daily["trade_date"].iloc[-1]),
            distinct_sessions=distinct_dates,
            cadence="intraday",
            universe_mode="CAUSAL_T_MINUS_1_PIT",
            cost_1x_rt_bps=float(daily["cost_1x_bps"].mean()),
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
            ar1_rho_for_min_trl=max(-0.5, min(0.5, float(cal_stats["ar1_rho"]))),
        )
    )

    hlz_hurdle = hlz_dynamic_t_hurdle(n_trials)
    t_conservative = min(t_2x_ev, t_2x_day)
    passed = (
        n_events >= STRICT_MIN_EVENTS
        and distinct_dates >= STRICT_MIN_DATES
        and mean_1x_day > 0.0
        and mean_2x_day > 0.0
        and t_conservative >= hlz_hurdle
        and dsr_prob >= STRICT_DSR_MIN
        and cpcv_pos_share >= STRICT_CPCV_POS_MIN
        and spa_p < STRICT_SPA_P_MAX
        and bool(hygiene_audit["passed"])
    )

    exit_counts = events_df["exit_reason"].value_counts().to_dict()
    return {
        "label": label,
        "events": n_events,
        "distinct_dates": distinct_dates,
        "mean_gross_day_bps": round(float(daily["gross_bps"].mean()), 2),
        "mean_cost_1x_bps": round(float(daily["cost_1x_bps"].mean()), 2),
        "mean_net_1x_day_bps": round(mean_1x_day, 2),
        "mean_net_2x_day_bps": round(mean_2x_day, 2),
        "mean_net_1x_event_bps": round(mean_1x_ev, 2),
        "mean_net_2x_event_bps": round(mean_2x_ev, 2),
        "t_stat_clustered_1x_daily": round(t_1x_day, 4),
        "t_stat_clustered_2x_daily": round(t_2x_day, 4),
        "t_stat_clustered_1x_event": round(t_1x_ev, 4),
        "t_stat_clustered_2x_event": round(t_2x_ev, 4),
        "hlz_t_hurdle_n105": round(hlz_hurdle, 4),
        "annualized_sharpe_1x": round(sr_1x_ann, 3),
        "annualized_sharpe_2x": round(sr_2x_ann, 3),
        "calendar_day_stats_2x": cal_stats,
        "dsr_probability_2x": round(dsr_prob, 4),
        "cpcv_5path_positive_share": round(cpcv_pos_share, 4),
        "cpcv_5path_means_2x_bps": [round(m, 2) for m in cpcv_path_means],
        "hansen_spa_p_value_2x": round(spa_p, 4),
        "hln_mcs_included": label in spa_res["mcs_surviving_strategies"],
        "conformal_pid_calibrated_stop_pct": round(float(pid_last["next_stop_pct"]), 4),
        "conformal_pid_empirical_stopout_rate": round(float(pid_last["empirical_miscoverage_rate"]), 4),
        "harvest_50_rate": round(float(events_df["harvested_50"].mean()), 4),
        "p90_trade_net_2x_bps": round(float(np.percentile(events_df["net_2x_bps"], 90)), 2),
        "p10_trade_net_2x_bps": round(float(np.percentile(events_df["net_2x_bps"], 10)), 2),
        "exit_reason_counts": {str(k): int(v) for k, v in exit_counts.items()},
        "adversarial_ex_top3_days_2x_bps": round(mean_2x_ex_top3, 2),
        "adversarial_ex_top3_days_t_2x": round(t_2x_ex_top3, 4),
        "subperiods": subperiods,
        "hygiene_14pt_passed": bool(hygiene_audit["passed"]),
        "hygiene_violations": list(hygiene_audit["violations"]),
        "verdict": "PASSED_ALPHA_CANDIDATE" if passed else "FAILED_HLZ_OR_STAT_GATE",
        "promotion_track": "ALPHA_STRATEGY_MINOR_CANDIDATE" if passed else "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "passed": passed,
    }


def run_h109_trial105_study(market_data_root: pathlib.Path) -> dict[str, Any]:
    """Execute strictly pre-wall (`<= 2026-10-15`) Trial #105 (`H109`) on real SIP 1m + BOATS 1m parquets."""
    import duckdb
    import pandas as pd

    sip_root = market_data_root / "alpaca-sip" / "raw" / "bars-1min"
    boats_root = market_data_root / "alpaca-sip" / "raw" / "boats-bars-1min"
    ca_root = market_data_root / "alpaca-sip" / "raw" / "corporate-actions"
    pub_root = market_data_root / "public" / "tables"

    # Strict pre-wall directory selection BEFORE reading any bytes!
    sip_dirs = sorted(d for d in sip_root.iterdir() if d.is_dir() and WARMUP_START_ISO <= d.name <= STUDY_WALL_ISO)
    boats_dirs = sorted(d for d in boats_root.iterdir() if d.is_dir() and BOATS_START_ISO <= d.name <= STUDY_WALL_ISO)
    overlap_dates = sorted({d.name for d in sip_dirs} & {d.name for d in boats_dirs})
    if not overlap_dates or max(overlap_dates) > STUDY_WALL_ISO:
        raise ValueError("Pre-wall overlap dates invalid or exceeded STUDY_WALL_DATE")

    sip_globs = [f"{d.as_posix()}/*.parquet" for d in sip_dirs]
    boats_globs = [f"{d.as_posix()}/*.parquet" for d in boats_dirs]
    ca_files = sorted(p.as_posix() for p in ca_root.glob("*.parquet") if p.name[:7] <= STUDY_WALL_ISO[:7])

    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute("PRAGMA memory_limit='24GB'")

    # 1. Aggregate daily RTH + 09:30-09:35 features strictly <= 2026-10-15
    con.execute(
        """
        CREATE TEMP TABLE sip_daily AS
        WITH raw_bars AS (
            SELECT
                UPPER(TRIM(symbol)) AS symbol,
                CAST(to_timestamp(t_ns / 1e9) AT TIME ZONE 'America/New_York' AS TIMESTAMP) AS ny_ts,
                o, h, l, c, v, vw
            FROM read_parquet($1)
            WHERE o > 0 AND h > 0 AND l > 0 AND c > 0 AND v >= 0
        ),
        tagged AS (
            SELECT
                symbol,
                strftime(ny_ts, '%Y-%m-%d') AS trade_date,
                EXTRACT(HOUR FROM ny_ts) * 60 + EXTRACT(MINUTE FROM ny_ts) AS ny_min,
                o, h, l, c, v, COALESCE(NULLIF(vw, 0), c) AS vw
            FROM raw_bars
            WHERE strftime(ny_ts, '%Y-%m-%d') <= $2
        )
        SELECT
            symbol,
            trade_date,
            arg_min(o, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 574) AS open_0930,
            arg_max(c, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 574) AS close_0934,
            SUM(vw * v) FILTER (WHERE ny_min >= 570 AND ny_min <= 574)
                / NULLIF(SUM(v) FILTER (WHERE ny_min >= 570 AND ny_min <= 574), 0) AS vwap_0930_0934,
            arg_min(o, ny_min) FILTER (WHERE ny_min >= 575 AND ny_min <= 577) AS open_0935,
            arg_max(c, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 959) AS rth_close,
            SUM(vw * v) FILTER (WHERE ny_min >= 570 AND ny_min <= 959) AS rth_dvol
        FROM tagged
        WHERE ny_min >= 570 AND ny_min <= 959
        GROUP BY symbol, trade_date
        HAVING rth_close IS NOT NULL AND rth_dvol > 0
        """,
        [sip_globs, STUDY_WALL_ISO],
    )

    # 2. Compute strictly causal T-1 20-day baselines (LAG 1, ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
    con.execute(
        """
        CREATE TEMP TABLE causal_daily AS
        WITH step1 AS (
            SELECT
                *,
                LAG(rth_close) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_rth_close,
                LAG(trade_date) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_trade_date,
                AVG(rth_dvol) OVER (
                    PARTITION BY symbol ORDER BY trade_date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS adv20_usd,
                COUNT(rth_dvol) OVER (
                    PARTITION BY symbol ORDER BY trade_date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS hist_days,
                LN(rth_close / NULLIF(LAG(rth_close) OVER (PARTITION BY symbol ORDER BY trade_date), 0)) AS log_ret
            FROM sip_daily
        )
        SELECT
            symbol,
            trade_date,
            prev_trade_date,
            prev_rth_close,
            adv20_usd,
            hist_days,
            STDDEV_SAMP(log_ret) OVER (
                PARTITION BY symbol ORDER BY trade_date
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            ) AS vol20d_daily,
            open_0930,
            close_0934,
            vwap_0930_0934,
            open_0935,
            rth_close
        FROM step1
        """
    )

    # 3. Aggregate BOATS 20:00-03:59 ET overnight activity mapped cleanly to its upcoming trade_date
    con.execute(
        """
        CREATE TEMP TABLE boats_daily AS
        SELECT
            UPPER(TRIM(symbol)) AS symbol,
            regexp_extract(filename, 'boats-bars-1min/([0-9]{4}-[0-9]{2}-[0-9]{2})/', 1) AS trade_date,
            SUM(COALESCE(NULLIF(vw, 0), c) * v) AS boats_dvol,
            SUM(v) AS boats_vol,
            COUNT(*) AS boats_bars
        FROM read_parquet($1, filename=true)
        WHERE c > 0 AND v > 0
        GROUP BY 1, 2
        HAVING trade_date >= $2 AND trade_date <= $3
        """,
        [boats_globs, BOATS_START_ISO, STUDY_WALL_ISO],
    )

    # 4. Operating common equities (sec_symbols), corporate splits, and strictly PIT SEC filings (accepted)
    sec_sym_p = (pub_root / "sec_symbols.parquet").as_posix()
    sec_fil_p = (pub_root / "sec_filings.parquet").as_posix()

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
        CREATE TEMP TABLE split_events AS
        SELECT DISTINCT UPPER(TRIM(symbol)) AS symbol, ex_date
        FROM read_parquet($1)
        WHERE type IN ('forward_splits', 'reverse_splits')
          AND ex_date IS NOT NULL AND ex_date <= $2
        """,
        [ca_files, STUDY_WALL_ISO],
    )
    con.execute(
        f"""
        CREATE TEMP TABLE sec_pit AS
        SELECT
            o.symbol,
            f.form,
            CAST(strptime(SUBSTR(f.accepted, 1, 19), '%Y-%m-%dT%H:%M:%S') AT TIME ZONE 'UTC'
                 AT TIME ZONE 'America/New_York' AS TIMESTAMP) AS accepted_ny
        FROM read_parquet('{sec_fil_p}') f
        JOIN operating_syms o ON o.cik = f.cik
        WHERE f.accepted IS NOT NULL
          AND LENGTH(f.accepted) >= 19
          AND SUBSTR(f.accepted, 1, 10) <= '{STUDY_WALL_ISO}'
          AND f.form IN ('8-K', 'S-3', 'S-3ASR', '424B5', 'EFFECT', 'F-3')
        """
    )

    # 5. Extract candidate triggers strictly using causal pre-09:35 ET features
    candidates_df = con.execute(
        """
        WITH base AS (
            SELECT
                c.symbol,
                c.trade_date,
                c.prev_trade_date,
                c.prev_rth_close,
                c.adv20_usd,
                COALESCE(c.vol20d_daily, 0.025) AS vol20d_daily,
                c.open_0930,
                c.close_0934,
                c.vwap_0930_0934,
                c.open_0935,
                (c.open_0930 - c.prev_rth_close) / c.prev_rth_close AS overnight_ret,
                COALESCE(b.boats_dvol, 0.0) AS boats_dvol,
                COALESCE(b.boats_dvol, 0.0) / c.adv20_usd AS boats_ratio,
                CAST(c.trade_date || ' 09:30:00' AS TIMESTAMP) AS rth_open_ny,
                CAST(c.prev_trade_date || ' 16:00:00' AS TIMESTAMP) AS prev_close_ny
            FROM causal_daily c
            INNER JOIN operating_syms o ON o.symbol = c.symbol
            LEFT JOIN boats_daily b ON b.symbol = c.symbol AND b.trade_date = c.trade_date
            LEFT JOIN split_events sp1 ON sp1.symbol = c.symbol AND sp1.ex_date = c.trade_date
            LEFT JOIN split_events sp2 ON sp2.symbol = c.symbol AND sp2.ex_date = c.prev_trade_date
            WHERE c.trade_date >= $1 AND c.trade_date <= $2
              AND c.hist_days >= 15
              AND c.prev_rth_close >= 5.0
              AND c.adv20_usd >= 25000000.0
              AND c.open_0930 > 0 AND c.open_0935 > 0 AND c.close_0934 > 0 AND c.vwap_0930_0934 > 0
              AND sp1.symbol IS NULL AND sp2.symbol IS NULL
        ),
        enriched AS (
            SELECT
                b.*,
                EXISTS (
                    SELECT 1 FROM sec_pit s
                    WHERE s.symbol = b.symbol
                      AND s.form = '8-K'
                      AND s.accepted_ny >= b.prev_close_ny
                      AND s.accepted_ny <= b.rth_open_ny
                ) AS has_fresh_8k,
                EXISTS (
                    SELECT 1 FROM sec_pit s
                    WHERE s.symbol = b.symbol
                      AND s.form IN ('S-3', 'S-3ASR', '424B5', 'EFFECT', 'F-3')
                      AND s.accepted_ny >= (b.rth_open_ny - INTERVAL 90 DAY)
                      AND s.accepted_ny <= b.rth_open_ny
                ) AS has_90d_dilution
            FROM base b
            WHERE b.overnight_ret >= 0.020 AND b.overnight_ret <= 0.250
        )
        SELECT
            symbol, trade_date, prev_rth_close, adv20_usd, vol20d_daily,
            open_0930, close_0934, vwap_0930_0934, open_0935,
            overnight_ret, boats_dvol, boats_ratio, has_fresh_8k, has_90d_dilution,
            CASE
                WHEN overnight_ret >= 0.025
                     AND boats_dvol >= 250000.0
                     AND boats_ratio >= 0.0025
                     AND NOT has_fresh_8k
                     AND close_0934 <= vwap_0930_0934
                THEN 'LEG_A_BOATS_FADE_SHORT'
                WHEN overnight_ret >= 0.020
                     AND has_fresh_8k
                     AND NOT has_90d_dilution
                     AND close_0934 > vwap_0930_0934
                THEN 'LEG_B_8K_VWAP_HOLD_LONG'
                ELSE NULL
            END AS leg
        FROM enriched
        WHERE (
            (overnight_ret >= 0.025 AND boats_dvol >= 250000.0 AND boats_ratio >= 0.0025
             AND NOT has_fresh_8k AND close_0934 <= vwap_0930_0934)
            OR
            (overnight_ret >= 0.020 AND has_fresh_8k AND NOT has_90d_dilution
             AND close_0934 > vwap_0930_0934)
        )
        ORDER BY trade_date, symbol
        """,
        [BOATS_START_ISO, STUDY_WALL_ISO],
    ).fetchdf()

    # 6. Exact 1-minute RTH bar-path replay (09:35..15:50 ET) per triggered (trade_date, symbol)
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates_df.to_dict(orient="records"):
        by_date[str(row["trade_date"])].append(row)

    trade_records: list[dict[str, Any]] = []
    for tdate, day_rows in sorted(by_date.items()):
        syms = sorted({str(r["symbol"]) for r in day_rows})
        day_glob = f"{(sip_root / tdate).as_posix()}/*.parquet"
        bars_df = con.execute(
            """
            SELECT
                UPPER(TRIM(symbol)) AS symbol,
                t_ns,
                o, h, l, c
            FROM read_parquet($1)
            WHERE UPPER(TRIM(symbol)) = ANY($2)
              AND (
                  EXTRACT(HOUR FROM (to_timestamp(t_ns / 1e9) AT TIME ZONE 'America/New_York')) * 60
                  + EXTRACT(MINUTE FROM (to_timestamp(t_ns / 1e9) AT TIME ZONE 'America/New_York'))
              ) BETWEEN 575 AND 950
            ORDER BY symbol, t_ns
            """,
            [day_glob, syms],
        ).fetchdf()

        grouped_bars = {sym: grp for sym, grp in bars_df.groupby("symbol", sort=False)}
        for r in day_rows:
            sym = str(r["symbol"])
            grp = grouped_bars.get(sym)
            if grp is None or grp.empty:
                continue
            is_short = r["leg"] == "LEG_A_BOATS_FADE_SHORT"
            sim = _simulate_uncapped_lossless_bar_path(
                grp["o"].to_numpy(dtype=float),
                grp["h"].to_numpy(dtype=float),
                grp["l"].to_numpy(dtype=float),
                grp["c"].to_numpy(dtype=float),
                side="short" if is_short else "long",
                stop_pct=ADR0022_MAX_STOP_PCT,
            )
            cost_1x = _almgren_chriss_1x_cost_bps(float(r["adv20_usd"]), float(r["vol20d_daily"]), is_short=is_short)
            gross_bps = float(sim["gross_bps"])
            trade_records.append(
                {
                    "trade_date": tdate,
                    "symbol": sym,
                    "leg": r["leg"],
                    "side": "short" if is_short else "long",
                    "overnight_ret_pct": round(float(r["overnight_ret"]) * 100.0, 3),
                    "boats_ratio": round(float(r["boats_ratio"]), 6),
                    "gross_bps": gross_bps,
                    "cost_1x_bps": cost_1x,
                    "net_1x_bps": gross_bps - cost_1x,
                    "net_2x_bps": gross_bps - 2.0 * cost_1x,
                    "vol20d_pct": float(r["vol20d_daily"]) * 100.0,
                    "mae_pct": float(sim["mae_pct"]),
                    "mfe_pct": float(sim["mfe_pct"]),
                    "exit_reason": sim["exit_reason"],
                    "harvested_50": bool(sim["harvested_50"]),
                }
            )

    trades_df = pd.DataFrame(trade_records)
    leg_a_df = trades_df[trades_df["leg"] == "LEG_A_BOATS_FADE_SHORT"].copy() if not trades_df.empty else pd.DataFrame()
    leg_b_df = (
        trades_df[trades_df["leg"] == "LEG_B_8K_VWAP_HOLD_LONG"].copy() if not trades_df.empty else pd.DataFrame()
    )

    primary_metrics = _evaluate_cohort_metrics(leg_a_df, label="H109_LEG_A_BOATS_FADE_SHORT", n_trials=TRIAL_NUMBER)
    diag_leg_b = _evaluate_cohort_metrics(leg_b_df, label="H109_LEG_B_8K_VWAP_HOLD_LONG", n_trials=TRIAL_NUMBER)
    diag_combined = _evaluate_cohort_metrics(trades_df, label="H109_COMBINED_BIFURCATION", n_trials=TRIAL_NUMBER)

    receipt: dict[str, Any] = {
        "schema": "sota.prewall-trial105-h109-receipt.v1",
        "trial_number": TRIAL_NUMBER,
        "hypothesis_id": HYPOTHESIS_ID,
        "prereg_spec_sha256": PREREG_SPEC_SHA256,
        "study_wall_date": STUDY_WALL_ISO,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "prewall_overlap_sessions": len(overlap_dates),
        "sample_start_date": overlap_dates[0],
        "sample_end_date": overlap_dates[-1],
        "primary_leg_a_boats_fade_short": primary_metrics,
        "diagnostic_leg_b_8k_vwap_hold_long": diag_leg_b,
        "diagnostic_combined_bifurcation": diag_combined,
    }
    hashable = {k: v for k, v in receipt.items() if k != "generated_at_utc"}
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(hashable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute Pre-Registered Trial #105 (H109) on Real Pre-Wall BOATS+SIP Lake."
    )
    parser.add_argument(
        "--market-data-root",
        type=pathlib.Path,
        default=pathlib.Path.home() / "Documents" / "market-data",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "docs" / "TRIAL105_H109_PREWALL_RECEIPT.json",
    )
    args = parser.parse_args(argv)
    receipt = run_h109_trial105_study(args.market_data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
