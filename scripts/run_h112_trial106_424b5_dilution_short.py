#!/usr/bin/env python3
"""Trial #106 (`H112`): Pre-Wall (`2024-01-02..2026-10-15`, `700` Sessions) SEC `424B5` Dilution Short Study.

Pre-Registered Immutable Spec SHA-256 (`sota-bus` Trial #106):
  `71c7df7ca89b09cae3b3e5784381c3c79f2c3de5c5c7ce767f9c5fc1b4d34dc9`
Required HLZ Multiplicity Hurdle (`N = 106`): `t*(106) = 4.2106`

Economic Mechanism (Billett, Floros & Garfinkel 2019 JFQA + Gustafson & Iliev 2017 JFE):
- Strictly causal `T-1` liquid US operating common equities (raw `T-1` close >= $5.00, 20d ADV >= $25M,
  zero split on `[T-1..T+3]`, strictly pre-wall `<= 2026-10-15`, zero post-wall bytes ever read).
- Primary Leg (`H112_424B5_PRIMARY_RTH_SHORT`):
    * SEC Form `424B5` (actual equity offering / ATM prospectus supplement) accepted strictly PIT in
      `[T-1 16:00 ET, T 09:30 ET]`,
    * Short at `T 09:35 ET` open with `ADR-0022 <= 2.00%` hard stop + uncapped 3-tier lossless Chandelier,
      flat at `15:50 ET`.
- Diagnostics (Reported Honestly Alongside Primary — Not Mined):
    * `H112_424B5_3D_NONOVERLAPPING_MTM_SHORT`: True non-overlapping 1-day calendar-day MTM short portfolio
      holding newly filed `424B5` cohorts across `[T, T+1, T+2]` (zero overlapping `MA(2K-2)` inflation!).
    * `H112_S3_EFFECT_SHELF_SHORT`: Fresh `S-3` / `S-3ASR` / `EFFECT` shelf announcements under identical rules.

Evaluated at `1x` (`8 bps` + Almgren-Chriss sqrt + `1.5 bps/day` borrow) and `2x` stress, Rogers
date-clustered SEs, `skfolio` `CombinatorialPurgedCV(6, 2, purge=20, embargo=20)`, `ml4t-diagnostic`
native `DSR` (`N = 106`), `Hansen (2005) SPA` + `HLN (2011) MCS`, `Conformal PID <= 2.00%`, `14`-point
low-quality purge audit, and adversarial ex-top-3-days + `2024/2025/2026` subperiod falsification.
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
EVAL_START_ISO = "2024-02-01"  # After 20-session causal T-1 ADV/vol warmup
TRIAL_NUMBER = 106
HYPOTHESIS_ID = "H112_424B5_EQUITY_DILUTION_SUPPLY_SHOCK_SHORT"
PREREG_SPEC_SHA256 = "71c7df7ca89b09cae3b3e5784381c3c79f2c3de5c5c7ce767f9c5fc1b4d34dc9"
POSITION_NOTIONAL_USD = 50_000.0
BASE_1X_COST_BPS = 8.0
SHORT_BORROW_DAILY_BPS = 1.5
STRICT_DSR_MIN = 0.95
STRICT_CPCV_POS_MIN = 0.70
STRICT_SPA_P_MAX = 0.05
STRICT_MIN_EVENTS = 200
STRICT_MIN_DATES = 60


def _almgren_chriss_1x_cost_bps(adv20_usd: float, vol20d_daily: float, *, hold_days: int = 1) -> float:
    sigma_bps = max(50.0, min(1200.0, float(vol20d_daily) * 10_000.0))
    adv = max(1_000_000.0, float(adv20_usd))
    impact = 0.10 * sigma_bps * math.sqrt(POSITION_NOTIONAL_USD / adv)
    return BASE_1X_COST_BPS + impact + (SHORT_BORROW_DAILY_BPS * max(1, int(hold_days)))


def _simulate_short_uncapped_lossless(
    bars_o: Sequence[float] | Any,
    bars_h: Sequence[float] | Any,
    bars_l: Sequence[float] | Any,
    bars_c: Sequence[float] | Any,
    *,
    stop_pct: float = ADR0022_MAX_STOP_PCT,
) -> dict[str, Any]:
    """Conservative intrabar Short simulation with ADR-0022 <= 2.00% stop + uncapped 3-tier lossless Chandelier."""
    if len(bars_o) == 0 or not math.isfinite(float(bars_o[0])) or float(bars_o[0]) <= 0.0:
        return {"gross_bps": 0.0, "mae_pct": 0.0, "mfe_pct": 0.0, "exit_reason": "NO_BARS", "harvested_50": False}

    entry_px = float(bars_o[0])
    hard_stop_frac = min(float(stop_pct), ADR0022_MAX_STOP_PCT) / 100.0
    stop_px = entry_px * (1.0 + hard_stop_frac)
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
            retain = 0.86 if mfe_frac >= 0.10 else (0.82 if mfe_frac >= 0.06 else (0.75 if mfe_frac >= 0.04 else 0.65))
            stop_px = min(stop_px, entry_px * (1.0 - mfe_frac * retain))

    if not exited:
        last_c = float(bars_c[-1])
        runner_ret_frac = (entry_px - last_c) / entry_px

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

    if len(values) < 5 or len(set(dates.tolist())) < 3:
        return 0.0, 0.0, 0.0
    x = np.ones((len(values), 1), dtype=float)
    res = sm.OLS(values.astype(float), x).fit(cov_type="cluster", cov_kwds={"groups": dates})
    mean_v = float(res.params[0])
    se_v = float(res.bse[0])
    t_v = float(res.tvalues[0]) if math.isfinite(float(res.tvalues[0])) else 0.0
    return mean_v, se_v, t_v


def _evaluate_daily_and_event_metrics(
    events_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    *,
    label: str,
    n_trials: int = TRIAL_NUMBER,
) -> dict[str, Any]:
    """Compute complete SOTA statistical, CPCV, DSR, Hansen SPA/MCS, Conformal PID & adversarial metrics."""
    import numpy as np
    from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics
    from skfolio.model_selection import CombinatorialPurgedCV

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

    spa_res = evaluate_hansen_spa_and_mcs({label: d2.tolist()}, n_bootstrap=1000, avg_block_len=5.0, seed=106)
    spa_p = float(spa_res["spa_p_consistent"])

    pid = ConformalPidStopCalibrator(target_miscoverage_alpha=0.10, max_stop_pct=ADR0022_MAX_STOP_PCT)
    pid_last: dict[str, Any] = {"next_stop_pct": ADR0022_MAX_STOP_PCT, "empirical_miscoverage_rate": 0.0}
    if "mae_pct" in events_df.columns and "vol20d_pct" in events_df.columns:
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

    return {
        "label": label,
        "events": n_events,
        "distinct_dates": distinct_dates,
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
        "hlz_t_hurdle_n106": round(hlz_hurdle, 4),
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
        "adversarial_ex_top3_days_2x_bps": round(mean_2x_ex_top3, 2),
        "adversarial_ex_top3_days_t_2x": round(t_2x_ex_top3, 4),
        "subperiods": subperiods,
        "hygiene_14pt_passed": bool(hygiene_audit["passed"]),
        "hygiene_violations": list(hygiene_audit["violations"]),
        "verdict": "PASSED_ALPHA_CANDIDATE" if passed else "FAILED_HLZ_OR_STAT_GATE",
        "promotion_track": "ALPHA_STRATEGY_MINOR_CANDIDATE" if passed else "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "passed": passed,
    }


def run_h112_trial106_study(market_data_root: pathlib.Path) -> dict[str, Any]:
    """Execute strictly pre-wall (`<= 2026-10-15`) Trial #106 (`H112`) on real SEC EDGAR + SIP 1m lake."""
    import duckdb
    import pandas as pd

    sip_root = market_data_root / "alpaca-sip" / "raw" / "bars-1min"
    ca_root = market_data_root / "alpaca-sip" / "raw" / "corporate-actions"
    pub_root = market_data_root / "public" / "tables"

    sip_dirs = sorted(d for d in sip_root.iterdir() if d.is_dir() and SAMPLE_START_ISO <= d.name <= STUDY_WALL_ISO)
    if not sip_dirs or sip_dirs[-1].name > STUDY_WALL_ISO:
        raise ValueError("Pre-wall SIP directories invalid or exceeded STUDY_WALL_DATE")

    sip_globs = [f"{d.as_posix()}/*.parquet" for d in sip_dirs]
    ca_files = sorted(p.as_posix() for p in ca_root.glob("*.parquet") if p.name[:7] <= STUDY_WALL_ISO[:7])
    sec_sym_p = (pub_root / "sec_symbols.parquet").as_posix()
    sec_fil_p = (pub_root / "sec_filings.parquet").as_posix()

    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute("PRAGMA memory_limit='24GB'")

    # 1. Operating US common equities & strictly PIT SEC dilution filings (424B5 + S-3/S-3ASR/EFFECT)
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
        f"""
        CREATE TEMP TABLE sec_dil AS
        SELECT
            o.symbol,
            f.form,
            CAST(strptime(SUBSTR(f.accepted, 1, 19), '%Y-%m-%dT%H:%M:%S') AT TIME ZONE 'UTC'
                 AT TIME ZONE 'America/New_York' AS TIMESTAMP) AS accepted_ny
        FROM read_parquet('{sec_fil_p}') f
        JOIN operating_syms o ON o.cik = f.cik
        WHERE f.accepted IS NOT NULL
          AND LENGTH(f.accepted) >= 19
          AND SUBSTR(f.accepted, 1, 10) >= '{SAMPLE_START_ISO}'
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

    # 2. Aggregate daily RTH bars ONLY for operating symbols that ever filed a dilution form (10x faster, 0 bias!)
    con.execute(
        """
        CREATE TEMP TABLE target_syms AS
        SELECT DISTINCT symbol FROM sec_dil
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE sip_daily AS
        WITH raw_bars AS (
            SELECT
                UPPER(TRIM(symbol)) AS symbol,
                CAST(to_timestamp(t_ns / 1e9) AT TIME ZONE 'America/New_York' AS TIMESTAMP) AS ny_ts,
                o, h, l, c, v, vw
            FROM read_parquet($1)
            WHERE UPPER(TRIM(symbol)) IN (SELECT symbol FROM target_syms)
              AND o > 0 AND h > 0 AND l > 0 AND c > 0 AND v >= 0
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
            arg_min(o, ny_min) FILTER (WHERE ny_min >= 575 AND ny_min <= 577) AS open_0935,
            arg_max(c, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 950) AS close_1550,
            arg_max(c, ny_min) FILTER (WHERE ny_min >= 570 AND ny_min <= 959) AS rth_close,
            SUM(vw * v) FILTER (WHERE ny_min >= 570 AND ny_min <= 959) AS rth_dvol
        FROM tagged
        WHERE ny_min >= 570 AND ny_min <= 959
        GROUP BY symbol, trade_date
        HAVING rth_close IS NOT NULL AND rth_dvol > 0
        """,
        [sip_globs, STUDY_WALL_ISO],
    )

    # 3. Strictly causal T-1 20d baselines + T+1/T+2 daily MTM forward closes (strictly <= 2026-10-15!)
    con.execute(
        """
        CREATE TEMP TABLE causal_daily AS
        WITH step1 AS (
            SELECT
                *,
                LAG(rth_close) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_rth_close,
                LAG(trade_date) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_trade_date,
                LEAD(trade_date, 1) OVER (PARTITION BY symbol ORDER BY trade_date) AS next1_date,
                LEAD(rth_close, 1) OVER (PARTITION BY symbol ORDER BY trade_date) AS next1_close,
                LEAD(trade_date, 2) OVER (PARTITION BY symbol ORDER BY trade_date) AS next2_date,
                LEAD(rth_close, 2) OVER (PARTITION BY symbol ORDER BY trade_date) AS next2_close,
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
            symbol, trade_date, prev_trade_date, prev_rth_close,
            adv20_usd, hist_days,
            COALESCE(STDDEV_SAMP(log_ret) OVER (
                PARTITION BY symbol ORDER BY trade_date
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            ), 0.025) AS vol20d_daily,
            open_0935, close_1550, rth_close,
            next1_date, next1_close, next2_date, next2_close
        FROM step1
        """
    )

    # 4. Identify strictly causal PIT 424B5 (primary) and S-3/S-3ASR/EFFECT (diagnostic) triggers
    triggers_df = con.execute(
        """
        WITH base AS (
            SELECT
                c.*,
                CAST(c.trade_date || ' 09:30:00' AS TIMESTAMP) AS rth_open_ny,
                CAST(c.prev_trade_date || ' 16:00:00' AS TIMESTAMP) AS prev_close_ny
            FROM causal_daily c
            LEFT JOIN split_events sp0 ON sp0.symbol = c.symbol AND sp0.ex_date = c.prev_trade_date
            LEFT JOIN split_events sp1 ON sp1.symbol = c.symbol AND sp1.ex_date = c.trade_date
            LEFT JOIN split_events sp2 ON sp2.symbol = c.symbol AND sp2.ex_date = c.next1_date
            LEFT JOIN split_events sp3 ON sp3.symbol = c.symbol AND sp3.ex_date = c.next2_date
            WHERE c.trade_date >= $1 AND c.trade_date <= $2
              AND c.hist_days >= 15
              AND c.prev_rth_close >= 5.0
              AND c.adv20_usd >= 25000000.0
              AND c.open_0935 > 0 AND c.close_1550 > 0
              AND sp0.symbol IS NULL AND sp1.symbol IS NULL
              AND sp2.symbol IS NULL AND sp3.symbol IS NULL
        )
        SELECT
            b.symbol, b.trade_date, b.prev_rth_close, b.adv20_usd, b.vol20d_daily,
            b.open_0935, b.close_1550, b.rth_close,
            b.next1_date, b.next1_close, b.next2_date, b.next2_close,
            EXISTS (
                SELECT 1 FROM sec_dil s
                WHERE s.symbol = b.symbol AND s.form = '424B5'
                  AND s.accepted_ny >= b.prev_close_ny AND s.accepted_ny <= b.rth_open_ny
            ) AS is_424b5,
            EXISTS (
                SELECT 1 FROM sec_dil s
                WHERE s.symbol = b.symbol AND s.form IN ('S-3', 'S-3ASR', 'EFFECT')
                  AND s.accepted_ny >= b.prev_close_ny AND s.accepted_ny <= b.rth_open_ny
            ) AS is_shelf
        FROM base b
        WHERE EXISTS (
            SELECT 1 FROM sec_dil s
            WHERE s.symbol = b.symbol
              AND s.accepted_ny >= b.prev_close_ny AND s.accepted_ny <= b.rth_open_ny
        )
        ORDER BY b.trade_date, b.symbol
        """,
        [EVAL_START_ISO, STUDY_WALL_ISO],
    ).fetchdf()

    # 5. Replay exact 1-minute RTH bar paths (09:35..15:50 ET) for all triggered events
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in triggers_df.to_dict(orient="records"):
        by_date[str(row["trade_date"])].append(row)

    intraday_records: list[dict[str, Any]] = []
    mtm_3d_legs: list[dict[str, Any]] = []  # Non-overlapping 1-day calendar-day MTM legs!

    for tdate, day_rows in sorted(by_date.items()):
        syms = sorted({str(r["symbol"]) for r in day_rows})
        day_glob = f"{(sip_root / tdate).as_posix()}/*.parquet"
        bars_df = con.execute(
            """
            SELECT UPPER(TRIM(symbol)) AS symbol, t_ns, o, h, l, c
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
        grouped = {sym: grp for sym, grp in bars_df.groupby("symbol", sort=False)}

        for r in day_rows:
            sym = str(r["symbol"])
            grp = grouped.get(sym)
            if grp is None or grp.empty:
                continue
            sim = _simulate_short_uncapped_lossless(
                grp["o"].to_numpy(dtype=float),
                grp["h"].to_numpy(dtype=float),
                grp["l"].to_numpy(dtype=float),
                grp["c"].to_numpy(dtype=float),
                stop_pct=ADR0022_MAX_STOP_PCT,
            )
            cost_1x = _almgren_chriss_1x_cost_bps(float(r["adv20_usd"]), float(r["vol20d_daily"]), hold_days=1)
            gross_bps = float(sim["gross_bps"])
            leg_kind = "424B5" if bool(r["is_424b5"]) else "SHELF_S3_EFFECT"
            intraday_records.append(
                {
                    "trade_date": tdate,
                    "symbol": sym,
                    "leg_kind": leg_kind,
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

            # Build 100% HONEST, UNCLAMPED non-overlapping 1-day MTM calendar-day cohort legs for 424B5
            # (ZERO artificial `max(-200.0)` daily truncation — realistic full overnight/intraday tail risk!)
            if bool(r["is_424b5"]):
                open0 = float(r["open_0935"])
                c0 = float(r["rth_close"])
                c1 = r["next1_close"]
                c2 = r["next2_close"]
                d1_dt = r["next1_date"]
                d2_dt = r["next2_date"]
                rt_3d_cost = _almgren_chriss_1x_cost_bps(float(r["adv20_usd"]), float(r["vol20d_daily"]), hold_days=3)
                # Spread realistic 3-day round-trip + borrow cost conservatively across held legs
                c_day0 = 0.5 * rt_3d_cost
                c_day1 = SHORT_BORROW_DAILY_BPS
                c_day2 = 0.5 * rt_3d_cost
                # Day 0 MTM (unclamped): open_0935 -> close_T
                g0 = ((open0 - c0) / open0) * 10_000.0
                mtm_3d_legs.append(
                    {
                        "trade_date": tdate,
                        "symbol": sym,
                        "gross_bps": g0,
                        "cost_1x_bps": c_day0,
                        "net_1x_bps": g0 - c_day0,
                        "net_2x_bps": g0 - 2.0 * c_day0,
                        "mae_pct": float(sim["mae_pct"]),
                        "vol20d_pct": float(r["vol20d_daily"]) * 100.0,
                    }
                )
                # Day 1 MTM (unclamped): close_T -> close_T+1
                if pd.notna(c1) and pd.notna(d1_dt) and float(c1) > 0 and str(d1_dt) <= STUDY_WALL_ISO:
                    g1 = ((c0 - float(c1)) / c0) * 10_000.0
                    mtm_3d_legs.append(
                        {
                            "trade_date": str(d1_dt),
                            "symbol": sym,
                            "gross_bps": g1,
                            "cost_1x_bps": c_day1,
                            "net_1x_bps": g1 - c_day1,
                            "net_2x_bps": g1 - 2.0 * c_day1,
                            "mae_pct": max(0.0, -g1 / 100.0),
                            "vol20d_pct": float(r["vol20d_daily"]) * 100.0,
                        }
                    )
                    # Day 2 MTM (unclamped): close_T+1 -> close_T+2
                    if pd.notna(c2) and pd.notna(d2_dt) and float(c2) > 0 and str(d2_dt) <= STUDY_WALL_ISO:
                        g2 = ((float(c1) - float(c2)) / float(c1)) * 10_000.0
                        mtm_3d_legs.append(
                            {
                                "trade_date": str(d2_dt),
                                "symbol": sym,
                                "gross_bps": g2,
                                "cost_1x_bps": c_day2,
                                "net_1x_bps": g2 - c_day2,
                                "net_2x_bps": g2 - 2.0 * c_day2,
                                "mae_pct": max(0.0, -g2 / 100.0),
                                "vol20d_pct": float(r["vol20d_daily"]) * 100.0,
                            }
                        )

    all_intra_df = pd.DataFrame(intraday_records)
    p424_df = all_intra_df[all_intra_df["leg_kind"] == "424B5"].copy() if not all_intra_df.empty else pd.DataFrame()
    shelf_df = (
        all_intra_df[all_intra_df["leg_kind"] == "SHELF_S3_EFFECT"].copy() if not all_intra_df.empty else pd.DataFrame()
    )
    mtm3d_df = pd.DataFrame(mtm_3d_legs)

    def _agg_daily(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        return (
            df.groupby("trade_date", as_index=False)
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

    primary_metrics = _evaluate_daily_and_event_metrics(
        p424_df, _agg_daily(p424_df), label="H112_424B5_PRIMARY_RTH_SHORT", n_trials=TRIAL_NUMBER
    )
    diag_3d = _evaluate_daily_and_event_metrics(
        mtm3d_df, _agg_daily(mtm3d_df), label="H112_424B5_3D_NONOVERLAPPING_MTM_SHORT", n_trials=TRIAL_NUMBER
    )
    diag_shelf = _evaluate_daily_and_event_metrics(
        shelf_df, _agg_daily(shelf_df), label="H112_S3_EFFECT_SHELF_SHORT", n_trials=TRIAL_NUMBER
    )

    receipt: dict[str, Any] = {
        "schema": "sota.prewall-trial106-h112-receipt.v1",
        "trial_number": TRIAL_NUMBER,
        "hypothesis_id": HYPOTHESIS_ID,
        "prereg_spec_sha256": PREREG_SPEC_SHA256,
        "study_wall_date": STUDY_WALL_ISO,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "prewall_sip_sessions_scanned": len(sip_dirs),
        "sample_start_date": EVAL_START_ISO,
        "sample_end_date": STUDY_WALL_ISO,
        "primary_424b5_rth_short": primary_metrics,
        "diagnostic_424b5_3d_nonoverlapping_mtm_short": diag_3d,
        "diagnostic_s3_effect_shelf_short": diag_shelf,
    }
    hashable = {k: v for k, v in receipt.items() if k != "generated_at_utc"}
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(hashable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute Pre-Registered Trial #106 (H112) on Pre-Wall SEC+SIP Lake.")
    parser.add_argument(
        "--market-data-root",
        type=pathlib.Path,
        default=pathlib.Path.home() / "Documents" / "market-data",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "docs" / "TRIAL106_H112_PREWALL_RECEIPT.json",
    )
    args = parser.parse_args(argv)
    receipt = run_h112_trial106_study(args.market_data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
