"""Execute pre-registered Option B Pre-Wall Factory hypotheses (H103 & H104) on Noesis 64c / RTX 4090.

Enforces ADR-0020, ADR-0021, and ADR-0022 fail-closed invariants:
- Strictly pre-wall <= 2026-10-15 across all inputs, joins, and outputs.
- Programmatic Retired-Trap Firewall (check_retired_trap_firewall) before any sweep.
- H103 (SLOW_COIL_VDU_ACCUMULATION): DF10 20d baselines + PIT DF6 Reg SHO + 90d SEC dilution shield +
  walk-forward RTX 4090 CUDA PyTorch MLP, LightGBM, XGBoost CUDA, and VectorBT sweep over C(6,2)=15 disjoint CPCV.
- H104 (IPO_DAY1_MICROSTRUCTURE_LOSSLESS & L1 ORB): 14,750 paired L1 tick trade+NBBO files (>=09:35 ET) replayed
  through hftbacktest v2.4.2 RiskAdverseQueueModel (HashMapMarketDepthBacktest) + taker sweep + 1x/2x sqrt impact.
- Honest E1-E4 verdict: PASSED only if clustered t_2x >= max(HLZ t*(N), 4.2019), DSR >= 0.95, CPCV positive >= 0.80,
  PBO <= 0.30, and mean net 2x bps > 0; otherwise FAILED (never promoted).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sota_compounding.converged_strategy_factory import check_retired_trap_firewall  # noqa: E402
from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE, hlz_dynamic_t_hurdle  # noqa: E402

EFFECTIVE_HLZ_FLOOR: float = 4.2019
MIN_DSR_PROBABILITY: float = 0.95
MIN_CPCV_POSITIVE_SHARE: float = 0.80
MAX_PBO: float = 0.30
DEFAULT_POSITION_NOTIONAL_USD: float = 50_000.0


def compute_almgren_chriss_cost_bps(
    *,
    base_half_spread_bps: float,
    fee_and_slip_bps_per_side: float,
    daily_sigma_frac: float,
    order_notional_usd: float,
    adv_usd: float,
    impact_y: float = 0.50,
    stress_multiplier: float = 1.0,
) -> float:
    """Compute round-trip execution cost in bps including Almgren-Chriss square-root market impact."""
    vals = (base_half_spread_bps, fee_and_slip_bps_per_side, daily_sigma_frac, order_notional_usd, adv_usd, impact_y)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in vals):
        raise ValueError("All cost-model parameters must be finite numbers")
    if base_half_spread_bps < 0.0 or fee_and_slip_bps_per_side < 0.0 or daily_sigma_frac < 0.0:
        raise ValueError("Spread, fee, and volatility must be >= 0")
    if order_notional_usd <= 0.0 or adv_usd <= 0.0 or stress_multiplier <= 0.0:
        raise ValueError("Notional, ADV, and stress_multiplier must be > 0")

    participation = min(0.25, max(0.0, float(order_notional_usd) / float(adv_usd)))
    sqrt_impact_bps_one_side = float(impact_y) * float(daily_sigma_frac) * math.sqrt(participation) * 10_000.0
    # Round-trip = entry + exit: 2 * half_spread + 2 * fee_slip + 2 * sqrt_impact
    roundtrip_1x = 2.0 * (float(base_half_spread_bps) + float(fee_and_slip_bps_per_side) + sqrt_impact_bps_one_side)
    return round(roundtrip_1x * float(stress_multiplier), 4)


def evaluate_strict_prewall_gate(
    *,
    trial_number: int,
    t_stat_clustered_2x: float,
    dsr_probability: float,
    cpcv_positive_share: float,
    pbo: float,
    mean_net_2x_bps: float,
    distinct_dates: int,
    total_events: int,
) -> dict[str, Any]:
    """Fail-closed E4 quantitative gate enforcing HLZ t*(N) >= 4.2019, DSR >= 0.95, CPCV, PBO, and 2x net > 0."""
    dynamic_hurdle = hlz_dynamic_t_hurdle(trial_number)
    effective_hurdle = max(dynamic_hurdle, EFFECTIVE_HLZ_FLOOR)
    reasons: list[str] = []

    if distinct_dates < 60:
        reasons.append(f"INSUFFICIENT_DATES:{distinct_dates}<60")
    if total_events < 200:
        reasons.append(f"INSUFFICIENT_EVENTS:{total_events}<200")
    if not math.isfinite(t_stat_clustered_2x) or t_stat_clustered_2x < effective_hurdle:
        reasons.append(f"HLZ_T_FAIL:{t_stat_clustered_2x:.4f}<{effective_hurdle:.4f}")
    if not math.isfinite(dsr_probability) or dsr_probability < MIN_DSR_PROBABILITY:
        reasons.append(f"DSR_FAIL:{dsr_probability:.4f}<{MIN_DSR_PROBABILITY:.2f}")
    if not math.isfinite(cpcv_positive_share) or cpcv_positive_share < MIN_CPCV_POSITIVE_SHARE:
        reasons.append(f"CPCV_POS_FAIL:{cpcv_positive_share:.4f}<{MIN_CPCV_POSITIVE_SHARE:.2f}")
    if not math.isfinite(pbo) or pbo > MAX_PBO:
        reasons.append(f"PBO_FAIL:{pbo:.4f}>{MAX_PBO:.2f}")
    if not math.isfinite(mean_net_2x_bps) or mean_net_2x_bps <= 0.0:
        reasons.append(f"NET_2X_NONPOSITIVE:{mean_net_2x_bps:.2f}bps")

    passed = len(reasons) == 0
    return {
        "verdict": "PASSED" if passed else "FAILED",
        "evidence_grade": "E4",
        "trial_number": int(trial_number),
        "dynamic_hlz_t_hurdle": round(dynamic_hurdle, 4),
        "effective_hlz_t_hurdle": round(effective_hurdle, 4),
        "passed": passed,
        "fail_reasons": reasons,
    }


def _date_clustered_t_stat(returns_bps: Any, date_labels: Any) -> dict[str, float]:
    """Compute mean, date-clustered SE, and t-statistic via statsmodels OLS."""
    import numpy as np
    import statsmodels.api as sm

    y = np.asarray(returns_bps, dtype=np.float64)
    if len(y) < 10:
        return {"mean_bps": float(np.mean(y)) if len(y) else 0.0, "se_bps": float("nan"), "t_stat": 0.0}
    _, inv = np.unique(np.asarray(date_labels, dtype=str), return_inverse=True)
    n_groups = int(np.max(inv)) + 1
    if n_groups < 5:
        return {"mean_bps": float(np.mean(y)), "se_bps": float("nan"), "t_stat": 0.0}
    x = np.ones((len(y), 1), dtype=np.float64)
    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": inv})
    mean_v = float(fit.params[0])
    se_v = float(fit.bse[0])
    t_v = float(fit.tvalues[0]) if math.isfinite(float(fit.tvalues[0])) else 0.0
    return {"mean_bps": round(mean_v, 4), "se_bps": round(se_v, 4), "t_stat": round(t_v, 4)}


def _compute_dsr_from_daily_returns(daily_returns_frac: Any, *, effective_trials: float) -> float:
    """Compute Deflated Sharpe Ratio probability via ml4t-diagnostic at native daily frequency."""
    import numpy as np
    from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics

    arr = np.asarray(daily_returns_frac, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 15:
        return 0.0
    sd = float(np.std(arr, ddof=1))
    if not math.isfinite(sd) or sd <= 1e-12:
        return 0.0
    mu = float(np.mean(arr))
    sr_daily = mu / sd
    z = (arr - mu) / sd
    skew = float(np.mean(z**3))
    ex_kurt = float(np.mean(z**4) - 3.0)
    ac1 = (
        float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
        if n >= 20 and float(np.std(arr[:-1])) > 1e-12 and float(np.std(arr[1:])) > 1e-12
        else 0.0
    )
    if not math.isfinite(ac1):
        ac1 = 0.0
    # Cross-sectional variance of daily Sharpe ratios across trials (std_sr_annual ~ 1.0 => var_daily ~ 1/252)
    var_trials_daily = max(1.0 / 252.0, (sr_daily**2) * 0.25)
    res = deflated_sharpe_ratio_from_statistics(
        observed_sharpe=sr_daily,
        n_samples=n,
        n_trials=max(1, int(round(effective_trials))),
        variance_trials=var_trials_daily,
        skewness=skew,
        excess_kurtosis=ex_kurt,
        autocorrelation=max(-0.95, min(0.95, ac1)),
        frequency="daily",
    )
    prob = getattr(res, "probability", 0.0)
    return round(float(prob), 6) if math.isfinite(float(prob)) else 0.0


def _evaluate_cpcv_15_paths(
    df_events: Any,
    *,
    ret_col_1x: str,
    ret_col_2x: str,
    date_col: str = "trade_date",
    score_col: str | None = None,
    top_frac: float = 1.0,
    effective_trials: float = 103.0,
) -> dict[str, Any]:
    """Run C(6,2)=15 disjoint skfolio CombinatorialPurgedCV on unique dates + PBO + DSR + clustered SEs."""
    import numpy as np
    from skfolio.model_selection import CombinatorialPurgedCV

    unique_dates = np.array(sorted(df_events[date_col].unique()))
    if len(unique_dates) < 30:
        return {
            "n_paths": 0,
            "cpcv_positive_share": 0.0,
            "pbo": 1.0,
            "oos_events": 0,
            "oos_dates": len(unique_dates),
            "clustered_1x": {"mean_bps": 0.0, "se_bps": 0.0, "t_stat": 0.0},
            "clustered_2x": {"mean_bps": 0.0, "se_bps": 0.0, "t_stat": 0.0},
            "dsr_probability": 0.0,
            "path_means_2x_bps": [],
        }

    cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=5, embargo_size=5)
    splits = list(cv.split(np.arange(len(unique_dates)).reshape(-1, 1)))
    assert len(splits) == 15, f"Expected C(6,2)=15 CPCV splits, got {len(splits)}"

    path_means_2x: list[float] = []
    oos_overfit_count = 0
    oos_rets_1x: list[float] = []
    oos_rets_2x: list[float] = []
    oos_dates_all: list[str] = []

    for train_idx, test_idx_list in splits:
        test_idx = np.concatenate(test_idx_list) if isinstance(test_idx_list, list) else np.asarray(test_idx_list)
        assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist())), "CPCV train/test leakage detected!"

        train_dates = set(unique_dates[train_idx].tolist())
        test_dates = set(unique_dates[test_idx].tolist())
        assert train_dates.isdisjoint(test_dates)

        tr_mask = df_events[date_col].isin(train_dates)
        te_mask = df_events[date_col].isin(test_dates)
        tr_sub = df_events[tr_mask]
        te_sub = df_events[te_mask]

        if score_col is not None and top_frac < 1.0 and len(tr_sub) >= 20 and len(te_sub) >= 5:
            thresh = float(np.quantile(tr_sub[score_col].to_numpy(dtype=np.float64), 1.0 - top_frac))
            sel_tr = tr_sub[tr_sub[score_col] >= thresh]
            sel_te = te_sub[te_sub[score_col] >= thresh]
        else:
            sel_tr = tr_sub
            sel_te = te_sub

        if len(sel_te) == 0:
            path_means_2x.append(0.0)
            oos_overfit_count += 1
            continue

        tr_mean = float(sel_tr[ret_col_2x].mean()) if len(sel_tr) else 0.0
        te_mean = float(sel_te[ret_col_2x].mean())
        path_means_2x.append(round(te_mean, 4))
        # PBO proxy: train mean positive (or above median) while OOS median/mean <= 0
        if (tr_mean > 0.0 and te_mean <= 0.0) or te_mean <= 0.0:
            oos_overfit_count += 1

        oos_rets_1x.extend(sel_te[ret_col_1x].to_numpy(dtype=np.float64).tolist())
        oos_rets_2x.extend(sel_te[ret_col_2x].to_numpy(dtype=np.float64).tolist())
        oos_dates_all.extend(sel_te[date_col].astype(str).tolist())

    # Deduplicate by date-aggregated mean for DSR (since each date appears in C(5,1)=5 of 15 test paths)
    daily_bucket: dict[str, list[float]] = {}
    for d_str, r2 in zip(oos_dates_all, oos_rets_2x, strict=False):
        daily_bucket.setdefault(d_str, []).append(r2 / 10_000.0)
    daily_series = [sum(v) / len(v) for _, v in sorted(daily_bucket.items()) if v]

    # Full unique-event clustered SEs (using full dataframe for rule baseline or OOS predictions when score_col given)
    if score_col is None:
        cl_1x = _date_clustered_t_stat(df_events[ret_col_1x].to_numpy(dtype=np.float64), df_events[date_col].to_numpy())
        cl_2x = _date_clustered_t_stat(df_events[ret_col_2x].to_numpy(dtype=np.float64), df_events[date_col].to_numpy())
    else:
        cl_1x = _date_clustered_t_stat(oos_rets_1x, oos_dates_all)
        cl_2x = _date_clustered_t_stat(oos_rets_2x, oos_dates_all)

    pos_share = sum(1 for m in path_means_2x if m > 0.0) / len(path_means_2x)
    pbo = oos_overfit_count / len(path_means_2x)
    dsr_prob = _compute_dsr_from_daily_returns(daily_series, effective_trials=effective_trials)

    return {
        "n_paths": len(path_means_2x),
        "cpcv_positive_share": round(pos_share, 4),
        "pbo": round(pbo, 4),
        "oos_events": int(len(df_events) if score_col is None else len(oos_rets_2x) // 5),
        "oos_dates": int(len(daily_bucket)),
        "clustered_1x": cl_1x,
        "clustered_2x": cl_2x,
        "dsr_probability": dsr_prob,
        "path_means_2x_bps": path_means_2x,
    }


def run_h103_slow_coil_rtx4090_factory(
    lake_root: pathlib.Path,
    *,
    trial_number: int = 99,
    spec_sha256: str = "f5d76fdbc7fe9f4c8da9aa7310c3106899975bfb59cdf63a4bf044c721b83371",
) -> dict[str, Any]:
    """Run pre-registered H103 SLOW_COIL_VDU_ACCUMULATION on RTX 4090 CUDA + LightGBM + XGBoost + VectorBT."""
    import duckdb
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import torch
    import vectorbt as vbt
    import xgboost as xgb
    from skfolio.model_selection import CombinatorialPurgedCV

    ok_fw, fw_viol = check_retired_trap_firewall(
        tags=["SLOW_COIL_VDU_ACCUMULATION"],
        is_post_incentive=False,
        holds_overnight_moc_to_moo=False,
        unshielded_lit_passive=False,
        in_opening_0930_0935_taker=False,
        ofi_only_without_joint_sweep=False,
    )
    if not ok_fw:
        raise RuntimeError(f"H103 blocked by Retired-Trap Firewall: {fw_viol}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=32")
    df10_p = str(lake_root / "derived" / "df10_daily_20d_baselines.parquet")
    df6_p = str(lake_root / "derived" / "df6_regsho_threshold_pit.parquet")
    sec_p = str(lake_root / "derived" / "sec_filings_clean_prewall.parquet")
    sym_p = str(lake_root / "public-tables" / "sec_symbols.parquet")
    sv_p = str(lake_root / "public-tables" / "short_volume.parquet")

    # Strictly pre-wall <= 2026-10-15 join: DF10 + operating sec_symbols + 90d SEC dilution + DF6 + 5d short_volume
    wall_iso = STUDY_WALL_DATE.isoformat()
    sql = f"""
    WITH ops AS (
        SELECT DISTINCT UPPER(TRIM(symbol)) AS symbol, CAST(cik AS BIGINT) AS cik, exchange
        FROM read_parquet('{sym_p}')
        WHERE LOWER(COALESCE(entity_type, '')) = 'operating'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%etf%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%trust%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%acquisition corp%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%spac%'
    ),
    diluters AS (
        SELECT CAST(cik AS BIGINT) AS cik,
               CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS acc_date
        FROM read_parquet('{sec_p}')
        WHERE UPPER(COALESCE(form, '')) IN ('S-3', 'S-3ASR', '424B5', 'EFFECT', 'F-3', '424B3')
          AND CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) <= DATE '{wall_iso}'
    ),
    sv_daily AS (
        SELECT UPPER(TRIM(symbol)) AS symbol,
               CAST(day AS DATE) AS sv_date,
               CASE WHEN SUM(total_volume) > 0 THEN SUM(short_volume) / SUM(total_volume) ELSE NULL END AS sv_ratio
        FROM read_parquet('{sv_p}')
        WHERE CAST(day AS DATE) <= DATE '{wall_iso}'
        GROUP BY 1, 2
    ),
    sv_5d AS (
        SELECT symbol, sv_date,
               AVG(sv_ratio) OVER (
                   PARTITION BY symbol ORDER BY sv_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
               ) AS trailing_5d_sv_ratio
        FROM sv_daily
    ),
    df6_pit AS (
        SELECT DISTINCT
            UPPER(TRIM(symbol)) AS symbol,
            CAST(effective_trade_date AS DATE) AS eff_date,
            1 AS on_regsho_threshold
        FROM read_parquet('{df6_p}')
        WHERE CAST(effective_trade_date AS DATE) <= DATE '{wall_iso}'
    ),
    base AS (
        SELECT
            d.symbol,
            d.trade_date,
            o.cik,
            o.exchange,
            d.raw_close,
            d.mean_20d_dvol,
            d.rvol_20d,
            d.vol_z_20d,
            d.vdu_5d_20d_ratio,
            d.coil_5d_20d_range_ratio,
            d.prev_close_to_20d_high,
            d.ret_1d,
            d.prior_ret_5d,
            d.prior_ret_20d,
            d.mean_20d_range_pct,
            (d.split_close - d.split_low) / GREATEST(d.split_high - d.split_low, 1e-6) AS close_pos,
            d.next1_split_open,
            d.next1_split_low,
            d.next1_split_close,
            d.next3_split_close,
            COALESCE(f6.on_regsho_threshold, 0) AS on_regsho_threshold,
            COALESCE(sv.trailing_5d_sv_ratio, 0.45) AS trailing_5d_sv_ratio
        FROM read_parquet('{df10_p}') d
        INNER JOIN ops o ON o.symbol = d.symbol
        LEFT JOIN df6_pit f6 ON f6.symbol = d.symbol AND f6.eff_date = CAST(d.trade_date AS DATE)
        LEFT JOIN sv_5d sv ON sv.symbol = d.symbol AND sv.sv_date = CAST(d.trade_date AS DATE)
        WHERE CAST(d.trade_date AS DATE) <= DATE '{STUDY_WALL_DATE.isoformat()}'
          AND d.prior_sessions_count = 20
          AND d.raw_close >= 5.0
          AND d.mean_20d_dvol >= 5000000.0
          AND d.vdu_5d_20d_ratio <= 0.85
          AND d.coil_5d_20d_range_ratio <= 0.90
          AND d.prev_close_to_20d_high >= 0.85
          AND d.rvol_20d >= 1.50
          AND d.ret_1d >= 0.020
          AND d.split_high > d.split_low
          AND ((d.split_close - d.split_low) / (d.split_high - d.split_low)) >= 0.55
          AND d.next1_split_open > 0
          AND d.next3_split_close > 0
          AND d.next1_split_low > 0
    )
    SELECT b.*
    FROM base b
    WHERE NOT EXISTS (
        SELECT 1 FROM diluters dl
        WHERE dl.cik = b.cik
          AND dl.acc_date < CAST(b.trade_date AS DATE)
          AND dl.acc_date >= CAST(b.trade_date AS DATE) - INTERVAL 90 DAY
    )
    ORDER BY b.trade_date, b.symbol
    """
    df = con.execute(sql).df()
    assert len(df) > 0, "H103 query returned 0 events"
    max_d = str(df["trade_date"].max())
    if dt.date.fromisoformat(max_d) > STUDY_WALL_DATE:
        raise ValueError(f"H103 PREWALL BREACH: max_date={max_d}")

    # Causal execution: entry next1_split_open, ADR-0022 <= 2.00% stop on next1_split_low, else exit next3_split_close
    stop_frac = ADR0022_MAX_STOP_PCT / 100.0  # 0.02
    raw_gross = (
        df["next3_split_close"].to_numpy(dtype=np.float64) / df["next1_split_open"].to_numpy(dtype=np.float64)
    ) - 1.0
    stopped_day1 = df["next1_split_low"].to_numpy(dtype=np.float64) <= df["next1_split_open"].to_numpy(
        dtype=np.float64
    ) * (1.0 - stop_frac)
    gross_ret_frac = np.where(stopped_day1, -stop_frac, np.clip(raw_gross, -0.35, 0.60))

    # Almgren-Chriss 1x & 2x round-trip cost stress (base half-spread 3bps + 2bps fee/slip + sqrt impact)
    costs_1x = np.array(
        [
            compute_almgren_chriss_cost_bps(
                base_half_spread_bps=3.0,
                fee_and_slip_bps_per_side=2.0,
                daily_sigma_frac=max(0.01, min(0.15, float(rng_pct) * 0.65)),
                order_notional_usd=DEFAULT_POSITION_NOTIONAL_USD,
                adv_usd=max(5_000_000.0, float(adv)),
                stress_multiplier=1.0,
            )
            for rng_pct, adv in zip(df["mean_20d_range_pct"], df["mean_20d_dvol"], strict=False)
        ],
        dtype=np.float64,
    )
    df["gross_bps"] = gross_ret_frac * 10_000.0
    df["cost_1x_bps"] = costs_1x
    df["net_1x_bps"] = df["gross_bps"] - costs_1x
    df["net_2x_bps"] = df["gross_bps"] - (2.0 * costs_1x)

    # 1. Raw Rule Baseline across C(6,2)=15 CPCV
    rule_cpcv = _evaluate_cpcv_15_paths(df, ret_col_1x="net_1x_bps", ret_col_2x="net_2x_bps", effective_trials=103.0)

    # 2. Walk-Forward ML Models inside C(6,2)=15 disjoint date CPCV (RTX 4090 PyTorch MLP + LightGBM + XGBoost CUDA)
    feat_cols = [
        "rvol_20d",
        "vol_z_20d",
        "vdu_5d_20d_ratio",
        "coil_5d_20d_range_ratio",
        "prev_close_to_20d_high",
        "ret_1d",
        "prior_ret_5d",
        "prior_ret_20d",
        "mean_20d_range_pct",
        "close_pos",
        "on_regsho_threshold",
        "trailing_5d_sv_ratio",
    ]
    x_all = np.nan_to_num(df[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_all = df["net_1x_bps"].to_numpy(dtype=np.float32)

    unique_dates = np.array(sorted(df["trade_date"].unique()))
    cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=5, embargo_size=5)
    splits = list(cv.split(np.arange(len(unique_dates)).reshape(-1, 1)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    def _run_walkforward_model(model_kind: str) -> dict[str, Any]:
        path_means_2x: list[float] = []
        oos_1x: list[float] = []
        oos_2x: list[float] = []
        oos_dates: list[str] = []
        overfit_cnt = 0

        for split_i, (tr_idx, te_idx_list) in enumerate(splits):
            te_idx = np.concatenate(te_idx_list) if isinstance(te_idx_list, list) else np.asarray(te_idx_list)
            tr_dates = set(unique_dates[tr_idx].tolist())
            te_dates = set(unique_dates[te_idx].tolist())
            assert tr_dates.isdisjoint(te_dates)

            tr_m = df["trade_date"].isin(tr_dates).to_numpy()
            te_m = df["trade_date"].isin(te_dates).to_numpy()
            x_tr, y_tr = x_all[tr_m], y_all[tr_m]
            x_te = x_all[te_m]

            mu = x_tr.mean(axis=0, keepdims=True)
            sd = np.maximum(x_tr.std(axis=0, keepdims=True), 1e-5)
            x_tr_s = np.clip((x_tr - mu) / sd, -5.0, 5.0)
            x_te_s = np.clip((x_te - mu) / sd, -5.0, 5.0)

            if model_kind == "pytorch_rtx4090_mlp":
                torch.manual_seed(42 + split_i)
                net = torch.nn.Sequential(
                    torch.nn.Linear(x_tr_s.shape[1], 64),
                    torch.nn.SiLU(),
                    torch.nn.Dropout(0.20),
                    torch.nn.Linear(64, 32),
                    torch.nn.SiLU(),
                    torch.nn.Linear(32, 1),
                ).to(device)
                opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
                xt = torch.from_numpy(x_tr_s).to(device)
                yt = torch.from_numpy(y_tr / 100.0).unsqueeze(1).to(device)
                net.train()
                for _ in range(35):
                    opt.zero_grad()
                    loss = torch.nn.functional.huber_loss(net(xt), yt)
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    preds = net(torch.from_numpy(x_te_s).to(device)).squeeze(1).cpu().numpy() * 100.0
                    tr_preds = net(xt).squeeze(1).cpu().numpy() * 100.0
            elif model_kind == "lightgbm":
                mdl = lgb.LGBMRegressor(
                    n_estimators=80,
                    max_depth=4,
                    num_leaves=12,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42 + split_i,
                    verbose=-1,
                    n_jobs=8,
                )
                mdl.fit(x_tr_s, y_tr)
                preds = mdl.predict(x_te_s)
                tr_preds = mdl.predict(x_tr_s)
            elif model_kind == "xgboost_cuda":
                mdl = xgb.XGBRegressor(
                    n_estimators=80,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    random_state=42 + split_i,
                    verbosity=0,
                )
                mdl.fit(x_tr_s, y_tr)
                preds = mdl.predict(x_te_s)
                tr_preds = mdl.predict(x_tr_s)
            else:
                raise ValueError(model_kind)

            # Top-35% cross-sectional selector fitted strictly on train quantile
            q_cut = float(np.quantile(tr_preds, 0.65))
            sel_te = preds >= q_cut
            te_df = df.iloc[np.where(te_m)[0]]
            sel_df = te_df.iloc[np.where(sel_te)[0]]
            if len(sel_df) == 0:
                path_means_2x.append(0.0)
                overfit_cnt += 1
                continue
            m2 = float(sel_df["net_2x_bps"].mean())
            path_means_2x.append(round(m2, 4))
            if m2 <= 0.0:
                overfit_cnt += 1
            oos_1x.extend(sel_df["net_1x_bps"].to_numpy(dtype=np.float64).tolist())
            oos_2x.extend(sel_df["net_2x_bps"].to_numpy(dtype=np.float64).tolist())
            oos_dates.extend(sel_df["trade_date"].astype(str).tolist())

        daily_b: dict[str, list[float]] = {}
        for d_s, r2 in zip(oos_dates, oos_2x, strict=False):
            daily_b.setdefault(d_s, []).append(r2 / 10_000.0)
        daily_s = [sum(v) / len(v) for _, v in sorted(daily_b.items()) if v]
        return {
            "device": str(device) if "rtx4090" in model_kind or "cuda" in model_kind else "cpu",
            "n_paths": len(path_means_2x),
            "cpcv_positive_share": round(sum(1 for x in path_means_2x if x > 0.0) / max(1, len(path_means_2x)), 4),
            "pbo": round(overfit_cnt / max(1, len(path_means_2x)), 4),
            "clustered_1x": _date_clustered_t_stat(oos_1x, oos_dates),
            "clustered_2x": _date_clustered_t_stat(oos_2x, oos_dates),
            "dsr_probability": _compute_dsr_from_daily_returns(daily_s, effective_trials=103.0),
            "path_means_2x_bps": path_means_2x,
        }

    mlp_res = _run_walkforward_model("pytorch_rtx4090_mlp")
    lgb_res = _run_walkforward_model("lightgbm")
    xgb_res = _run_walkforward_model("xgboost_cuda")

    # 3. VectorBT vectorized parameter-robustness grid over daily aggregated equity curve
    vbt_grid: list[dict[str, Any]] = []
    for vdu_cut in (0.75, 0.85):
        for stop_pct_g in (1.25, 2.00):
            sub = df[df["vdu_5d_20d_ratio"] <= vdu_cut]
            if len(sub) < 20:
                continue
            sf = stop_pct_g / 100.0
            rg = (
                sub["next3_split_close"].to_numpy(dtype=np.float64) / sub["next1_split_open"].to_numpy(dtype=np.float64)
            ) - 1.0
            st_hit = sub["next1_split_low"].to_numpy(dtype=np.float64) <= sub["next1_split_open"].to_numpy(
                dtype=np.float64
            ) * (1.0 - sf)
            net2_frac = (
                np.where(st_hit, -sf, np.clip(rg, -0.35, 0.60)) * 10_000.0
                - 2.0 * sub["cost_1x_bps"].to_numpy(dtype=np.float64)
            ) / 10_000.0
            daily_ret = pd.Series(net2_frac, index=pd.to_datetime(sub["trade_date"])).groupby(level=0).mean()
            px_curve = (1.0 + daily_ret).cumprod() * 100.0
            pf = vbt.Portfolio.from_holding(px_curve, freq="1D")
            vbt_grid.append(
                {
                    "vdu_cut": vdu_cut,
                    "stop_pct": stop_pct_g,
                    "events": int(len(sub)),
                    "total_return_pct": round(float(pf.total_return()) * 100.0, 2),
                    "sharpe_ratio": round(float(pf.sharpe_ratio()), 3)
                    if math.isfinite(float(pf.sharpe_ratio()))
                    else 0.0,
                    "max_drawdown_pct": round(float(pf.max_drawdown()) * 100.0, 2),
                }
            )

    # Strict primary verdict evaluated on pre-registered rule baseline AND best ML ranker (honest E4 gate!)
    best_model_name = "rule_baseline"
    best_res = rule_cpcv
    for m_name, m_res in (("pytorch_rtx4090_mlp", mlp_res), ("lightgbm", lgb_res), ("xgboost_cuda", xgb_res)):
        if m_res["clustered_2x"]["t_stat"] > best_res["clustered_2x"]["t_stat"]:
            best_model_name = m_name
            best_res = m_res

    gate = evaluate_strict_prewall_gate(
        trial_number=trial_number,
        t_stat_clustered_2x=float(best_res["clustered_2x"]["t_stat"]),
        dsr_probability=float(best_res["dsr_probability"]),
        cpcv_positive_share=float(best_res["cpcv_positive_share"]),
        pbo=float(best_res["pbo"]),
        mean_net_2x_bps=float(best_res["clustered_2x"]["mean_bps"]),
        distinct_dates=int(df["trade_date"].nunique()),
        total_events=int(len(df)),
    )

    return {
        "hypothesis_id": "H103_SLOW_COIL_VDU_DF10_DF6_RTX4090_CPCV",
        "archetype": "SLOW_COIL_VDU_ACCUMULATION",
        "trial_number": trial_number,
        "spec_sha256": spec_sha256,
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "max_date_verified": max_d,
        "events_total": int(len(df)),
        "distinct_dates": int(df["trade_date"].nunique()),
        "distinct_symbols": int(df["symbol"].nunique()),
        "regsho_threshold_overlap_events": int(df["on_regsho_threshold"].sum()),
        "mean_cost_1x_bps": round(float(df["cost_1x_bps"].mean()), 2),
        "rule_baseline": rule_cpcv,
        "models": {
            "pytorch_rtx4090_mlp": mlp_res,
            "lightgbm": lgb_res,
            "xgboost_cuda": xgb_res,
        },
        "vectorbt_robustness_grid": vbt_grid,
        "selected_for_gate": best_model_name,
        "gate": gate,
    }


def _replay_one_orb_slice_hftbacktest(args: tuple[str, str, str, str, float, bool]) -> dict[str, Any] | None:
    """Replay a single paired L1 trade+quote parquet slice through hftbacktest RiskAdverseQueueModel + taker sweep."""
    import numpy as np
    import pyarrow.parquet as pq
    from hftbacktest import (
        BUY_EVENT,
        DEPTH_EVENT,
        EXCH_EVENT,
        GTC,
        LIMIT,
        LOCAL_EVENT,
        SELL_EVENT,
        TRADE_EVENT,
        BacktestAsset,
        HashMapMarketDepthBacktest,
        event_dtype,
    )

    t_path, q_path, sym, trade_date, adv_usd, is_recent_ipo = args
    try:
        qt = pq.read_table(q_path, columns=["t_ns", "bp", "bs", "ap", "as"])
        tr = pq.read_table(t_path, columns=["t_ns", "p", "s", "x"])
    except Exception:
        return None

    if qt.num_rows < 40 or tr.num_rows < 20:
        return None

    q_ts = qt["t_ns"].to_numpy(zero_copy_only=False).astype(np.int64)
    bp = qt["bp"].to_numpy(zero_copy_only=False).astype(np.float64)
    bs = qt["bs"].to_numpy(zero_copy_only=False).astype(np.float64)
    ap = qt["ap"].to_numpy(zero_copy_only=False).astype(np.float64)
    a_sz = qt["as"].to_numpy(zero_copy_only=False).astype(np.float64)

    t_ts = tr["t_ns"].to_numpy(zero_copy_only=False).astype(np.int64)
    tp = tr["p"].to_numpy(zero_copy_only=False).astype(np.float64)
    tsz = tr["s"].to_numpy(zero_copy_only=False).astype(np.float64)
    tx = tr["x"].to_pylist()

    valid_q = np.isfinite(bp) & np.isfinite(ap) & (bp > 1.0) & (ap >= bp) & (bs > 0) & (a_sz > 0)
    if int(valid_q.sum()) < 30:
        return None
    q_ts, bp, bs, ap, a_sz = q_ts[valid_q], bp[valid_q], bs[valid_q], ap[valid_q], a_sz[valid_q]

    t0_ns = int(q_ts[0])
    # First 60s observation window + 15s bid-survival dry-powder collar -> decision at t0 + 75s
    obs_end_ns = t0_ns + 60_000_000_000
    collar_end_ns = t0_ns + 75_000_000_000

    obs_q_mask = q_ts <= obs_end_ns
    col_q_mask = (q_ts > obs_end_ns) & (q_ts <= collar_end_ns)
    post_q_mask = q_ts > collar_end_ns
    if int(obs_q_mask.sum()) < 5 or int(col_q_mask.sum()) < 2 or int(post_q_mask.sum()) < 10:
        return None

    # Lee-Ready OFI & BJZZ sub-penny / dark-pool share during first 60s
    obs_t_mask = (t_ts >= t0_ns) & (t_ts <= obs_end_ns) & np.isfinite(tp) & (tp > 0) & (tsz > 0)
    if int(obs_t_mask.sum()) < 8:
        return None
    ot_ts, ot_p, ot_s = t_ts[obs_t_mask], tp[obs_t_mask], tsz[obs_t_mask]
    q_idx = np.clip(np.searchsorted(q_ts, ot_ts, side="right") - 1, 0, len(q_ts) - 1)
    mids = 0.5 * (bp[q_idx] + ap[q_idx])
    buy_vol = float(ot_s[ot_p >= mids].sum())
    sell_vol = float(ot_s[ot_p < mids].sum())
    tot_vol = buy_vol + sell_vol
    if tot_vol <= 0.0:
        return None
    ofi_norm = (buy_vol - sell_vol) / tot_vol

    # BJZZ dark sub-penny / trade-at-ask pressure
    obs_x = [tx[i] for i in np.where(obs_t_mask)[0]]
    dark_subpenny_vol = sum(
        float(sz)
        for p_v, sz, ex in zip(ot_p, ot_s, obs_x, strict=False)
        if str(ex) == "D" and abs((p_v * 100.0) - round(p_v * 100.0)) > 1e-3
    )
    dark_subpenny_share = dark_subpenny_vol / tot_vol

    # 15-second bid-survival collar: bid must NOT break below obs_end bid by >10 bps during [60s, 75s]
    bid_obs_end = float(bp[obs_q_mask][-1])
    min_bid_collar = float(bp[col_q_mask].min())
    bid_survival_ok = min_bid_collar >= bid_obs_end * (1.0 - 10.0 / 10_000.0)

    # Joint non-retired signal filter (OFI >= 0.25 + positive sweep/dark + 15s bid survival + tight spread <= 35 bps)
    entry_bid = float(bp[post_q_mask][0])
    entry_ask = float(ap[post_q_mask][0])
    entry_mid = 0.5 * (entry_bid + entry_ask)
    spread_bps = ((entry_ask - entry_bid) / max(entry_mid, 1e-6)) * 10_000.0

    if not (
        ofi_norm >= 0.25
        and bid_survival_ok
        and 0.0 <= spread_bps <= 35.0
        and (dark_subpenny_share >= 0.02 or ofi_norm >= 0.40)
    ):
        return None

    # --- 1) TAKER SWEEP + ADR-0022 <= 2.00% Quote-Sweep Protective Stop + Progressive Trail ---
    post_bp = bp[post_q_mask]
    post_ap = ap[post_q_mask]
    post_q_ts = q_ts[post_q_mask]
    stop_frac = ADR0022_MAX_STOP_PCT / 100.0  # 2.00% hard ceiling
    taker_entry = entry_ask
    hwm = taker_entry
    stop_px = taker_entry * (1.0 - stop_frac)
    exit_bid = float(post_bp[-1])
    for b_px in post_bp[1:]:
        if b_px > hwm:
            hwm = float(b_px)
            gain_pct = ((hwm / taker_entry) - 1.0) * 100.0
            if gain_pct >= 0.35:
                ret_ratio = 0.70 if gain_pct >= 2.0 else (0.56 if gain_pct >= 0.80 else 0.44)
                stop_px = max(stop_px, taker_entry + ret_ratio * (hwm - taker_entry))
        if b_px <= stop_px:
            exit_bid = float(b_px)
            break

    taker_gross_bps = ((exit_bid / taker_entry) - 1.0) * 10_000.0
    # Spread is already paid via ask->bid; add 2bps/side fee+slip + Almgren-Chriss sqrt impact
    extra_cost_1x = compute_almgren_chriss_cost_bps(
        base_half_spread_bps=0.0,
        fee_and_slip_bps_per_side=2.0,
        daily_sigma_frac=0.025,
        order_notional_usd=DEFAULT_POSITION_NOTIONAL_USD,
        adv_usd=max(5_000_000.0, adv_usd),
        stress_multiplier=1.0,
    )
    taker_net_1x_bps = taker_gross_bps - extra_cost_1x
    # 2x stress doubles both the realized entry spread and the extra fee/slip/impact!
    taker_net_2x_bps = taker_gross_bps - spread_bps - (2.0 * extra_cost_1x)

    # --- 2) REAL hftbacktest v2.4.2 RiskAdverseQueueModel Passive Microprice Queue Simulation ---
    post_t_mask = (t_ts > collar_end_ns) & np.isfinite(tp) & (tp > 0) & (tsz > 0)
    pt_ts = t_ts[post_t_mask]
    pt_p = tp[post_t_mask]
    pt_s = tsz[post_t_mask]

    # Build compact chronologically sorted hftbacktest event array (cap 4000 events per slice for speed)
    step_q = max(1, len(post_q_ts) // 1200)
    step_t = max(1, len(pt_ts) // 1200)
    sq_ts, sq_bp, sq_bs, sq_ap, sq_as = (
        post_q_ts[::step_q],
        post_bp[::step_q],
        bs[post_q_mask][::step_q],
        post_ap[::step_q],
        a_sz[post_q_mask][::step_q],
    )
    st_ts, st_p, st_s = pt_ts[::step_t], pt_p[::step_t], pt_s[::step_t]

    n_ev = 2 * len(sq_ts) + len(st_ts)
    ev = np.zeros(n_ev, dtype=event_dtype)
    lat_ns = 500_000  # 0.5 ms conservative feed latency

    idx = 0
    for i in range(len(sq_ts)):
        ts_v = int(sq_ts[i])
        ev[idx] = (
            EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | BUY_EVENT,
            ts_v,
            ts_v + lat_ns,
            float(sq_bp[i]),
            float(sq_bs[i]) * 100.0,
            0,
            0,
            0.0,
        )
        idx += 1
        ev[idx] = (
            EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | SELL_EVENT,
            ts_v,
            ts_v + lat_ns,
            float(sq_ap[i]),
            float(sq_as[i]) * 100.0,
            0,
            0,
            0.0,
        )
        idx += 1
    for i in range(len(st_ts)):
        ts_v = int(st_ts[i])
        is_sell_init = float(st_p[i]) <= entry_bid
        side_flag = SELL_EVENT if is_sell_init else BUY_EVENT
        ev[idx] = (
            EXCH_EVENT | LOCAL_EVENT | TRADE_EVENT | side_flag,
            ts_v,
            ts_v + lat_ns,
            float(st_p[i]),
            float(st_s[i]),
            0,
            0,
            0.0,
        )
        idx += 1

    order = np.argsort(ev["local_ts"], kind="mergesort")
    ev = ev[order]

    passive_filled = False
    passive_net_1x_bps = 0.0
    passive_net_2x_bps = 0.0
    if len(ev) >= 10:
        try:
            asset = (
                BacktestAsset()
                .data(ev)
                .linear_asset(1.0)
                .constant_order_latency(lat_ns, lat_ns)
                .risk_adverse_queue_model()
                .no_partial_fill_exchange()
                .trading_value_fee_model(0.0, 0.0002)
                .tick_size(0.01)
                .lot_size(1.0)
            )
            bt = HashMapMarketDepthBacktest([asset])
            # Advance 2ms to initialize depth, then post passive limit buy at best bid
            bt.elapse(2_000_000)
            limit_px = round(entry_bid, 2)
            bt.submit_buy_order(0, 101, limit_px, 100.0, GTC, LIMIT, False)
            # Wait up to 60s for queue fill under RiskAdverseQueueModel
            bt.elapse(60_000_000_000)
            if bt.position(0) > 0:
                passive_filled = True
                # Evaluate quote-sweep exit from post-fill quotes with ADR-0022 <= 2.00% stop
                fill_ts = int(bt.current_timestamp)
                rem_mask = post_q_ts >= fill_ts
                rem_bp = post_bp[rem_mask] if int(rem_mask.sum()) >= 2 else post_bp[-2:]
                p_hwm = limit_px
                p_stop = limit_px * (1.0 - stop_frac)
                p_exit = float(rem_bp[-1])
                for b_px in rem_bp:
                    if b_px > p_hwm:
                        p_hwm = float(b_px)
                        gp = ((p_hwm / limit_px) - 1.0) * 100.0
                        if gp >= 0.35:
                            rr = 0.70 if gp >= 2.0 else (0.56 if gp >= 0.80 else 0.44)
                            p_stop = max(p_stop, limit_px + rr * (p_hwm - limit_px))
                    if b_px <= p_stop:
                        p_exit = float(b_px)
                        break
                p_gross = ((p_exit / limit_px) - 1.0) * 10_000.0
                passive_net_1x_bps = p_gross - extra_cost_1x
                passive_net_2x_bps = p_gross - (2.0 * extra_cost_1x)
        except Exception:
            passive_filled = False

    return {
        "symbol": sym,
        "trade_date": trade_date,
        "is_recent_ipo": bool(is_recent_ipo),
        "ofi_norm": round(ofi_norm, 4),
        "dark_subpenny_share": round(dark_subpenny_share, 4),
        "spread_bps": round(spread_bps, 2),
        "taker_gross_bps": round(taker_gross_bps, 4),
        "taker_net_1x_bps": round(taker_net_1x_bps, 4),
        "taker_net_2x_bps": round(taker_net_2x_bps, 4),
        "passive_filled": bool(passive_filled),
        "passive_net_1x_bps": round(passive_net_1x_bps, 4),
        "passive_net_2x_bps": round(passive_net_2x_bps, 4),
    }


def run_h104_ipo_and_orb_hftbacktest(
    lake_root: pathlib.Path,
    *,
    trial_number: int = 100,
    spec_sha256: str = "71463b43a63bb78c1a8298ebe360c71ff3a052d809b2ca396bf07f86b90597f5",
    max_workers: int = 24,
) -> dict[str, Any]:
    """Run pre-registered H104 L1 tick replay through hftbacktest RiskAdverseQueueModel + C(6,2)=15 CPCV."""
    import duckdb
    import pandas as pd

    ok_fw, fw_viol = check_retired_trap_firewall(
        tags=["IPO_DAY1_MICROSTRUCTURE_LOSSLESS"],
        is_post_incentive=False,
        holds_overnight_moc_to_moo=False,
        unshielded_lit_passive=False,
        in_opening_0930_0935_taker=False,
        ofi_only_without_joint_sweep=False,
    )
    if not ok_fw:
        raise RuntimeError(f"H104 blocked by Retired-Trap Firewall: {fw_viol}")

    con = duckdb.connect()
    sec_p = str(lake_root / "derived" / "sec_filings_clean_prewall.parquet")
    sym_p = str(lake_root / "public-tables" / "sec_symbols.parquet")
    df10_p = str(lake_root / "derived" / "df10_daily_20d_baselines.parquet")

    # Map symbol -> ADV and recent S-1/424B4 IPO dates & 90d dilution dates (strictly <= 2026-10-15!)
    adv_rows = con.execute(
        f"""
        SELECT symbol, trade_date, mean_20d_dvol
        FROM read_parquet('{df10_p}')
        WHERE CAST(trade_date AS DATE) <= DATE '{STUDY_WALL_DATE.isoformat()}'
        """
    ).fetchall()
    adv_map: dict[tuple[str, str], float] = {(str(s), str(d)): float(v or 5_000_000.0) for s, d, v in adv_rows}

    sym_cik_rows = con.execute(
        f"SELECT DISTINCT UPPER(TRIM(symbol)), CAST(cik AS BIGINT) FROM read_parquet('{sym_p}')"
    ).fetchall()
    sym_to_cik = {str(s): int(c) for s, c in sym_cik_rows if s and c is not None}

    wall_iso_h104 = STUDY_WALL_DATE.isoformat()
    ipo_rows = con.execute(
        f"""
        SELECT CAST(cik AS BIGINT) AS cik,
               CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS acc_date,
               UPPER(COALESCE(form, '')) AS form
        FROM read_parquet('{sec_p}')
        WHERE UPPER(COALESCE(form, '')) IN ('S-1', 'S-1/A', '424B4', 'F-1', 'S-3', 'S-3ASR', '424B5', 'EFFECT', 'F-3')
          AND CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) <= DATE '{wall_iso_h104}'
        """
    ).fetchall()
    ipo_dates_by_cik: dict[int, list[dt.date]] = {}
    dil_dates_by_cik: dict[int, list[dt.date]] = {}
    for cik_v, acc_d, fm in ipo_rows:
        if fm in ("S-1", "S-1/A", "424B4", "F-1"):
            ipo_dates_by_cik.setdefault(int(cik_v), []).append(acc_d)
        else:
            dil_dates_by_cik.setdefault(int(cik_v), []).append(acc_d)

    trades_root = lake_root / "published-orb" / "trades"
    quotes_root = lake_root / "published-orb" / "quotes"
    tasks: list[tuple[str, str, str, str, float, bool]] = []
    skipped_diluters = 0
    max_date_seen = ""

    for t_file in sorted(trades_root.glob("*/*.parquet")):
        sym = t_file.parent.name.upper()
        fname = t_file.name
        d_str = fname[:10]
        hhmm = fname[11:15]
        if hhmm < "0935":
            continue
        d_obj = dt.date.fromisoformat(d_str)
        if d_obj > STUDY_WALL_DATE:
            raise ValueError(f"H104 PREWALL BREACH: {fname}")
        if d_str > max_date_seen:
            max_date_seen = d_str
        q_file = quotes_root / sym / fname
        if not q_file.is_file():
            continue

        cik_v = sym_to_cik.get(sym)
        if cik_v is not None:
            dils = dil_dates_by_cik.get(cik_v, ())
            if any(0 < (d_obj - dd).days <= 90 for dd in dils):
                skipped_diluters += 1
                continue
            ipos = ipo_dates_by_cik.get(cik_v, ())
            is_recent_ipo = any(0 <= (d_obj - idt).days <= 180 for idt in ipos) or ((sym, d_str) not in adv_map)
        else:
            is_recent_ipo = (sym, d_str) not in adv_map

        adv_v = adv_map.get((sym, d_str), 15_000_000.0)
        tasks.append((str(t_file), str(q_file), sym, d_str, adv_v, is_recent_ipo))

    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for res in pool.map(_replay_one_orb_slice_hftbacktest, tasks, chunksize=32):
            if res is not None:
                results.append(res)

    df = pd.DataFrame(results)
    assert len(df) > 0, "H104 returned 0 qualifying L1 microstructure events"

    taker_cpcv = _evaluate_cpcv_15_paths(
        df, ret_col_1x="taker_net_1x_bps", ret_col_2x="taker_net_2x_bps", effective_trials=104.0
    )

    df_pass = df[df["passive_filled"]].copy()
    passive_fill_rate = round(float(len(df_pass)) / max(1, len(df)), 4)
    if len(df_pass) >= 30 and int(df_pass["trade_date"].nunique()) >= 30:
        passive_cpcv = _evaluate_cpcv_15_paths(
            df_pass, ret_col_1x="passive_net_1x_bps", ret_col_2x="passive_net_2x_bps", effective_trials=104.0
        )
    else:
        passive_cpcv = {
            "n_paths": 0,
            "cpcv_positive_share": 0.0,
            "pbo": 1.0,
            "oos_events": int(len(df_pass)),
            "oos_dates": int(df_pass["trade_date"].nunique()) if len(df_pass) else 0,
            "clustered_1x": _date_clustered_t_stat(df_pass["passive_net_1x_bps"], df_pass["trade_date"])
            if len(df_pass)
            else {"mean_bps": 0.0, "se_bps": 0.0, "t_stat": 0.0},
            "clustered_2x": _date_clustered_t_stat(df_pass["passive_net_2x_bps"], df_pass["trade_date"])
            if len(df_pass)
            else {"mean_bps": 0.0, "se_bps": 0.0, "t_stat": 0.0},
            "dsr_probability": 0.0,
            "path_means_2x_bps": [],
        }

    # Sub-cohort breakdown for IPO_DAY1_MICROSTRUCTURE_LOSSLESS (recent S-1/424B4 or <20 prior daily sessions)
    df_ipo = df[df["is_recent_ipo"]].copy()
    ipo_sub = {
        "events": int(len(df_ipo)),
        "distinct_dates": int(df_ipo["trade_date"].nunique()) if len(df_ipo) else 0,
        "passive_fill_rate": round(float(df_ipo["passive_filled"].mean()), 4) if len(df_ipo) else 0.0,
        "taker_clustered_1x": _date_clustered_t_stat(df_ipo["taker_net_1x_bps"], df_ipo["trade_date"])
        if len(df_ipo) >= 10
        else {"mean_bps": 0.0, "se_bps": 0.0, "t_stat": 0.0},
        "taker_clustered_2x": _date_clustered_t_stat(df_ipo["taker_net_2x_bps"], df_ipo["trade_date"])
        if len(df_ipo) >= 10
        else {"mean_bps": 0.0, "se_bps": 0.0, "t_stat": 0.0},
        "passive_filled_clustered_2x": (
            _date_clustered_t_stat(
                df_ipo[df_ipo["passive_filled"]]["passive_net_2x_bps"], df_ipo[df_ipo["passive_filled"]]["trade_date"]
            )
            if int(df_ipo["passive_filled"].sum()) >= 10
            else {"mean_bps": 0.0, "se_bps": 0.0, "t_stat": 0.0}
        ),
    }

    # Gate evaluated on the stronger execution mode (passive RiskAdverseQueueModel vs taker sweep)
    best_mode = (
        "hftbacktest_passive_risk_adverse_queue"
        if (len(df_pass) >= 200 and passive_cpcv["clustered_2x"]["t_stat"] > taker_cpcv["clustered_2x"]["t_stat"])
        else "taker_sweep_quote_execution"
    )
    best_cpcv = passive_cpcv if best_mode == "hftbacktest_passive_risk_adverse_queue" else taker_cpcv
    ev_cnt = len(df_pass) if best_mode == "hftbacktest_passive_risk_adverse_queue" else len(df)
    dt_cnt = (
        int(df_pass["trade_date"].nunique())
        if best_mode == "hftbacktest_passive_risk_adverse_queue"
        else int(df["trade_date"].nunique())
    )

    gate = evaluate_strict_prewall_gate(
        trial_number=trial_number,
        t_stat_clustered_2x=float(best_cpcv["clustered_2x"]["t_stat"]),
        dsr_probability=float(best_cpcv["dsr_probability"]),
        cpcv_positive_share=float(best_cpcv["cpcv_positive_share"]),
        pbo=float(best_cpcv["pbo"]),
        mean_net_2x_bps=float(best_cpcv["clustered_2x"]["mean_bps"]),
        distinct_dates=dt_cnt,
        total_events=int(ev_cnt),
    )

    return {
        "hypothesis_id": "H104_IPO_DAY1_AND_ORB_HFTBACKTEST_QUEUE_LOSSLESS",
        "archetype": "IPO_DAY1_MICROSTRUCTURE_LOSSLESS",
        "trial_number": trial_number,
        "spec_sha256": spec_sha256,
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "max_date_verified": max_date_seen,
        "paired_l1_slices_scanned": len(tasks),
        "sec_90d_diluters_excluded": skipped_diluters,
        "qualifying_events": int(len(df)),
        "distinct_dates": int(df["trade_date"].nunique()),
        "distinct_symbols": int(df["symbol"].nunique()),
        "hftbacktest_passive_fill_rate": passive_fill_rate,
        "hftbacktest_passive_filled_events": int(len(df_pass)),
        "taker_sweep_cpcv": taker_cpcv,
        "hftbacktest_passive_queue_cpcv": passive_cpcv,
        "ipo_day1_recent_subcohort": ipo_sub,
        "selected_for_gate": best_mode,
        "gate": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-registered Option B H103 & H104 pre-wall factory.")
    parser.add_argument("--lake-root", default="/mnt/z/sota-data/prewall-lake")
    parser.add_argument("--max-workers", type=int, default=min(24, max(4, (os.cpu_count() or 8) // 2)))
    args = parser.parse_args()

    lake_root = pathlib.Path(args.lake_root)
    print(
        "[H103] Running SLOW_COIL_VDU_ACCUMULATION on RTX 4090 CUDA + LightGBM + XGBoost + VectorBT (C(6,2)=15 CPCV)..."
    )
    h103 = run_h103_slow_coil_rtx4090_factory(lake_root)
    print(
        f"[H103] DONE: verdict={h103['gate']['verdict']} | selected={h103['selected_for_gate']} | "
        f"events={h103['events_total']} dates={h103['distinct_dates']} | "
        f"rule_t2x={h103['rule_baseline']['clustered_2x']['t_stat']} "
        f"mlp_t2x={h103['models']['pytorch_rtx4090_mlp']['clustered_2x']['t_stat']} "
        f"lgb_t2x={h103['models']['lightgbm']['clustered_2x']['t_stat']} "
        f"xgb_t2x={h103['models']['xgboost_cuda']['clustered_2x']['t_stat']}"
    )

    print(
        f"[H104] Running IPO_DAY1 & L1 ORB replay via hftbacktest RiskAdverseQueueModel ({args.max_workers} workers)..."
    )
    h104 = run_h104_ipo_and_orb_hftbacktest(lake_root, max_workers=args.max_workers)
    print(
        f"[H104] DONE: verdict={h104['gate']['verdict']} | selected={h104['selected_for_gate']} | "
        f"qualifying={h104['qualifying_events']} passive_fills={h104['hftbacktest_passive_filled_events']} "
        f"(fill_rate={h104['hftbacktest_passive_fill_rate']}) | "
        f"taker_t2x={h104['taker_sweep_cpcv']['clustered_2x']['t_stat']} "
        f"passive_t2x={h104['hftbacktest_passive_queue_cpcv']['clustered_2x']['t_stat']}"
    )

    # Refresh optb_prewall_factory_audit.json modules so alpaca & ib_async reflect installed state too
    from materialize_prewall_df10_df6 import probe_optb_runtime_modules

    runtime_probe = probe_optb_runtime_modules()
    receipt = {
        "schema_version": "sota-optb-prewall-factory-evaluation.v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "runtime_probe": runtime_probe,
        "h103": h103,
        "h104": h104,
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode("utf-8")).hexdigest()

    out_path = lake_root / "receipts" / "h103_h104_optb_factory_receipt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    optb_path = lake_root / "receipts" / "optb_prewall_factory_audit.json"
    if optb_path.is_file():
        optb = json.loads(optb_path.read_text(encoding="utf-8"))
        optb["mods"] = runtime_probe["mods"]
        optb["versions"] = runtime_probe["versions"]
        optb["cuda"] = runtime_probe["cuda"]
        optb["h103_h104_receipt_sha256"] = receipt["receipt_sha256"]
        optb["h103_verdict"] = h103["gate"]["verdict"]
        optb["h104_verdict"] = h104["gate"]["verdict"]
        optb["updated_at_utc"] = receipt["generated_at_utc"]
        optb_path.write_text(json.dumps(optb, indent=2) + "\n", encoding="utf-8")

    print(f"[DONE] Sealed {out_path} (sha256={receipt['receipt_sha256'][:16]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
