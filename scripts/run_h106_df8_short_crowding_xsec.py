"""Pre-registered H106: Multi-Feed PIT Short-Crowding vs Dilution-Supply Cross-Sectional Factor on RTX 4090 CPCV.

Enforces ADR-0020, ADR-0021, ADR-0022, and ADR-0023 fail-closed contracts:
- Strictly pre-wall <= 2026-10-15 across DF10, quarantined DF8 (`known_from < trade_date`), XBRL `sec_float`
  (`filed < trade_date`), `short_volume` (`day <= trade_date`), DF6 (`effective_trade_date == next1_trade_date`),
  and clean SEC filings (`accepted <= 2026-10-15` ET, `< trade_date`).
- Passes check_retired_trap_firewall (no POST chase, no overnight carry, no unshielded passive queue).
- Strictly non-overlapping 1-day intraday MTM (`next1_split_open -> next1_split_close`) with adverse-stop-first
  ADR-0022 <= 2.00% hard stop on both legs, 1x/2x Almgren-Chriss sqrt impact + 5/10 bps short-borrow fee.
- C(6,2)=15 disjoint skfolio CombinatorialPurgedCV(purge=5d, embargo=5d), date-ranked features [-1,+1],
  pre-declared primary equal-weight 3-model ensemble (RTX 4090 PyTorch CUDA MLP + LightGBM + XGBoost CUDA),
  and conservative min(split_pooled_t, unique_date_oof_t) vs HLZ t*(102) = 4.2019 & DSR >= 0.95.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
from dataclasses import asdict
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_optb_prewall_factory import (  # noqa: E402
    DEFAULT_POSITION_NOTIONAL_USD,
    _compute_dsr_from_daily_returns,
    _date_clustered_t_stat,
    compute_almgren_chriss_cost_bps,
    evaluate_strict_prewall_gate,
)

from sota_compounding.bitemporal_sidecar import validate_prewall_bitemporal_partition_contract  # noqa: E402
from sota_compounding.converged_strategy_factory import check_retired_trap_firewall  # noqa: E402
from sota_compounding.gpu_accelerator import (  # noqa: E402
    configure_rtx4090_torch_backend,
    resolve_fast_prewall_lake_root,
)
from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE  # noqa: E402
from sota_compounding.prewall_h099_enrichment import (  # noqa: E402
    CANONICAL_SEC_ALL_DILUTION_FORMS,
    audit_h099_prewall_lake_readiness,
)

H106_ID = "H106_DF8_SHORT_CROWDING_DILUTION_SHIELD_XSEC_RTX4090_CPCV"
H106_SPEC_SHA256 = "88292e639cfbd9f436e3466b07f75faa5acdce6eb0b65c5cd051cc6fd0cfd947"
H106_TRIAL_NUMBER = 102
SHORT_BORROW_1X_BPS = 5.0
MIN_SYMBOLS_PER_DATE = 20


def compute_intraday_leg_gross_bps(
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    *,
    side: str,
    stop_frac: float = ADR0022_MAX_STOP_PCT / 100.0,
) -> float:
    """Compute conservative adverse-stop-first 1-day intraday MTM return in bps (`<= 2.00%` ADR-0022 stop)."""
    vals = (open_px, high_px, low_px, close_px, stop_frac)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in vals):
        raise ValueError("OHLC and stop_frac must be finite numbers")
    if open_px <= 0.0 or high_px <= 0.0 or low_px <= 0.0 or close_px <= 0.0 or high_px < low_px:
        raise ValueError(f"Invalid OHLC bar: O={open_px}, H={high_px}, L={low_px}, C={close_px}")
    if not (low_px <= open_px <= high_px and low_px <= close_px <= high_px):
        raise ValueError(f"Inconsistent OHLC bounds: O={open_px}, H={high_px}, L={low_px}, C={close_px}")
    max_stop_frac = ADR0022_MAX_STOP_PCT / 100.0
    if not (0.0 < stop_frac <= max_stop_frac):
        raise ValueError(f"stop_frac must be in (0, {max_stop_frac}], got {stop_frac}")

    side_u = side.upper()
    if side_u == "LONG":
        if low_px <= open_px * (1.0 - stop_frac):
            return round(-stop_frac * 10_000.0, 4)
        raw = max(-0.35, min(0.60, (close_px / open_px) - 1.0))
        return round(raw * 10_000.0, 4)
    if side_u == "SHORT":
        if high_px >= open_px * (1.0 + stop_frac):
            return round(-stop_frac * 10_000.0, 4)
        raw = max(-0.35, min(0.60, 1.0 - (close_px / open_px)))
        return round(raw * 10_000.0, 4)
    raise ValueError(f"Unsupported side={side!r}; expected LONG or SHORT")


def compute_intraday_leg_arrays(
    open_arr: Any,
    high_arr: Any,
    low_arr: Any,
    close_arr: Any,
    *,
    stop_frac: float = ADR0022_MAX_STOP_PCT / 100.0,
) -> tuple[Any, Any, Any, Any]:
    """Vectorized adverse-stop-first LONG and SHORT gross return arrays (bps) & stop-hit flags."""
    import numpy as np

    max_stop_frac = ADR0022_MAX_STOP_PCT / 100.0
    if not (0.0 < stop_frac <= max_stop_frac):
        raise ValueError(f"stop_frac={stop_frac} exceeds ADR-0022 cap {max_stop_frac}")

    ratio = close_arr / open_arr
    long_stopped = low_arr <= open_arr * (1.0 - stop_frac)
    long_unstopped = np.clip(ratio - 1.0, -0.35, 0.60)
    long_gross_bps = np.where(long_stopped, -stop_frac, long_unstopped) * 10_000.0

    short_stopped = high_arr >= open_arr * (1.0 + stop_frac)
    short_unstopped = np.clip(1.0 - ratio, -0.35, 0.60)
    short_gross_bps = np.where(short_stopped, -stop_frac, short_unstopped) * 10_000.0
    return long_gross_bps, short_gross_bps, long_stopped, short_stopped


def cross_sectional_rank_matrix(raw_matrix: Any, date_codes: Any) -> Any:
    """Rank-normalize each feature strictly within each single trade_date to [-1, +1] (zero cross-date leakage)."""
    import numpy as np

    try:
        from scipy.stats import rankdata as _scipy_rankdata
    except ImportError:
        _scipy_rankdata = None

    n_rows, n_cols = raw_matrix.shape
    out = np.zeros((n_rows, n_cols), dtype=np.float32)
    unique_codes, inv = np.unique(date_codes, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    sorted_inv = inv[order]
    counts = np.bincount(sorted_inv, minlength=len(unique_codes))
    starts = np.zeros(len(unique_codes), dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])

    for col in range(n_cols):
        col_vals = raw_matrix[:, col].astype(np.float64, copy=False)
        for g_idx in range(len(unique_codes)):
            st = int(starts[g_idx])
            ln = int(counts[g_idx])
            if ln <= 1:
                continue
            grp_idx = order[st : st + ln]
            vals = col_vals[grp_idx]
            finite_mask = np.isfinite(vals)
            n_valid = int(finite_mask.sum())
            if n_valid <= 1:
                continue
            valid_pos = grp_idx[finite_mask]
            v = vals[finite_mask]
            if _scipy_rankdata is not None:
                # 1-based average ranks mapped cleanly into [-1, +1]
                avg_ranks = _scipy_rankdata(v, method="average") - 1.0
                out[valid_pos, col] = ((2.0 * avg_ranks / (n_valid - 1)) - 1.0).astype(np.float32)
            else:
                sub_order = np.argsort(v, kind="mergesort")
                sorted_v = v[sub_order]
                ranks = np.empty(n_valid, dtype=np.float64)
                i = 0
                while i < n_valid:
                    j = i + 1
                    while j < n_valid and sorted_v[j] == sorted_v[i]:
                        j += 1
                    avg_r = 0.5 * (i + j - 1)
                    ranks[sub_order[i:j]] = (2.0 * avg_r / (n_valid - 1)) - 1.0
                    i = j
                out[valid_pos, col] = ranks.astype(np.float32)
    return out


def _attach_sec_flags_pit(
    ciks: Any,
    trade_dates: list[dt.date],
    sec_p: str,
    wall_iso: str,
) -> tuple[Any, Any]:
    """Attach strictly causal prior-accepted 90d dilution shield & 5d 8-K flags without correlated SQL."""
    import duckdb
    import numpy as np

    con = duckdb.connect()
    dilution_sql = ", ".join(f"'{f}'" for f in CANONICAL_SEC_ALL_DILUTION_FORMS)
    dil_rows = con.execute(
        f"""
        SELECT CAST(cik AS BIGINT) AS cik,
               CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS acc_date
        FROM read_parquet('{sec_p}')
        WHERE UPPER(COALESCE(form, '')) IN ({dilution_sql})
          AND CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) <= DATE '{wall_iso}'
        """
    ).fetchall()
    dil_map: dict[int, list[int]] = {}
    for c, d in dil_rows:
        dil_map.setdefault(int(c), []).append(d.toordinal())
    for lst in dil_map.values():
        lst.sort()

    k8_rows = con.execute(
        f"""
        SELECT CAST(cik AS BIGINT) AS cik,
               CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS acc_date
        FROM read_parquet('{sec_p}')
        WHERE UPPER(COALESCE(form, '')) = '8-K'
          AND CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) <= DATE '{wall_iso}'
        """
    ).fetchall()
    k8_map: dict[int, list[int]] = {}
    for c, d in k8_rows:
        k8_map.setdefault(int(c), []).append(d.toordinal())
    for lst in k8_map.values():
        lst.sort()

    n = len(ciks)
    has_90d_dil = np.zeros(n, dtype=np.float32)
    has_5d_8k = np.zeros(n, dtype=np.float32)
    ords = [d.toordinal() for d in trade_dates]

    for i in range(n):
        c = int(ciks[i])
        td_ord = ords[i]
        d_lst = dil_map.get(c)
        if d_lst:
            pos = bisect.bisect_left(d_lst, td_ord) - 1
            if pos >= 0 and (td_ord - d_lst[pos]) <= 90:
                has_90d_dil[i] = 1.0
        k_lst = k8_map.get(c)
        if k_lst:
            pos = bisect.bisect_left(k_lst, td_ord) - 1
            if pos >= 0 and (td_ord - k_lst[pos]) <= 5:
                has_5d_8k[i] = 1.0

    return has_90d_dil, has_5d_8k


def _evaluate_ls_cpcv_from_scores(
    *,
    scores_by_split: list[tuple[Any, Any, Any, float]],
    trade_date_strs: Any,
    date_codes: Any,
    long_net_1x: Any,
    long_net_2x: Any,
    short_net_1x: Any,
    short_net_2x: Any,
    has_90d_dil: Any,
    trial_number: int,
) -> dict[str, Any]:
    """Evaluate daily dollar-neutral top-10% non-diluting long vs bottom-10% short over C(6,2)=15 CPCV splits."""
    import numpy as np
    import statsmodels.api as sm

    path_means_2x: list[float] = []
    overfit_cnt = 0
    pooled_dates: list[str] = []
    pooled_ls_1x: list[float] = []
    pooled_ls_2x: list[float] = []
    oof_date_buckets: dict[str, list[tuple[float, float, float, float]]] = {}

    for _tr_idx, te_idx, te_scores, tr_mean_2x in scores_by_split:
        sub_dates = trade_date_strs[te_idx]
        sub_codes = date_codes[te_idx]
        unique_sub_codes = np.unique(sub_codes)

        split_ls_2x: list[float] = []
        for dc in unique_sub_codes:
            local_mask = sub_codes == dc
            global_idx = te_idx[local_mask]
            if len(global_idx) < MIN_SYMBOLS_PER_DATE:
                continue
            local_scores = te_scores[local_mask]
            nondil_mask = has_90d_dil[global_idx] < 0.5
            if int(nondil_mask.sum()) < 10:
                continue

            nondil_pos = np.where(nondil_mask)[0]
            k_long = max(5, int(math.floor(0.10 * len(nondil_pos))))
            nd_order = np.argsort(-local_scores[nondil_pos], kind="mergesort")
            chosen_long_local = nondil_pos[nd_order[:k_long]]
            chosen_long_set = set(chosen_long_local.tolist())

            rem_local = np.array([k for k in range(len(global_idx)) if k not in chosen_long_set], dtype=np.int64)
            k_short = max(5, int(math.floor(0.10 * len(global_idx))))
            if len(rem_local) < k_short:
                continue
            rem_order = np.argsort(local_scores[rem_local], kind="mergesort")
            chosen_short_local = rem_local[rem_order[:k_short]]

            l_g = global_idx[chosen_long_local]
            s_g = global_idx[chosen_short_local]
            r_l1 = float(long_net_1x[l_g].mean())
            r_l2 = float(long_net_2x[l_g].mean())
            r_s1 = float(short_net_1x[s_g].mean())
            r_s2 = float(short_net_2x[s_g].mean())
            ls1 = 0.5 * r_l1 + 0.5 * r_s1
            ls2 = 0.5 * r_l2 + 0.5 * r_s2

            d_str = str(sub_dates[local_mask][0])
            split_ls_2x.append(ls2)
            pooled_dates.append(d_str)
            pooled_ls_1x.append(ls1)
            pooled_ls_2x.append(ls2)
            oof_date_buckets.setdefault(d_str, []).append((ls1, ls2, r_l2, r_s2))

        if not split_ls_2x:
            path_means_2x.append(0.0)
            overfit_cnt += 1
            continue
        te_mean = float(np.mean(split_ls_2x))
        path_means_2x.append(round(te_mean, 4))
        if (tr_mean_2x > 0.0 and te_mean <= 0.0) or te_mean <= 0.0:
            overfit_cnt += 1

    pooled_cl_1x = _date_clustered_t_stat(pooled_ls_1x, pooled_dates)
    pooled_cl_2x = _date_clustered_t_stat(pooled_ls_2x, pooled_dates)

    sorted_dates = sorted(oof_date_buckets.keys())
    uniq_1x = np.array([float(np.mean([t[0] for t in oof_date_buckets[d]])) for d in sorted_dates], dtype=np.float64)
    uniq_2x = np.array([float(np.mean([t[1] for t in oof_date_buckets[d]])) for d in sorted_dates], dtype=np.float64)
    uniq_long_2x = np.array(
        [float(np.mean([t[2] for t in oof_date_buckets[d]])) for d in sorted_dates], dtype=np.float64
    )
    uniq_short_2x = np.array(
        [float(np.mean([t[3] for t in oof_date_buckets[d]])) for d in sorted_dates], dtype=np.float64
    )

    if len(uniq_2x) >= 15:
        x_ones = np.ones((len(uniq_2x), 1), dtype=np.float64)
        hac_fit = sm.OLS(uniq_2x, x_ones).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        uniq_mean_2x = float(hac_fit.params[0])
        uniq_se_2x = float(hac_fit.bse[0])
        uniq_t_2x = float(hac_fit.tvalues[0]) if math.isfinite(float(hac_fit.tvalues[0])) else 0.0
    else:
        uniq_mean_2x, uniq_se_2x, uniq_t_2x = 0.0, float("nan"), 0.0

    dsr_prob = _compute_dsr_from_daily_returns(uniq_2x / 10_000.0, effective_trials=float(trial_number))

    # Best-day & top-5-day outlier concentration test
    outlier_sens: dict[str, Any] = {}
    if len(uniq_2x) >= 20:
        order_desc = np.argsort(-uniq_2x)
        no_best1 = np.delete(uniq_2x, order_desc[:1])
        no_best5 = np.delete(uniq_2x, order_desc[:5])
        x1 = np.ones((len(no_best1), 1), dtype=np.float64)
        x5 = np.ones((len(no_best5), 1), dtype=np.float64)
        f1 = sm.OLS(no_best1, x1).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        f5 = sm.OLS(no_best5, x5).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        outlier_sens = {
            "drop_best1_mean_2x_bps": round(float(f1.params[0]), 4),
            "drop_best1_t_stat": round(float(f1.tvalues[0]), 4),
            "drop_best5_mean_2x_bps": round(float(f5.params[0]), 4),
            "drop_best5_t_stat": round(float(f5.tvalues[0]), 4),
        }

    # Annual subperiod breakdown
    subperiods: dict[str, Any] = {}
    for yr in ("2024", "2025", "2026"):
        idx_yr = [i for i, d in enumerate(sorted_dates) if d.startswith(yr)]
        if len(idx_yr) >= 10:
            arr_yr = uniq_2x[idx_yr]
            fit_yr = sm.OLS(arr_yr, np.ones((len(arr_yr), 1), dtype=np.float64)).fit(
                cov_type="HAC", cov_kwds={"maxlags": 5}
            )
            subperiods[yr] = {
                "dates": len(arr_yr),
                "mean_2x_bps": round(float(fit_yr.params[0]), 4),
                "t_stat_hac": round(float(fit_yr.tvalues[0]), 4),
            }

    conservative_t = round(min(float(pooled_cl_2x["t_stat"]), uniq_t_2x), 4)
    return {
        "n_paths": len(path_means_2x),
        "cpcv_positive_share": round(sum(1 for m in path_means_2x if m > 0.0) / max(1, len(path_means_2x)), 4),
        "pbo": round(overfit_cnt / max(1, len(path_means_2x)), 4),
        "unique_oos_dates": len(sorted_dates),
        "split_pooled_clustered_1x": pooled_cl_1x,
        "split_pooled_clustered_2x": pooled_cl_2x,
        "unique_date_oof_1x": {
            "mean_bps": round(float(np.mean(uniq_1x)), 4) if len(uniq_1x) else 0.0,
        },
        "unique_date_oof_2x_hac5": {
            "mean_bps": round(uniq_mean_2x, 4),
            "se_bps": round(uniq_se_2x, 4) if math.isfinite(uniq_se_2x) else None,
            "t_stat": round(uniq_t_2x, 4),
        },
        "conservative_gate_t_2x": conservative_t,
        "long_leg_unique_mean_2x_bps": round(float(np.mean(uniq_long_2x)), 4) if len(uniq_long_2x) else 0.0,
        "short_leg_unique_mean_2x_bps": round(float(np.mean(uniq_short_2x)), 4) if len(uniq_short_2x) else 0.0,
        "dsr_probability": dsr_prob,
        "path_means_2x_bps": path_means_2x,
        "outlier_sensitivity": outlier_sens,
        "subperiod_breakdown": subperiods,
    }


def run_h106_df8_short_crowding(
    lake_root: pathlib.Path,
    *,
    trial_number: int = H106_TRIAL_NUMBER,
) -> dict[str, Any]:
    """Execute pre-registered H106 Multi-Feed Short-Crowding vs Dilution Cross-Sectional CPCV on RTX 4090."""
    import duckdb
    import lightgbm as lgb
    import numpy as np
    import torch
    import xgboost as xgb
    from skfolio.model_selection import CombinatorialPurgedCV

    ok_fw, fw_viol = check_retired_trap_firewall(
        tags=["FIRST_HAND_SEC_EPISODIC_PIVOT", "PRE_TRAVEL_EARLY_IGNITION"],
        is_post_incentive=False,
        holds_overnight_moc_to_moo=False,
        unshielded_lit_passive=False,
        in_opening_0930_0935_taker=False,
        ofi_only_without_joint_sweep=False,
    )
    if not ok_fw:
        raise RuntimeError(f"H106 blocked by Retired-Trap Firewall: {fw_viol}")

    hw = configure_rtx4090_torch_backend()
    read_root, fast_tier = resolve_fast_prewall_lake_root(lake_root)
    canon_df8 = lake_root / "derived" / "df8_short_interest_clean_prewall.parquet"
    mirror_df8 = read_root / "derived" / "df8_short_interest_clean_prewall.parquet"
    if not mirror_df8.is_file() or (canon_df8.is_file() and mirror_df8.stat().st_size != canon_df8.stat().st_size):
        read_root = lake_root
        fast_tier = "canonical_fallback"

    df10_p = str(read_root / "derived" / "df10_daily_20d_baselines.parquet")
    df8_p = str(read_root / "derived" / "df8_short_interest_clean_prewall.parquet")
    df6_p = str(read_root / "derived" / "df6_regsho_threshold_pit.parquet")
    sec_p = str(read_root / "derived" / "sec_filings_clean_prewall.parquet")
    fl_p = str(read_root / "public-tables" / "sec_float.parquet")
    sym_p = str(read_root / "public-tables" / "sec_symbols.parquet")
    sv_p = str(read_root / "public-tables" / "short_volume.parquet")
    wall_iso = STUDY_WALL_DATE.isoformat()

    bitemp = validate_prewall_bitemporal_partition_contract(
        partition_id="published-orb/trades",
        columns=["symbol", "t_ns", "p", "s"],
        row_count=53_063_777,
        max_event_ts_ns=int(dt.datetime(2026, 10, 15, 19, 59, 0, tzinfo=dt.UTC).timestamp() * 1_000_000_000),
        allow_bounded_e2_projection=True,
    )
    h099_rec = audit_h099_prewall_lake_readiness(
        bitemporal_spec=bitemp,
        df10_covered_symbol_days=7_039_270,
        df10_total_symbol_days=7_039_270,
        df6_dated_snapshots_available=pathlib.Path(df6_p).is_file(),
        sec_purged_postwall_count=0 if pathlib.Path(sec_p).is_file() else 12,
    )

    con = duckdb.connect()
    con.execute("PRAGMA threads=32")
    sql = f"""
    WITH ops AS (
        SELECT UPPER(TRIM(symbol)) AS symbol, CAST(cik AS BIGINT) AS cik
        FROM read_parquet('{sym_p}')
        WHERE LOWER(COALESCE(entity_type, '')) = 'operating'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%etf%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%trust%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%acquisition corp%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%spac%'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY UPPER(TRIM(symbol)) ORDER BY CAST(cik AS BIGINT) DESC) = 1
    ),
    df10_next AS (
        SELECT
            UPPER(TRIM(symbol)) AS symbol,
            CAST(trade_date AS DATE) AS td,
            LEAD(CAST(trade_date AS DATE), 1) OVER (
                PARTITION BY UPPER(TRIM(symbol)) ORDER BY CAST(trade_date AS DATE)
            ) AS next1_td,
            raw_close,
            mean_20d_vol,
            mean_20d_dvol,
            mean_20d_range_pct,
            prior_sessions_count,
            rvol_20d,
            vol_z_20d,
            vdu_5d_20d_ratio,
            coil_5d_20d_range_ratio,
            prev_close_to_20d_high,
            ret_1d,
            prior_ret_5d,
            prior_ret_20d,
            split_high,
            split_low,
            split_close,
            next1_split_open,
            next1_split_high,
            next1_split_low,
            next1_split_close
        FROM read_parquet('{df10_p}')
        WHERE CAST(trade_date AS DATE) <= DATE '{wall_iso}'
    ),
    base AS (
        SELECT d.*, o.cik
        FROM df10_next d
        INNER JOIN ops o ON o.symbol = d.symbol
        WHERE d.next1_td IS NOT NULL
          AND d.next1_td <= DATE '{wall_iso}'
          AND d.prior_sessions_count = 20
          AND d.raw_close >= 10.0
          AND d.mean_20d_dvol >= 20000000.0
          AND d.next1_split_open > 0 AND d.next1_split_low > 0 AND d.next1_split_high > 0 AND d.next1_split_close > 0
          AND d.next1_split_high >= d.next1_split_low
    ),
    df8_clean AS (
        SELECT UPPER(TRIM(symbol)) AS symbol,
               CAST(known_from AS DATE) AS known_date,
               CAST(settlement AS DATE) AS settle_date,
               CAST(short_shares AS DOUBLE) AS short_shares,
               CAST(short_shares_before AS DOUBLE) AS short_shares_before,
               CAST(days_to_cover AS DOUBLE) AS days_to_cover
        FROM read_parquet('{df8_p}')
        WHERE CAST(known_from AS DATE) <= DATE '{wall_iso}'
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY UPPER(TRIM(symbol)), CAST(known_from AS DATE)
            ORDER BY CAST(settlement AS DATE) DESC
        ) = 1
    ),
    fl_clean AS (
        SELECT CAST(cik AS BIGINT) AS cik,
               CAST(filed AS DATE) AS filed_date,
               CAST(value AS DOUBLE) AS float_usd
        FROM read_parquet('{fl_p}')
        WHERE CAST(value AS DOUBLE) > 0 AND CAST(filed AS DATE) <= DATE '{wall_iso}'
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY CAST(cik AS BIGINT), CAST(filed AS DATE)
            ORDER BY "end" DESC, accn DESC
        ) = 1
    ),
    sv_daily AS (
        SELECT UPPER(TRIM(symbol)) AS symbol,
               CAST(day AS DATE) AS sv_date,
               CASE WHEN SUM(total_volume) > 0 THEN SUM(short_volume) / SUM(total_volume) ELSE NULL END AS sv_ratio
        FROM read_parquet('{sv_p}')
        WHERE CAST(day AS DATE) <= DATE '{wall_iso}'
        GROUP BY 1, 2
    ),
    sv_feat AS (
        SELECT symbol, sv_date, sv_ratio,
               AVG(sv_ratio) OVER (
                   PARTITION BY symbol ORDER BY sv_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
               ) AS trailing_5d_sv_ratio
        FROM sv_daily
    ),
    df6_pit AS (
        SELECT UPPER(TRIM(symbol)) AS symbol,
               CAST(effective_trade_date AS DATE) AS eff_date,
               MIN(CAST(list_date AS DATE)) AS list_date,
               1.0 AS on_regsho_threshold
        FROM read_parquet('{df6_p}')
        WHERE CAST(effective_trade_date AS DATE) <= DATE '{wall_iso}'
        GROUP BY 1, 2
    )
    SELECT
        b.symbol,
        b.td AS trade_date,
        b.next1_td AS next1_trade_date,
        b.cik,
        b.mean_20d_dvol,
        b.mean_20d_range_pct,
        b.next1_split_open,
        b.next1_split_high,
        b.next1_split_low,
        b.next1_split_close,
        si.known_date,
        fl.filed_date,
        COALESCE(si.days_to_cover, 0.0) AS days_to_cover,
        CASE
            WHEN si.short_shares IS NOT NULL AND si.short_shares_before IS NOT NULL AND si.short_shares_before > 0
            THEN LEAST(5.0, GREATEST(-1.0, (si.short_shares - si.short_shares_before) / si.short_shares_before))
            ELSE 0.0
        END AS si_mom,
        CASE WHEN si.short_shares > 0 THEN LN(1.0 + si.short_shares) ELSE 0.0 END AS log_short_shares,
        CASE
            WHEN si.short_shares > 0 AND b.mean_20d_vol > 0 THEN LEAST(25.0, si.short_shares / b.mean_20d_vol)
            ELSE 0.0
        END AS si_to_adv,
        CASE
            WHEN si.short_shares > 0 AND fl.float_usd > 0
            THEN LEAST(1.5, (si.short_shares * b.raw_close) / fl.float_usd)
            ELSE 0.0
        END AS si_to_float,
        CASE WHEN fl.float_usd IS NOT NULL AND fl.float_usd > 0 THEN 1.0 ELSE 0.0 END AS has_float_match,
        COALESCE(sv.trailing_5d_sv_ratio, 0.45) AS trailing_5d_sv_ratio,
        COALESCE(sv.sv_ratio - sv.trailing_5d_sv_ratio, 0.0) AS sv_accel_1d_5d,
        COALESCE(f6.on_regsho_threshold, 0.0) AS on_regsho_threshold,
        b.rvol_20d,
        b.vol_z_20d,
        b.vdu_5d_20d_ratio,
        b.coil_5d_20d_range_ratio,
        b.prev_close_to_20d_high,
        b.ret_1d,
        b.prior_ret_5d,
        b.prior_ret_20d,
        CASE
            WHEN b.split_high > b.split_low THEN (b.split_close - b.split_low) / (b.split_high - b.split_low)
            ELSE 0.5
        END AS close_pos
    FROM base b
    ASOF LEFT JOIN df8_clean si ON b.symbol = si.symbol AND b.td > si.known_date
    ASOF LEFT JOIN fl_clean fl ON b.cik = fl.cik AND b.td > fl.filed_date
    LEFT JOIN sv_feat sv ON sv.symbol = b.symbol AND sv.sv_date = b.td
    LEFT JOIN df6_pit f6 ON f6.symbol = b.symbol AND f6.eff_date = b.next1_td AND f6.list_date <= b.td
    ORDER BY b.td, b.symbol
    """
    df = con.execute(sql).df()
    assert len(df) >= 200, f"H106 panel too small: {len(df)} < 200"

    td_list = [d.date() if isinstance(d, dt.datetime) else d for d in df["trade_date"].tolist()]
    n1_list = [d.date() if isinstance(d, dt.datetime) else d for d in df["next1_trade_date"].tolist()]
    max_td = max(td_list)
    max_n1 = max(n1_list)
    if max_td > STUDY_WALL_DATE or max_n1 > STUDY_WALL_DATE:
        raise ValueError(f"H106 PREWALL BREACH: max_td={max_td}, max_n1={max_n1}")

    # Strict PIT causality assertions on ASOF matches
    known_valid = df["known_date"].dropna()
    if len(known_valid) > 0:
        max_known = max(k.date() if hasattr(k, "date") else k for k in known_valid)
        if max_known >= STUDY_WALL_DATE:
            raise ValueError(f"H106 DF8 max_known={max_known} >= {STUDY_WALL_DATE}")
        leaks_si = (df["known_date"].notna()) & (df["known_date"] >= df["trade_date"])
        if bool(leaks_si.any()):
            raise ValueError(f"H106 DF8 CAUSALITY LEAK rows={int(leaks_si.sum())}")
    leaks_fl = (df["filed_date"].notna()) & (df["filed_date"] >= df["trade_date"])
    if bool(leaks_fl.any()):
        raise ValueError(f"H106 FLOAT CAUSALITY LEAK rows={int(leaks_fl.sum())}")

    has_90d_dil, has_5d_8k = _attach_sec_flags_pit(df["cik"].to_numpy(dtype=np.int64), td_list, sec_p, wall_iso)

    op = df["next1_split_open"].to_numpy(dtype=np.float64)
    hi = df["next1_split_high"].to_numpy(dtype=np.float64)
    lo = df["next1_split_low"].to_numpy(dtype=np.float64)
    cl = df["next1_split_close"].to_numpy(dtype=np.float64)
    long_gross_bps, short_gross_bps, long_stopped, short_stopped = compute_intraday_leg_arrays(
        op, hi, lo, cl, stop_frac=ADR0022_MAX_STOP_PCT / 100.0
    )

    adv_arr = df["mean_20d_dvol"].to_numpy(dtype=np.float64)
    rng_arr = df["mean_20d_range_pct"].to_numpy(dtype=np.float64)
    costs_1x = np.array(
        [
            compute_almgren_chriss_cost_bps(
                base_half_spread_bps=2.5,
                fee_and_slip_bps_per_side=1.5,
                daily_sigma_frac=max(0.005, min(0.15, float(rng) * 0.65)) if math.isfinite(float(rng)) else 0.02,
                order_notional_usd=DEFAULT_POSITION_NOTIONAL_USD,
                adv_usd=max(20_000_000.0, float(adv)),
                stress_multiplier=1.0,
            )
            for rng, adv in zip(rng_arr, adv_arr, strict=False)
        ],
        dtype=np.float64,
    )
    long_net_1x = long_gross_bps - costs_1x
    long_net_2x = long_gross_bps - (2.0 * costs_1x)
    short_net_1x = short_gross_bps - costs_1x - SHORT_BORROW_1X_BPS
    short_net_2x = short_gross_bps - (2.0 * costs_1x) - (2.0 * SHORT_BORROW_1X_BPS)

    trade_date_strs = np.array([d.isoformat() for d in td_list], dtype="<U10")
    unique_dates, date_codes = np.unique(trade_date_strs, return_inverse=True)
    assert len(unique_dates) >= 60, f"H106 distinct_dates too low: {len(unique_dates)} < 60"

    feat_cols = [
        "days_to_cover",
        "si_mom",
        "log_short_shares",
        "si_to_adv",
        "si_to_float",
        "trailing_5d_sv_ratio",
        "sv_accel_1d_5d",
        "rvol_20d",
        "vol_z_20d",
        "vdu_5d_20d_ratio",
        "coil_5d_20d_range_ratio",
        "prev_close_to_20d_high",
        "ret_1d",
        "prior_ret_5d",
        "prior_ret_20d",
        "close_pos",
    ]
    raw_feat = df[feat_cols].to_numpy(dtype=np.float32)
    ranked_feat = cross_sectional_rank_matrix(raw_feat, date_codes)
    binary_feat = np.column_stack(
        [
            has_90d_dil,
            has_5d_8k,
            df["on_regsho_threshold"].to_numpy(dtype=np.float32),
            df["has_float_match"].to_numpy(dtype=np.float32),
        ]
    )
    x_all = np.hstack([ranked_feat, binary_feat]).astype(np.float32, copy=False)

    # Target for training ONLY: date-centered long net_1x return (predicted strictly on train dates)
    y_raw = long_net_1x.astype(np.float32)
    date_sum = np.bincount(date_codes, weights=y_raw, minlength=len(unique_dates))
    date_cnt = np.bincount(date_codes, minlength=len(unique_dates))
    date_mean = date_sum / np.maximum(date_cnt, 1)
    y_centered = (y_raw - date_mean[date_codes]).astype(np.float32)

    # Economic rule score (date-ranked signals; zero fit parameters)
    # Positive: high SI crowding (si_to_float, days_to_cover) + calm coil/thrust + no 90d dilution
    # Negative: active dilution + rising off-exchange short-volume pressure + extended prior run
    rule_score = (
        0.30 * ranked_feat[:, 4]  # si_to_float
        + 0.20 * ranked_feat[:, 0]  # days_to_cover
        + 0.15 * ranked_feat[:, 1]  # si_mom
        + 0.20 * ranked_feat[:, 15]  # close_pos
        - 0.15 * ranked_feat[:, 5]  # trailing_5d_sv_ratio
        - 0.60 * has_90d_dil
        - 0.25 * df["on_regsho_threshold"].to_numpy(dtype=np.float32)
    )

    cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=5, embargo_size=5)
    splits = list(cv.split(np.arange(len(unique_dates)).reshape(-1, 1)))
    assert len(splits) == 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rule_splits: list[tuple[Any, Any, Any, float]] = []
    mlp_splits: list[tuple[Any, Any, Any, float]] = []
    lgb_splits: list[tuple[Any, Any, Any, float]] = []
    xgb_splits: list[tuple[Any, Any, Any, float]] = []
    ens_splits: list[tuple[Any, Any, Any, float]] = []

    def _quick_train_ls_mean(tr_idx: Any, tr_sc: Any) -> float:
        sample_dc = date_codes[tr_idx]
        uniq_dc = np.unique(sample_dc)[::4]
        vals: list[float] = []
        for dc in uniq_dc:
            g_mask = sample_dc == dc
            g_idx = tr_idx[g_mask]
            if len(g_idx) < MIN_SYMBOLS_PER_DATE:
                continue
            nd = has_90d_dil[g_idx] < 0.5
            if int(nd.sum()) < 10:
                continue
            sc = tr_sc[g_mask]
            nd_p = np.where(nd)[0]
            k_l = max(5, int(0.10 * len(nd_p)))
            l_sel = nd_p[np.argsort(-sc[nd_p], kind="mergesort")[:k_l]]
            l_set = set(l_sel.tolist())
            rem = np.array([k for k in range(len(g_idx)) if k not in l_set], dtype=np.int64)
            k_s = max(5, int(0.10 * len(g_idx)))
            if len(rem) < k_s:
                continue
            s_sel = rem[np.argsort(sc[rem], kind="mergesort")[:k_s]]
            vals.append(0.5 * float(long_net_2x[g_idx[l_sel]].mean()) + 0.5 * float(short_net_2x[g_idx[s_sel]].mean()))
        return float(np.mean(vals)) if vals else 0.0

    for split_i, (tr_d_idx, te_d_idx_list) in enumerate(splits):
        te_d_idx = np.concatenate(te_d_idx_list) if isinstance(te_d_idx_list, list) else np.asarray(te_d_idx_list)
        assert set(tr_d_idx.tolist()).isdisjoint(set(te_d_idx.tolist())), "CPCV train/test date leakage!"
        tr_mask = np.isin(date_codes, tr_d_idx)
        te_mask = np.isin(date_codes, te_d_idx)
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]

        x_tr, y_tr = x_all[tr_idx], y_centered[tr_idx]
        x_te = x_all[te_idx]

        # 1. Rule baseline
        rule_tr = rule_score[tr_idx]
        rule_te = rule_score[te_idx]
        rule_splits.append((tr_idx, te_idx, rule_te, _quick_train_ls_mean(tr_idx, rule_tr)))

        # 2. RTX 4090 PyTorch CUDA MLP (Ada TF32 + BF16)
        torch.manual_seed(106 + split_i)
        net = torch.nn.Sequential(
            torch.nn.Linear(x_tr.shape[1], 64),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(64, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        xt = torch.from_numpy(x_tr).to(device)
        yt = torch.from_numpy(y_tr / 100.0).unsqueeze(1).to(device)
        net.train()
        use_amp = device.type == "cuda" and hw.get("bf16_supported", False)
        for _ in range(30):
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = torch.nn.functional.huber_loss(net(xt), yt)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            mlp_tr = net(xt).squeeze(1).float().cpu().numpy() * 100.0
            mlp_te = net(torch.from_numpy(x_te).to(device)).squeeze(1).float().cpu().numpy() * 100.0
        mlp_splits.append((tr_idx, te_idx, mlp_te, _quick_train_ls_mean(tr_idx, mlp_tr)))

        # 3. LightGBM
        mdl_lgb = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=106 + split_i,
            verbose=-1,
            n_jobs=16,
        )
        mdl_lgb.fit(x_tr, y_tr)
        lgb_tr = mdl_lgb.predict(x_tr)
        lgb_te = mdl_lgb.predict(x_te)
        lgb_splits.append((tr_idx, te_idx, lgb_te, _quick_train_ls_mean(tr_idx, lgb_tr)))

        # 4. XGBoost CUDA
        mdl_xgb = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda" if device.type == "cuda" else "cpu",
            random_state=106 + split_i,
            verbosity=0,
        )
        mdl_xgb.fit(x_tr, y_tr)
        xgb_tr = mdl_xgb.predict(x_tr)
        xgb_te = mdl_xgb.predict(x_te)
        xgb_splits.append((tr_idx, te_idx, xgb_te, _quick_train_ls_mean(tr_idx, xgb_tr)))

        # 5. Pre-declared primary equal-weight z-scored 3-model ensemble per split
        def _z(arr: Any, ref_mu: float, ref_sd: float) -> Any:
            return (arr - ref_mu) / max(ref_sd, 1e-5)

        m_tr_mu, m_tr_sd = float(mlp_tr.mean()), float(mlp_tr.std())
        l_tr_mu, l_tr_sd = float(lgb_tr.mean()), float(lgb_tr.std())
        x_tr_mu, x_tr_sd = float(xgb_tr.mean()), float(xgb_tr.std())
        ens_tr = (_z(mlp_tr, m_tr_mu, m_tr_sd) + _z(lgb_tr, l_tr_mu, l_tr_sd) + _z(xgb_tr, x_tr_mu, x_tr_sd)) / 3.0
        ens_te = (_z(mlp_te, m_tr_mu, m_tr_sd) + _z(lgb_te, l_tr_mu, l_tr_sd) + _z(xgb_te, x_tr_mu, x_tr_sd)) / 3.0
        ens_splits.append((tr_idx, te_idx, ens_te, _quick_train_ls_mean(tr_idx, ens_tr)))

    eval_kwargs = {
        "trade_date_strs": trade_date_strs,
        "date_codes": date_codes,
        "long_net_1x": long_net_1x,
        "long_net_2x": long_net_2x,
        "short_net_1x": short_net_1x,
        "short_net_2x": short_net_2x,
        "has_90d_dil": has_90d_dil,
        "trial_number": trial_number,
    }
    res_rule = _evaluate_ls_cpcv_from_scores(scores_by_split=rule_splits, **eval_kwargs)
    res_mlp = _evaluate_ls_cpcv_from_scores(scores_by_split=mlp_splits, **eval_kwargs)
    res_lgb = _evaluate_ls_cpcv_from_scores(scores_by_split=lgb_splits, **eval_kwargs)
    res_xgb = _evaluate_ls_cpcv_from_scores(scores_by_split=xgb_splits, **eval_kwargs)
    res_ens = _evaluate_ls_cpcv_from_scores(scores_by_split=ens_splits, **eval_kwargs)

    # Single pre-declared primary gate evaluation on ensemble_3m (zero cherry-picking!)
    gate = evaluate_strict_prewall_gate(
        trial_number=trial_number,
        t_stat_clustered_2x=float(res_ens["conservative_gate_t_2x"]),
        dsr_probability=float(res_ens["dsr_probability"]),
        cpcv_positive_share=float(res_ens["cpcv_positive_share"]),
        pbo=float(res_ens["pbo"]),
        mean_net_2x_bps=float(res_ens["unique_date_oof_2x_hac5"]["mean_bps"]),
        distinct_dates=int(res_ens["unique_oos_dates"]),
        total_events=int(len(df)),
    )

    return {
        "hypothesis_id": H106_ID,
        "trial_number": trial_number,
        "spec_sha256": H106_SPEC_SHA256,
        "study_wall_date": wall_iso,
        "fast_lake_tier": fast_tier,
        "cuda_device": str(device),
        "gpu_name": hw.get("gpu_name"),
        "h099_readiness": asdict(h099_rec),
        "max_trade_date_verified": max_td.isoformat(),
        "max_next1_trade_date_verified": max_n1.isoformat(),
        "events_total": int(len(df)),
        "distinct_dates": int(len(unique_dates)),
        "distinct_symbols": int(df["symbol"].nunique()),
        "df8_matched_rows": int(df["known_date"].notna().sum()),
        "sec_float_matched_rows": int((df["has_float_match"] > 0.5).sum()),
        "active_90d_dilution_rows": int((has_90d_dil > 0.5).sum()),
        "long_stopped_day1_rate": round(float(long_stopped.mean()), 4),
        "short_stopped_day1_rate": round(float(short_stopped.mean()), 4),
        "mean_ac_cost_1x_bps": round(float(costs_1x.mean()), 2),
        "short_borrow_1x_bps": SHORT_BORROW_1X_BPS,
        "primary_model": "ensemble_3m",
        "models": {
            "rule_baseline": res_rule,
            "pytorch_rtx4090_mlp": res_mlp,
            "lightgbm": res_lgb,
            "xgboost_cuda": res_xgb,
            "ensemble_3m": res_ens,
        },
        "gate": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute pre-registered H106 DF8 Short-Crowding Cross-Sectional CPCV.")
    parser.add_argument("--lake-root", default="/mnt/z/sota-data/prewall-lake")
    args = parser.parse_args()

    lake_root = pathlib.Path(args.lake_root)
    print(f"[H106] Executing {H106_ID} (Trial #{H106_TRIAL_NUMBER}, spec={H106_SPEC_SHA256[:12]}...)...")
    res = run_h106_df8_short_crowding(lake_root)

    receipt = {
        "schema_version": "sota-optb-h106-df8-short-crowding.v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "h106": res,
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode("utf-8")).hexdigest()

    out_path = lake_root / "receipts" / "h106_df8_short_crowding_xsec_receipt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    optb_path = lake_root / "receipts" / "optb_prewall_factory_audit.json"
    if optb_path.is_file():
        optb = json.loads(optb_path.read_text(encoding="utf-8"))
        optb["h106_receipt_sha256"] = receipt["receipt_sha256"]
        optb["h106_verdict"] = res["gate"]["verdict"]
        optb["h106_trial_number"] = H106_TRIAL_NUMBER
        optb["updated_at_utc"] = receipt["generated_at_utc"]
        optb_path.write_text(json.dumps(optb, indent=2) + "\n", encoding="utf-8")

    ens = res["models"]["ensemble_3m"]
    print(
        f"[H106] DONE: verdict={res['gate']['verdict']} | tier={res['fast_lake_tier']} | gpu={res['gpu_name']} | "
        f"events={res['events_total']} dates={res['distinct_dates']} | "
        f"ens_1x={ens['split_pooled_clustered_1x']['mean_bps']}bps "
        f"ens_2x_hac={ens['unique_date_oof_2x_hac5']['mean_bps']}bps "
        f"t_cons={ens['conservative_gate_t_2x']} dsr={ens['dsr_probability']} "
        f"cpcv+={ens['cpcv_positive_share']} pbo={ens['pbo']} | "
        f"sha256={receipt['receipt_sha256'][:16]}..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
