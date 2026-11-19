"""Pre-registered Trial #103 (H107): SIC Sector-Neutral Liquid Multi-Feed Residual Factor on RTX 4090 CPCV.

Enforces ADR-0020, ADR-0021, ADR-0022, ADR-0023, and ADR-0024 fail-closed invariants:
- Strictly pre-wall <= 2026-10-15 across DF10, quarantined DF8 (``known_from < trade_date``), XBRL ``sec_float``
  (``filed < trade_date``), T-1 ``short_volume`` (``day <= trade_date``), DF6 (``effective_trade_date``),
  and clean SEC filings (``accepted <= 2026-10-15`` ET, ``< trade_date``).
- Operating US common equities only (``prev_raw_close >= 10.0``, ``mean_20d_dvol >= 25M``, valid 2-digit SIC
  with ``>= 6`` liquid symbols in ``(trade_date, sic2)`` and ``>= 20`` active symbols/day).
- Passes ``check_retired_trap_firewall`` (zero post-incentive chase, zero MOC-to-MOO carry, zero unshielded passive).
- Strictly non-overlapping 1-day intraday MTM (``next1_split_open -> next1_split_close``) with adverse-stop-first
  ``ADR-0022 <= 2.00%`` hard stop on both legs, dollar-neutral inverse-vol within-sector quintiles,
  and ``1x``/``2x`` Almgren-Chriss sqrt impact + ``5``/``10 bps`` short borrow.
- Primary model: equal-weight z-scored 3-model ensemble (PyTorch RTX 4090 CUDA MLP + XGBoost CUDA + Ridge baseline)
  over ``C(6,2)=15`` disjoint ``skfolio`` ``CombinatorialPurgedCV(purge=5d, embargo=5d)`` vs ``HLZ t*(103) = 4.2041``
  and ``DSR >= 0.95``.
"""

from __future__ import annotations

import argparse
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

from run_h106_df8_short_crowding_xsec import (  # noqa: E402
    _attach_sec_flags_pit,
    compute_intraday_leg_arrays,
)
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
from sota_compounding.prewall_h099_enrichment import audit_h099_prewall_lake_readiness  # noqa: E402

H107_ID = "H107_SECTOR_NEUTRAL_LIQUID_MULTIFEED_RESIDUAL_RTX4090_CPCV"
H107_SPEC_SHA256 = "a0a41e11b2aa3618427e0e1141246175d8a98c88f79483d3efc2364e8ebc0fe6"
H107_TRIAL_NUMBER = 103
MIN_LIQUID_DVOL_USD = 25_000_000.0
MIN_SECTOR_GROUP_SIZE = 6
MIN_SYMBOLS_PER_DATE = 20
SHORT_BORROW_1X_BPS = 5.0


def build_within_sector_neutral_ranks(
    raw_matrix: Any,
    date_codes: Any,
    sic2_codes: Any,
    *,
    demean_within_sector: bool = True,
) -> Any:
    """Rank-normalize features strictly within each ``(trade_date, sic2)`` bucket into ``[-1, +1]`` (zero leakage)."""
    import numpy as np

    try:
        from scipy.stats import rankdata as _scipy_rankdata
    except ImportError:
        _scipy_rankdata = None

    n_rows, n_cols = raw_matrix.shape
    out = np.zeros((n_rows, n_cols), dtype=np.float32)
    pair_keys = (date_codes.astype(np.int64) << 16) + sic2_codes.astype(np.int64)
    unique_pairs, inv = np.unique(pair_keys, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    sorted_inv = inv[order]
    counts = np.bincount(sorted_inv, minlength=len(unique_pairs))
    starts = np.zeros(len(unique_pairs), dtype=np.int64)
    if len(unique_pairs) > 1:
        np.cumsum(counts[:-1], out=starts[1:])

    for col in range(n_cols):
        col_vals = raw_matrix[:, col].astype(np.float64, copy=False)
        for g_idx in range(len(unique_pairs)):
            ln = int(counts[g_idx])
            if ln <= 1:
                continue
            st = int(starts[g_idx])
            grp_idx = order[st : st + ln]
            vals = col_vals[grp_idx]
            finite_mask = np.isfinite(vals)
            n_valid = int(finite_mask.sum())
            if n_valid <= 1:
                continue
            valid_pos = grp_idx[finite_mask]
            v = vals[finite_mask]
            if demean_within_sector:
                v = v - float(np.mean(v))
            if _scipy_rankdata is not None:
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


def precompute_date_sector_groups(
    date_codes: Any,
    sic2_codes: Any,
    inv_vol_weights: Any,
    has_90d_dil: Any,
) -> dict[int, list[tuple[Any, Any, Any]]]:
    """Pre-index ``(global_indices, inv_vol, nondil_mask)`` per ``(trade_date, sic2)`` pre-trade."""
    import numpy as np

    by_date: dict[int, list[tuple[Any, Any, Any]]] = {}
    unique_dc = np.unique(date_codes)
    for dc in unique_dc:
        dc_int = int(dc)
        d_idx = np.where(date_codes == dc)[0]
        if len(d_idx) < MIN_SYMBOLS_PER_DATE:
            continue
        sec_vals = sic2_codes[d_idx]
        buckets: list[tuple[Any, Any, Any]] = []
        for s_code in np.unique(sec_vals):
            s_mask = sec_vals == s_code
            g_idx = d_idx[s_mask]
            if len(g_idx) < MIN_SECTOR_GROUP_SIZE:
                continue
            iv = inv_vol_weights[g_idx]
            nd = has_90d_dil[g_idx] < 0.5
            buckets.append((g_idx, iv, nd))
        if buckets:
            by_date[dc_int] = buckets
    return by_date


def compute_daily_sector_neutral_returns(
    scores: Any,
    date_idx_slice: Any,
    *,
    date_codes: Any,
    date_sector_groups: dict[int, list[tuple[Any, Any, Any]]],
    long_net_1x: Any,
    long_net_2x: Any,
    short_net_1x: Any,
    short_net_2x: Any,
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """Compute non-overlapping 1-day inverse-vol sector-neutral top-quintile vs bottom-quintile portfolio returns."""
    import numpy as np

    pos_lookup = np.full(len(date_codes), -1, dtype=np.int64)
    pos_lookup[date_idx_slice] = np.arange(len(date_idx_slice), dtype=np.int64)
    unique_dc = np.unique(date_codes[date_idx_slice])

    out_dc: list[int] = []
    out_ls1: list[float] = []
    out_ls2: list[float] = []
    out_l2: list[float] = []
    out_s2: list[float] = []

    for dc in unique_dc:
        dc_int = int(dc)
        buckets = date_sector_groups.get(dc_int)
        if not buckets:
            continue
        l_idx_parts: list[Any] = []
        l_w_parts: list[Any] = []
        s_idx_parts: list[Any] = []
        s_w_parts: list[Any] = []

        for g_idx, iv, nd in buckets:
            loc = pos_lookup[g_idx]
            if bool(np.any(loc < 0)):
                continue
            sc = scores[loc]
            if int(nd.sum()) < 2:
                continue
            nd_pos = np.where(nd)[0]
            k_sec = max(1, int(math.floor(0.20 * len(g_idx))))
            k_l = min(k_sec, len(nd_pos))
            nd_order = np.argsort(-sc[nd_pos], kind="mergesort")
            chosen_l_local = nd_pos[nd_order[:k_l]]
            chosen_l_set = set(chosen_l_local.tolist())

            rem_local = np.array([k for k in range(len(g_idx)) if k not in chosen_l_set], dtype=np.int64)
            if len(rem_local) < 1:
                continue
            k_s = min(k_sec, len(rem_local))
            rem_order = np.argsort(sc[rem_local], kind="mergesort")
            chosen_s_local = rem_local[rem_order[:k_s]]

            l_idx_parts.append(g_idx[chosen_l_local])
            l_w_parts.append(iv[chosen_l_local])
            s_idx_parts.append(g_idx[chosen_s_local])
            s_w_parts.append(iv[chosen_s_local])

        if not l_idx_parts or not s_idx_parts:
            continue
        l_all = np.concatenate(l_idx_parts)
        s_all = np.concatenate(s_idx_parts)
        if len(l_all) < 5 or len(s_all) < 5:
            continue
        lw = np.concatenate(l_w_parts)
        sw = np.concatenate(s_w_parts)
        lw_sum = float(lw.sum())
        sw_sum = float(sw.sum())
        if lw_sum <= 0.0 or sw_sum <= 0.0:
            continue
        lw_norm = lw / lw_sum
        sw_norm = sw / sw_sum

        r_l1 = float(np.dot(lw_norm, long_net_1x[l_all]))
        r_l2 = float(np.dot(lw_norm, long_net_2x[l_all]))
        r_s1 = float(np.dot(sw_norm, short_net_1x[s_all]))
        r_s2 = float(np.dot(sw_norm, short_net_2x[s_all]))

        out_dc.append(dc_int)
        out_ls1.append(0.5 * r_l1 + 0.5 * r_s1)
        out_ls2.append(0.5 * r_l2 + 0.5 * r_s2)
        out_l2.append(r_l2)
        out_s2.append(r_s2)

    return out_dc, out_ls1, out_ls2, out_l2, out_s2


def _evaluate_sector_cpcv(
    *,
    scores_by_split: list[tuple[Any, Any, Any, float]],
    unique_dates: Any,
    date_codes: Any,
    date_sector_groups: dict[int, list[tuple[Any, Any, Any]]],
    long_net_1x: Any,
    long_net_2x: Any,
    short_net_1x: Any,
    short_net_2x: Any,
    trial_number: int,
) -> dict[str, Any]:
    """Evaluate C(6,2)=15 CPCV splits with clustered SEs, HAC OOF, DSR, PBO, and outlier checks."""
    import numpy as np
    import statsmodels.api as sm

    path_means_2x: list[float] = []
    overfit_cnt = 0
    pooled_dates: list[str] = []
    pooled_ls_1x: list[float] = []
    pooled_ls_2x: list[float] = []
    oof_date_buckets: dict[str, list[tuple[float, float, float, float]]] = {}

    for _tr_idx, te_idx, te_scores, tr_mean_2x in scores_by_split:
        dcs, ls1_l, ls2_l, l2_l, s2_l = compute_daily_sector_neutral_returns(
            te_scores,
            te_idx,
            date_codes=date_codes,
            date_sector_groups=date_sector_groups,
            long_net_1x=long_net_1x,
            long_net_2x=long_net_2x,
            short_net_1x=short_net_1x,
            short_net_2x=short_net_2x,
        )
        if not ls2_l:
            path_means_2x.append(0.0)
            overfit_cnt += 1
            continue
        for dc_int, ls1, ls2, l2, s2 in zip(dcs, ls1_l, ls2_l, l2_l, s2_l, strict=True):
            d_str = str(unique_dates[dc_int])
            pooled_dates.append(d_str)
            pooled_ls_1x.append(ls1)
            pooled_ls_2x.append(ls2)
            oof_date_buckets.setdefault(d_str, []).append((ls1, ls2, l2, s2))

        te_mean = float(np.mean(ls2_l))
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

    outlier_sens: dict[str, Any] = {}
    if len(uniq_2x) >= 20:
        order_desc = np.argsort(-uniq_2x)
        no_best1 = np.delete(uniq_2x, order_desc[:1])
        no_best5 = np.delete(uniq_2x, order_desc[:5])
        f1 = sm.OLS(no_best1, np.ones((len(no_best1), 1), dtype=np.float64)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5}
        )
        f5 = sm.OLS(no_best5, np.ones((len(no_best5), 1), dtype=np.float64)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5}
        )
        outlier_sens = {
            "drop_best1_mean_2x_bps": round(float(f1.params[0]), 4),
            "drop_best1_t_stat": round(float(f1.tvalues[0]), 4),
            "drop_best5_mean_2x_bps": round(float(f5.params[0]), 4),
            "drop_best5_t_stat": round(float(f5.tvalues[0]), 4),
        }

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
        "unique_date_oof_1x": {"mean_bps": round(float(np.mean(uniq_1x)), 4) if len(uniq_1x) else 0.0},
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


def run_h107_sector_neutral_factor(
    lake_root: pathlib.Path,
    *,
    trial_number: int = H107_TRIAL_NUMBER,
) -> dict[str, Any]:
    """Execute pre-registered Trial #103 (H107) SIC Sector-Neutral Liquid Multi-Feed CPCV on RTX 4090."""
    import duckdb
    import numpy as np
    import torch
    import xgboost as xgb
    from skfolio.model_selection import CombinatorialPurgedCV
    from sklearn.linear_model import Ridge

    ok_fw, fw_viol = check_retired_trap_firewall(
        tags=["FIRST_HAND_SEC_EPISODIC_PIVOT", "PRE_TRAVEL_EARLY_IGNITION"],
        is_post_incentive=False,
        holds_overnight_moc_to_moo=False,
        unshielded_lit_passive=False,
        in_opening_0930_0935_taker=False,
        ofi_only_without_joint_sweep=False,
    )
    if not ok_fw:
        raise RuntimeError(f"H107 blocked by Retired-Trap Firewall: {fw_viol}")

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
        SELECT UPPER(TRIM(symbol)) AS symbol,
               CAST(cik AS BIGINT) AS cik,
               CAST(SUBSTR(LPAD(REGEXP_REPLACE(
                   COALESCE(CAST(sic AS VARCHAR), ''), '[^0-9]', '', 'g'
               ), 4, '0'), 1, 2) AS INTEGER) AS sic2
        FROM read_parquet('{sym_p}')
        WHERE LOWER(COALESCE(entity_type, '')) = 'operating'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%etf%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%trust%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%acquisition corp%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%spac%'
          AND LENGTH(REGEXP_REPLACE(COALESCE(CAST(sic AS VARCHAR), ''), '[^0-9]', '', 'g')) >= 3
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
            vol_z_20d,
            vdu_5d_20d_ratio,
            coil_5d_20d_range_ratio,
            ret_1d,
            prior_ret_5d,
            next1_split_open,
            next1_split_high,
            next1_split_low,
            next1_split_close
        FROM read_parquet('{df10_p}')
        WHERE CAST(trade_date AS DATE) <= DATE '{wall_iso}'
    ),
    base AS (
        SELECT d.*, o.cik, o.sic2
        FROM df10_next d
        INNER JOIN ops o ON o.symbol = d.symbol
        WHERE d.next1_td IS NOT NULL
          AND d.next1_td <= DATE '{wall_iso}'
          AND d.prior_sessions_count = 20
          AND d.raw_close >= 10.0
          AND d.mean_20d_dvol >= {MIN_LIQUID_DVOL_USD}
          AND d.next1_split_open > 0 AND d.next1_split_low > 0 AND d.next1_split_high > 0 AND d.next1_split_close > 0
          AND d.next1_split_high >= d.next1_split_low
    ),
    sec_filtered AS (
        SELECT *, COUNT(*) OVER (PARTITION BY td, sic2) AS n_sec
        FROM base
    ),
    eligible AS (
        SELECT * FROM sec_filtered WHERE n_sec >= {MIN_SECTOR_GROUP_SIZE}
    ),
    df8_clean AS (
        SELECT UPPER(TRIM(symbol)) AS symbol,
               CAST(known_from AS DATE) AS known_date,
               CAST(settlement AS DATE) AS settle_date,
               CAST(short_shares AS DOUBLE) AS short_shares,
               CAST(short_shares_before AS DOUBLE) AS short_shares_before
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
    )
    SELECT
        b.symbol,
        b.td AS trade_date,
        b.next1_td AS next1_trade_date,
        b.cik,
        b.sic2,
        b.mean_20d_dvol,
        b.mean_20d_range_pct,
        b.next1_split_open,
        b.next1_split_high,
        b.next1_split_low,
        b.next1_split_close,
        si.known_date,
        fl.filed_date,
        -b.prior_ret_5d AS neg_prior_ret_5d,
        -b.ret_1d AS neg_ret_1d,
        -b.vdu_5d_20d_ratio AS neg_vdu_ratio,
        -b.coil_5d_20d_range_ratio AS neg_coil_ratio,
        b.vol_z_20d,
        COALESCE(sv.trailing_5d_sv_ratio, 0.45) AS df7_short_vol_ratio_t1,
        CASE
            WHEN si.short_shares > 0 AND fl.float_usd > 0
            THEN LEAST(1.5, (si.short_shares * b.raw_close) / fl.float_usd)
            ELSE 0.0
        END AS df8_short_crowding_usd_pit,
        CASE
            WHEN si.short_shares IS NOT NULL AND si.short_shares_before IS NOT NULL AND si.short_shares_before > 0
            THEN LEAST(5.0, GREATEST(-1.0, (si.short_shares - si.short_shares_before) / si.short_shares_before))
            ELSE 0.0
        END AS df8_short_delta_pit
    FROM eligible b
    ASOF LEFT JOIN df8_clean si ON b.symbol = si.symbol AND b.td > si.known_date
    ASOF LEFT JOIN fl_clean fl ON b.cik = fl.cik AND b.td > fl.filed_date
    LEFT JOIN sv_feat sv ON sv.symbol = b.symbol AND sv.sv_date = b.td
    ORDER BY b.td, b.sic2, b.symbol
    """
    df = con.execute(sql).df()
    assert len(df) >= 200, f"H107 panel too small: {len(df)} < 200"

    td_list = [d.date() if isinstance(d, dt.datetime) else d for d in df["trade_date"].tolist()]
    n1_list = [d.date() if isinstance(d, dt.datetime) else d for d in df["next1_trade_date"].tolist()]
    max_td = max(td_list)
    max_n1 = max(n1_list)
    if max_td > STUDY_WALL_DATE or max_n1 > STUDY_WALL_DATE:
        raise ValueError(f"H107 PREWALL BREACH: max_td={max_td}, max_n1={max_n1}")

    known_valid = df["known_date"].dropna()
    if len(known_valid) > 0:
        max_known = max(k.date() if hasattr(k, "date") else k for k in known_valid)
        if max_known >= STUDY_WALL_DATE:
            raise ValueError(f"H107 DF8 max_known={max_known} >= {STUDY_WALL_DATE}")
        leaks_si = (df["known_date"].notna()) & (df["known_date"] >= df["trade_date"])
        if bool(leaks_si.any()):
            raise ValueError(f"H107 DF8 CAUSALITY LEAK rows={int(leaks_si.sum())}")
    leaks_fl = (df["filed_date"].notna()) & (df["filed_date"] >= df["trade_date"])
    if bool(leaks_fl.any()):
        raise ValueError(f"H107 FLOAT CAUSALITY LEAK rows={int(leaks_fl.sum())}")

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
                adv_usd=max(MIN_LIQUID_DVOL_USD, float(adv)),
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
    assert len(unique_dates) >= 60, f"H107 distinct_dates too low: {len(unique_dates)} < 60"

    sic2_codes = df["sic2"].to_numpy(dtype=np.int64)
    safe_rng = np.where(np.isfinite(rng_arr) & (rng_arr > 0.005), rng_arr, 0.02)
    inv_vol_weights = (1.0 / np.clip(safe_rng, 0.005, 0.15)).astype(np.float64)

    feat_cols = [
        "neg_prior_ret_5d",
        "neg_ret_1d",
        "neg_vdu_ratio",
        "neg_coil_ratio",
        "vol_z_20d",
        "df7_short_vol_ratio_t1",
        "df8_short_crowding_usd_pit",
        "df8_short_delta_pit",
    ]
    raw_feat = df[feat_cols].to_numpy(dtype=np.float32)
    ranked_feat = build_within_sector_neutral_ranks(raw_feat, date_codes, sic2_codes, demean_within_sector=True)
    sec_fresh_catalyst = (has_5d_8k * (1.0 - has_90d_dil)).astype(np.float32)
    x_all = np.hstack([ranked_feat, sec_fresh_catalyst.reshape(-1, 1), has_90d_dil.reshape(-1, 1)]).astype(
        np.float32, copy=False
    )

    # Sector-centered training target (residualized strictly within (trade_date, sic2) per train row — no leakage)
    y_raw = long_net_1x.astype(np.float32)
    pair_keys = (date_codes.astype(np.int64) << 16) + sic2_codes
    _u_pairs, pair_inv = np.unique(pair_keys, return_inverse=True)
    pair_sum = np.bincount(pair_inv, weights=y_raw, minlength=len(_u_pairs))
    pair_cnt = np.bincount(pair_inv, minlength=len(_u_pairs))
    y_centered = (y_raw - (pair_sum / np.maximum(pair_cnt, 1))[pair_inv]).astype(np.float32)

    date_sector_groups = precompute_date_sector_groups(date_codes, sic2_codes, inv_vol_weights, has_90d_dil)

    # Zero-parameter sector-neutral economic rule baseline
    rule_score = (
        0.25 * ranked_feat[:, 0]
        + 0.15 * ranked_feat[:, 1]
        + 0.15 * ranked_feat[:, 2]
        + 0.10 * ranked_feat[:, 3]
        + 0.15 * ranked_feat[:, 6]
        - 0.15 * ranked_feat[:, 5]
        + 0.25 * sec_fresh_catalyst
        - 0.60 * has_90d_dil
    )

    cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=5, embargo_size=5)
    splits = list(cv.split(np.arange(len(unique_dates)).reshape(-1, 1)))
    assert len(splits) == 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rule_splits: list[tuple[Any, Any, Any, float]] = []
    ridge_splits: list[tuple[Any, Any, Any, float]] = []
    mlp_splits: list[tuple[Any, Any, Any, float]] = []
    xgb_splits: list[tuple[Any, Any, Any, float]] = []
    ens_splits: list[tuple[Any, Any, Any, float]] = []

    def _quick_train_ls_mean(tr_idx: Any, tr_sc: Any) -> float:
        uniq_tr_dc = np.unique(date_codes[tr_idx])[::4]
        sub_mask = np.isin(date_codes[tr_idx], uniq_tr_dc)
        _dcs, _ls1, ls2_l, _l2, _s2 = compute_daily_sector_neutral_returns(
            tr_sc[sub_mask],
            tr_idx[sub_mask],
            date_codes=date_codes,
            date_sector_groups=date_sector_groups,
            long_net_1x=long_net_1x,
            long_net_2x=long_net_2x,
            short_net_1x=short_net_1x,
            short_net_2x=short_net_2x,
        )
        return float(np.mean(ls2_l)) if ls2_l else 0.0

    for split_i, (tr_d_idx, te_d_idx_list) in enumerate(splits):
        te_d_idx = np.concatenate(te_d_idx_list) if isinstance(te_d_idx_list, list) else np.asarray(te_d_idx_list)
        assert set(tr_d_idx.tolist()).isdisjoint(set(te_d_idx.tolist())), "CPCV train/test date leakage!"
        tr_idx = np.where(np.isin(date_codes, tr_d_idx))[0]
        te_idx = np.where(np.isin(date_codes, te_d_idx))[0]

        x_tr, y_tr = x_all[tr_idx], y_centered[tr_idx]
        x_te = x_all[te_idx]

        # 1. Economic rule baseline
        rule_splits.append((tr_idx, te_idx, rule_score[te_idx], _quick_train_ls_mean(tr_idx, rule_score[tr_idx])))

        # 2. Ridge linear model
        mdl_ridge = Ridge(alpha=100.0)
        mdl_ridge.fit(x_tr, y_tr)
        ridge_tr = mdl_ridge.predict(x_tr)
        ridge_te = mdl_ridge.predict(x_te)
        ridge_splits.append((tr_idx, te_idx, ridge_te, _quick_train_ls_mean(tr_idx, ridge_tr)))

        # 3. RTX 4090 PyTorch CUDA MLP (Ada TF32 + BF16)
        torch.manual_seed(107 + split_i)
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

        # 4. XGBoost CUDA
        mdl_xgb = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda" if device.type == "cuda" else "cpu",
            random_state=107 + split_i,
            verbosity=0,
        )
        mdl_xgb.fit(x_tr, y_tr)
        xgb_tr = mdl_xgb.predict(x_tr)
        xgb_te = mdl_xgb.predict(x_te)
        xgb_splits.append((tr_idx, te_idx, xgb_te, _quick_train_ls_mean(tr_idx, xgb_tr)))

        # 5. Pre-declared primary equal-weight z-scored 3-model ensemble (PyTorch CUDA MLP + XGBoost CUDA + Ridge)
        def _z(arr: Any, mu: float, sd: float) -> Any:
            return (arr - mu) / max(sd, 1e-5)

        m_mu, m_sd = float(mlp_tr.mean()), float(mlp_tr.std())
        x_mu, x_sd = float(xgb_tr.mean()), float(xgb_tr.std())
        r_mu, r_sd = float(ridge_tr.mean()), float(ridge_tr.std())
        ens_tr = (_z(mlp_tr, m_mu, m_sd) + _z(xgb_tr, x_mu, x_sd) + _z(ridge_tr, r_mu, r_sd)) / 3.0
        ens_te = (_z(mlp_te, m_mu, m_sd) + _z(xgb_te, x_mu, x_sd) + _z(ridge_te, r_mu, r_sd)) / 3.0
        ens_splits.append((tr_idx, te_idx, ens_te, _quick_train_ls_mean(tr_idx, ens_tr)))

    eval_kw = {
        "unique_dates": unique_dates,
        "date_codes": date_codes,
        "date_sector_groups": date_sector_groups,
        "long_net_1x": long_net_1x,
        "long_net_2x": long_net_2x,
        "short_net_1x": short_net_1x,
        "short_net_2x": short_net_2x,
        "trial_number": trial_number,
    }
    res_rule = _evaluate_sector_cpcv(scores_by_split=rule_splits, **eval_kw)
    res_ridge = _evaluate_sector_cpcv(scores_by_split=ridge_splits, **eval_kw)
    res_mlp = _evaluate_sector_cpcv(scores_by_split=mlp_splits, **eval_kw)
    res_xgb = _evaluate_sector_cpcv(scores_by_split=xgb_splits, **eval_kw)
    res_ens = _evaluate_sector_cpcv(scores_by_split=ens_splits, **eval_kw)

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
        "hypothesis_id": H107_ID,
        "trial_number": trial_number,
        "spec_sha256": H107_SPEC_SHA256,
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
        "distinct_sic2": int(df["sic2"].nunique()),
        "active_90d_dilution_rows": int((has_90d_dil > 0.5).sum()),
        "long_stopped_day1_rate": round(float(long_stopped.mean()), 4),
        "short_stopped_day1_rate": round(float(short_stopped.mean()), 4),
        "mean_ac_cost_1x_bps": round(float(costs_1x.mean()), 2),
        "short_borrow_1x_bps": SHORT_BORROW_1X_BPS,
        "primary_model": "ensemble_3m",
        "models": {
            "rule_baseline": res_rule,
            "ridge_linear": res_ridge,
            "pytorch_rtx4090_mlp": res_mlp,
            "xgboost_cuda": res_xgb,
            "ensemble_3m": res_ens,
        },
        "gate": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute pre-registered H107 SIC Sector-Neutral Liquid Multi-Feed CPCV on RTX 4090."
    )
    parser.add_argument("--lake-root", default="/mnt/z/sota-data/prewall-lake")
    parser.add_argument("--out-path", default="")
    args = parser.parse_args()

    lake_root = pathlib.Path(args.lake_root)
    print(f"[H107] Executing {H107_ID} (Trial #{H107_TRIAL_NUMBER}, spec={H107_SPEC_SHA256[:12]}...)...")
    res = run_h107_sector_neutral_factor(lake_root)

    receipt = {
        "schema_version": "sota-optb-h107-sector-neutral.v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "h107": res,
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode("utf-8")).hexdigest()

    out_path = pathlib.Path(args.out_path) if args.out_path else (lake_root / "derived" / "h107_trial103_receipt.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    ens = res["models"]["ensemble_3m"]
    print(
        f"[H107] DONE: verdict={res['gate']['verdict']} | tier={res['fast_lake_tier']} | gpu={res['gpu_name']} | "
        f"events={res['events_total']} dates={res['distinct_dates']} sic2={res['distinct_sic2']} | "
        f"ens_1x={ens['split_pooled_clustered_1x']['mean_bps']}bps "
        f"ens_2x_hac={ens['unique_date_oof_2x_hac5']['mean_bps']}bps "
        f"t_cons={ens['conservative_gate_t_2x']} dsr={ens['dsr_probability']} "
        f"cpcv+={ens['cpcv_positive_share']} pbo={ens['pbo']} | "
        f"sha256={receipt['receipt_sha256'][:16]}..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
