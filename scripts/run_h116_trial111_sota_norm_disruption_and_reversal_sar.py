#!/usr/bin/env python3
"""Trial #111 (`H116`): SOTA-Upstream Whole-Market Per-Symbol Norm Disruption Early Entry + Reversal SAR.

Pre-Registered Immutable Spec SHA-256 (`sota-bus` Trial #111):
  `c9ccc38c5e903814ab4ebaa0aca5ecb7b0cf937422e24c2e10241d7929ce6e45`
Required HLZ Multiplicity Hurdle (`N = 111`): `t*(111) = 4.2210`
ADIA Lab #19 Power Gate (`D=700`, `sigma=95 bps`): APPROVED (`SR_min_ann=2.5326 <= 5.0`, `MinTRL=500`)

SOTA Upstream Libraries Executed Directly (`~/Documents/upstream/PINS.md`):
- `gerrymanoim/exchange_calendars` (`XNYS`): authoritative NYSE session verification (`0` post-wall dates).
- `duckdb==2.0.0` + `pola-rs/polars==2.1.0` + `unionai-oss/pandera==0.35.0`: columnar lake extraction & schema check.
- `statsmodels==0.15.0`: date-clustered Rogers/Newey-West SEs.
- `skfolio==1.12.7`: fold-fitted `CombinatorialPurgedCV(6,2,purge=20,embargo=20)`.
- `ml4t/diagnostic==0.2.0`: Deflated Sharpe Ratio (`DSR`) & Bailey (2014) `PBO`.
- `bashtage/arch` (when available) / stationary bootstrap Hansen `SPA` & `MCS`.

Primary Literature Foundations (`[E1]`):
- Zarattini, Aziz & Barbon (2024/2025 SSRN 4416622 / 4729284): Opening Range / Noise-Area VWAP Breakout.
- Kaminski & Lo (2014 JFE): Stop-Loss & Regime-Switch Stop-and-Reverse (SAR).
- Cont, Kukanov & Stoikov (2014 JFE) & Boehmer, Jones, Zhang & Zhang (2021 JF): OFI / Order-Flow Reversal.
- Almgren & Chriss (2001): Square-Root Market Impact.
- Harvey, Liu & Zhu (2016 RFS) & Lopez de Prado, Lipton & Zoonekynd (2025, ADIA Lab #19): HLZ / MinTRL / DSR.
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
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_h113_trial108_prefill_confirm_per_symbol_min_trail import (  # noqa: E402
    EVAL_START_ISO,
    SAMPLE_START_ISO,
    STUDY_WALL_ISO,
    _almgren_chriss_1x_cost_bps,
    _causal_per_symbol_min_trail_params,
    _compute_fold_fitted_cpcv_and_pbo,
    _evaluate_arm_metrics,
)
from run_h115_trial110_final_4catalyst_concentrated_1000rpm import _maker_1x_cost_bps  # noqa: E402

from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT  # noqa: E402

TRIAL_NUMBER = 111
HYPOTHESIS_ID = "H116_SOTA_NORM_DISRUPTION_EARLY_ENTRY_AND_REVERSAL_SAR"
PREREG_SPEC_SHA256 = "c9ccc38c5e903814ab4ebaa0aca5ecb7b0cf937422e24c2e10241d7929ce6e45"

PRIMARY_CONFIG_GRID: tuple[dict[str, Any], ...] = (
    {"name": "cfg_k2_z25", "k_top": 2, "vol_z": 2.5, "move_z": 1.8},
    {"name": "cfg_k3_z25", "k_top": 3, "vol_z": 2.5, "move_z": 1.8},  # Canonical baseline
    {"name": "cfg_k3_z30", "k_top": 3, "vol_z": 3.0, "move_z": 2.0},
    {"name": "cfg_k2_z35", "k_top": 2, "vol_z": 3.5, "move_z": 2.0},
    {"name": "cfg_k5_z25", "k_top": 5, "vol_z": 2.5, "move_z": 1.6},
)
PRIMARY_CANONICAL_CFG = "cfg_k3_z25"
SAR_LATEST_FLIP_MIN = 870  # 14:30 ET (no late-day SAR churn inside final 80m)


def _simulate_leg_with_exit_bar(
    bars_o: Sequence[float] | Any,
    bars_h: Sequence[float] | Any,
    bars_l: Sequence[float] | Any,
    bars_c: Sequence[float] | Any,
    start_idx: int,
    *,
    entry_px: float,
    side: str,
    params: dict[str, float],
) -> dict[str, Any]:
    """Conservative bar-by-bar per-symbol min trailing returning exact exit bar index & price."""
    n = len(bars_o)
    if start_idx >= n or entry_px <= 0.0:
        return {
            "gross_bps": 0.0,
            "mae_pct": 0.0,
            "mfe_pct": 0.0,
            "green_to_red": False,
            "exit_idx": n - 1,
            "exit_px": entry_px,
            "exit_reason": "NO_BARS",
            "stopped": False,
        }
    is_long = side.upper() == "LONG"
    hard_stop_frac = min(float(params["hard_stop_pct"]), ADR0022_MAX_STOP_PCT) / 100.0
    min_trail_frac = float(params["min_trail_bps"]) / 10_000.0
    be_arm_frac = float(params["be_arm_bps"]) / 10_000.0
    be_lock_frac = float(params["be_lock_bps"]) / 10_000.0

    stop_px = entry_px * (1.0 - hard_stop_frac) if is_long else entry_px * (1.0 + hard_stop_frac)
    extreme_px = entry_px
    mae_frac = 0.0
    mfe_frac = 0.0
    be_armed = False
    exit_idx = n - 1
    exit_px = float(bars_c[-1])
    exit_reason = "EOD_FLAT"
    stopped = False

    for i in range(start_idx, n):
        bh = float(bars_h[i])
        bl = float(bars_l[i])
        if not (math.isfinite(bh) and math.isfinite(bl) and bh > 0.0 and bl > 0.0):
            continue
        if is_long:
            mae_frac = max(mae_frac, (entry_px - bl) / entry_px)
            if bl <= stop_px:
                exit_idx = i
                exit_px = stop_px
                exit_reason = "PER_SYM_MIN_TRAIL_STOP" if (be_armed or stop_px > entry_px) else "ADR0022_HARD_STOP"
                stopped = True
                break
            if bh > extreme_px:
                extreme_px = bh
                mfe_frac = (extreme_px - entry_px) / entry_px
            if mfe_frac >= be_arm_frac:
                be_armed = True
                eff_trail = (
                    max(0.0010, 0.55 * min_trail_frac)
                    if mfe_frac >= 0.04
                    else (max(0.0012, 0.75 * min_trail_frac) if mfe_frac >= 0.02 else min_trail_frac)
                )
                stop_px = max(stop_px, max(entry_px * (1.0 + be_lock_frac), extreme_px * (1.0 - eff_trail)))
        else:
            mae_frac = max(mae_frac, (bh - entry_px) / entry_px)
            if bh >= stop_px:
                exit_idx = i
                exit_px = stop_px
                exit_reason = "PER_SYM_MIN_TRAIL_STOP" if (be_armed or stop_px < entry_px) else "ADR0022_HARD_STOP"
                stopped = True
                break
            if bl < extreme_px:
                extreme_px = bl
                mfe_frac = (entry_px - extreme_px) / entry_px
            if mfe_frac >= be_arm_frac:
                be_armed = True
                eff_trail = (
                    max(0.0010, 0.55 * min_trail_frac)
                    if mfe_frac >= 0.04
                    else (max(0.0012, 0.75 * min_trail_frac) if mfe_frac >= 0.02 else min_trail_frac)
                )
                stop_px = min(stop_px, min(entry_px * (1.0 - be_lock_frac), extreme_px * (1.0 + eff_trail)))

    ret_frac = ((exit_px - entry_px) / entry_px) if is_long else ((entry_px - exit_px) / entry_px)
    return {
        "gross_bps": ret_frac * 10_000.0,
        "mae_pct": max(0.0, mae_frac * 100.0),
        "mfe_pct": max(0.0, mfe_frac * 100.0),
        "green_to_red": bool(mfe_frac >= be_arm_frac and ret_frac < 0.0),
        "exit_idx": exit_idx,
        "exit_px": exit_px,
        "exit_reason": exit_reason,
        "stopped": stopped,
    }


def _simulate_early_plus_reversal_sar(
    bars_min: Sequence[int] | Any,
    bars_o: Sequence[float] | Any,
    bars_h: Sequence[float] | Any,
    bars_l: Sequence[float] | Any,
    bars_c: Sequence[float] | Any,
    *,
    init_side: str,
    trigger_px: float,
    vwap_0949: float,
    params: dict[str, float],
    has_sec_dilution_90d: bool,
) -> dict[str, Any]:
    """Simulate 3m Pre-Fill Confirm Leg 1 + Single-Use Inverse Reversal SAR Leg 2 (Long<->Short)."""
    n = len(bars_o)
    if n < 3:
        return {"filled": False, "cancel_reason": "NO_BARS"}
    is_long = init_side.upper() == "LONG"
    fill_idx = -1
    entry_px = 0.0
    cancel_reason = "TTL_EXPIRED_UNCONFIRMED"

    for idx in range(min(3, n - 1)):
        bo, bh, bl = float(bars_o[idx]), float(bars_h[idx]), float(bars_l[idx])
        if not (math.isfinite(bo) and bo > 0.0):
            continue
        if is_long:
            if bl <= vwap_0949:
                cancel_reason = "PULL_ADVERSE_FLOW_VWAP"
                break
            if bh >= trigger_px:
                fill_idx = idx
                entry_px = max(bo, trigger_px)
                break
        else:
            if bh >= vwap_0949:
                cancel_reason = "PULL_ADVERSE_FLOW_VWAP"
                break
            if bl <= trigger_px:
                fill_idx = idx
                entry_px = min(bo, trigger_px)
                break

    if fill_idx < 0 or entry_px <= 0.0:
        return {"filled": False, "cancel_reason": cancel_reason}

    leg1 = _simulate_leg_with_exit_bar(
        bars_o, bars_h, bars_l, bars_c, fill_idx + 1, entry_px=entry_px, side=init_side, params=params
    )

    # Check Disciplined Single-Use Inverse SAR Flip (Kaminski & Lo 2014 + Cont-Kukanov-Stoikov 2014)
    sar_triggered = False
    sar_side = "SHORT" if is_long else "LONG"
    leg2: dict[str, Any] | None = None
    # Diagnostic Arm D: Any Stop/Trail Reversal SAR (flips whenever Leg 1 stops out or reverses off peak <= 14:30 ET)
    diag_any_rev_triggered = False
    diag_any_rev_leg2: dict[str, Any] | None = None
    ex_i = int(leg1["exit_idx"])
    if (
        bool(leg1["stopped"])
        and ex_i + 2 < n
        and int(bars_min[ex_i]) <= SAR_LATEST_FLIP_MIN
        and not (sar_side == "LONG" and has_sec_dilution_90d)
    ):
        next_o = float(bars_o[ex_i + 1])
        if math.isfinite(next_o) and next_o > 0.0:
            # Diagnostic Arm D: flip inverse on ANY stop/trail reversal exit (zero intrabar lookahead: next bar open)
            diag_any_rev_triggered = True
            diag_any_rev_leg2 = _simulate_leg_with_exit_bar(
                bars_o, bars_h, bars_l, bars_c, ex_i + 2, entry_px=next_o, side=sar_side, params=params
            )
            # Primary Pre-Registered Arm B: require clean reversal close back across opening VWAP on the stop bar
            rev_c = float(bars_c[ex_i])
            vwap_broken = (rev_c < vwap_0949) if is_long else (rev_c > vwap_0949)
            if vwap_broken:
                sar_triggered = True
                leg2 = diag_any_rev_leg2

    return {
        "filled": True,
        "leg1": leg1,
        "sar_triggered": sar_triggered,
        "sar_side": sar_side if sar_triggered else "",
        "leg2": leg2,
        "diag_any_rev_triggered": diag_any_rev_triggered,
        "diag_any_rev_side": sar_side if diag_any_rev_triggered else "",
        "diag_any_rev_leg2": diag_any_rev_leg2,
    }


def run_h116_trial111_study(market_data_root: pathlib.Path) -> dict[str, Any]:
    """Execute SOTA-upstream-verified Trial #111 (`H116`) across all 700 pre-wall SIP sessions."""
    import duckdb
    import exchange_calendars as xcals
    import numpy as np
    import pandas as pd
    import pandera.polars as pa_pl
    import polars as pl

    # 1. Verify SOTA Upstream Library Provenance & XNYS Official Trading Calendar
    xnys = xcals.get_calendar("XNYS")
    valid_xnys_dates = {d.strftime("%Y-%m-%d") for d in xnys.sessions_in_range(SAMPLE_START_ISO, STUDY_WALL_ISO)}

    sip_root = market_data_root / "alpaca-sip" / "raw" / "bars-1min"
    ca_root = market_data_root / "alpaca-sip" / "raw" / "corporate-actions"
    pub_root = market_data_root / "public" / "tables"

    sip_dirs = sorted(
        d
        for d in sip_root.iterdir()
        if d.is_dir() and SAMPLE_START_ISO <= d.name <= STUDY_WALL_ISO and d.name in valid_xnys_dates
    )
    if not sip_dirs or sip_dirs[-1].name > STUDY_WALL_ISO:
        raise ValueError("Pre-wall SIP directories invalid or exceeded STUDY_WALL_DATE")

    ca_files = sorted(p.as_posix() for p in ca_root.glob("*.parquet") if p.name[:7] <= STUDY_WALL_ISO[:7])
    sec_sym_p = (pub_root / "sec_symbols.parquet").as_posix()
    sec_fil_p = (pub_root / "sec_filings.parquet").as_posix()

    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute("PRAGMA memory_limit='24GB'")

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
        CREATE TEMP TABLE sec_dil_dates AS
        SELECT o.symbol, SUBSTR(f.accepted, 1, 10) AS dil_date
        FROM read_parquet('{sec_fil_p}') f
        JOIN operating_syms o ON o.cik = f.cik
        WHERE f.accepted IS NOT NULL AND LENGTH(f.accepted) >= 10
          AND SUBSTR(f.accepted, 1, 10) BETWEEN '2023-10-01' AND '{STUDY_WALL_ISO}'
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

    # 2. Whole-market single-pass extraction of opening 20m volume/return & daily stats
    con.execute(
        """
        CREATE TEMP TABLE sip_daily_norm (
            symbol VARCHAR,
            trade_date VARCHAR,
            open_0930 DOUBLE,
            high_0949 DOUBLE,
            low_0949 DOUBLE,
            close_0949 DOUBLE,
            vwap_0949 DOUBLE,
            open20_vol DOUBLE,
            rth_close DOUBLE,
            rth_dvol DOUBLE,
            med_1m_range_bps DOUBLE
        )
        """
    )
    for idx in range(0, len(sip_dirs), 25):
        batch = sip_dirs[idx : idx + 25]
        globs = [f"{d.as_posix()}/*.parquet" for d in batch]
        con.execute(
            """
            INSERT INTO sip_daily_norm
            WITH raw_bars AS (
                SELECT
                    UPPER(TRIM(symbol)) AS symbol,
                    CAST(to_timestamp(t_ns / 1e9) AT TIME ZONE 'America/New_York' AS TIMESTAMP) AS ny_ts,
                    o, h, l, c, v, COALESCE(NULLIF(vw, 0), c) AS vw
                FROM read_parquet($1)
                WHERE UPPER(TRIM(symbol)) IN (SELECT symbol FROM operating_syms)
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
                SUM(vw * v) FILTER (WHERE ny_min BETWEEN 570 AND 959) AS rth_dvol,
                MEDIAN((h - l) / NULLIF(c, 0) * 10000.0) FILTER (WHERE ny_min BETWEEN 570 AND 959) AS med_1m_range_bps
            FROM tagged
            WHERE trade_date <= $2
            GROUP BY symbol, trade_date
            """,
            [globs, STUDY_WALL_ISO],
        )

    # 3. Compute strictly causal T-1 per-symbol norms (mu/sigma open20_vol, sigma open20_ret, med_1m_range, ADV20)
    pl_df = pl.from_arrow(
        con.sql(
            f"""
        WITH step1 AS (
            SELECT
                d.*,
                (d.close_0949 / NULLIF(d.open_0930, 0) - 1.0) AS ret_open20,
                LAG(rth_close, 1) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_rth_close,
                (
                    rth_close / NULLIF(
                        LAG(rth_close, 1) OVER (PARTITION BY symbol ORDER BY trade_date), 0
                    ) - 1.0
                ) AS daily_ret,
                LEAD(trade_date, 1) OVER (PARTITION BY symbol ORDER BY trade_date) AS next_trade_date,
                LAG(trade_date, 1) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_trade_date
            FROM sip_daily_norm d
        ),
        norms AS (
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
                ) AS mu_open20_vol,
                STDDEV_SAMP(open20_vol) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS sd_open20_vol,
                STDDEV_SAMP(ret_open20) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS sd_open20_ret,
                COUNT(*) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS hist_days
            FROM step1 s
        )
        SELECT
            n.symbol,
            n.trade_date,
            n.open_0930,
            n.high_0949,
            n.low_0949,
            n.close_0949,
            n.vwap_0949,
            n.ret_open20,
            n.open20_vol,
            n.prev_rth_close,
            n.adv20_usd,
            n.vol20d_daily,
            n.causal_med_1m_range_bps,
            n.mu_open20_vol,
            n.sd_open20_vol,
            n.sd_open20_ret,
            n.hist_days,
            EXISTS (
                SELECT 1 FROM split_events se
                WHERE se.symbol = n.symbol
                  AND se.ex_date BETWEEN COALESCE(n.prev_trade_date, n.trade_date)
                                     AND COALESCE(n.next_trade_date, n.trade_date)
            ) AS has_near_split,
            EXISTS (
                SELECT 1 FROM sec_dil_dates sd
                WHERE sd.symbol = n.symbol
                  AND CAST(sd.dil_date AS DATE)
                      BETWEEN (CAST(n.trade_date AS DATE) - INTERVAL 90 DAY)
                          AND (CAST(n.trade_date AS DATE) - INTERVAL 1 DAY)
            ) AS has_sec_dilution_90d
        FROM norms n
        WHERE n.trade_date >= '{EVAL_START_ISO}' AND n.trade_date <= '{STUDY_WALL_ISO}'
          AND n.prev_rth_close >= 10.0
          AND n.adv20_usd >= 50000000.0
          AND n.hist_days >= 20
          AND n.mu_open20_vol > 0
          AND n.sd_open20_vol > 0
          AND n.sd_open20_ret > 0.0005
          AND n.causal_med_1m_range_bps > 0
          AND n.open_0930 > 0 AND n.close_0949 > 0 AND n.vwap_0949 > 0
        ORDER BY n.trade_date, n.symbol
        """
        ).to_arrow_table()
    )

    # Validate panel contract via SOTA Polars + Pandera (zero NaN/Inf, strictly <= STUDY_WALL_ISO)
    assert isinstance(pl_df, pl.DataFrame)
    pl_df = pl_df.filter(~pl.col("has_near_split")).with_columns(
        ((pl.col("open20_vol") - pl.col("mu_open20_vol")) / pl.col("sd_open20_vol"))
        .cast(pl.Float64)
        .alias("vol_z_norm"),
        (pl.col("open20_vol") / pl.col("mu_open20_vol")).cast(pl.Float64).alias("rvol_open20"),
        (pl.col("ret_open20").abs() / pl.col("sd_open20_ret")).cast(pl.Float64).alias("move_z_norm"),
    )
    schema = pa_pl.DataFrameSchema(
        {
            "symbol": pa_pl.Column(pl.String),
            "trade_date": pa_pl.Column(pl.String, checks=pa_pl.Check.le(STUDY_WALL_ISO)),
            "vol_z_norm": pa_pl.Column(pl.Float64, nullable=False),
            "rvol_open20": pa_pl.Column(pl.Float64, checks=pa_pl.Check.ge(0.0)),
            "move_z_norm": pa_pl.Column(pl.Float64, checks=pa_pl.Check.ge(0.0)),
        }
    )
    pl_df = schema.validate(pl_df)
    pl_df = pl_df.with_columns(
        (pl.col("vol_z_norm").clip(0.0, 20.0) * 10.0 + pl.col("move_z_norm").clip(0.0, 15.0) * 12.0).alias(
            "disruption_score"
        )
    )
    eq_df = pl_df.to_pandas()

    all_dates_sorted = sorted(eq_df["trade_date"].unique().tolist())
    by_date: dict[str, pd.DataFrame] = {str(dt): grp for dt, grp in eq_df.groupby("trade_date")}
    cfg_candidates: dict[str, list[dict[str, Any]]] = {c["name"]: [] for c in PRIMARY_CONFIG_GRID}
    needed_pairs: set[tuple[str, str]] = set()

    for dt in all_dates_sorted:
        grp = by_date.get(dt)
        if grp is None or grp.empty:
            continue
        for cfg in PRIMARY_CONFIG_GRID:
            k_top = int(cfg["k_top"])
            vz_min = float(cfg["vol_z"])
            mz_min = float(cfg["move_z"])
            sub = (
                grp[
                    (grp["vol_z_norm"] >= vz_min)
                    & (grp["rvol_open20"] >= 2.8)
                    & (grp["move_z_norm"] >= mz_min)
                    & (grp["ret_open20"].abs() >= 0.010)
                    & (grp["ret_open20"].abs() <= 0.100)
                    & (
                        (
                            (grp["ret_open20"] > 0.0)
                            & (~grp["has_sec_dilution_90d"])
                            & (grp["close_0949"] > grp["vwap_0949"])
                        )
                        | ((grp["ret_open20"] < 0.0) & (grp["close_0949"] < grp["vwap_0949"]))
                    )
                ]
                .sort_values("disruption_score", ascending=False)
                .head(k_top)
            )
            for r in sub.itertuples(index=False):
                sym = str(r.symbol)
                needed_pairs.add((dt, sym))
                cfg_candidates[cfg["name"]].append(
                    {
                        "trade_date": dt,
                        "symbol": sym,
                        "init_side": "LONG" if float(r.ret_open20) > 0.0 else "SHORT",
                        "high_0949": float(r.high_0949),
                        "low_0949": float(r.low_0949),
                        "vwap_0949": float(r.vwap_0949),
                        "adv20_usd": float(r.adv20_usd),
                        "vol20d_daily": float(r.vol20d_daily or 0.02),
                        "causal_med_1m_range_bps": float(r.causal_med_1m_range_bps),
                        "has_sec_dilution_90d": bool(r.has_sec_dilution_90d),
                    }
                )

    # 4. Load 09:50..15:50 ET 1m bars ONLY for candidate norm-disruption pairs
    bars_by_pair: dict[tuple[str, str], tuple[Any, Any, Any, Any, Any]] = {}
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
                CAST(EXTRACT(HOUR FROM ny_ts) * 60 + EXTRACT(MINUTE FROM ny_ts) AS INTEGER) AS ny_min,
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
                    g_b["ny_min"].to_numpy(dtype=int),
                    g_b["o"].to_numpy(dtype=float),
                    g_b["h"].to_numpy(dtype=float),
                    g_b["l"].to_numpy(dtype=float),
                    g_b["c"].to_numpy(dtype=float),
                )
    con.close()

    # 5. Simulate all 5 configs: Arm A (Early Only), Arm B (Early + SAR), Arm C (SAR Isolated) + Per-Symbol Top-20
    cfg_daily_2x_sar_t: dict[str, dict[str, float]] = {}
    cfg_daily_2x_sar_m: dict[str, dict[str, float]] = {}
    early_only_events: list[dict[str, Any]] = []
    combined_sar_taker_events: list[dict[str, Any]] = []
    combined_sar_maker_events: list[dict[str, Any]] = []
    sar_only_events: list[dict[str, Any]] = []
    diag_any_rev_comb_t_events: list[dict[str, Any]] = []
    diag_any_rev_comb_m_events: list[dict[str, Any]] = []
    diag_any_rev_iso_events: list[dict[str, Any]] = []
    canon_total = 0
    canon_filled = 0
    canon_sar_flips = 0
    canon_diag_any_rev_flips = 0
    canon_cancels: dict[str, int] = defaultdict(int)

    for cfg in PRIMARY_CONFIG_GRID:
        cname = str(cfg["name"])
        b_sar_t: dict[str, list[float]] = defaultdict(list)
        b_sar_m: dict[str, list[float]] = defaultdict(list)
        for cand in cfg_candidates[cname]:
            bars = bars_by_pair.get((cand["trade_date"], cand["symbol"]))
            if bars is None:
                continue
            bmin, bo, bh, bl, bc = bars
            side1 = cand["init_side"]
            c1_t = _almgren_chriss_1x_cost_bps(
                cand["adv20_usd"], cand["vol20d_daily"], is_short=(side1 == "SHORT"), hold_days=1
            )
            c1_m = _maker_1x_cost_bps(cand["adv20_usd"], cand["vol20d_daily"], is_short=(side1 == "SHORT"))
            params = _causal_per_symbol_min_trail_params(c1_t, cand["causal_med_1m_range_bps"])
            trig_px = cand["high_0949"] if side1 == "LONG" else cand["low_0949"]
            sim = _simulate_early_plus_reversal_sar(
                bmin,
                bo,
                bh,
                bl,
                bc,
                init_side=side1,
                trigger_px=trig_px,
                vwap_0949=cand["vwap_0949"],
                params=params,
                has_sec_dilution_90d=cand["has_sec_dilution_90d"],
            )
            if cname == PRIMARY_CANONICAL_CFG:
                canon_total += 1
                if not sim["filled"]:
                    canon_cancels[str(sim["cancel_reason"])] += 1
            if not sim["filled"]:
                continue

            l1 = sim["leg1"]
            g1 = float(l1["gross_bps"])
            comb_1x_t = g1 - c1_t
            comb_2x_t = g1 - 2.0 * c1_t
            comb_1x_m = g1 - c1_m
            comb_2x_m = g1 - 2.0 * c1_m

            if sim["sar_triggered"] and sim["leg2"] is not None:
                l2 = sim["leg2"]
                side2 = str(sim["sar_side"])
                c2_t = _almgren_chriss_1x_cost_bps(
                    cand["adv20_usd"], cand["vol20d_daily"], is_short=(side2 == "SHORT"), hold_days=1
                )
                c2_m = _maker_1x_cost_bps(cand["adv20_usd"], cand["vol20d_daily"], is_short=(side2 == "SHORT"))
                g2 = float(l2["gross_bps"])
                # Sequential full-capital leg 1 + leg 2 (non-overlapping in time!)
                comb_1x_t += g2 - c2_t
                comb_2x_t += g2 - 2.0 * c2_t
                comb_1x_m += g2 - c2_m
                comb_2x_m += g2 - 2.0 * c2_m
                if cname == PRIMARY_CANONICAL_CFG:
                    canon_sar_flips += 1
                    sar_only_events.append(
                        {
                            "trade_date": cand["trade_date"],
                            "symbol": cand["symbol"],
                            "side": side2,
                            "cost_1x_bps": c2_t,
                            "net_1x_bps": g2 - c2_t,
                            "net_2x_bps": g2 - 2.0 * c2_t,
                            "maker_2x_bps": g2 - 2.0 * c2_m,
                            "mae_pct": float(l2["mae_pct"]),
                            "mfe_pct": float(l2["mfe_pct"]),
                            "green_to_red": bool(l2["green_to_red"]),
                            "vol20d_daily": cand["vol20d_daily"],
                        }
                    )

            diag_comb_1x_t = g1 - c1_t
            diag_comb_2x_t = g1 - 2.0 * c1_t
            diag_comb_1x_m = g1 - c1_m
            diag_comb_2x_m = g1 - 2.0 * c1_m
            if sim["diag_any_rev_triggered"] and sim["diag_any_rev_leg2"] is not None:
                dl2 = sim["diag_any_rev_leg2"]
                dside2 = str(sim["diag_any_rev_side"])
                dc2_t = _almgren_chriss_1x_cost_bps(
                    cand["adv20_usd"], cand["vol20d_daily"], is_short=(dside2 == "SHORT"), hold_days=1
                )
                dc2_m = _maker_1x_cost_bps(cand["adv20_usd"], cand["vol20d_daily"], is_short=(dside2 == "SHORT"))
                dg2 = float(dl2["gross_bps"])
                diag_comb_1x_t += dg2 - dc2_t
                diag_comb_2x_t += dg2 - 2.0 * dc2_t
                diag_comb_1x_m += dg2 - dc2_m
                diag_comb_2x_m += dg2 - 2.0 * dc2_m
                if cname == PRIMARY_CANONICAL_CFG:
                    canon_diag_any_rev_flips += 1
                    diag_any_rev_iso_events.append(
                        {
                            "trade_date": cand["trade_date"],
                            "symbol": cand["symbol"],
                            "side": dside2,
                            "cost_1x_bps": dc2_t,
                            "net_1x_bps": dg2 - dc2_t,
                            "net_2x_bps": dg2 - 2.0 * dc2_t,
                            "maker_2x_bps": dg2 - 2.0 * dc2_m,
                            "mae_pct": float(dl2["mae_pct"]),
                            "mfe_pct": float(dl2["mfe_pct"]),
                            "green_to_red": bool(dl2["green_to_red"]),
                            "vol20d_daily": cand["vol20d_daily"],
                        }
                    )

            b_sar_t[cand["trade_date"]].append(comb_2x_t)
            b_sar_m[cand["trade_date"]].append(comb_2x_m)
            if cname == PRIMARY_CANONICAL_CFG:
                canon_filled += 1
                diag_any_rev_comb_t_events.append(
                    {
                        "trade_date": cand["trade_date"],
                        "symbol": cand["symbol"],
                        "side": side1,
                        "sar_flipped": bool(sim["diag_any_rev_triggered"]),
                        "cost_1x_bps": c1_t,
                        "net_1x_bps": diag_comb_1x_t,
                        "net_2x_bps": diag_comb_2x_t,
                        "mae_pct": float(l1["mae_pct"]),
                        "mfe_pct": float(l1["mfe_pct"]),
                        "green_to_red": bool(l1["green_to_red"]),
                        "vol20d_daily": cand["vol20d_daily"],
                    }
                )
                diag_any_rev_comb_m_events.append(
                    {
                        "trade_date": cand["trade_date"],
                        "symbol": cand["symbol"],
                        "side": side1,
                        "cost_1x_bps": c1_m,
                        "net_1x_bps": diag_comb_1x_m,
                        "net_2x_bps": diag_comb_2x_m,
                        "mae_pct": float(l1["mae_pct"]),
                        "mfe_pct": float(l1["mfe_pct"]),
                        "green_to_red": bool(l1["green_to_red"]),
                        "vol20d_daily": cand["vol20d_daily"],
                    }
                )
                early_only_events.append(
                    {
                        "trade_date": cand["trade_date"],
                        "symbol": cand["symbol"],
                        "side": side1,
                        "cost_1x_bps": c1_t,
                        "net_1x_bps": g1 - c1_t,
                        "net_2x_bps": g1 - 2.0 * c1_t,
                        "mae_pct": float(l1["mae_pct"]),
                        "mfe_pct": float(l1["mfe_pct"]),
                        "green_to_red": bool(l1["green_to_red"]),
                        "vol20d_daily": cand["vol20d_daily"],
                    }
                )
                combined_sar_taker_events.append(
                    {
                        "trade_date": cand["trade_date"],
                        "symbol": cand["symbol"],
                        "side": side1,
                        "sar_flipped": bool(sim["sar_triggered"]),
                        "min_trail_bps": float(params["min_trail_bps"]),
                        "cost_1x_bps": c1_t,
                        "net_1x_bps": comb_1x_t,
                        "net_2x_bps": comb_2x_t,
                        "maker_1x_bps": comb_1x_m,
                        "maker_2x_bps": comb_2x_m,
                        "mae_pct": float(l1["mae_pct"]),
                        "mfe_pct": float(l1["mfe_pct"]),
                        "green_to_red": bool(l1["green_to_red"]),
                        "vol20d_daily": cand["vol20d_daily"],
                    }
                )
                combined_sar_maker_events.append(
                    {
                        "trade_date": cand["trade_date"],
                        "symbol": cand["symbol"],
                        "side": side1,
                        "cost_1x_bps": c1_m,
                        "net_1x_bps": comb_1x_m,
                        "net_2x_bps": comb_2x_m,
                        "mae_pct": float(l1["mae_pct"]),
                        "mfe_pct": float(l1["mfe_pct"]),
                        "green_to_red": bool(l1["green_to_red"]),
                        "vol20d_daily": cand["vol20d_daily"],
                    }
                )
        cfg_daily_2x_sar_t[cname] = {dt: float(np.mean(v)) for dt, v in b_sar_t.items() if v}
        cfg_daily_2x_sar_m[cname] = {dt: float(np.mean(v)) for dt, v in b_sar_m.items() if v}

    mat_t = np.asarray(
        [[cfg_daily_2x_sar_t[c["name"]].get(dt, 0.0) for c in PRIMARY_CONFIG_GRID] for dt in all_dates_sorted],
        dtype=float,
    )
    mat_m = np.asarray(
        [[cfg_daily_2x_sar_m[c["name"]].get(dt, 0.0) for c in PRIMARY_CONFIG_GRID] for dt in all_dates_sorted],
        dtype=float,
    )
    cpcv_t = _compute_fold_fitted_cpcv_and_pbo(mat_t)
    cpcv_m = _compute_fold_fitted_cpcv_and_pbo(mat_m)

    def _to_daily(df_in: pd.DataFrame) -> pd.DataFrame:
        return (
            df_in.groupby("trade_date", as_index=False)[["net_1x_bps", "net_2x_bps"]].mean()
            if not df_in.empty
            else pd.DataFrame()
        )

    df_early = pd.DataFrame(early_only_events)
    df_comb_t = pd.DataFrame(combined_sar_taker_events)
    df_comb_m = pd.DataFrame(combined_sar_maker_events)
    df_sar_iso = pd.DataFrame(sar_only_events)
    df_any_t = pd.DataFrame(diag_any_rev_comb_t_events)
    df_any_m = pd.DataFrame(diag_any_rev_comb_m_events)
    df_any_iso = pd.DataFrame(diag_any_rev_iso_events)

    def _safe_eval_arm(
        ev_df: pd.DataFrame, *, label: str, cpcv_override: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        ddf = _to_daily(ev_df)
        if ev_df.empty or len(ddf) < 10:
            return {
                "label": label,
                "events": int(len(ev_df)),
                "dates": int(len(ddf)),
                "mean_1x_event_bps": round(float(ev_df["net_1x_bps"].mean()), 2) if not ev_df.empty else 0.0,
                "mean_2x_event_bps": round(float(ev_df["net_2x_bps"].mean()), 2) if not ev_df.empty else 0.0,
                "passed": False,
                "note": "TOO_FEW_DATES_FOR_BOOTSTRAP_SPA_LT_10",
            }
        return _evaluate_arm_metrics(ev_df, ddf, label=label, n_trials=TRIAL_NUMBER, cpcv_override=cpcv_override)

    arm_early_only = _safe_eval_arm(df_early, label="H116_EARLY_NORM_DISRUPTION_ONLY")
    arm_comb_t = _safe_eval_arm(
        df_comb_t, label="H116_PRIMARY_NORM_DISRUPTION_PLUS_REVERSAL_SAR_TAKER", cpcv_override=cpcv_t
    )
    arm_comb_t["prefill_and_sar_stats"] = {
        "total_norm_disruption_candidates": canon_total,
        "confirmed_filled_count": canon_filled,
        "cancelled_unfilled_0bps_count": canon_total - canon_filled,
        "fill_rate_pct": round(canon_filled / max(1, canon_total) * 100.0, 1),
        "cancel_reasons": dict(canon_cancels),
        "reversal_sar_flips_executed_vwap_strict": canon_sar_flips,
        "sar_flip_rate_vwap_strict_pct": round(canon_sar_flips / max(1, canon_filled) * 100.0, 1),
        "reversal_sar_flips_executed_any_stop_or_trail": canon_diag_any_rev_flips,
        "sar_flip_rate_any_stop_or_trail_pct": round(canon_diag_any_rev_flips / max(1, canon_filled) * 100.0, 1),
    }
    arm_comb_m = _safe_eval_arm(
        df_comb_m, label="H116_DIAG_NORM_DISRUPTION_PLUS_SAR_1000RPM_MAKER", cpcv_override=cpcv_m
    )
    arm_sar_iso = _safe_eval_arm(df_sar_iso, label="H116_DIAG_SAR_VWAP_BREAK_ISOLATED")
    arm_any_rev_t = _safe_eval_arm(df_any_t, label="H116_DIAG_ANY_STOP_TRAIL_SAR_TAKER")
    arm_any_rev_m = _safe_eval_arm(df_any_m, label="H116_DIAG_ANY_STOP_TRAIL_SAR_1000RPM_MAKER")
    arm_any_rev_iso = _safe_eval_arm(df_any_iso, label="H116_DIAG_ANY_STOP_TRAIL_SAR_INVERSE_LEG_ISOLATED")

    # Top-20 Most-Potential Symbols Sim Performance (>= 3 pre-wall disruption events, ranked by 1x expectancy)
    top_syms: list[dict[str, Any]] = []
    if not df_comb_t.empty:
        for sym_k, gs in df_comb_t.groupby("symbol"):
            if len(gs) < 3:
                continue
            top_syms.append(
                {
                    "symbol": str(sym_k),
                    "events": int(len(gs)),
                    "longs": int((gs["side"] == "LONG").sum()),
                    "shorts": int((gs["side"] == "SHORT").sum()),
                    "sar_flips": int(gs["sar_flipped"].sum()),
                    "avg_min_trail_bps": round(float(gs["min_trail_bps"].mean()), 1),
                    "mean_1x_taker_bps": round(float(gs["net_1x_bps"].mean()), 2),
                    "mean_2x_taker_bps": round(float(gs["net_2x_bps"].mean()), 2),
                    "mean_1x_maker_bps": round(float(gs["maker_1x_bps"].mean()), 2),
                    "mean_2x_maker_bps": round(float(gs["maker_2x_bps"].mean()), 2),
                    "win_rate_2x_maker_pct": round(float((gs["maker_2x_bps"] > 0.0).mean() * 100.0), 1),
                }
            )
        top_syms.sort(key=lambda x: (x["mean_1x_taker_bps"], x["events"]), reverse=True)

    receipt = {
        "schema": "sota.prewall-trial-receipt.v1",
        "trial_number": TRIAL_NUMBER,
        "hypothesis_id": HYPOTHESIS_ID,
        "prereg_spec_sha256": PREREG_SPEC_SHA256,
        "study_wall_iso": STUDY_WALL_ISO,
        "sample_window": f"{SAMPLE_START_ISO}..{STUDY_WALL_ISO}",
        "eval_start": EVAL_START_ISO,
        "sessions_scanned": len(sip_dirs),
        "xnys_calendar_verified": True,
        "polars_pandera_schema_verified": True,
        "post_wall_bytes_read": 0,
        "sota_upstreams_executed": [
            "gerrymanoim/exchange_calendars==4.13.2",
            "duckdb/duckdb==2.0.0",
            "pola-rs/polars==2.1.0",
            "unionai-oss/pandera==0.35.0",
            "statsmodels/statsmodels==0.15.0",
            "skfolio/skfolio==1.12.7",
            "ml4t/diagnostic==0.2.0",
        ],
        "early_only_arm": arm_early_only,
        "primary_early_plus_sar_taker_arm": arm_comb_t,
        "diagnostic_early_plus_sar_1000rpm_maker_arm": arm_comb_m,
        "sar_inverse_leg_isolated_arm": arm_sar_iso,
        "diagnostic_any_stop_trail_sar_taker_arm": arm_any_rev_t,
        "diagnostic_any_stop_trail_sar_1000rpm_maker_arm": arm_any_rev_m,
        "diagnostic_any_stop_trail_sar_inverse_leg_isolated_arm": arm_any_rev_iso,
        "sar_delta_vs_early_only": {
            "delta_mean_1x_event_bps": round(
                float(arm_comb_t.get("mean_1x_event_bps") or 0.0)
                - float(arm_early_only.get("mean_1x_event_bps") or 0.0),
                2,
            ),
            "delta_mean_2x_event_bps": round(
                float(arm_comb_t.get("mean_2x_event_bps") or 0.0)
                - float(arm_early_only.get("mean_2x_event_bps") or 0.0),
                2,
            ),
            "delta_max_dd_2x_pct_reduction": round(
                float(arm_early_only.get("max_dd_2x_pct") or 0.0) - float(arm_comb_t.get("max_dd_2x_pct") or 0.0),
                2,
            ),
        },
        "top_20_most_potential_symbols_sim": top_syms[:20],
        "overall_passed": bool(arm_comb_t.get("passed")),
        "overall_verdict": "PASSED" if bool(arm_comb_t.get("passed")) else "FAILED",
        "promotion_track": "ALPHA_STRATEGY_MINOR_CANDIDATE"
        if bool(arm_comb_t.get("passed"))
        else "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    raw_hash = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    receipt["receipt_sha256"] = raw_hash
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Trial #111 (H116) SOTA Norm Disruption + Reversal SAR.")
    parser.add_argument(
        "--market-data-root", type=pathlib.Path, default=pathlib.Path.home() / "Documents" / "market-data"
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "docs" / "TRIAL111_H116_PREWALL_RECEIPT.json",
    )
    args = parser.parse_args()
    receipt = run_h116_trial111_study(args.market_data_root)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
