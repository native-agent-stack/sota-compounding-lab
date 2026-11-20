#!/usr/bin/env python3
"""Trial #109 (`H114`): Multi-Feed DF1..DF9 Concentrated Extreme Long/Short Screening x Pre-Fill 0 bps & Min Trail.

Pre-Registered Immutable Spec SHA-256 (`sota-bus` Trial #109):
  `2dbfbc0c92830be297e7963be54c38be29b8ff75a48b85329c860fd3a0ac6dc8`
Required HLZ Multiplicity Hurdle (`N = 109`): `t*(109) = 4.2169`
ADIA Lab #19 Power Gate (`D=700`, `sigma=95 bps`): APPROVED (`SR_min_ann=2.5301 <= 5.0`, `MinTRL=499`)

Multi-Feed PIT Screening Stack (`2024-01-02..2026-10-15`, `0` post-wall bytes):
- `DF1` SIP 1-minute bars (`09:30..15:50 ET`) + `DF3` Corporate-Action Split Guard `[T-1..T+1]`
- `DF4` Trailing 20d `09:30..09:49` clock-bucket `RVOL_tod`
- `DF5` PIT SEC EDGAR `8-K` catalyst (`accepted` within 36h before `09:30 ET`) & `90d` `424B5`/`S-3`/`S-3ASR`/`EFFECT`
- `DF7` FINRA `CNMSshvol` `T-1` daily short-volume ratio (`short_volume / total_volume`, `day = prev_trade_date`)
- `DF8` FINRA Rule 4560 settled short interest (`days_to_cover`, strictly `known_from <= prev_trade_date`)
- `DF9` Live opening `SPY` + `UVXY` volatility/risk-off regime
- Execution: `3m` Pre-Fill Directional Confirmation (`0.00 bps` cancel-if-wrong) + Causal `T-1` Per-Symbol Min Trail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import defaultdict
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
    _simulate_blind_uniform_baseline,
    _simulate_prefill_and_per_symbol_min_trail,
)

TRIAL_NUMBER = 109
HYPOTHESIS_ID = "H114_MULTIFEED_DF1_DF9_CONCENTRATED_LONGSHORT_MIN_TRAIL"
PREREG_SPEC_SHA256 = "2dbfbc0c92830be297e7963be54c38be29b8ff75a48b85329c860fd3a0ac6dc8"

PRIMARY_CONFIG_GRID: tuple[dict[str, Any], ...] = (
    {"name": "cfg_k2_r22", "k_side": 2, "rvol_long": 2.2, "rvol_short": 1.8},  # Canonical baseline
    {"name": "cfg_k3_r22", "k_side": 3, "rvol_long": 2.2, "rvol_short": 1.8},
    {"name": "cfg_k2_r28", "k_side": 2, "rvol_long": 2.8, "rvol_short": 2.2},
    {"name": "cfg_k1_r25", "k_side": 1, "rvol_long": 2.5, "rvol_short": 2.0},
    {"name": "cfg_k4_r20", "k_side": 4, "rvol_long": 2.0, "rvol_short": 1.6},
)
PRIMARY_CANONICAL_CFG = "cfg_k2_r22"


def run_h114_trial109_study(market_data_root: pathlib.Path) -> dict[str, Any]:
    """Execute strictly pre-wall (`<= 2026-10-15`) multi-feed Trial #109 (`H114`) on 700d lake."""
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
    sh_vol_p = (pub_root / "short_volume.parquet").as_posix()
    sh_int_p = (pub_root / "short_interest.parquet").as_posix()

    con = duckdb.connect()
    con.execute("PRAGMA threads=16")
    con.execute("PRAGMA memory_limit='24GB'")

    # 1. Operating equities + DF3 splits + DF5 SEC 8-K & dilution + DF7 FINRA short_vol + DF8 FINRA SI
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
          AND SUBSTR(f.accepted, 1, 10) BETWEEN '2023-10-01' AND '{STUDY_WALL_ISO}'
          AND f.form IN ('424B5', 'S-3', 'S-3ASR', 'EFFECT')
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE sec_8k_events AS
        SELECT
            o.symbol,
            CAST(strptime(SUBSTR(f.accepted, 1, 19), '%Y-%m-%dT%H:%M:%S') AS TIMESTAMP) AS acc_utc
        FROM read_parquet('{sec_fil_p}') f
        JOIN operating_syms o ON o.cik = f.cik
        WHERE f.accepted IS NOT NULL AND LENGTH(f.accepted) >= 19
          AND SUBSTR(f.accepted, 1, 10) BETWEEN '2023-12-25' AND '{STUDY_WALL_ISO}'
          AND f.form = '8-K'
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE finra_shvol AS
        SELECT
            UPPER(TRIM(symbol)) AS symbol,
            day,
            SUM(short_volume) / NULLIF(SUM(total_volume), 0) AS short_vol_ratio
        FROM read_parquet('{sh_vol_p}')
        WHERE day BETWEEN '2023-12-01' AND '{STUDY_WALL_ISO}'
          AND total_volume > 0
        GROUP BY 1, 2
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE finra_si AS
        SELECT
            UPPER(TRIM(symbol)) AS symbol,
            known_from,
            MAX(days_to_cover) AS days_to_cover
        FROM read_parquet('{sh_int_p}')
        WHERE known_from IS NOT NULL
          AND known_from BETWEEN '2023-11-01' AND '{STUDY_WALL_ISO}'
          AND days_to_cover > 0
        GROUP BY 1, 2
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

    # 2. Single-pass batched DF1/DF4 extraction across all 700 sessions
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
                SUM(vw * v) FILTER (WHERE ny_min BETWEEN 570 AND 959) AS rth_dvol,
                MEDIAN((h - l) / NULLIF(c, 0) * 10000.0) FILTER (WHERE ny_min BETWEEN 570 AND 959) AS med_1m_range_bps
            FROM tagged
            WHERE trade_date <= $2
            GROUP BY symbol, trade_date
            """,
            [globs, STUDY_WALL_ISO],
        )

    # 3. Join strictly causal T-1 DF1+DF3+DF4+DF5+DF7+DF8 features
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
                COALESCE(sv.short_vol_ratio, 0.0) AS t1_short_vol_ratio,
                COALESCE(
                    (
                        SELECT si.days_to_cover FROM finra_si si
                        WHERE si.symbol = b.symbol
                          AND si.known_from <= COALESCE(b.prev_trade_date, b.trade_date)
                        ORDER BY si.known_from DESC
                        LIMIT 1
                    ),
                    0.0
                ) AS t8_days_to_cover,
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
                EXISTS (
                    SELECT 1 FROM sec_8k_events s8
                    WHERE s8.symbol = b.symbol
                      AND s8.acc_utc BETWEEN (CAST(b.trade_date || ' 13:30:00' AS TIMESTAMP) - INTERVAL 36 HOUR)
                                         AND CAST(b.trade_date || ' 13:30:00' AS TIMESTAMP)
                ) AS has_fresh_8k_36h,
                (b.symbol IN (SELECT symbol FROM operating_syms)) AS is_operating
            FROM base b
            LEFT JOIN finra_shvol sv
              ON sv.symbol = b.symbol AND sv.day = b.prev_trade_date
        )
        SELECT * FROM enriched
        WHERE trade_date >= '{EVAL_START_ISO}' AND trade_date <= '{STUDY_WALL_ISO}'
        ORDER BY trade_date, symbol
        """
    ).df()

    all_dates_sorted = sorted(daily_df["trade_date"].unique().tolist())

    # Extract DF9 opening SPY & UVXY returns (09:30 -> 09:49)
    def _extract_anchor_ret(sym_name: str) -> dict[str, float]:
        out_m: dict[str, float] = {}
        for row in daily_df[daily_df["symbol"] == sym_name].itertuples(index=False):
            o0 = float(row.open_0930 or 0.0)
            c49 = float(row.close_0949 or 0.0)
            if o0 > 0.0 and c49 > 0.0:
                out_m[str(row.trade_date)] = (c49 - o0) / o0
        return out_m

    spy_ret_map = _extract_anchor_ret("SPY")
    uvxy_ret_map = _extract_anchor_ret("UVXY")

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

    # Composite multi-feed conviction score (strictly causal DF4+DF5+DF7+DF8)
    eq_df["long_mf_score"] = (
        eq_df["rvol_open20"].clip(upper=12.0) * 10.0
        + eq_df["has_fresh_8k_36h"].astype(float) * 35.0
        + (eq_df["t1_short_vol_ratio"] >= 0.50).astype(float) * 20.0
        + (eq_df["t8_days_to_cover"] >= 3.0).astype(float) * 20.0
    )
    eq_df["short_mf_score"] = (
        eq_df["rvol_open20"].clip(upper=12.0) * 10.0
        + eq_df["has_sec_dilution_90d"].astype(float) * 40.0
        + (eq_df["t1_short_vol_ratio"] >= 0.45).astype(float) * 20.0
        + (-eq_df["ret_open20"] * 100.0).clip(lower=0.0, upper=8.0) * 3.0
    )

    by_date: dict[str, pd.DataFrame] = {str(dt): grp for dt, grp in eq_df.groupby("trade_date")}
    cfg_candidates: dict[str, list[dict[str, Any]]] = {c["name"]: [] for c in PRIMARY_CONFIG_GRID}
    unfiltered_pv_candidates: list[dict[str, Any]] = []
    needed_pairs: set[tuple[str, str]] = set()

    for dt in all_dates_sorted:
        spy_r = spy_ret_map.get(dt)
        uvxy_r = uvxy_ret_map.get(dt, 0.0)
        grp = by_date.get(dt)
        if spy_r is None or grp is None or grp.empty:
            continue
        non_panic = (spy_r >= -0.0025) and (uvxy_r <= 0.030)
        risk_off = (spy_r <= -0.0030) or (uvxy_r >= 0.020)

        for cfg in PRIMARY_CONFIG_GRID:
            k_side = int(cfg["k_side"])
            rl = float(cfg["rvol_long"])
            rs = float(cfg["rvol_short"])
            day_rows: list[tuple[Any, str, str]] = []

            if non_panic:
                long_sub = (
                    grp[
                        (~grp["has_sec_dilution_90d"])
                        & (grp["rvol_open20"] >= rl)
                        & (grp["ret_open20"] >= 0.008)
                        & (grp["ret_open20"] <= 0.080)
                        & (grp["close_0949"] > grp["vwap_0949"])
                        & (grp["close_pos20"] >= 0.68)
                        & (
                            grp["has_fresh_8k_36h"]
                            | (grp["t1_short_vol_ratio"] >= 0.50)
                            | (grp["t8_days_to_cover"] >= 3.0)
                        )
                    ]
                    .sort_values("long_mf_score", ascending=False)
                    .head(k_side)
                )
                for r in long_sub.itertuples(index=False):
                    cat = (
                        "8K_CATALYST"
                        if bool(r.has_fresh_8k_36h)
                        else ("FINRA_SQUEEZE_FUEL" if float(r.t8_days_to_cover) >= 3.0 else "FINRA_SHVOL_ACCUM")
                    )
                    day_rows.append((r, "LONG", cat))

            short_sub = (
                grp[
                    (grp["rvol_open20"] >= rs)
                    & (grp["ret_open20"] <= -0.008)
                    & (grp["ret_open20"] >= -0.080)
                    & (grp["close_0949"] < grp["vwap_0949"])
                    & (grp["close_pos20"] <= 0.32)
                    & (grp["has_sec_dilution_90d"] | (risk_off & (grp["t1_short_vol_ratio"] >= 0.45)))
                ]
                .sort_values("short_mf_score", ascending=False)
                .head(k_side)
            )
            for r in short_sub.itertuples(index=False):
                cat = "SEC_DILUTION_OVERHANG" if bool(r.has_sec_dilution_90d) else "VIX_RISKOFF_FINRA_DIST"
                day_rows.append((r, "SHORT", cat))

            for r, side, cat in day_rows:
                sym = str(r.symbol)
                needed_pairs.add((dt, sym))
                cfg_candidates[cfg["name"]].append(
                    {
                        "trade_date": dt,
                        "symbol": sym,
                        "side": side,
                        "catalyst": cat,
                        "high_0949": float(r.high_0949),
                        "low_0949": float(r.low_0949),
                        "vwap_0949": float(r.vwap_0949),
                        "adv20_usd": float(r.adv20_usd),
                        "vol20d_daily": float(r.vol20d_daily or 0.02),
                        "causal_med_1m_range_bps": float(r.causal_med_1m_range_bps),
                    }
                )

        # Unfiltered price/volume-only ablation baseline (no DF5/DF7/DF8/DF9 multi-feed screen)
        pv_long = (
            grp[
                (grp["rvol_open20"] >= 2.2)
                & (grp["ret_open20"] >= 0.008)
                & (grp["ret_open20"] <= 0.080)
                & (grp["close_0949"] > grp["vwap_0949"])
                & (grp["close_pos20"] >= 0.68)
            ]
            .sort_values("rvol_open20", ascending=False)
            .head(2)
        )
        pv_short = (
            grp[
                (grp["rvol_open20"] >= 1.8)
                & (grp["ret_open20"] <= -0.008)
                & (grp["ret_open20"] >= -0.080)
                & (grp["close_0949"] < grp["vwap_0949"])
                & (grp["close_pos20"] <= 0.32)
            ]
            .sort_values("rvol_open20", ascending=False)
            .head(2)
        )
        for sub_df, side in ((pv_long, "LONG"), (pv_short, "SHORT")):
            for r in sub_df.itertuples(index=False):
                sym = str(r.symbol)
                needed_pairs.add((dt, sym))
                unfiltered_pv_candidates.append(
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

    # 4. Load 09:50..15:50 ET 1m bars ONLY for candidate pairs
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

    # 5. Simulate all 5 configs + multi-feed catalyst attribution + unfiltered PV ablation
    cfg_daily_2x_series: dict[str, dict[str, float]] = {}
    canonical_events: list[dict[str, Any]] = []
    canonical_total_candidates = 0
    canonical_filled_count = 0
    canonical_cancel_reasons: dict[str, int] = defaultdict(int)
    canonical_cancelled_cf_2x: list[float] = []

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
                if not sim["filled"]:
                    canonical_cancel_reasons[str(sim["cancel_reason"])] += 1
                    base_cf = _simulate_blind_uniform_baseline(bo, bh, bl, bc, side=side)
                    canonical_cancelled_cf_2x.append(float(base_cf["gross_bps"]) - cost_2x)
            if sim["filled"]:
                net_1x = float(sim["gross_bps"]) - cost_1x
                net_2x = float(sim["gross_bps"]) - cost_2x
                daily_buckets_2x[dt].append(net_2x)
                if cname == PRIMARY_CANONICAL_CFG:
                    canonical_filled_count += 1
                    canonical_events.append(
                        {
                            "trade_date": dt,
                            "symbol": sym,
                            "side": side,
                            "catalyst": cand["catalyst"],
                            "gross_bps": float(sim["gross_bps"]),
                            "net_1x_bps": net_1x,
                            "net_2x_bps": net_2x,
                            "cost_1x_bps": cost_1x,
                            "mae_pct": float(sim["mae_pct"]),
                            "mfe_pct": float(sim["mfe_pct"]),
                            "green_to_red": bool(sim["green_to_red"]),
                            "vol20d_daily": cand["vol20d_daily"],
                        }
                    )
        cfg_daily_2x_series[cname] = {dt: float(np.mean(vals)) for dt, vals in daily_buckets_2x.items() if vals}

    mat_rows = [[cfg_daily_2x_series[c["name"]].get(dt, 0.0) for c in PRIMARY_CONFIG_GRID] for dt in all_dates_sorted]
    cpcv_primary = _compute_fold_fitted_cpcv_and_pbo(np.asarray(mat_rows, dtype=float))

    ev_prim_df = pd.DataFrame(canonical_events)
    daily_prim_df = (
        ev_prim_df.groupby("trade_date", as_index=False)[["net_1x_bps", "net_2x_bps"]].mean()
        if not ev_prim_df.empty
        else pd.DataFrame()
    )
    primary_metrics = _evaluate_arm_metrics(
        ev_prim_df,
        daily_prim_df,
        label="H114_MULTIFEED_DF1_DF9_CONCENTRATED_LONGSHORT_MIN_TRAIL",
        n_trials=TRIAL_NUMBER,
        cpcv_override=cpcv_primary,
    )
    primary_metrics["prefill_confirmation_stats"] = {
        "total_signal_candidates": canonical_total_candidates,
        "confirmed_filled_count": canonical_filled_count,
        "cancelled_unfilled_0bps_count": canonical_total_candidates - canonical_filled_count,
        "fill_rate_pct": round(canonical_filled_count / max(1, canonical_total_candidates) * 100.0, 1),
        "cancel_reasons": dict(canonical_cancel_reasons),
        "cancelled_counterfactual_mean_2x_bps": round(float(np.mean(canonical_cancelled_cf_2x)), 2)
        if canonical_cancelled_cf_2x
        else 0.0,
    }
    # Catalyst-level breakdown across DF5/DF7/DF8/DF9 buckets
    cat_breakdown: dict[str, dict[str, Any]] = {}
    if not ev_prim_df.empty:
        for cat_k, g_c in ev_prim_df.groupby("catalyst"):
            cat_breakdown[str(cat_k)] = {
                "events": int(len(g_c)),
                "dates": int(g_c["trade_date"].nunique()),
                "mean_1x_bps": round(float(g_c["net_1x_bps"].mean()), 2),
                "mean_2x_bps": round(float(g_c["net_2x_bps"].mean()), 2),
                "win_rate_2x_pct": round(float((g_c["net_2x_bps"] > 0.0).mean() * 100.0), 1),
            }
    primary_metrics["catalyst_breakdown"] = cat_breakdown

    # Unfiltered PV-only blind baseline ablation
    pv_events: list[dict[str, Any]] = []
    for cand in unfiltered_pv_candidates:
        bars = bars_by_pair.get((cand["trade_date"], cand["symbol"]))
        if bars is None:
            continue
        bo, bh, bl, bc = bars
        cost_1x = _almgren_chriss_1x_cost_bps(
            cand["adv20_usd"], cand["vol20d_daily"], is_short=(cand["side"] == "SHORT"), hold_days=1
        )
        base_sim = _simulate_blind_uniform_baseline(bo, bh, bl, bc, side=cand["side"])
        pv_events.append(
            {
                "trade_date": cand["trade_date"],
                "symbol": cand["symbol"],
                "side": cand["side"],
                "net_1x_bps": float(base_sim["gross_bps"]) - cost_1x,
                "net_2x_bps": float(base_sim["gross_bps"]) - 2.0 * cost_1x,
                "cost_1x_bps": cost_1x,
                "mae_pct": float(base_sim["mae_pct"]),
                "mfe_pct": float(base_sim["mfe_pct"]),
                "green_to_red": bool(base_sim["green_to_red"]),
                "vol20d_daily": cand["vol20d_daily"],
            }
        )
    ev_pv_df = pd.DataFrame(pv_events)
    daily_pv_df = (
        ev_pv_df.groupby("trade_date", as_index=False)[["net_1x_bps", "net_2x_bps"]].mean()
        if not ev_pv_df.empty
        else pd.DataFrame()
    )
    pv_ablation_metrics = _evaluate_arm_metrics(
        ev_pv_df,
        daily_pv_df,
        label="H114_ABLATION_PRICE_VOL_ONLY_BLIND_UNIFORM",
        n_trials=TRIAL_NUMBER,
    )

    ablation_delta = {
        "delta_mean_1x_event_bps": round(
            float(primary_metrics.get("mean_1x_event_bps") or 0.0)
            - float(pv_ablation_metrics.get("mean_1x_event_bps") or 0.0),
            2,
        ),
        "delta_mean_2x_daily_bps": round(
            float(primary_metrics.get("mean_2x_daily_bps") or 0.0)
            - float(pv_ablation_metrics.get("mean_2x_daily_bps") or 0.0),
            2,
        ),
        "delta_max_dd_2x_pct_reduction": round(
            float(pv_ablation_metrics.get("max_dd_2x_pct") or 0.0) - float(primary_metrics.get("max_dd_2x_pct") or 0.0),
            2,
        ),
        "delta_green_to_red_pct_reduction": round(
            float(pv_ablation_metrics.get("green_to_red_pct") or 0.0)
            - float(primary_metrics.get("green_to_red_pct") or 0.0),
            1,
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
        "feeds_integrated": [
            "DF1_SIP_1M",
            "DF3_SPLITS",
            "DF4_RVOL_OPEN20",
            "DF5_SEC_8K_AND_424B5_S3_PIT",
            "DF7_FINRA_CNMSSHVOL_T1",
            "DF8_FINRA_SI_T8_KNOWN_FROM",
            "DF9_SPY_UVXY_REGIME",
        ],
        "primary_arm": primary_metrics,
        "pv_only_ablation_arm": pv_ablation_metrics,
        "ablation_delta_multifeed_screen_vs_pv_only": ablation_delta,
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
    parser = argparse.ArgumentParser(
        description="Run Trial #109 (H114) Multi-Feed DF1..DF9 study on 700d pre-wall lake."
    )
    parser.add_argument(
        "--market-data-root", type=pathlib.Path, default=pathlib.Path.home() / "Documents" / "market-data"
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "docs" / "TRIAL109_H114_PREWALL_RECEIPT.json",
    )
    args = parser.parse_args()
    receipt = run_h114_trial109_study(args.market_data_root)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
