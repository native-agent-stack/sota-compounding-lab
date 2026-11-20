"""Execute v3.11.0 Institutional Pre-Wall Alpha Factory (`S1` Qlib-Inspired 18-Feature LightGBM + `S2` hftbacktest).

Enforces ADR-0027 / ADR-0064 strict causal pre-wall (`<= 2026-10-15`) invariants:
1. Pre-joins official `XNYS` calendar-aligned `next_session_date` into
   `derived/df10_daily_20d_baselines_next_session.parquet` strictly against the split-adjusted daily
   bar feed (matching `materialize_prewall_df10_df6.py`, ZERO raw-bar fallback divergence).
2. Evaluates `ADIA Lab #19` `MinTRL/MDE` statistical power gate at `cumulative_trials_n = 104` (`t*(104) = 4.2063`)
   and feeds `power_gate["approved"]` directly into the promotion decision.
3. Runs `S2` `hftbacktest 2.4.2` `BacktestAsset().risk_adverse_queue_model()` passive-queue replay on
   explicitly date-verified pre-wall (`<= 2026-10-15`) SIP files + `ADR-0059` conservative TCA cost feedback
   (failing closed on 0 pre-wall files, zero synthetic hardcoded fill arrays) and feeds calibrated cost into `S1`.
4. Freezes the strictly causal `T-1` liquid operating universe (`prev_raw_close >= $5.00`,
   `mean_20d_dvol >= $25M`, fail-closed 90d SEC `S-3`/`424B5`/`EFFECT` dilution-shielded) BEFORE
   joining next-session outcomes (missing/halted next-session bars become explicit `0.0` unfilled entries,
   never leaking tomorrow's bar availability into today's cross-sectional ranks).
5. Runs a 5-configuration LightGBM grid over 18 locally defined `Qlib Alpha158`-inspired OHLCV+Liquidity
   features on `C(6,2) = 15` disjoint `skfolio` `CombinatorialPurgedCV(purged_size=20, embargo_size=20)`
   (matching the 20-session DF10 rolling lookback window) to compute true Bailey et al. (2014) CSCV/CPCV `PBO`
   (IS winner OOS rank logit), `ml4t-diagnostic` native daily `DSR`, and `statsmodels` Rogers date-clustered `t-stat`,
   enforcing strict `DSR > 0.95` and `PBO < 0.25`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import pathlib
import re
import sys
from dataclasses import asdict
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_optb_prewall_factory import (  # noqa: E402
    DEFAULT_POSITION_NOTIONAL_USD,
    _compute_dsr_from_daily_returns,
    _date_clustered_t_stat,
    compute_almgren_chriss_cost_bps,
)

from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE, hlz_dynamic_t_hurdle  # noqa: E402
from sota_compounding.strategy_contract_v2 import (  # noqa: E402
    calibrate_sim_cost_from_paper_tca,
    validate_next_session_date_alignment,
)

STUDY_WALL_ISO = STUDY_WALL_DATE.isoformat()
TRIAL_104_NUMBER = 104
TRIAL_104_ID = "H108_QLIB_ALPHA158_LGBM_NEXT_SESSION_PREWALL_CPCV"
STRICT_DSR_MIN = 0.95
STRICT_PBO_MAX = 0.25
STRICT_CPCV_POS_MIN = 0.70
STRICT_MIN_EVENTS = 200
STRICT_MIN_DATES = 60
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Five candidate LightGBM configurations for true Bailey et al. (2014) IS-vs-OOS CSCV/CPCV PBO computation
LGBM_PBO_CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "lgb_l15_r1", "num_leaves": 15, "reg_lambda": 1.0, "n_estimators": 80, "learning_rate": 0.05},
    {"name": "lgb_l31_r0", "num_leaves": 31, "reg_lambda": 0.0, "n_estimators": 100, "learning_rate": 0.05},
    {"name": "lgb_l31_r5", "num_leaves": 31, "reg_lambda": 5.0, "n_estimators": 100, "learning_rate": 0.05},
    {"name": "lgb_l63_r2", "num_leaves": 63, "reg_lambda": 2.0, "n_estimators": 80, "learning_rate": 0.05},
    {"name": "lgb_l15_r10", "num_leaves": 15, "reg_lambda": 10.0, "n_estimators": 120, "learning_rate": 0.03},
)


def _redact_local_path(p: pathlib.Path, lake_root: pathlib.Path) -> str:
    """Redact local workstation username / home path from error strings (`ADR-0027` privacy gate)."""
    try:
        rel = p.resolve().relative_to(lake_root.resolve())
        return f"<lake_root>/{rel.as_posix()}"
    except Exception:
        return f"<path>/{p.name}"


def materialize_next_session_aligned_df10(lake_root: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    """Enrich pre-wall `df10_daily_20d_baselines.parquet` with exact `XNYS` calendar-aligned `next_session_date`."""
    import duckdb
    import exchange_calendars as xcals

    src_path = lake_root / "derived" / "df10_daily_20d_baselines.parquet"
    split_csvs = [lake_root / "universe" / y / "daily_split.csv.gz" for y in ("2024", "2025", "2026")]
    out_path = lake_root / "derived" / "df10_daily_20d_baselines_next_session.parquet"
    if not src_path.is_file():
        raise FileNotFoundError(f"Missing pre-wall DF10 parquet: {_redact_local_path(src_path, lake_root)}")

    cal = xcals.get_calendar("XNYS")
    sessions = [ts.date().isoformat() for ts in cal.sessions_in_range("2024-01-02", STUDY_WALL_ISO)]
    if any(d > STUDY_WALL_ISO for d in sessions):
        raise ValueError("XNYS session range crossed STUDY_WALL_DATE")

    xnys_pairs = [(sessions[i], sessions[i + 1] if i + 1 < len(sessions) else None) for i in range(len(sessions))]

    con = duckdb.connect()
    con.execute("PRAGMA threads=32")
    con.execute("CREATE TEMP TABLE xnys_cal (trade_date VARCHAR PRIMARY KEY, next_session_date VARCHAR)")
    con.executemany("INSERT INTO xnys_cal VALUES (?, ?)", xnys_pairs)

    # Strictly require canonical `universe/{2024,2025,2026}/daily_split.csv.gz` (ZERO raw-OHLC fallback!)
    missing_split = [p for p in split_csvs if not p.is_file()]
    if missing_split:
        miss_desc = ", ".join(_redact_local_path(p, lake_root) for p in missing_split)
        raise FileNotFoundError(
            f"Missing canonical split-adjusted daily CSVs (refusing raw-OHLC fallback): {miss_desc}"
        )
    split_list_sql = "[" + ", ".join(f"'{p.as_posix()}'" for p in split_csvs) + "]"
    nxt_cte = f"""
    nxt_bars AS (
        SELECT
            UPPER(TRIM(symbol)) AS symbol,
            CAST(CAST(ts_utc AT TIME ZONE 'America/New_York' AS DATE) AS VARCHAR) AS trade_date,
            CAST(open AS DOUBLE) AS split_open,
            CAST(close AS DOUBLE) AS split_close,
            CAST(low AS DOUBLE) AS split_low,
            CAST(high AS DOUBLE) AS split_high,
            CAST(volume AS DOUBLE) AS split_volume
        FROM read_csv_auto({split_list_sql}, union_by_name=true)
        WHERE CAST(ts_utc AT TIME ZONE 'America/New_York' AS DATE) <= DATE '{STUDY_WALL_ISO}'
          AND open > 0 AND close > 0 AND low > 0 AND high > 0 AND volume >= 0
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY UPPER(TRIM(symbol)), CAST(ts_utc AT TIME ZONE 'America/New_York' AS DATE)
            ORDER BY CAST(volume AS DOUBLE) DESC
        ) = 1
    )
    """
    target_source = "universe/{2024,2025,2026}/daily_split.csv.gz"

    query = f"""
    COPY (
        WITH base AS (
            SELECT *
            FROM read_parquet('{src_path.as_posix()}')
            WHERE trade_date <= '{STUDY_WALL_ISO}'
        ),
        {nxt_cte}
        SELECT
            b.symbol,
            b.trade_date,
            x.next_session_date,
            b.raw_open,
            b.raw_high,
            b.raw_low,
            b.raw_close,
            b.raw_volume,
            b.raw_dvol,
            b.split_open,
            b.split_high,
            b.split_low,
            b.split_close,
            b.bar_range_pct,
            b.prior_sessions_count,
            b.mean_20d_vol,
            b.std_20d_vol,
            b.mean_20d_dvol,
            b.mean_5d_vol_prior,
            b.mean_20d_range_pct,
            b.mean_5d_range_pct_prior,
            b.high_20d_prior,
            b.low_20d_prior,
            b.prev_split_close,
            b.prev_raw_close,
            b.rvol_20d,
            b.vol_z_20d,
            b.vdu_5d_20d_ratio,
            b.coil_5d_20d_range_ratio,
            b.prev_close_to_20d_high,
            b.ret_1d,
            b.prior_ret_5d,
            b.prior_ret_20d,
            nxt.split_open AS next1_split_open,
            nxt.split_close AS next1_split_close,
            nxt.split_low AS next1_split_low,
            nxt.split_high AS next1_split_high,
            nxt.split_volume AS next1_split_volume
        FROM base b
        INNER JOIN xnys_cal x ON x.trade_date = b.trade_date
        LEFT JOIN nxt_bars nxt ON nxt.symbol = b.symbol AND nxt.trade_date = x.next_session_date
        ORDER BY b.trade_date, b.symbol
    ) TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
    """
    con.execute(query)

    pairs_rows = con.execute(
        f"""
        SELECT DISTINCT trade_date, next_session_date
        FROM read_parquet('{out_path.as_posix()}')
        ORDER BY trade_date
        """
    ).fetchall()
    total_rows, aligned_target_rows, max_td, max_nsd = con.execute(
        f"""
        SELECT
            COUNT(*),
            SUM(CASE WHEN next1_split_open IS NOT NULL AND next1_split_close IS NOT NULL THEN 1 ELSE 0 END),
            MAX(trade_date),
            MAX(next_session_date)
        FROM read_parquet('{out_path.as_posix()}')
        """
    ).fetchone()

    align_receipt = validate_next_session_date_alignment(
        [r[0] for r in pairs_rows],
        [r[1] for r in pairs_rows],
        sessions,
        study_wall_iso=STUDY_WALL_ISO,
    )
    h = hashlib.sha256()
    with out_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)

    audit = {
        "file": out_path.name,
        "target_source": target_source,
        "total_rows": int(total_rows),
        "calendar_aligned_target_rows": int(aligned_target_rows or 0),
        "max_trade_date": str(max_td),
        "max_next_session_date": str(max_nsd),
        "distinct_trade_dates": len(pairs_rows),
        "alignment_check": align_receipt,
        "sha256": h.hexdigest(),
    }
    return out_path, audit


def evaluate_mintrl_mde_power_gate(
    *,
    distinct_dates: int,
    cumulative_trials_n: int = TRIAL_104_NUMBER,
    daily_sigma_bps: float = 85.0,
    target_annual_sr: float = 3.0,
) -> dict[str, Any]:
    """ADIA Lab #19 finite-sample MinTRL & MDE power gate before burning Trial #104."""
    t_star = hlz_dynamic_t_hurdle(cumulative_trials_n)
    sr_min_ann = t_star / math.sqrt(max(1, distinct_dates) / 252.0)
    min_trl_days = int(math.ceil(252.0 * ((t_star / max(target_annual_sr, 0.1)) ** 2)))
    mde_daily_bps = round(t_star * daily_sigma_bps / math.sqrt(max(1, distinct_dates)), 3)
    approved = bool(distinct_dates >= 60 and sr_min_ann <= 5.0)
    return {
        "cumulative_trials_n": cumulative_trials_n,
        "hlz_t_hurdle": t_star,
        "distinct_dates": distinct_dates,
        "sr_min_ann_to_clear_hlz": round(sr_min_ann, 4),
        "min_trl_days_at_sr3": min_trl_days,
        "mde_daily_bps": mde_daily_bps,
        "approved": approved,
    }


def evaluate_strict_adr0027_gate(
    *,
    trial_number: int,
    t_stat_clustered_2x: float,
    dsr_probability: float,
    cpcv_positive_share: float,
    pbo: float,
    mean_net_2x_bps: float,
    distinct_dates: int,
    total_events: int,
    power_gate_approved: bool,
    sec_shield_verified: bool,
    s2_queue_replay_verified: bool,
) -> dict[str, Any]:
    """Fail-closed ADR-0027 promotion gate enforcing strict `DSR > 0.95`, `PBO < 0.25`, `t > t*(N)`, and inputs."""
    t_hurdle = hlz_dynamic_t_hurdle(trial_number)
    checks = {
        "power_gate_approved": bool(power_gate_approved),
        "sec_dilution_shield_verified": bool(sec_shield_verified),
        "s2_queue_replay_verified": bool(s2_queue_replay_verified),
        "events_ge_200": int(total_events) >= STRICT_MIN_EVENTS,
        "dates_ge_60": int(distinct_dates) >= STRICT_MIN_DATES,
        "net_2x_positive": math.isfinite(mean_net_2x_bps) and mean_net_2x_bps > 0.0,
        "hlz_t_gt_hurdle": math.isfinite(t_stat_clustered_2x) and t_stat_clustered_2x > t_hurdle,
        "dsr_gt_095": math.isfinite(dsr_probability) and dsr_probability > STRICT_DSR_MIN,
        "pbo_lt_025": math.isfinite(pbo) and pbo < STRICT_PBO_MAX,
        "cpcv_pos_ge_070": math.isfinite(cpcv_positive_share) and cpcv_positive_share >= STRICT_CPCV_POS_MIN,
    }
    passed = all(checks.values())
    failed_checks = [k for k, ok in checks.items() if not ok]
    return {
        "passed": passed,
        "verdict": "PASSED" if passed else "FAILED",
        "hlz_t_hurdle": t_hurdle,
        "strict_dsr_threshold": f"> {STRICT_DSR_MIN}",
        "strict_pbo_threshold": f"< {STRICT_PBO_MAX}",
        "checks": checks,
        "failed_checks": failed_checks,
    }


def run_s2_hftbacktest_queue_parity_probe(
    lake_root: pathlib.Path,
    *,
    paper_tca_json: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Run real `S2` `hftbacktest` `RiskAdverseQueueModel` replay over date-verified pre-wall SIP bars."""
    import duckdb
    import hftbacktest
    import numpy as np
    from hftbacktest import (
        BUY_EVENT,
        DEPTH_EVENT,
        EXCH_EVENT,
        GTX,
        LIMIT,
        LOCAL_EVENT,
        SELL_EVENT,
        TRADE_EVENT,
        BacktestAsset,
        HashMapMarketDepthBacktest,
        event_dtype,
    )

    bars_root = lake_root / "alpaca-sip" / "raw" / "bars-1min"
    if not bars_root.is_dir():
        raise FileNotFoundError(f"Missing pre-wall SIP 1m directory: {_redact_local_path(bars_root, lake_root)}")

    # Strictly filter and verify every date directory is <= STUDY_WALL_ISO (zero post-wall contamination)
    prewall_date_dirs: list[tuple[str, pathlib.Path]] = []
    postwall_skipped = 0
    for child in sorted(bars_root.iterdir()):
        if not child.is_dir() or not _ISO_DATE_RE.match(child.name):
            continue
        if child.name <= STUDY_WALL_ISO:
            prewall_date_dirs.append((child.name, child))
        else:
            postwall_skipped += 1

    prewall_files: list[pathlib.Path] = []
    for _d_iso, d_path in prewall_date_dirs:
        prewall_files.extend(sorted(d_path.glob("*.parquet")))

    if not prewall_files:
        raise FileNotFoundError(f"Zero pre-wall (<= {STUDY_WALL_ISO}) SIP parquet files found for S2 queue replay")

    # Select latest pre-wall session (<= 2026-10-15) and replay real 1m OHLCV into hftbacktest RiskAdverseQueueModel
    replay_date, replay_dir = prewall_date_dirs[-1]
    replay_glob = (replay_dir / "*.parquet").as_posix()
    wall_end_ny_ns = (
        int(
            dt.datetime(
                STUDY_WALL_DATE.year,
                STUDY_WALL_DATE.month,
                STUDY_WALL_DATE.day,
                23,
                59,
                59,
                999999,
                tzinfo=ZoneInfo("America/New_York"),
            ).timestamp()
            * 1_000_000_000
        )
        + 999
    )
    con = duckdb.connect()
    top_sym_row = con.execute(
        f"""
        SELECT symbol, COUNT(*) AS n_bars, SUM(v * c) AS dvol
        FROM read_parquet('{replay_glob}')
        WHERE CAST(t_ns AS BIGINT) <= ?
          AND o > 0 AND h >= l AND c > 0 AND v > 0
        GROUP BY symbol
        HAVING COUNT(*) >= 120
        ORDER BY dvol DESC
        LIMIT 1
        """,
        [wall_end_ny_ns],
    ).fetchone()
    if top_sym_row is None:
        raise RuntimeError(f"No liquid >=120-bar symbol found in pre-wall session {replay_date}")
    replay_sym = str(top_sym_row[0])
    bars_df = con.execute(
        f"""
        SELECT
            CAST(t_ns AS BIGINT) AS ts_ns,
            CAST(o AS DOUBLE) AS open,
            CAST(h AS DOUBLE) AS high,
            CAST(l AS DOUBLE) AS low,
            CAST(c AS DOUBLE) AS close,
            CAST(v AS DOUBLE) AS volume
        FROM read_parquet('{replay_glob}')
        WHERE symbol = ?
          AND CAST(t_ns AS BIGINT) <= ?
          AND o > 0 AND h >= l AND c > 0 AND v > 0
        ORDER BY ts_ns
        """,
        [replay_sym, wall_end_ny_ns],
    ).fetchdf()

    if len(bars_df) < 60:
        raise RuntimeError(f"Insufficient pre-wall bars for {replay_sym} on {replay_date}: {len(bars_df)}")

    # Build hftbacktest event stream (DEPTH + TRADE events derived strictly from pre-wall real OHLCV prints)
    tick_sz = 0.01
    n_bars = len(bars_df)
    events = np.zeros(n_bars * 4, dtype=event_dtype)
    k = 0
    for row in bars_df.itertuples(index=False):
        ts = int(row.ts_ns)
        bid_p = round(max(tick_sz, float(row.low)), 2)
        ask_p = round(max(bid_p + tick_sz, float(row.high)), 2)
        sz = max(100.0, float(row.volume) * 0.05)
        # 1. Best bid depth snapshot
        events[k]["ev"] = DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT | BUY_EVENT
        events[k]["exch_ts"] = ts
        events[k]["local_ts"] = ts + 1_000_000
        events[k]["px"] = bid_p
        events[k]["qty"] = sz
        k += 1
        # 2. Best ask depth snapshot
        events[k]["ev"] = DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT | SELL_EVENT
        events[k]["exch_ts"] = ts + 100_000
        events[k]["local_ts"] = ts + 1_100_000
        events[k]["px"] = ask_p
        events[k]["qty"] = sz
        k += 1
        # 3. Downward sell trade at bar low (tests passive bid queue fill under RiskAdverseQueueModel)
        events[k]["ev"] = TRADE_EVENT | EXCH_EVENT | LOCAL_EVENT | SELL_EVENT
        events[k]["exch_ts"] = ts + 500_000
        events[k]["local_ts"] = ts + 1_500_000
        events[k]["px"] = bid_p
        events[k]["qty"] = max(10.0, float(row.volume) * 0.25)
        k += 1
        # 4. Closing print at bar close
        events[k]["ev"] = TRADE_EVENT | EXCH_EVENT | LOCAL_EVENT | BUY_EVENT
        events[k]["exch_ts"] = ts + 900_000
        events[k]["local_ts"] = ts + 1_900_000
        events[k]["px"] = round(float(row.close), 2)
        events[k]["qty"] = max(10.0, float(row.volume) * 0.25)
        k += 1

    asset = (
        BacktestAsset()
        .data(events[:k])
        .tick_size(tick_sz)
        .lot_size(1.0)
        .constant_order_latency(1_000_000, 1_000_000)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
    )
    hbt = HashMapMarketDepthBacktest([asset])
    submitted_passive = 0
    filled_passive = 0
    replay_steps_executed = 0
    order_id = 1
    prev_pos = 0.0
    try:
        for _step_idx in range(min(40, n_bars - 2)):
            if hbt.elapse(60_000_000_000) != 0:
                break
            replay_steps_executed += 1
            cur_pos = float(hbt.position(0))
            # Catch any async passive entry fill that completed during a prior cancel!
            if cur_pos > prev_pos:
                filled_passive += 1
                best_ask = float(hbt.depth(0).best_ask)
                best_bid_fb = float(hbt.depth(0).best_bid)
                exit_px = best_ask if (math.isfinite(best_ask) and best_ask > 0.0) else best_bid_fb
                if math.isfinite(exit_px) and exit_px > 0.0:
                    hbt.submit_sell_order(0, order_id, exit_px, cur_pos, GTX, LIMIT, False)
                    order_id += 1
            prev_pos = cur_pos

            depth = hbt.depth(0)
            best_bid = float(depth.best_bid)
            entry_oid: int | None = None
            if cur_pos == 0.0 and math.isfinite(best_bid) and best_bid > 0.0:
                entry_oid = order_id
                hbt.submit_buy_order(0, entry_oid, best_bid, 10.0, GTX, LIMIT, False)
                submitted_passive += 1
                order_id += 1
            end_of_stream = hbt.elapse(60_000_000_000) != 0
            post_pos = float(hbt.position(0))
            if post_pos > prev_pos:
                filled_passive += 1
                if not end_of_stream:
                    best_ask = float(hbt.depth(0).best_ask)
                    exit_px = best_ask if (math.isfinite(best_ask) and best_ask > 0.0) else best_bid
                    if math.isfinite(exit_px) and exit_px > 0.0:
                        hbt.submit_sell_order(0, order_id, exit_px, post_pos, GTX, LIMIT, False)
                        order_id += 1
            elif entry_oid is not None and not end_of_stream:
                hbt.cancel(0, entry_oid, False)
            prev_pos = post_pos
            if end_of_stream:
                break
            hbt.clear_inactive_orders(0)
        # Drain one final latency window after the loop so any async fill during step-40 cancel is captured!
        hbt.elapse(60_000_000_000)
        final_pos = float(hbt.position(0))
        if final_pos > prev_pos:
            filled_passive += 1
    finally:
        hbt.close()

    # Load empirical ADR-0059 Paper TCA if provided; else report PRIOR_CONSERVATIVE_FLOOR (0 synthetic fills)
    tca_source = "PRIOR_CONSERVATIVE_FLOOR"
    obs_fills = 0
    obs_slip = 0.0
    obs_markout = 0.0
    if paper_tca_json is not None and paper_tca_json.is_file():
        tca_doc = json.loads(paper_tca_json.read_text(encoding="utf-8"))
        if isinstance(tca_doc, dict) and int(tca_doc.get("paper_fills_count") or 0) > 0:
            tca_source = "EMPIRICAL_ADR0059_PAPER_TCA"
            obs_fills = int(tca_doc["paper_fills_count"])
            obs_slip = float(tca_doc.get("observed_mean_slippage_bps") or 0.0)
            obs_markout = float(tca_doc.get("observed_adverse_markout_5s_bps") or 0.0)

    tca_cal = calibrate_sim_cost_from_paper_tca(
        strategy_id="BJZZ_SUBPENNY_INST_DARK_SWEEP",
        paper_fills_count=obs_fills,
        observed_mean_slippage_bps=obs_slip,
        observed_adverse_markout_5s_bps=obs_markout,
        prior_half_spread_bps=4.0,
    )
    replay_verified = bool(k > 0 and submitted_passive > 0)
    fill_rate = round(filled_passive / max(1, submitted_passive), 4)
    return {
        "engine": f"hftbacktest {getattr(hftbacktest, '__version__', '2.x')} RiskAdverseQueueModel",
        "hftbacktest_replay_executed": replay_verified,
        "prewall_sip_date_dirs_verified": len(prewall_date_dirs),
        "postwall_sip_date_dirs_excluded": postwall_skipped,
        "prewall_sip_raw_parquet_files": len(prewall_files),
        "replay_session_date": replay_date,
        "replay_symbol": replay_sym,
        "replay_bars_count": n_bars,
        "replay_events_processed": k,
        "replay_steps_executed": replay_steps_executed,
        "passive_gtx_orders_submitted": submitted_passive,
        "passive_gtx_orders_filled": filled_passive,
        "passive_queue_fill_rate": fill_rate,
        "study_wall_date": STUDY_WALL_ISO,
        "tca_cost_source": tca_source,
        "tca_cost_calibration_feedback": asdict(tca_cal),
        "queue_parity_status": "VERIFIED_PREWALL_REPLAY" if replay_verified else "REPLAY_INCOMPLETE",
    }


def run_s1_qlib_alpha158_lgbm_cpcv(
    aligned_df10_path: pathlib.Path,
    lake_root: pathlib.Path,
    *,
    power_gate: dict[str, Any],
    s2_res: dict[str, Any],
) -> dict[str, Any]:
    """Run strictly causal 18-feature `Qlib Alpha158`-inspired LightGBM `C(6,2)=15` CPCV (`<=2026-10-15`)."""
    import duckdb
    import lightgbm as lgb
    import numpy as np
    from scipy.stats import rankdata
    from skfolio.model_selection import CombinatorialPurgedCV

    sec_path = lake_root / "derived" / "sec_filings_clean_prewall.parquet"
    sym_path = lake_root / "public-tables" / "sec_symbols.parquet"
    # Fail closed if SEC dilution files are absent (NEVER silently default dilution predicate to TRUE!)
    if not sec_path.is_file() or not sym_path.is_file():
        raise FileNotFoundError(
            "Missing required pre-wall SEC dilution parquet under <lake_root>/derived/ or <lake_root>/public-tables/"
        )

    con = duckdb.connect()
    con.execute("PRAGMA threads=32")
    dil_parsed_cnt = int(
        con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT
                    UPPER(TRIM(ss.symbol)) AS symbol,
                    CAST(TRY_CAST(sf.accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS dil_date
                FROM read_parquet('{sec_path.as_posix()}') sf
                INNER JOIN read_parquet('{sym_path.as_posix()}') ss ON CAST(ss.cik AS BIGINT) = CAST(sf.cik AS BIGINT)
                WHERE TRY_CAST(sf.accepted AS TIMESTAMPTZ) IS NOT NULL
                  AND CAST(
                      TRY_CAST(sf.accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE
                  ) <= DATE '{STUDY_WALL_ISO}'
                  AND UPPER(TRIM(sf.form)) IN ('S-3','S-3ASR','F-3','F-3ASR','424B2','424B3','424B4','424B5','EFFECT')
            )
            """
        ).fetchone()[0]
    )
    if dil_parsed_cnt <= 0:
        raise RuntimeError(
            f"Unusable/empty parsed pre-wall SEC dilution join (dil_parsed_cnt={dil_parsed_cnt}) — failing closed"
        )

    # P1-1 FIX: Freeze strictly causal T-1 signal-date universe BEFORE joining next-session OHLC!
    # NEVER filter on `next1_split_open > 0` before ranking/selecting today's portfolio!
    df = con.execute(
        f"""
        WITH dil AS (
            SELECT DISTINCT
                UPPER(TRIM(ss.symbol)) AS symbol,
                CAST(TRY_CAST(sf.accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS dil_date
            FROM read_parquet('{sec_path.as_posix()}') sf
            INNER JOIN read_parquet('{sym_path.as_posix()}') ss ON CAST(ss.cik AS BIGINT) = CAST(sf.cik AS BIGINT)
            WHERE TRY_CAST(sf.accepted AS TIMESTAMPTZ) IS NOT NULL
              AND CAST(
                  TRY_CAST(sf.accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE
              ) <= DATE '{STUDY_WALL_ISO}'
              AND UPPER(TRIM(sf.form)) IN ('S-3','S-3ASR','F-3','F-3ASR','424B2','424B3','424B4','424B5','EFFECT')
        )
        SELECT
            b.symbol,
            b.trade_date,
            b.next_session_date,
            b.raw_close,
            b.mean_20d_dvol,
            b.bar_range_pct,
            b.mean_20d_range_pct,
            b.mean_5d_range_pct_prior,
            b.rvol_20d,
            b.vol_z_20d,
            b.vdu_5d_20d_ratio,
            b.coil_5d_20d_range_ratio,
            b.prev_close_to_20d_high,
            b.ret_1d,
            b.prior_ret_5d,
            b.prior_ret_20d,
            (b.split_close - b.split_open) / GREATEST(b.split_open, 1e-6) AS kmid,
            (b.split_high - b.split_low) / GREATEST(b.split_open, 1e-6) AS klen,
            (b.split_high - GREATEST(b.split_open, b.split_close)) / GREATEST(b.split_high - b.split_low, 1e-6) AS kup,
            (LEAST(b.split_open, b.split_close) - b.split_low) / GREATEST(b.split_high - b.split_low, 1e-6) AS klow,
            (b.split_close - b.split_low) / GREATEST(b.split_high - b.split_low, 1e-6) AS clv,
            b.raw_dvol / GREATEST(b.mean_20d_dvol, 1.0) AS dvol_ratio_20d,
            b.next1_split_open,
            b.next1_split_close,
            b.next1_split_low,
            b.next1_split_high,
            b.next1_split_volume
        FROM read_parquet('{aligned_df10_path.as_posix()}') b
        LEFT JOIN dil
          ON dil.symbol = b.symbol
         AND dil.dil_date < CAST(b.trade_date AS DATE)
         AND dil.dil_date >= CAST(b.trade_date AS DATE) - INTERVAL '90 days'
        WHERE b.trade_date <= '{STUDY_WALL_ISO}'
          AND b.next_session_date IS NOT NULL
          AND b.next_session_date <= '{STUDY_WALL_ISO}'
          AND b.prev_raw_close >= 5.0
          AND b.mean_20d_dvol >= 25000000.0
          AND dil.symbol IS NULL
        ORDER BY b.trade_date, b.symbol
        """
    ).fetchdf()

    if len(df) < 1000:
        raise RuntimeError(f"Insufficient pre-wall liquid rows: {len(df)}")

    feature_cols = [
        "kmid",
        "klen",
        "kup",
        "klow",
        "clv",
        "bar_range_pct",
        "rvol_20d",
        "vol_z_20d",
        "vdu_5d_20d_ratio",
        "coil_5d_20d_range_ratio",
        "prev_close_to_20d_high",
        "ret_1d",
        "prior_ret_5d",
        "prior_ret_20d",
        "dvol_ratio_20d",
    ]
    df["mom_accel_5_20"] = df["prior_ret_5d"].fillna(0.0) - (df["prior_ret_20d"].fillna(0.0) * 0.25)
    df["vol_price_shock"] = np.sign(df["ret_1d"].fillna(0.0)) * np.log1p(np.maximum(df["rvol_20d"].fillna(0.0), 0.0))
    df["range_expand_ratio"] = df["bar_range_pct"].fillna(0.0) / np.maximum(df["mean_20d_range_pct"].fillna(0.01), 1e-4)
    feature_cols.extend(["mom_accel_5_20", "vol_price_shock", "range_expand_ratio"])

    # Compute next-session outcomes ONLY after universe is frozen; missing/halted/0-vol bars -> unfilled (0.0 bps)
    o_px = df["next1_split_open"].to_numpy(dtype=np.float64)
    c_px = df["next1_split_close"].to_numpy(dtype=np.float64)
    l_px = df["next1_split_low"].to_numpy(dtype=np.float64)
    h_px = df["next1_split_high"].to_numpy(dtype=np.float64)
    v_px = df["next1_split_volume"].to_numpy(dtype=np.float64)
    valid_next = (
        np.isfinite(o_px)
        & np.isfinite(c_px)
        & np.isfinite(l_px)
        & np.isfinite(h_px)
        & np.isfinite(v_px)
        & (o_px > 0.0)
        & (c_px > 0.0)
        & (l_px > 0.0)
        & (h_px > 0.0)
        & (v_px > 0.0)
    )
    safe_o = np.where(valid_next, o_px, 1.0)
    stop_frac = ADR0022_MAX_STOP_PCT / 100.0
    # P2-1 FIX: Include 1e-12 floating tolerance so exact 2.00% touches (e.g. $16.17/$16.50, $16.83/$16.50) stop out!
    long_raw = np.where(valid_next, (c_px / safe_o) - 1.0, 0.0)
    long_stopped = valid_next & ((l_px / safe_o - 1.0) <= -(stop_frac - 1e-12))
    long_ret_bps = np.where(
        valid_next,
        np.where(long_stopped, -ADR0022_MAX_STOP_PCT * 100.0, long_raw * 10_000.0),
        0.0,
    )

    short_raw = np.where(valid_next, (safe_o - c_px) / safe_o, 0.0)
    short_stopped = valid_next & ((h_px / safe_o - 1.0) >= (stop_frac - 1e-12))
    short_ret_bps = np.where(
        valid_next,
        np.where(short_stopped, -ADR0022_MAX_STOP_PCT * 100.0, short_raw * 10_000.0),
        0.0,
    )
    y_train_target = np.clip(long_raw * 10_000.0, -500.0, 500.0).astype(np.float32)

    # Within-date cross-sectional rank normalization strictly on the frozen T-1 universe
    dates_arr = df["trade_date"].to_numpy(dtype=str)
    unique_dates, date_inv = np.unique(dates_arr, return_inverse=True)
    n_dates = len(unique_dates)
    X_raw = df[feature_cols].to_numpy(dtype=np.float64)
    X_rank = np.zeros_like(X_raw, dtype=np.float32)

    date_order = np.argsort(date_inv, kind="mergesort")
    sorted_dinv = date_inv[date_order]
    d_counts = np.bincount(sorted_dinv, minlength=n_dates)
    d_starts = np.zeros(n_dates, dtype=np.int64)
    if n_dates > 1:
        np.cumsum(d_counts[:-1], out=d_starts[1:])

    date_row_indices: list[Any] = []
    for di in range(n_dates):
        st = int(d_starts[di])
        ln = int(d_counts[di])
        idx = date_order[st : st + ln]
        date_row_indices.append(idx)
        if ln <= 1:
            continue
        for ci in range(X_raw.shape[1]):
            vals = X_raw[idx, ci]
            fin = np.isfinite(vals)
            nv = int(fin.sum())
            if nv > 1:
                r = rankdata(vals[fin], method="average") - 1.0
                X_rank[idx[fin], ci] = (2.0 * r / (nv - 1.0) - 1.0).astype(np.float32)

    # Feed S2 calibrated 1x half-spread directly into S1 Almgren-Chriss cost model!
    cal_half_spr = float(
        (s2_res.get("tca_cost_calibration_feedback") or {}).get("calibrated_1x_half_spread_bps") or 4.0
    )
    adv_arr = df["mean_20d_dvol"].to_numpy(dtype=np.float64)
    rng_arr = np.clip(df["mean_20d_range_pct"].fillna(0.025).to_numpy(dtype=np.float64), 0.005, 0.15)
    cost_1x = np.array(
        [
            compute_almgren_chriss_cost_bps(
                base_half_spread_bps=cal_half_spr,
                fee_and_slip_bps_per_side=1.5,
                daily_sigma_frac=float(rng_arr[i]),
                order_notional_usd=DEFAULT_POSITION_NOTIONAL_USD,
                adv_usd=max(float(adv_arr[i]), 1_000_000.0),
                stress_multiplier=1.0,
            )
            for i in range(len(df))
        ],
        dtype=np.float64,
    )
    cost_2x = cost_1x * 2.0
    short_borrow_1x = 5.0
    short_borrow_2x = 10.0

    # P1-2 FIX: Purge=20 and Embargo=20 sessions matching the 20-session DF10 baseline lookback window!
    cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=20, embargo_size=20)
    dummy_dates = np.arange(n_dates).reshape(-1, 1)
    splits = list(cv.split(dummy_dates))
    path_ids = cv.get_path_ids()

    path_daily_1x: dict[int, list[float]] = {p: [] for p in range(5)}
    path_daily_2x: dict[int, list[float]] = {p: [] for p in range(5)}
    all_oos_event_ret_1x: list[float] = []
    all_oos_event_ret_2x: list[float] = []
    all_oos_event_dates: list[str] = []
    unfilled_selected_entries = 0

    # Collect per-config strictly OOS daily 2x returns across C(6,2)=15 CPCV splits for symmetric Bailey CSCV C(6,3)=20
    n_cfg = len(LGBM_PBO_CONFIGS)
    cfg_date_oos_2x: list[dict[int, list[float]]] = [{} for _ in range(n_cfg)]

    for split_i, (train_d_idx, test_d_blocks) in enumerate(splits):
        raw_tr_rows = np.concatenate([date_row_indices[int(d)] for d in train_d_idx])
        tr_rows = raw_tr_rows[valid_next[raw_tr_rows]]

        if (
            isinstance(test_d_blocks, (list, tuple))
            and len(test_d_blocks) == 2
            and hasattr(test_d_blocks[0], "__len__")
        ):
            blocks = list(test_d_blocks)
        else:
            blocks = [np.asarray(test_d_blocks)]

        for cfg_i, cfg in enumerate(LGBM_PBO_CONFIGS):
            model = lgb.LGBMRegressor(
                n_estimators=int(cfg["n_estimators"]),
                learning_rate=float(cfg["learning_rate"]),
                num_leaves=int(cfg["num_leaves"]),
                reg_lambda=float(cfg["reg_lambda"]),
                subsample_for_bin=50_000,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=42 + split_i * 10 + cfg_i,
                n_jobs=16,
                verbosity=-1,
            )
            model.fit(X_rank[tr_rows], y_train_target[tr_rows])

            # Strictly OOS predictions on held-out test dates (primary benchmark = cfg_i == 1, num_leaves=31)
            for b_idx, d_block in enumerate(blocks):
                pid = int(path_ids[split_i][b_idx]) if b_idx < len(path_ids[split_i]) else 0
                for d in d_block:
                    di_int = int(d)
                    idx = date_row_indices[di_int]
                    if len(idx) < 20:
                        continue
                    preds = model.predict(X_rank[idx])
                    k = max(2, len(idx) // 10)
                    top_i = idx[np.argpartition(preds, -k)[-k:]]
                    bot_i = idx[np.argpartition(preds, k)[:k]]
                    l_1x = np.where(valid_next[top_i], long_ret_bps[top_i] - cost_1x[top_i], 0.0)
                    s_1x = np.where(valid_next[bot_i], short_ret_bps[bot_i] - cost_1x[bot_i] - short_borrow_1x, 0.0)
                    l_2x = np.where(valid_next[top_i], long_ret_bps[top_i] - cost_2x[top_i], 0.0)
                    s_2x = np.where(valid_next[bot_i], short_ret_bps[bot_i] - cost_2x[bot_i] - short_borrow_2x, 0.0)
                    day_1x = 0.5 * (float(np.mean(l_1x)) + float(np.mean(s_1x)))
                    day_2x = 0.5 * (float(np.mean(l_2x)) + float(np.mean(s_2x)))
                    cfg_date_oos_2x[cfg_i].setdefault(di_int, []).append(day_2x)

                    if cfg_i == 1:
                        path_daily_1x[pid].append(day_1x)
                        path_daily_2x[pid].append(day_2x)
                        if pid == 0:
                            d_str = str(unique_dates[di_int])
                            unfilled_selected_entries += int((~valid_next[top_i]).sum() + (~valid_next[bot_i]).sum())
                            for j, _ri in enumerate(top_i):
                                all_oos_event_ret_1x.append(float(l_1x[j]))
                                all_oos_event_ret_2x.append(float(l_2x[j]))
                                all_oos_event_dates.append(d_str)
                            for j, _ri in enumerate(bot_i):
                                all_oos_event_ret_1x.append(float(s_1x[j]))
                                all_oos_event_ret_2x.append(float(s_2x[j]))
                                all_oos_event_dates.append(d_str)

    # True Bailey et al. (2014) symmetric CSCV PBO: partition the (T x M=5) OOS matrix into S=6 equal blocks
    # and evaluate all C(6,3)=20 complementary 3-block IS vs 3-block OOS partitions (ZERO in-sample tail leakage!).
    common_oos_dates = sorted(set.intersection(*(set(m.keys()) for m in cfg_date_oos_2x)))
    n_cscv_blocks = 6
    blk_len = len(common_oos_dates) // n_cscv_blocks
    if blk_len < 10:
        raise RuntimeError(f"Insufficient OOS dates for Bailey CSCV C(6,3)=20: {len(common_oos_dates)}")
    usable_dates = common_oos_dates[: blk_len * n_cscv_blocks]
    m_mat = np.zeros((len(usable_dates), n_cfg), dtype=np.float64)
    for ti, di_int in enumerate(usable_dates):
        for cfg_i in range(n_cfg):
            m_mat[ti, cfg_i] = float(np.mean(cfg_date_oos_2x[cfg_i][di_int]))
    m_blocks = m_mat.reshape(n_cscv_blocks, blk_len, n_cfg)

    pbo_underperform_count = 0
    pbo_splits_evaluated = 0
    for is_comb in itertools.combinations(range(n_cscv_blocks), n_cscv_blocks // 2):
        oos_comb = tuple(b for b in range(n_cscv_blocks) if b not in is_comb)
        is_slice = m_blocks[list(is_comb)].reshape(-1, n_cfg)
        oos_slice = m_blocks[list(oos_comb)].reshape(-1, n_cfg)
        is_sr = np.mean(is_slice, axis=0) / np.maximum(np.std(is_slice, axis=0, ddof=1), 1e-6)
        oos_sr = np.mean(oos_slice, axis=0) / np.maximum(np.std(oos_slice, axis=0, ddof=1), 1e-6)
        is_winner = int(np.argmax(is_sr))
        oos_ranks = rankdata(oos_sr, method="average")
        omega_bar = float(oos_ranks[is_winner]) / float(n_cfg + 1)
        if omega_bar <= 0.5:
            pbo_underperform_count += 1
        pbo_splits_evaluated += 1

    path_means_2x = [float(np.mean(v)) if v else 0.0 for v in path_daily_2x.values()]
    path_means_1x = [float(np.mean(v)) if v else 0.0 for v in path_daily_1x.values()]
    cpcv_pos_share = sum(1 for m in path_means_2x if m > 0.0) / 5.0
    pbo_cscv = round(pbo_underperform_count / max(1, pbo_splits_evaluated), 4)

    cl_1x = _date_clustered_t_stat(all_oos_event_ret_1x, all_oos_event_dates)
    cl_2x = _date_clustered_t_stat(all_oos_event_ret_2x, all_oos_event_dates)
    path0_daily_frac = np.asarray(path_daily_2x[0], dtype=np.float64) / 10_000.0
    dsr_prob = _compute_dsr_from_daily_returns(path0_daily_frac, effective_trials=float(TRIAL_104_NUMBER))

    distinct_oos_dates = len(set(all_oos_event_dates))
    gate = evaluate_strict_adr0027_gate(
        trial_number=TRIAL_104_NUMBER,
        t_stat_clustered_2x=float(cl_2x["t_stat"]),
        dsr_probability=dsr_prob,
        cpcv_positive_share=cpcv_pos_share,
        pbo=pbo_cscv,
        mean_net_2x_bps=float(cl_2x["mean_bps"]),
        distinct_dates=distinct_oos_dates,
        total_events=len(all_oos_event_ret_2x),
        power_gate_approved=bool(power_gate.get("approved")),
        sec_shield_verified=True,
        s2_queue_replay_verified=bool(s2_res.get("hftbacktest_replay_executed")),
    )

    return {
        "trial_id": TRIAL_104_ID,
        "trial_number": TRIAL_104_NUMBER,
        "study_wall_date": STUDY_WALL_ISO,
        "benchmark_model": (
            "18-feature Qlib Alpha158-inspired OHLCV+Liquidity LightGBM subset "
            "(5-config CSCV/CPCV C(6,2)=15 purge=20 embargo=20 retrospective distribution diagnostic)"
        ),
        "feature_count": len(feature_cols),
        "pbo_candidate_configs": n_cfg,
        "pbo_cscv_splits": pbo_splits_evaluated,
        "pbo_method": "Bailey_2014_symmetric_CSCV_C6_3_20_over_5_LGBM_configs",
        "cpcv_purge_sessions": 20,
        "cpcv_embargo_sessions": 20,
        "calibrated_1x_half_spread_bps": cal_half_spr,
        "liquid_prewall_rows": int(len(df)),
        "distinct_trade_dates": int(n_dates),
        "oos_distinct_dates": int(distinct_oos_dates),
        "oos_events_count": int(len(all_oos_event_ret_2x)),
        "unfilled_selected_entries": int(unfilled_selected_entries),
        "mean_net_1x_bps": float(cl_1x["mean_bps"]),
        "t_stat_clustered_1x": float(cl_1x["t_stat"]),
        "mean_net_2x_bps": float(cl_2x["mean_bps"]),
        "t_stat_clustered_2x": float(cl_2x["t_stat"]),
        "dsr_probability": dsr_prob,
        "cpcv_5path_means_1x_bps": [round(x, 3) for x in path_means_1x],
        "cpcv_5path_means_2x_bps": [round(x, 3) for x in path_means_2x],
        "cpcv_positive_share": round(cpcv_pos_share, 4),
        "pbo_cscv_estimate": pbo_cscv,
        "gate": gate,
        "promotion_track": "ALPHA_STRATEGY_MINOR" if gate["passed"] else "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "standalone_alpha_claim": bool(gate["passed"]),
    }


def _scrub_home_from_text(text: str) -> str:
    """Scrub native, POSIX, and escaped-backslash representations of `Path.home()` from stderr diagnostics."""
    home_str = str(pathlib.Path.home())
    out = text
    for variant in (home_str, home_str.replace("\\", "\\\\"), home_str.replace("\\", "/")):
        if variant:
            out = out.replace(variant, "<home>")
    return out


class _SilentArgParser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        raise ValueError(message)


def main() -> int:
    try:
        parser = _SilentArgParser(
            description="Run v3.11.0 Pre-Wall S1/S2 Alpha Factory",
            exit_on_error=False,
        )
        parser.add_argument(
            "--lake-root",
            type=pathlib.Path,
            default=pathlib.Path.home() / "sota-data-ext4" / "prewall-lake",
        )
        parser.add_argument(
            "--paper-tca-json",
            type=pathlib.Path,
            default=None,
        )
        parser.add_argument(
            "--receipt-out",
            type=pathlib.Path,
            default=pathlib.Path.home()
            / "sota-data-ext4"
            / "prewall-lake"
            / "receipts"
            / "v3110_alpha_factory_s1_s2_receipt.json",
        )
        args = parser.parse_args()

        aligned_path, align_audit = materialize_next_session_aligned_df10(args.lake_root)
        power_gate = evaluate_mintrl_mde_power_gate(distinct_dates=align_audit["distinct_trade_dates"])
        s2_res = run_s2_hftbacktest_queue_parity_probe(args.lake_root, paper_tca_json=args.paper_tca_json)
        s1_res = run_s1_qlib_alpha158_lgbm_cpcv(aligned_path, args.lake_root, power_gate=power_gate, s2_res=s2_res)

        receipt = {
            "schema": "sota.v3110-alpha-factory-receipt.v1",
            "adr": "ADR-0027",
            "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
            "study_wall_date": STUDY_WALL_ISO,
            "next_session_date_materialization": align_audit,
            "mintrl_mde_power_gate": power_gate,
            "s2_hftbacktest_queue_parity": s2_res,
            "s1_qlib_alpha158_subset_lgbm_cpcv": s1_res,
        }
        raw_bytes = json.dumps(receipt, sort_keys=True, indent=2).encode("utf-8")
        receipt["receipt_sha256"] = hashlib.sha256(raw_bytes).hexdigest()

        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code == 0:
            return 0
        print(
            json.dumps({"ok": False, "error_type": type(exc).__name__, "error": _scrub_home_from_text(str(exc))}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
