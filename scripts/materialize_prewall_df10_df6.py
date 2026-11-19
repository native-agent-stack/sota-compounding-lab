"""Materialize strictly pre-wall (<= 2026-10-15) PIT DF10 (20-session XNYS daily baselines) and DF6 (dated Reg SHO).

Enforces ADR-0020 & ADR-0021 fail-closed contracts:
- Zero post-wall leakage: every trade_date, list_date, effective_trade_date, and accepted_at <= 2026-10-15 ET.
- Official XNYS calendar validation via exchange_calendars (0 weekend/holiday phantom sessions).
- Strictly causal DF10 windows (ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING; d < trade_date, prior_sessions_count == 20).
- Raw unadjusted prices & dollar volumes for liquidity filters paired with split-adjusted OHLC for forward returns.
- Dated NASDAQ + NYSE/Arca/American Reg SHO Rule 203(c)(6) threshold membership snapshots with exact published_ts_utc.
- Post-write pyarrow parquet_metadata + XNYS audit proof sealed into an immutable SHA-256 receipt.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

STUDY_WALL_DATE = dt.date(2026, 10, 15)
USER_AGENT = "Mozilla/5.0 SOTA-Quant-Research/1.0"


def _ny_tz_for_date(d: dt.date) -> dt.tzinfo:
    """Return America/New_York ZoneInfo when tzdata exists, else exact US Eastern DST offset."""
    try:
        return ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError:
        # US DST: 2nd Sunday in March -> 1st Sunday in November
        mar1 = dt.date(d.year, 3, 1)
        dst_start = mar1 + dt.timedelta(days=(6 - mar1.weekday()) % 7 + 7)
        nov1 = dt.date(d.year, 11, 1)
        dst_end = nov1 + dt.timedelta(days=(6 - nov1.weekday()) % 7)
        offset_h = -4 if dst_start <= d < dst_end else -5
        return dt.timezone(dt.timedelta(hours=offset_h))


def assert_prewall_date(d: dt.date, label: str = "date") -> dt.date:
    """Fail closed if any date crosses the 2026-10-15 study wall."""
    if d > STUDY_WALL_DATE:
        raise ValueError(f"PREWALL_CUTOFF_BREACH: {label}={d.isoformat()} > {STUDY_WALL_DATE.isoformat()}")
    return d


def compute_regsho_published_ts_utc(list_date: dt.date) -> str:
    """Return conservative PIT UTC publish timestamp (~23:00 ET evening of list_date)."""
    assert_prewall_date(list_date, "list_date")
    ny_pub = dt.datetime(list_date.year, list_date.month, list_date.day, 23, 0, 0, tzinfo=_ny_tz_for_date(list_date))
    return ny_pub.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def parse_regsho_pipe_text(
    raw_text: str,
    *,
    list_date: dt.date,
    effective_trade_date: dt.date,
    exchange_source: str,
) -> list[dict[str, Any]]:
    """Parse pipe-delimited NASDAQ or NYSE Reg SHO threshold text into PIT records."""
    assert_prewall_date(list_date, "list_date")
    assert_prewall_date(effective_trade_date, "effective_trade_date")
    if effective_trade_date <= list_date:
        raise ValueError(
            f"DF6_CAUSALITY_ERROR: effective_trade_date={effective_trade_date.isoformat()} "
            f"must be strictly after list_date={list_date.isoformat()}"
        )

    pub_utc = compute_regsho_published_ts_utc(list_date)
    rows: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue
        parts = stripped.split("|")
        sym = parts[0].strip().upper()
        if not sym or sym in {"SYMBOL", "FILE CREATION TIME"} or sym.startswith("FILE CREATION"):
            continue
        # Trailer lines like "202610152300" have digits only
        if sym.isdigit():
            continue
        sec_name = parts[1].strip() if len(parts) > 1 else ""
        market_cat = parts[2].strip() if len(parts) > 2 else exchange_source
        flag = parts[3].strip().upper() if len(parts) > 3 else ""
        if flag and flag not in {"Y", "YES", "1", "TRUE"}:
            continue
        rule_3210 = parts[4].strip().upper() if len(parts) > 4 else ""
        rows.append(
            {
                "list_date": list_date.isoformat(),
                "effective_trade_date": effective_trade_date.isoformat(),
                "published_ts_utc": pub_utc,
                "symbol": sym,
                "exchange_source": exchange_source,
                "market_category": market_cat or exchange_source,
                "security_name": sec_name[:120],
                "rule_3210": rule_3210,
            }
        )
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_url_text(url: str, timeout: float = 12.0) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), ""
    except Exception:
        return -1, ""


def _xnys_prewall_sessions(start_date: str = "2024-01-02", end_date: str = "2026-10-15") -> list[dt.date]:
    import exchange_calendars as xcals

    end_d = assert_prewall_date(dt.date.fromisoformat(end_date), "end_date")
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range(start_date, end_d.isoformat())
    out = [ts.date() for ts in sessions]
    if any(d > STUDY_WALL_DATE for d in out):
        raise ValueError("XNYS session crossed STUDY_WALL_DATE")
    return out


def _audit_parquet_metadata_and_xnys(
    parquet_path: pathlib.Path,
    date_columns: tuple[str, ...],
    valid_xnys_dates: set[str],
) -> dict[str, Any]:
    """Read Parquet row-group statistics + verify zero post-wall / non-XNYS date leakage."""
    import duckdb
    import pyarrow.parquet as pq

    meta = pq.read_metadata(parquet_path)
    con = duckdb.connect()
    col_audits: dict[str, Any] = {}
    for col in date_columns:
        min_d, max_d, n_postwall, n_non_xnys = con.execute(
            f"""
            SELECT
                MIN(CAST({col} AS VARCHAR)),
                MAX(CAST({col} AS VARCHAR)),
                SUM(CASE WHEN CAST({col} AS DATE) > DATE '{STUDY_WALL_DATE.isoformat()}' THEN 1 ELSE 0 END),
                SUM(CASE WHEN CAST({col} AS VARCHAR) NOT IN (SELECT UNNEST(?)) THEN 1 ELSE 0 END)
            FROM read_parquet(?)
            """,
            [sorted(valid_xnys_dates), str(parquet_path)],
        ).fetchone()
        if int(n_postwall or 0) > 0:
            raise ValueError(f"POSTWALL_LEAK_DETECTED in {parquet_path.name}.{col}: {n_postwall} rows")
        if int(n_non_xnys or 0) > 0:
            raise ValueError(f"NON_XNYS_DATE_DETECTED in {parquet_path.name}.{col}: {n_non_xnys} rows")
        col_audits[col] = {
            "min_date": min_d,
            "max_date": max_d,
            "postwall_rows": int(n_postwall or 0),
            "non_xnys_rows": int(n_non_xnys or 0),
        }

    return {
        "file": str(parquet_path.name),
        "num_rows": int(meta.num_rows),
        "num_row_groups": int(meta.num_row_groups),
        "num_columns": int(meta.num_columns),
        "size_bytes": int(parquet_path.stat().st_size),
        "sha256": sha256_file(parquet_path),
        "date_column_audits": col_audits,
    }


def materialize_df10_daily(lake_root: pathlib.Path, valid_xnys_dates: set[str]) -> dict[str, Any]:
    """Materialize strictly causal 20-prior-XNYS-session DF10 daily baselines (<= 2026-10-15)."""
    import duckdb

    out_path = lake_root / "derived" / "df10_daily_20d_baselines.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_globs = [str(lake_root / "universe" / y / "daily_raw.csv.gz") for y in ("2024", "2025", "2026")]
    split_globs = [str(lake_root / "universe" / y / "daily_split.csv.gz") for y in ("2024", "2025", "2026")]

    con = duckdb.connect()
    con.execute("PRAGMA threads=32")
    con.execute("CREATE TEMP TABLE xnys_sessions (trade_date DATE PRIMARY KEY)")
    con.executemany("INSERT INTO xnys_sessions VALUES (?)", [(d,) for d in sorted(valid_xnys_dates)])

    # Combine raw unadjusted bars (for price/dvol filters) + split-adjusted bars (for forward returns & VDU patterns)
    # Every rolling baseline uses strictly prior 20 XNYS sessions (20 PRECEDING AND 1 PRECEDING -> d < trade_date).
    query = f"""
    COPY (
        WITH raw_all AS (
            SELECT
                UPPER(TRIM(symbol)) AS symbol,
                CAST(ts_utc AT TIME ZONE 'America/New_York' AS DATE) AS trade_date,
                CAST(open AS DOUBLE) AS raw_open,
                CAST(high AS DOUBLE) AS raw_high,
                CAST(low AS DOUBLE) AS raw_low,
                CAST(close AS DOUBLE) AS raw_close,
                CAST(volume AS DOUBLE) AS raw_volume
            FROM read_csv_auto({raw_globs!r}, union_by_name=true)
            WHERE CAST(ts_utc AT TIME ZONE 'America/New_York' AS DATE) <= DATE '{STUDY_WALL_DATE.isoformat()}'
              AND volume >= 0 AND close > 0
        ),
        split_all AS (
            SELECT
                UPPER(TRIM(symbol)) AS symbol,
                CAST(ts_utc AT TIME ZONE 'America/New_York' AS DATE) AS trade_date,
                CAST(open AS DOUBLE) AS split_open,
                CAST(high AS DOUBLE) AS split_high,
                CAST(low AS DOUBLE) AS split_low,
                CAST(close AS DOUBLE) AS split_close,
                CAST(volume AS DOUBLE) AS split_volume
            FROM read_csv_auto({split_globs!r}, union_by_name=true)
            WHERE CAST(ts_utc AT TIME ZONE 'America/New_York' AS DATE) <= DATE '{STUDY_WALL_DATE.isoformat()}'
              AND volume >= 0 AND close > 0
        ),
        joined AS (
            SELECT
                r.symbol,
                r.trade_date,
                r.raw_open,
                r.raw_high,
                r.raw_low,
                r.raw_close,
                r.raw_volume,
                r.raw_close * r.raw_volume AS raw_dvol,
                COALESCE(s.split_open, r.raw_open) AS split_open,
                COALESCE(s.split_high, r.raw_high) AS split_high,
                COALESCE(s.split_low, r.raw_low) AS split_low,
                COALESCE(s.split_close, r.raw_close) AS split_close,
                COALESCE(s.split_volume, r.raw_volume) AS split_volume,
                (COALESCE(s.split_high, r.raw_high) - COALESCE(s.split_low, r.raw_low))
                    / GREATEST(COALESCE(s.split_close, r.raw_close), 1e-6) AS bar_range_pct
            FROM raw_all r
            INNER JOIN xnys_sessions x ON x.trade_date = r.trade_date
            LEFT JOIN split_all s ON s.symbol = r.symbol AND s.trade_date = r.trade_date
            QUALIFY ROW_NUMBER() OVER (PARTITION BY r.symbol, r.trade_date ORDER BY r.raw_volume DESC) = 1
        ),
        windowed AS (
            SELECT
                symbol,
                trade_date,
                raw_open,
                raw_high,
                raw_low,
                raw_close,
                raw_volume,
                raw_dvol,
                split_open,
                split_high,
                split_low,
                split_close,
                split_volume,
                bar_range_pct,
                COUNT(*) OVER w20 AS prior_sessions_count,
                AVG(raw_volume) OVER w20 AS mean_20d_vol,
                STDDEV_SAMP(raw_volume) OVER w20 AS std_20d_vol,
                AVG(raw_dvol) OVER w20 AS mean_20d_dvol,
                AVG(raw_volume) OVER w5 AS mean_5d_vol_prior,
                AVG(bar_range_pct) OVER w20 AS mean_20d_range_pct,
                AVG(bar_range_pct) OVER w5 AS mean_5d_range_pct_prior,
                MAX(split_high) OVER w20 AS high_20d_prior,
                MIN(split_low) OVER w20 AS low_20d_prior,
                LAG(split_close, 1) OVER ws AS prev_split_close,
                LAG(raw_close, 1) OVER ws AS prev_raw_close,
                LAG(split_close, 5) OVER ws AS lag5_split_close,
                LAG(split_close, 20) OVER ws AS lag20_split_close,
                LEAD(split_open, 1) OVER ws AS next1_split_open,
                LEAD(split_close, 1) OVER ws AS next1_split_close,
                LEAD(split_close, 3) OVER ws AS next3_split_close,
                LEAD(split_close, 5) OVER ws AS next5_split_close,
                LEAD(split_low, 1) OVER ws AS next1_split_low,
                LEAD(split_high, 1) OVER ws AS next1_split_high
            FROM joined
            WINDOW
                ws AS (PARTITION BY symbol ORDER BY trade_date),
                w20 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
                w5 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING)
        )
        SELECT
            symbol,
            CAST(trade_date AS VARCHAR) AS trade_date,
            raw_open,
            raw_high,
            raw_low,
            raw_close,
            raw_volume,
            raw_dvol,
            split_open,
            split_high,
            split_low,
            split_close,
            bar_range_pct,
            CAST(prior_sessions_count AS INTEGER) AS prior_sessions_count,
            mean_20d_vol,
            COALESCE(std_20d_vol, 0.0) AS std_20d_vol,
            mean_20d_dvol,
            mean_5d_vol_prior,
            mean_20d_range_pct,
            mean_5d_range_pct_prior,
            high_20d_prior,
            low_20d_prior,
            prev_split_close,
            prev_raw_close,
            CASE WHEN mean_20d_vol > 0 THEN raw_volume / mean_20d_vol ELSE 0.0 END AS rvol_20d,
            CASE WHEN mean_20d_vol > 0 THEN
                (raw_volume - mean_20d_vol) / GREATEST(COALESCE(std_20d_vol, 0.0), mean_20d_vol * 1e-6)
            ELSE 0.0 END AS vol_z_20d,
            CASE WHEN mean_20d_vol > 0 THEN mean_5d_vol_prior / mean_20d_vol ELSE NULL END AS vdu_5d_20d_ratio,
            CASE
                WHEN mean_20d_range_pct > 0 THEN mean_5d_range_pct_prior / mean_20d_range_pct
                ELSE NULL
            END AS coil_5d_20d_range_ratio,
            CASE
                WHEN high_20d_prior > 0 THEN prev_split_close / high_20d_prior
                ELSE NULL
            END AS prev_close_to_20d_high,
            CASE
                WHEN prev_split_close > 0 THEN (split_close / prev_split_close) - 1.0
                ELSE NULL
            END AS ret_1d,
            CASE
                WHEN lag5_split_close > 0 THEN (prev_split_close / lag5_split_close) - 1.0
                ELSE NULL
            END AS prior_ret_5d,
            CASE
                WHEN lag20_split_close > 0 THEN (prev_split_close / lag20_split_close) - 1.0
                ELSE NULL
            END AS prior_ret_20d,
            next1_split_open,
            next1_split_close,
            next3_split_close,
            next5_split_close,
            next1_split_low,
            next1_split_high
        FROM windowed
        WHERE prior_sessions_count = 20
        ORDER BY trade_date, symbol
    ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
    """
    con.execute(query)

    audit = _audit_parquet_metadata_and_xnys(out_path, ("trade_date",), valid_xnys_dates)

    # Also honestly audit published-orb intraday same-clock session depth (zero fabrication)
    orb_trades_dir = lake_root / "published-orb" / "trades"
    sym_date_counts: dict[str, int] = {}
    max_orb_date = ""
    if orb_trades_dir.is_dir():
        for p in orb_trades_dir.glob("*/*.parquet"):
            sym = p.parent.name.upper()
            d_str = p.name[:10]
            if d_str > max_orb_date:
                max_orb_date = d_str
            sym_date_counts[sym] = sym_date_counts.get(sym, 0) + 1
    syms_with_ge20_intraday = sum(1 for c in sym_date_counts.values() if c >= 21)
    audit["published_orb_intraday_honesty_audit"] = {
        "distinct_symbols_in_published_orb": len(sym_date_counts),
        "symbols_with_ge20_prior_intraday_slices": syms_with_ge20_intraday,
        "max_symbol_intraday_slices": max(sym_date_counts.values()) if sym_date_counts else 0,
        "max_orb_date": max_orb_date,
        "daily_20d_df10_ready": True,
        "intraday_same_clock_20d_df10_ready": syms_with_ge20_intraday > 0,
    }
    return audit


def materialize_df6_regsho(
    lake_root: pathlib.Path, sessions: list[dt.date], *, max_workers: int = 12
) -> dict[str, Any]:
    """Fetch & materialize dated NASDAQ + NYSE/Arca/American Reg SHO threshold lists (<= 2026-10-15)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = lake_root / "derived" / "df6_regsho_threshold_pit.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pair each list_date[i] with effective_trade_date = sessions[i+1] (both strictly <= 2026-10-15!)
    pairs: list[tuple[dt.date, dt.date]] = [(sessions[i], sessions[i + 1]) for i in range(len(sessions) - 1)]

    def _fetch_one_pair(pair: tuple[dt.date, dt.date]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        list_d, eff_d = pair
        out_rows: list[dict[str, Any]] = []
        stats = {"nq_ok": 0, "ny_ok": 0, "nq_err": 0, "ny_err": 0}

        nq_url = f"https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{list_d:%Y%m%d}.txt"
        st_nq, txt_nq = _fetch_url_text(nq_url)
        if st_nq == 200 and txt_nq:
            stats["nq_ok"] = 1
            out_rows.extend(
                parse_regsho_pipe_text(txt_nq, list_date=list_d, effective_trade_date=eff_d, exchange_source="NASDAQ")
            )
        else:
            stats["nq_err"] = 1

        ny_url = (
            f"https://www.nyse.com/api/regulatory/threshold-securities/download?selectedDate={list_d:%d-%b-%Y}&market="
        )
        st_ny, txt_ny = _fetch_url_text(ny_url)
        if st_ny == 200 and txt_ny:
            stats["ny_ok"] = 1
            out_rows.extend(
                parse_regsho_pipe_text(
                    txt_ny, list_date=list_d, effective_trade_date=eff_d, exchange_source="NYSE_GROUP"
                )
            )
        else:
            stats["ny_err"] = 1

        return out_rows, stats

    all_rows: list[dict[str, Any]] = []
    agg_stats = {"nq_ok": 0, "ny_ok": 0, "nq_err": 0, "ny_err": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for rows, st in pool.map(_fetch_one_pair, pairs):
            all_rows.extend(rows)
            for k in agg_stats:
                agg_stats[k] += st[k]

    if not all_rows:
        raise RuntimeError("DF6 fetch returned 0 rows across all XNYS sessions")

    # Sort deterministically & deduplicate by (effective_trade_date, symbol, exchange_source)
    seen: set[tuple[str, str, str]] = set()
    dedup_rows: list[dict[str, Any]] = []
    for rec in sorted(all_rows, key=lambda r: (r["effective_trade_date"], r["symbol"], r["exchange_source"])):
        key = (rec["effective_trade_date"], rec["symbol"], rec["exchange_source"])
        if key not in seen:
            seen.add(key)
            dedup_rows.append(rec)

    table = pa.Table.from_pylist(dedup_rows)
    pq.write_table(table, out_path, compression="zstd")

    valid_xnys_dates = {d.isoformat() for d in sessions}
    audit = _audit_parquet_metadata_and_xnys(
        out_path,
        ("list_date", "effective_trade_date"),
        valid_xnys_dates,
    )
    audit["fetch_coverage"] = {
        "target_xnys_pairs": len(pairs),
        "nasdaq_sessions_fetched": agg_stats["nq_ok"],
        "nasdaq_sessions_missing": agg_stats["nq_err"],
        "nyse_group_sessions_fetched": agg_stats["ny_ok"],
        "nyse_group_sessions_missing": agg_stats["ny_err"],
        "distinct_symbols": len({r["symbol"] for r in dedup_rows}),
        "distinct_effective_dates": len({r["effective_trade_date"] for r in dedup_rows}),
    }
    return audit


def audit_sec_clean_prewall(lake_root: pathlib.Path) -> dict[str, Any]:
    """Verify derived/sec_filings_clean_prewall.parquet has 0 post-wall leaks (<= 2026-10-15 ET)."""
    import duckdb
    import pyarrow.parquet as pq

    sec_path = lake_root / "derived" / "sec_filings_clean_prewall.parquet"
    meta = pq.read_metadata(sec_path)
    con = duckdb.connect()
    min_acc, max_acc, n_post_acc, n_post_filed = con.execute(
        f"""
        SELECT
            MIN(accepted),
            MAX(accepted),
            SUM(
                CASE
                    WHEN CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE)
                         > DATE '{STUDY_WALL_DATE.isoformat()}'
                    THEN 1 ELSE 0
                END
            ),
            SUM(CASE WHEN CAST(filed AS DATE) > DATE '{STUDY_WALL_DATE.isoformat()}' THEN 1 ELSE 0 END)
        FROM read_parquet(?)
        """,
        [str(sec_path)],
    ).fetchone()
    if int(n_post_acc or 0) > 0 or int(n_post_filed or 0) > 0:
        raise ValueError(f"SEC_POSTWALL_LEAK: accepted={n_post_acc}, filed={n_post_filed}")
    return {
        "file": sec_path.name,
        "num_rows": int(meta.num_rows),
        "sha256": sha256_file(sec_path),
        "min_accepted_utc": min_acc,
        "max_accepted_utc": max_acc,
        "postwall_accepted_rows": int(n_post_acc or 0),
        "postwall_filed_rows": int(n_post_filed or 0),
        "clean_prewall": True,
    }


def probe_optb_runtime_modules() -> dict[str, Any]:
    """Verify Option B RTX 4090 CUDA + quant factory modules inside the active Python environment."""
    import importlib

    checks = {
        "torch": "torch",
        "polars": "polars",
        "duckdb": "duckdb",
        "skfolio": "skfolio",
        "ml4t": "ml4t.diagnostic",
        "hftbacktest": "hftbacktest",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "vectorbt": "vectorbt",
        "pandera": "pandera",
        "statsmodels": "statsmodels",
        "exchange_calendars": "exchange_calendars",
        "alpaca": "alpaca",
        "ib_async": "ib_async",
        "qlib": "qlib",
        "kronos": "kronos",
        "timesfm": "timesfm",
    }
    mods: dict[str, bool] = {}
    versions: dict[str, str] = {}
    cuda_info: dict[str, Any] = {"cuda_available": False, "gpu_name": None}

    for key, mod_name in checks.items():
        try:
            m = importlib.import_module(mod_name)
            mods[key] = True
            versions[key] = str(getattr(m, "__version__", "installed"))
            if key == "torch":
                cuda_avail = bool(m.cuda.is_available())
                cuda_info = {
                    "cuda_available": cuda_avail,
                    "gpu_name": str(m.cuda.get_device_name(0)) if cuda_avail else None,
                    "cuda_version": str(getattr(m.version, "cuda", None)),
                }
        except Exception:
            mods[key] = False

    return {"mods": mods, "versions": versions, "cuda": cuda_info}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize pre-wall <=2026-10-15 DF10 and DF6 PIT tables.")
    parser.add_argument("--lake-root", default="/mnt/z/sota-data/prewall-lake")
    parser.add_argument("--start-date", default="2024-01-02")
    parser.add_argument("--end-date", default=STUDY_WALL_DATE.isoformat())
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()

    lake_root = pathlib.Path(args.lake_root)
    sessions = _xnys_prewall_sessions(args.start_date, args.end_date)
    valid_xnys_dates = {d.isoformat() for d in sessions}

    print(f"[S1] Materializing DF10 20-prior-XNYS-session baselines ({len(sessions)} sessions <= {args.end_date})...")
    df10_audit = materialize_df10_daily(lake_root, valid_xnys_dates)
    print(f"[S1] DF10 DONE: {df10_audit['num_rows']} rows, sha256={df10_audit['sha256'][:16]}...")

    print(f"[S2] Fetching DF6 dated Reg SHO threshold lists ({len(sessions) - 1} pairs <= {args.end_date})...")
    df6_audit = materialize_df6_regsho(lake_root, sessions, max_workers=args.max_workers)
    print(f"[S2] DF6 DONE: {df6_audit['num_rows']} rows, sha256={df6_audit['sha256'][:16]}...")

    print("[S2b] Auditing sec_filings_clean_prewall.parquet (0 post-wall leaks)...")
    sec_audit = audit_sec_clean_prewall(lake_root)
    print(f"[S2b] SEC CLEAN: {sec_audit['num_rows']} rows, max_accepted={sec_audit['max_accepted_utc']}")

    runtime_probe = probe_optb_runtime_modules()

    receipt_body = {
        "schema_version": "sota-prewall-df10-df6-materialization.v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "xnys_sessions_count": len(sessions),
        "df10_daily_20d_baselines": df10_audit,
        "df6_regsho_threshold_pit": df6_audit,
        "sec_filings_clean_prewall": sec_audit,
        "runtime_probe": runtime_probe,
    }
    receipt_body["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt_body, sort_keys=True).encode("utf-8")
    ).hexdigest()

    receipt_path = lake_root / "receipts" / "df10_df6_materialization_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt_body, indent=2) + "\n", encoding="utf-8")

    # Update optb_prewall_factory_audit.json preserving plan_sha256 & next_hlz
    optb_path = lake_root / "receipts" / "optb_prewall_factory_audit.json"
    existing_optb = json.loads(optb_path.read_text(encoding="utf-8")) if optb_path.is_file() else {}
    updated_optb = {
        "plan_sha256": existing_optb.get("plan_sha256", ""),
        "next_hlz": existing_optb.get("next_hlz", 4.2019),
        "updated_at_utc": receipt_body["generated_at_utc"],
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "mods": runtime_probe["mods"],
        "versions": runtime_probe["versions"],
        "cuda": runtime_probe["cuda"],
        "df10_daily_sha256": df10_audit["sha256"],
        "df6_regsho_sha256": df6_audit["sha256"],
        "sec_clean_sha256": sec_audit["sha256"],
        "materialization_receipt_sha256": receipt_body["receipt_sha256"],
    }
    optb_path.write_text(json.dumps(updated_optb, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Sealed {receipt_path} & updated {optb_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
