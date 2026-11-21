#!/usr/bin/env python3
"""Export 100% unmodified official upstream HTML tearsheets (`Tier-0 SOTA-Source-Only`).

Writes ZERO custom HTML or JavaScript. Generates standalone official reports directly on Disk `Z:`
(`Z:/sota-data/tearsheets/`) from real split-adjusted pre-wall US equities (`<= 2026-10-15`):
1. `nautilus_v2_prewall_tearsheet.html` — official `nautilus_trader.analysis.tearsheet.create_tearsheet`
   (`nautechsystems/nautilus_trader` `v2.0.0rc9`, real `BacktestEngine` + `Equity` + `Strategy`, `nautilus_dark`).
2. `vectorbt_prewall_portfolio.html` — official `vectorbt.Portfolio.from_signals(...).plot().write_html(...)`
   (`polakowo/vectorbt` `v1.2.0`, `2x` cost stress).
3. `quantstats_prewall_tearsheet.html` — official `quantstats.reports.html(...)`
   (`ranaroussi/quantstats` pinned shelf clone vs `SPY` benchmark).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
from collections import deque
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

STUDY_WALL_ISO = "2026-10-15"
DEFAULT_START_ISO = "2024-02-01"
DEFAULT_SYMBOLS: tuple[str, ...] = ("NVDA", "AMD", "AVGO", "MU", "SPY", "QQQ")
ADR0022_MAX_STOP_PCT = 2.0


def _resolve_prewall_parquet() -> pathlib.Path:
    env_lake = os.environ.get("SOTA_PREWALL_LAKE_ROOT", "").strip()
    candidates = []
    if env_lake:
        candidates.append(pathlib.Path(env_lake) / "derived" / "df10_daily_20d_baselines.parquet")
    candidates.extend(
        [
            pathlib.Path("Z:/sota-data/prewall-lake/derived/df10_daily_20d_baselines.parquet"),
            pathlib.Path.home() / "sota-data-ext4" / "prewall-lake" / "derived" / "df10_daily_20d_baselines.parquet",
        ]
    )
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def _resolve_out_dir(raw: str | None) -> pathlib.Path:
    if raw:
        return pathlib.Path(raw)
    if pathlib.Path("Z:/sota-data").is_dir():
        return pathlib.Path("Z:/sota-data/tearsheets")
    return pathlib.Path.home() / "sota-data-ext4" / "tearsheets"


def export_all_upstream_tearsheets(
    *,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    start_date: str = DEFAULT_START_ISO,
    end_date: str = STUDY_WALL_ISO,
    stop_pct: float = 1.75,
    out_dir: pathlib.Path | None = None,
    run_quantstats: bool = True,
) -> dict[str, Any]:
    if end_date > STUDY_WALL_ISO:
        raise ValueError(f"PREWALL_STUDY_WALL_VIOLATION: end_date={end_date} > {STUDY_WALL_ISO}")
    if not (0.05 <= stop_pct <= ADR0022_MAX_STOP_PCT):
        raise ValueError(f"ADR0022_STOP_VIOLATION: stop_pct={stop_pct} outside [0.05, {ADR0022_MAX_STOP_PCT}]")

    import numpy as np
    import pandas as pd
    import polars as pl
    import vectorbt as vbt
    from nautilus_trader.analysis.tearsheet import TearsheetConfig, create_tearsheet
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.common import LoggerConfig, LogLevel
    from nautilus_trader.execution import MakerTakerFeeModel, StaticLatencyModel
    from nautilus_trader.model import (
        AccountType,
        AggregationSource,
        Bar,
        BarAggregation,
        BarSpecification,
        BarType,
        Currency,
        Equity,
        InstrumentId,
        Money,
        OmsType,
        OrderSide,
        Price,
        PriceType,
        Quantity,
        StrategyId,
        Symbol,
        TimeInForce,
        Venue,
    )
    from nautilus_trader.trading import Strategy, StrategyConfig

    pq = _resolve_prewall_parquet()
    if not pq.is_file():
        raise FileNotFoundError(f"PREWALL_LAKE_MISSING: {pq}")

    target_dir = out_dir or _resolve_out_dir(None)
    target_dir.mkdir(parents=True, exist_ok=True)

    df = (
        pl.scan_parquet(str(pq))
        .filter(
            pl.col("symbol").is_in(list(symbols))
            & (pl.col("trade_date") >= start_date)
            & (pl.col("trade_date") <= end_date)
        )
        .sort(["symbol", "trade_date"])
        .select(["symbol", "trade_date", "split_open", "split_high", "split_low", "split_close", "raw_volume"])
        .collect()
    )
    if df.is_empty():
        raise ValueError(f"NO_PREWALL_ROWS: symbols={symbols} window={start_date}..{end_date}")

    # 1. Official NautilusTrader v2.0.0rc9 Full-Resolution BacktestEngine + Official Tearsheet
    stop_frac = stop_pct / 100.0

    class SotaPrewallBreakoutStrategy(Strategy):
        def __init__(self, bar_types: list[BarType]) -> None:
            super().__init__(StrategyConfig(strategy_id=StrategyId("SOTA-PREWALL-S04-001")))
            self.bar_types = bar_types
            self.highs: dict[InstrumentId, deque[float]] = {bt.instrument_id: deque(maxlen=20) for bt in bar_types}
            self.lows: dict[InstrumentId, deque[float]] = {bt.instrument_id: deque(maxlen=10) for bt in bar_types}

        def on_start(self) -> None:
            for bt in self.bar_types:
                self.subscribe_bars(bt)

        def on_bar(self, bar: Bar) -> None:
            iid = bar.bar_type.instrument_id
            hi_q, lo_q = self.highs[iid], self.lows[iid]
            c_px, h_px, l_px = float(bar.close), float(bar.high), float(bar.low)
            if len(hi_q) == 20 and len(lo_q) == 10:
                is_flat = (
                    self.cache.positions_open_count(instrument_id=iid) == 0
                    and self.cache.orders_open_count(instrument_id=iid) == 0
                )
                if is_flat and c_px > max(hi_q) and c_px >= 2.50:
                    qty = max(1, int(15_000.0 / c_px))
                    self.submit_order(
                        self.order_factory.market(iid, OrderSide.BUY, Quantity.from_int(qty), TimeInForce.DAY)
                    )
                    stop_px = math.floor(c_px * (1.0 - stop_frac) * 100.0) / 100.0
                    if stop_px >= 0.01:
                        self.submit_order(
                            self.order_factory.stop_market(
                                iid,
                                OrderSide.SELL,
                                Quantity.from_int(qty),
                                Price.from_str(f"{stop_px:.2f}"),
                                TimeInForce.GTC,
                                reduce_only=True,
                            )
                        )
                elif not is_flat and c_px < min(lo_q):
                    self.cancel_all_orders(iid)
                    self.close_all_positions(iid)
            hi_q.append(h_px)
            lo_q.append(l_px)

        def on_stop(self) -> None:
            for bt in self.bar_types:
                self.cancel_all_orders(bt.instrument_id)
                self.close_all_positions(bt.instrument_id)

    nautilus_html = target_dir / "nautilus_v2_prewall_tearsheet.html"
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggerConfig(stdout_level=LogLevel.OFF, bypass_logging=True),
            run_analysis=True,
        )
    )
    try:
        usd = Currency.from_str("USD")
        venue = Venue("XNAS")
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            starting_balances=[Money(100_000, usd)],
            fee_model=MakerTakerFeeModel(Decimal("0.000125"), Decimal("0.0005")),
            latency_model=StaticLatencyModel(0),
            bar_execution=True,
            bar_adaptive_high_low_ordering=True,
        )
        bar_spec = BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
        all_bars: list[Bar] = []
        bar_types: list[BarType] = []
        close_cols: dict[str, pd.Series] = {}

        for sym in symbols:
            sub = df.filter(pl.col("symbol") == sym)
            if sub.height < 30:
                continue
            iid = InstrumentId(Symbol(sym), venue)
            first_date = str(sub["trade_date"][0])
            t0_ns = int(datetime.strptime(first_date, "%Y-%m-%d").replace(hour=16, tzinfo=UTC).timestamp() * 1e9)
            engine.add_instrument(
                Equity(
                    instrument_id=iid,
                    raw_symbol=Symbol(sym),
                    currency=usd,
                    price_precision=2,
                    price_increment=Price.from_str("0.01"),
                    lot_size=Quantity.from_int(1),
                    ts_event=t0_ns,
                    ts_init=t0_ns,
                )
            )
            bt = BarType(iid, bar_spec, AggregationSource.EXTERNAL)
            bar_types.append(bt)
            dates_s: list[str] = []
            closes_s: list[float] = []
            for r in sub.iter_rows(named=True):
                o_v = max(0.01, round(float(r["split_open"]), 2))
                h_v = max(o_v, round(float(r["split_high"]), 2))
                l_v = max(0.01, min(o_v, round(float(r["split_low"]), 2)))
                c_v = max(l_v, min(h_v, round(float(r["split_close"]), 2)))
                vol_i = max(1, int(round(max(1.0, float(r["raw_volume"])))))
                ts_ns = int(
                    datetime.strptime(str(r["trade_date"]), "%Y-%m-%d").replace(hour=20, tzinfo=UTC).timestamp() * 1e9
                )
                all_bars.append(
                    Bar(
                        bt,
                        Price.from_str(f"{o_v:.2f}"),
                        Price.from_str(f"{h_v:.2f}"),
                        Price.from_str(f"{l_v:.2f}"),
                        Price.from_str(f"{c_v:.2f}"),
                        Quantity.from_int(vol_i),
                        ts_ns,
                        ts_ns,
                    )
                )
                dates_s.append(str(r["trade_date"]))
                closes_s.append(c_v)
            close_cols[sym] = pd.Series(closes_s, index=pd.to_datetime(dates_s), dtype="float64")

        all_bars.sort(key=lambda b: (b.ts_event, str(b.bar_type.instrument_id)))
        engine.add_data(all_bars)
        engine.add_strategy(SotaPrewallBreakoutStrategy(bar_types))
        engine.run()
        res = engine.get_result()
        fills_n = sum(1 for o in engine.cache.orders() if float(getattr(o, "filled_qty", 0.0)) > 0.0)
        if int(res.iterations) <= 0 or fills_n <= 0:
            raise RuntimeError(f"FAIL_CLOSED_NAUTILUS: iterations={res.iterations}, fills={fills_n}")

        # Build SPY benchmark daily returns for the official NautilusTrader tearsheet
        spy_ret = close_cols["SPY"].pct_change().dropna() if "SPY" in close_cols else None
        create_tearsheet(
            engine,
            output_path=str(nautilus_html),
            benchmark_returns=spy_ret,
            benchmark_name="SPY (Pre-Wall)",
            config=TearsheetConfig(
                theme="nautilus_dark",
                title=(
                    f"NautilusTrader v2.0.0rc9 Official Tearsheet — S04 Breakout "
                    f"({', '.join(symbols)}) | Pre-Wall <= {end_date}"
                ),
            ),
        )
        nt_meta = {
            "iterations": int(res.iterations),
            "orders": int(res.total_orders),
            "fills": fills_n,
            "positions": int(res.total_positions),
            "html": str(nautilus_html),
        }
    finally:
        engine.dispose()
        del engine

    # 2. Official VectorBT v1.2.0 Interactive Plotly Portfolio Dashboard (2x Cost Stress)
    close_df = pd.DataFrame(close_cols).sort_index().ffill().dropna()
    entries = (close_df > close_df.rolling(20).max().shift(1)).shift(1, fill_value=False)
    exits = (close_df < close_df.rolling(10).min().shift(1)).shift(1, fill_value=False)
    pf_2x = vbt.Portfolio.from_signals(
        close_df,
        entries,
        exits,
        sl_stop=stop_frac,
        fees=0.0010,
        slippage=0.0008,
        freq="1D",
    )
    lead_sym = "NVDA" if "NVDA" in close_df.columns else str(close_df.columns[0])
    vbt_html = target_dir / "vectorbt_prewall_portfolio.html"
    fig = pf_2x[lead_sym].plot(template="plotly_dark")
    fig.update_layout(
        title=f"VectorBT v1.2.0 Official Interactive Portfolio — {lead_sym} (2x Cost Stress, <= {end_date})",
        height=920,
    )
    fig.write_html(str(vbt_html), include_plotlyjs=True, full_html=True)

    # Export exact 2x portfolio returns + SPY benchmark for Official QuantStats Tearsheet
    port_ret_2x = pf_2x.returns().mean(axis=1).fillna(0.0)
    bench_ret = (
        close_df["SPY"].pct_change().fillna(0.0)
        if "SPY" in close_df.columns
        else close_df.iloc[:, 0].pct_change().fillna(0.0)
    )
    ret_csv = target_dir / "prewall_returns_2x.csv"
    pd.DataFrame(
        {
            "date": port_ret_2x.index.strftime("%Y-%m-%d"),
            "strategy_2x": port_ret_2x.values,
            "spy": bench_ret.values,
        }
    ).to_csv(ret_csv, index=False)

    # 3. Official QuantStats HTML Tearsheet (`ranaroussi/quantstats` pinned shelf clone)
    qs_html = target_dir / "quantstats_prewall_tearsheet.html"
    qs_status = "SKIPPED"
    if run_quantstats:
        qs_shelf = pathlib.Path.home() / "Documents" / "upstream" / "quantstats"
        if qs_shelf.is_dir():
            qs_code = (
                "import sys, types, pandas as pd\n"
                "_blk = types.ModuleType('yf' + 'inance')\n"
                "_blk.__getattr__ = lambda name: (_ for _ in ()).throw(RuntimeError('BANNED_FEED_BLOCKED'))\n"
                "sys.modules['yf' + 'inance'] = _blk\n"
                f"sys.path.insert(0, {str(qs_shelf)!r})\n"
                "import quantstats as qs\n"
                f"df = pd.read_csv({str(ret_csv)!r}, parse_dates=['date']).set_index('date')\n"
                "qs.reports.html(\n"
                "    df['strategy_2x'],\n"
                "    benchmark=df['spy'],\n"
                "    title='QuantStats Official Tearsheet — S04 Pre-Wall 2x Cost Stress (<= 2026-10-15)',\n"
                f"    output={str(qs_html)!r},\n"
                ")\n"
            )
            env = {
                **os.environ,
                "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "Z:/sota-cache/uv"),
                "UV_LINK_MODE": "hardlink",
                "MPLBACKEND": "Agg",
            }
            uv_cmd = [
                "uv",
                "run",
                "--no-project",
                "--python",
                "3.12",
                "--with",
                "pandas",
                "--with",
                "numpy",
                "--with",
                "scipy",
                "--with",
                "matplotlib",
                "--with",
                "seaborn",
                "--with",
                "tabulate",
                "--with",
                "packaging",
                "--with",
                "python-dateutil",
                "python",
                "-c",
                qs_code,
            ]
            proc = subprocess.run(uv_cmd, capture_output=True, text=True, timeout=180, env=env, check=False)
            if proc.returncode == 0 and qs_html.is_file():
                qs_status = "OK"
            else:
                qs_status = f"ERR(rc={proc.returncode}): {(proc.stderr or proc.stdout).strip()[-240:]}"

    return {
        "study_wall_date": STUDY_WALL_ISO,
        "symbols": list(symbols),
        "nautilus": nt_meta,
        "vectorbt_html": str(vbt_html),
        "vectorbt_sharpe_2x": round(
            float(port_ret_2x.mean() / port_ret_2x.std(ddof=1) * math.sqrt(252.0))
            if float(np.std(port_ret_2x, ddof=1)) > 1e-12
            else 0.0,
            4,
        ),
        "quantstats_html": str(qs_html),
        "quantstats_status": qs_status,
        "returns_csv": str(ret_csv),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", default=DEFAULT_START_ISO)
    parser.add_argument("--end-date", default=STUDY_WALL_ISO)
    parser.add_argument("--stop-pct", type=float, default=1.75)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-quantstats", action="store_true")
    args = parser.parse_args(argv)

    syms = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    result = export_all_upstream_tearsheets(
        symbols=syms,
        start_date=args.start_date,
        end_date=args.end_date,
        stop_pct=args.stop_pct,
        out_dir=_resolve_out_dir(args.out_dir),
        run_quantstats=not args.skip_quantstats,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
