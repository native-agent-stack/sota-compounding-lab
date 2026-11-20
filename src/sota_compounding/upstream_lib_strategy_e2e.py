"""SOTA Upstream-Library Strategy E2E Sim & Regime-Conditioned Per-Symbol Factor Engine (`ADR-0035`).

Unifies four institutional layers into a zero-dependency, fail-closed, dimensionless core for
`https://github.com/native-agent-stack/sota-compounding-lab`:

1. **4-Tier Canonical Upstream Platform & 12-Strategy Library Registry:**
   Maps the 18 immutable core shelf pins (`~/Documents/upstream/PINS.md`) plus frontier quality-first
   upstreams (`bashtage/arch`, `microsoft/qlib`, `Nixtla/conformalforecast`, `NVIDIA/cudf`) across
   Vectorized/GPU Discovery -> Event-Driven Queue-Parity Sim (`NautilusTrader v2.0.0rc9` + `VectorBT v1.2.0`
   + `hftbacktest RiskAdverseQueueModel`) -> Statistical Falsification (`skfolio`, `ml4t-diagnostic`,
   `statsmodels`, `arch`) -> PIT Execution (`alpaca-py`, `ib_async`, `edgartools`, `perspective`).
2. **3-Horizon Soft Probabilistic Market-Situation Classifier (`classify_market_situation`):**
   Fuses Structural Macro (1-2y cross-asset ratios), Tactical Breadth & VIX Term Structure (2-8w),
   Intraday Soft `H107` Log-Odds (`UVXY`/`SVIX`/beta dispersion), and Session Window (`PRE_MARKET`,
   `RTH_CORE`, `POST_CLOSE`, `OVERNIGHT_BOATS`) with a Cederburg-safe asymmetric volatility crash brake
   strictly capped `<= 1.0x`.
3. **Deterministic Queue-Parity E2E Pre-Wall Bar Simulator (`simulate_queue_parity_e2e_bar_series`):**
   Enforces `STOP_CHECK_BEFORE_MFE_UPDATE` conservative intrabar ordering, Almgren-Chriss `1x`/`2x`
   square-root impact, `ADR-0022 <= 2.00%` Reg NMS Rule 612 tick-quantized stops, cadence-matched
   lossless micro-trails (never choking `DAILY_SWING` runners), and `take_profit_cap_enabled = False`.
4. **Hierarchical Empirical-Bayes Per-Symbol Factor Allocator (`apply_regime_conditioned_per_symbol_factors`):**
   Combines pre-wall per-symbol execution physics (RTH Long, RTH Short, BOATS 24/5 Garman-Klass vol &
   5m pullback quantiles), Hierarchical EB shrinkage toward peer-cluster priors (`Jensen, Kelly &
   Pedersen 2023 JF`, `kappa = 20..40`), Fixed-Share Hedge strategy tilts, and Conformal PID adaptive
   stops strictly inside `[0.05%, 2.00%]`, while hard-refusing live-capital ports (`:4001`, `:7496`,
   `:8796`, `:8798`), quarantined bleeders, `90d` SEC `S-3`/`424B5` diluters on Longs, and `ADR-0032`
   Trial #105 blind BOATS gap-fade shorts.

Governance: `PROTECTIVE_FILTER_OR_GOVERNANCE`, `standalone_alpha_claim = False`,
`hlz_trial_consumed = False`, `paper_only_live_refused = True`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Literal

from sota_compounding.advanced_merit_sim_convergence import (
    ConformalPidStopCalibrator,
    StrategyClusterObservation,
    compute_asymmetric_vol_crash_brake,
    compute_hierarchical_eb_merit_pool,
    update_fixed_share_hedge_weights,
)
from sota_compounding.converged_strategy_factory import (
    check_retired_trap_firewall,
    is_post_ant_strategy_quarantined,
)
from sota_compounding.lifecycle import (
    ADR0022_MAX_STOP_PCT,
    STUDY_WALL_DATE,
    hlz_dynamic_t_hurdle,
    quantize_nms_stop,
)
from sota_compounding.tca_queue_parity import FORBIDDEN_LIVE_LANES, PAPER_ACCOUNT_LANES

UPSTREAM_LIB_E2E_SCHEMA = "sota.upstream-lib-strategy-e2e-regime-persymbol.v1"
UPSTREAM_LIB_E2E_ADR = "ADR-0035"
UPSTREAM_LIB_E2E_VERSION = "1.0.0"

FORBIDDEN_LIVE_LANES_EXTENDED: frozenset[str] = frozenset(
    {
        *FORBIDDEN_LIVE_LANES,
        ":8796",
        "4001",
        "7496",
        "8796",
        "8798",
        "api.alpaca.markets",
        "live",
        "LIVE",
    }
)
ALLOWED_PAPER_VENUE_LANES: frozenset[str] = frozenset(PAPER_ACCOUNT_LANES)

MIN_HARD_STOP_PCT: float = 0.05
DEFAULT_EB_KAPPA: float = 25.0
MIN_PRICE_USD_FILTER: float = 2.50

_VALID_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,6}$")

VALID_SESSIONS: frozenset[str] = frozenset({"PRE_MARKET", "RTH_CORE", "POST_CLOSE", "OVERNIGHT_BOATS"})

SOTA_UPSTREAM_BACKTEST_PLATFORMS: dict[str, tuple[dict[str, str], ...]] = {
    "TIER_1_VECTORIZED_GPU_DISCOVERY": (
        {
            "name": "polars",
            "repo": "pola-rs/polars",
            "pin_sha": "7f902bd3ffd27ff82d14278a09a295ff2f7aed92",
            "version": "py-2.1.0",
            "role": "Zero-copy Parquet pre-wall lake scans & 20d XNYS clock-bucket RVOL profiles",
        },
        {
            "name": "duckdb",
            "repo": "duckdb/duckdb",
            "pin_sha": "d54cbbe3df17e8e5ce68312f483b4155836112aa",
            "version": "v2.0.0",
            "role": "Columnar SQL PIT joins over SIP 1m, FINRA RegSHO T-1, and SEC EDGAR tables",
        },
        {
            "name": "vectorbt",
            "repo": "polakowo/vectorbt",
            "pin_sha": "a52a984b631624bb9daf41cb19f532fe559cf0ef",
            "version": "v1.2.0",
            "role": "Vectorized multi-parameter plateau sweeps & transaction-cost sensitivity grids",
        },
        {
            "name": "qlib",
            "repo": "microsoft/qlib",
            "pin_sha": "54355232463878d2eebb91fe0ee5fa7fa1f5976c",
            "version": "v0.9.8",
            "role": "Cross-sectional Rank-IC factor benchmarks (Alpha158) & DF12 feedback",
        },
        {
            "name": "cudf_torch_rtx4090",
            "repo": "NVIDIA/cudf",
            "pin_sha": "ece1ff92c8e5a61ac87f6cc43ef5cf763659377d",
            "version": "cu132",
            "role": "Noesis RTX 4090 CUDA batched scoring & stationary block bootstrap acceleration",
        },
    ),
    "TIER_2_EVENT_DRIVEN_QUEUE_PARITY_SIM": (
        {
            "name": "nautilus_trader",
            "repo": "nautechsystems/nautilus_trader",
            "pin_sha": "da6b57aefdd62b44f675e434ffcfefca7f4025dd",
            "version": "v2.0.0rc9",
            "role": "Rust-core event-driven BacktestEngine with sim-to-paper strategy code parity",
        },
        {
            "name": "nautilus-alpaca",
            "repo": "gmatrunich/nautilus-alpaca",
            "pin_sha": "59685d56631b981018a189b5ced1a7000a50bfbd",
            "version": "main",
            "role": "Reference Alpaca venue adapter for NautilusTrader paper parity",
        },
        {
            "name": "hftbacktest",
            "repo": "nkaz001/hftbacktest",
            "pin_sha": "5f3ec40b2afb764e0fea112f941ed85523ef4e88",
            "version": "main",
            "role": "RiskAdverseQueueModel passive L1 queue depletion & Glosten-Milgrom adverse selection",
        },
        {
            "name": "exchange_calendars",
            "repo": "gerrymanoim/exchange_calendars",
            "pin_sha": "dbe38b1f6887434bbdd1a7d2df6ff8f1742a048a",
            "version": "4.13.2",
            "role": "Official XNYS session, half-day, and FINRA Rule 4560 T+8 business-day clock",
        },
        {
            "name": "pandera",
            "repo": "unionai-oss/pandera",
            "pin_sha": "02b375ab43f39f8c5c5108695c1791fa04f99a26",
            "version": "v0.35.0",
            "role": "Strict Polars/Arrow DataFrame finite Float64 & study-wall schema validation",
        },
    ),
    "TIER_3_STATISTICAL_FALSIFICATION_PORTFOLIO": (
        {
            "name": "skfolio",
            "repo": "skfolio/skfolio",
            "pin_sha": "90e3fb907fab39d244d899f3caf4e9676889309c",
            "version": "v1.12.7",
            "role": "CombinatorialPurgedCV C(6,2)=15 purged+embargoed splits & drawdown sizing",
        },
        {
            "name": "ml4t-diagnostic",
            "repo": "ml4t/diagnostic",
            "pin_sha": "19e76072e64a18c9ba56130e4a7bcc0bc871f57e",
            "version": "v0.2.0",
            "role": "Deflated Sharpe Ratio (DSR), Bailey PBO, ADIA Lab #19 MinTRL & White RC",
        },
        {
            "name": "statsmodels",
            "repo": "statsmodels/statsmodels",
            "pin_sha": "278ff9950636cdd4939b4055e339a8e681d79cab",
            "version": "v0.15.0",
            "role": "Date-clustered Rogers & Newey-West Bartlett HAC inference on daily MTM legs",
        },
        {
            "name": "arch",
            "repo": "bashtage/arch",
            "pin_sha": "d3e858fc1c99932bb6936cbaf0d1b45b136266e5",
            "version": "v9.0.0",
            "role": "Hansen (2005) SPA & Hansen-Lunde-Nason (2011) MCS stationary block bootstrap",
        },
        {
            "name": "conformalforecast",
            "repo": "Nixtla/conformalforecast",
            "pin_sha": "59922b75adbc377342f68c453ff0d8fc6dcc7532",
            "version": "main",
            "role": "Conformal PID / Adaptive Conformal Inference online stop calibration <= 2.00%",
        },
    ),
    "TIER_4_PIT_FEEDS_PAPER_EXECUTION_BLOTTER": (
        {
            "name": "alpaca-py",
            "repo": "alpacahq/alpaca-py",
            "pin_sha": "c42add7c0a7026e2ead3ed64e76f4def2b99fda4",
            "version": "v0.45.0",
            "role": "DF1 consolidated SIP + DF2 overnight BOATS + paper-only bracket routing",
        },
        {
            "name": "ib_async",
            "repo": "ib-api-reloaded/ib_async",
            "pin_sha": "ab629f34c1823ea4c1542f32f07377208a86bdfd",
            "version": "v2.1.0",
            "role": "IBKR Paper (:4002) outsideRth stops, ETB short-locate & L1 cross-broker parity",
        },
        {
            "name": "edgartools",
            "repo": "dgunning/edgartools",
            "pin_sha": "574c2bd15f6c629bceef03b6550ea8ac8abf3473",
            "version": "v6.2.0",
            "role": "PIT SEC EDGAR acceptanceDateTime 8-K/13D catalysts & 90d S-3/424B5 dilution shield",
        },
        {
            "name": "perspective",
            "repo": "finos/perspective",
            "pin_sha": "0264ec13cd28ef4a2a136c8ef021c3c112bc8990",
            "version": "v5.9.2",
            "role": "Streaming WebAssembly institutional execution, TCA & risk blotter",
        },
        {
            "name": "lightweight-charts",
            "repo": "tradingview/lightweight-charts",
            "pin_sha": "c29c51e1e5bc61981da5832c97797cb8283d120f",
            "version": "master",
            "role": "60fps Canvas candlestick, VWAP & Reg NMS stop-ratchet visualization",
        },
    ),
}

SOTA_UPSTREAM_STRATEGY_LIBRARY: dict[str, dict[str, Any]] = {
    "ZARATTINI_2024_5M_ORB": {
        "strategy_id": "ZARATTINI_2024_5M_ORB",
        "family": "INTRADAY_MOMENTUM",
        "cadence": "INTRADAY_5M",
        "direction": "BIDIRECTIONAL",
        "upstream_engines": ("vectorbt", "nautilus_trader", "polars", "skfolio", "ml4t-diagnostic"),
        "allowed_sessions": ("RTH_CORE",),
        "favored_regimes": ("BROADENING_BULL", "NARROW_CONCENTRATION", "TRANSITIONAL"),
        "suppressed_regimes": ("OVERNIGHT_BOATS",),
        "max_spread_bps": 25.0,
        "base_stop_pct": 1.35,
        "max_stop_pct": 1.60,
        "take_profit_cap_enabled": False,
        "citation": "Zarattini, Barbon & Aziz (2024, SSRN 4729284) — Stocks-in-Play 5m ORB",
    },
    "ZARATTINI_2025_NOISE_VWAP": {
        "strategy_id": "ZARATTINI_2025_NOISE_VWAP",
        "family": "INTRADAY_MOMENTUM",
        "cadence": "INTRADAY_5M",
        "direction": "BIDIRECTIONAL",
        "upstream_engines": ("vectorbt", "nautilus_trader", "polars", "statsmodels", "arch"),
        "allowed_sessions": ("RTH_CORE",),
        "favored_regimes": ("NARROW_CONCENTRATION", "BROADENING_BULL", "HIGH_VOL_STRESS"),
        "suppressed_regimes": ("OVERNIGHT_BOATS",),
        "allowed_symbols": ("SPY", "QQQ", "IWM", "SMH"),
        "max_spread_bps": 5.0,
        "base_stop_pct": 0.45,
        "max_stop_pct": 0.65,
        "take_profit_cap_enabled": False,
        "citation": "Zarattini, Aziz & Barbon (2025, SFI 24-97) — 14d Noise-Area + VWAP 30m",
    },
    "BJZZ_SUBPENNY_INST_DARK_SWEEP": {
        "strategy_id": "BJZZ_SUBPENNY_INST_DARK_SWEEP",
        "family": "MICROSTRUCTURE",
        "cadence": "INTRADAY_5M",
        "direction": "LONG",
        "upstream_engines": ("hftbacktest", "nautilus_trader", "polars", "alpaca-py"),
        "allowed_sessions": ("PRE_MARKET", "RTH_CORE", "POST_CLOSE", "OVERNIGHT_BOATS"),
        "favored_regimes": ("BROADENING_BULL", "NARROW_CONCENTRATION", "OVERNIGHT_BOATS", "TRANSITIONAL"),
        "suppressed_regimes": ("CONTRACTION_RISK_OFF",),
        "max_spread_bps": 28.0,
        "base_stop_pct": 1.10,
        "max_stop_pct": 1.60,
        "take_profit_cap_enabled": False,
        "citation": "Boehmer, Jones, Zhang & Zhang (2021, JF) — Sub-Penny TRF vs Institutional Dark Sweep",
    },
    "H090_TIGHT_SPREAD_HIGH_OFI_PASSIVE_BID": {
        "strategy_id": "H090_TIGHT_SPREAD_HIGH_OFI_PASSIVE_BID",
        "family": "MICROSTRUCTURE",
        "cadence": "INTRADAY_5M",
        "direction": "LONG",
        "upstream_engines": ("hftbacktest", "nautilus_trader", "polars", "statsmodels"),
        "allowed_sessions": ("RTH_CORE", "PRE_MARKET"),
        "favored_regimes": ("BROADENING_BULL", "NARROW_CONCENTRATION", "TRANSITIONAL"),
        "suppressed_regimes": ("CONTRACTION_RISK_OFF", "OVERNIGHT_BOATS"),
        "max_spread_bps": 10.0,
        "base_stop_pct": 0.85,
        "max_stop_pct": 1.25,
        "take_profit_cap_enabled": False,
        "citation": "Cont, Kukanov & Stoikov (2014, JFE) OFI + hftbacktest RiskAdverseQueueModel",
    },
    "H01_PRE_TRAVEL_POCKET_PIVOT": {
        "strategy_id": "H01_PRE_TRAVEL_POCKET_PIVOT",
        "family": "CATALYST_SUPPLY_SHOCK",
        "cadence": "INTRADAY_5M",
        "direction": "LONG",
        "upstream_engines": ("polars", "duckdb", "vectorbt", "edgartools"),
        "allowed_sessions": ("PRE_MARKET", "RTH_CORE", "OVERNIGHT_BOATS"),
        "favored_regimes": ("BROADENING_BULL", "TRANSITIONAL"),
        "suppressed_regimes": ("CONTRACTION_RISK_OFF",),
        "max_spread_bps": 22.0,
        "base_stop_pct": 1.25,
        "max_stop_pct": 1.85,
        "take_profit_cap_enabled": False,
        "citation": "Pre-Travel Early Ignition (calm 0..+5.5% price, RVOL_tod >= 2.2x, Zero-Sell-Pressure)",
    },
    "H08_FIRST_HAND_CATALYST_IGNITION": {
        "strategy_id": "H08_FIRST_HAND_CATALYST_IGNITION",
        "family": "CATALYST_SUPPLY_SHOCK",
        "cadence": "INTRADAY_5M",
        "direction": "LONG",
        "upstream_engines": ("edgartools", "polars", "duckdb", "nautilus_trader"),
        "allowed_sessions": ("PRE_MARKET", "RTH_CORE", "POST_CLOSE", "OVERNIGHT_BOATS"),
        "favored_regimes": ("BROADENING_BULL", "NARROW_CONCENTRATION", "OVERNIGHT_BOATS", "TRANSITIONAL"),
        "suppressed_regimes": (),
        "max_spread_bps": 28.0,
        "base_stop_pct": 1.40,
        "max_stop_pct": 2.00,
        "take_profit_cap_enabled": False,
        "citation": "First-Hand PIT SEC EDGAR 8-K/13D acceptanceDateTime (15m decay + 90d dilution shield)",
    },
    "S03_STOCKBEE_4PCT_BURST_EP": {
        "strategy_id": "S03_STOCKBEE_4PCT_BURST_EP",
        "family": "SWING_COIL",
        "cadence": "DAILY_SWING",
        "direction": "LONG",
        "upstream_engines": ("vectorbt", "skfolio", "ml4t-diagnostic", "arch", "edgartools"),
        "allowed_sessions": ("RTH_CORE",),
        "favored_regimes": ("BROADENING_BULL", "TRANSITIONAL"),
        "suppressed_regimes": ("CONTRACTION_RISK_OFF", "OVERNIGHT_BOATS"),
        "max_spread_bps": 25.0,
        "base_stop_pct": 1.75,
        "max_stop_pct": 2.00,
        "take_profit_cap_enabled": False,
        "citation": "Stockbee 4% Momentum Burst & Episodic Pivot (MCS M* swing survivor)",
    },
    "S04_MINERVINI_VCP_QULLAMAGGIE": {
        "strategy_id": "S04_MINERVINI_VCP_QULLAMAGGIE",
        "family": "SWING_COIL",
        "cadence": "DAILY_SWING",
        "direction": "BIDIRECTIONAL",
        "upstream_engines": ("vectorbt", "skfolio", "ml4t-diagnostic", "arch", "statsmodels"),
        "allowed_sessions": ("RTH_CORE",),
        "favored_regimes": ("BROADENING_BULL", "NARROW_CONCENTRATION", "HIGH_VOL_STRESS"),
        "suppressed_regimes": ("OVERNIGHT_BOATS",),
        "max_spread_bps": 22.0,
        "base_stop_pct": 1.65,
        "max_stop_pct": 2.00,
        "take_profit_cap_enabled": False,
        "citation": "Minervini Stage-2 VCP + Qullamaggie Exhaustion (MCS M* + positive 1x/2x swing)",
    },
    "S09_CONNORS_IBS_CONFORMAL_MR": {
        "strategy_id": "S09_CONNORS_IBS_CONFORMAL_MR",
        "family": "SWING_COIL",
        "cadence": "DAILY_SWING",
        "direction": "LONG",
        "upstream_engines": ("vectorbt", "skfolio", "arch", "conformalforecast", "statsmodels"),
        "allowed_sessions": ("RTH_CORE",),
        "favored_regimes": ("HIGH_VOL_STRESS", "TRANSITIONAL", "BROADENING_BULL"),
        "suppressed_regimes": ("OVERNIGHT_BOATS",),
        "max_spread_bps": 18.0,
        "base_stop_pct": 1.50,
        "max_stop_pct": 2.00,
        "take_profit_cap_enabled": False,
        "citation": "Connors IBS Mean Reversion + Conformal PID Stops (MCS M* + positive 1x/2x swing)",
    },
    "S12_PEAD_DRIFT_COIL_BREAKOUT": {
        "strategy_id": "S12_PEAD_DRIFT_COIL_BREAKOUT",
        "family": "SWING_COIL",
        "cadence": "DAILY_SWING",
        "direction": "LONG",
        "upstream_engines": ("edgartools", "vectorbt", "skfolio", "arch", "statsmodels"),
        "allowed_sessions": ("RTH_CORE",),
        "favored_regimes": ("BROADENING_BULL", "NARROW_CONCENTRATION", "TRANSITIONAL"),
        "suppressed_regimes": ("CONTRACTION_RISK_OFF", "OVERNIGHT_BOATS"),
        "max_spread_bps": 22.0,
        "base_stop_pct": 1.60,
        "max_stop_pct": 2.00,
        "take_profit_cap_enabled": False,
        "citation": "Bernard & Thomas (1989, JAR) Post-Earnings Announcement Drift Coil (MCS M*)",
    },
    "VIX_SPIKE_AND_CRUSH_REGIME_SURFER": {
        "strategy_id": "VIX_SPIKE_AND_CRUSH_REGIME_SURFER",
        "family": "VOLATILITY_REGIME",
        "cadence": "INTRADAY_5M",
        "direction": "BIDIRECTIONAL",
        "upstream_engines": ("vectorbt", "nautilus_trader", "statsmodels", "arch"),
        "allowed_sessions": ("PRE_MARKET", "RTH_CORE", "POST_CLOSE"),
        "favored_regimes": ("HIGH_VOL_STRESS", "CONTRACTION_RISK_OFF"),
        "suppressed_regimes": ("OVERNIGHT_BOATS",),
        "allowed_symbols": ("SPY", "QQQ", "IWM", "UVXY", "VXX", "SVIX", "VIXY", "VXZ", "SQQQ", "SOXS"),
        "max_spread_bps": 18.0,
        "base_stop_pct": 1.00,
        "max_stop_pct": 1.25,
        "take_profit_cap_enabled": False,
        "citation": "VIX Term-Structure Backwardation / Crush Regime Surfer (H107 soft log-odds)",
    },
    "SEC_DILUTION_OVERHANG_ETB_SHORT": {
        "strategy_id": "SEC_DILUTION_OVERHANG_ETB_SHORT",
        "family": "CATALYST_SUPPLY_SHOCK",
        "cadence": "INTRADAY_5M",
        "direction": "SHORT",
        "upstream_engines": ("edgartools", "duckdb", "polars", "ib_async"),
        "allowed_sessions": ("RTH_CORE",),
        "favored_regimes": ("HIGH_VOL_STRESS", "CONTRACTION_RISK_OFF", "NARROW_CONCENTRATION", "TRANSITIONAL"),
        "suppressed_regimes": ("OVERNIGHT_BOATS",),
        "max_spread_bps": 20.0,
        "base_stop_pct": 1.35,
        "max_stop_pct": 1.80,
        "take_profit_cap_enabled": False,
        "citation": "PIT 90d SEC S-3/424B5 Overhang + ETB/SSR Breakdown Short (ADR-0033 uncapped trail)",
    },
}

PEER_CLUSTER_DEFAULT_PRIORS_BPS: dict[str, float] = {
    "MEGA_CAP_TECH": 2.5,
    "HIGH_BETA_SEMI": 1.5,
    "INDEX_CORE_ETF": 3.0,
    "VOL_HEDGE_ETF": 2.0,
    "CATALYST_MIDCAP": 1.0,
    "BROAD_LIQUID": 0.5,
}

_MEGA_CAP_TECH: frozenset[str] = frozenset({"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NFLX", "ORCL", "CRM"})
_HIGH_BETA_SEMI: frozenset[str] = frozenset(
    {"NVDA", "AMD", "AVGO", "MU", "TSM", "ARM", "MRVL", "SMCI", "ASML", "LRCX", "AMAT", "KLAC", "ANET", "INTC", "SNDK"}
)
_INDEX_CORE_ETF: frozenset[str] = frozenset({"SPY", "QQQ", "IWM", "DIA", "SMH", "SOXX", "XLF", "XLE", "XLK", "XLV"})
_VOL_HEDGE_ETF: frozenset[str] = frozenset(
    {"UVXY", "VXX", "VIXY", "VXZ", "SVIX", "SQQQ", "SOXS", "SPXU", "TQQQ", "SOXL", "FNGD", "SH", "SDS", "QID"}
)


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number (got {value!r})")
    try:
        f = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} overflows float64 (got {value!r})") from exc
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite (got {value!r})")
    return 0.0 if f == 0.0 else f


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _validate_paper_venue_lane(venue_lane: str) -> str:
    lane = str(venue_lane or "").strip()
    if not lane:
        raise ValueError("venue_lane must be a non-empty sanctioned paper lane")
    lower = lane.lower()
    if lane in FORBIDDEN_LIVE_LANES_EXTENDED or lower in FORBIDDEN_LIVE_LANES_EXTENDED or "live" in lower:
        raise ValueError(f"FAIL_CLOSED_LIVE_CAPITAL_REFUSAL: forbidden live venue {venue_lane!r}")
    if lane not in ALLOWED_PAPER_VENUE_LANES:
        raise ValueError(f"Unsupported venue_lane {venue_lane!r}; must be one of {sorted(ALLOWED_PAPER_VENUE_LANES)}")
    return lane


def infer_peer_cluster(symbol: str, *, session_dvol_usd: float = 0.0, has_catalyst: bool = False) -> str:
    """Assign a deterministic peer cluster for Hierarchical Empirical-Bayes prior pooling."""
    sym = str(symbol or "").strip().upper()
    if sym in _INDEX_CORE_ETF:
        return "INDEX_CORE_ETF"
    if sym in _VOL_HEDGE_ETF:
        return "VOL_HEDGE_ETF"
    if sym in _HIGH_BETA_SEMI:
        return "HIGH_BETA_SEMI"
    if sym in _MEGA_CAP_TECH:
        return "MEGA_CAP_TECH"
    if has_catalyst or (0.0 < session_dvol_usd < 150_000_000.0):
        return "CATALYST_MIDCAP"
    return "BROAD_LIQUID"


@dataclass(frozen=True, slots=True)
class MarketSituationInput:
    """3-horizon market situation telemetry (Macro 1-2y + Tactical Breadth/VIX + Intraday Soft H107)."""

    session_segment: Literal["PRE_MARKET", "RTH_CORE", "POST_CLOSE", "OVERNIGHT_BOATS"] = "RTH_CORE"
    # Structural Macro (1-2y cross-asset z-scores / spreads)
    rsp_spy_trend_z: float = 0.0
    yield_curve_10y_2y_bps: float = 20.0
    hyg_lqd_trend_z: float = 0.0
    iwm_spy_trend_z: float = 0.0
    spy_tlt_corr: float = -0.20
    xly_xlp_trend_z: float = 0.0
    # Tactical Breadth & VIX Term Structure (2-8w)
    breadth_score_0_100: float = 55.0
    distribution_days_25d: int = 2
    vix_spot: float = 18.0
    vix_9d: float = 17.5
    vix_3m: float = 20.0
    # Intraday Soft H107 Micro-Regime
    uvxy_move_pct: float = 0.0
    svix_move_pct: float = 0.0
    spy_move_pct: float = 0.0
    qqq_move_pct: float = 0.0
    realized_vol_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class MarketSituationAssessment:
    """Classified multi-horizon market regime with Cederburg crash brake and strategy tilts."""

    session_segment: str
    structural_macro_regime: str
    tactical_vix_regime: str
    composite_market_situation: str
    vix_term_ratio_9d_3m: float
    soft_vix_log_odds: float
    cederburg_crash_brake_mult: float
    regime_stop_scale: float
    regime_spread_scale: float
    short_overlay_enabled: bool
    favored_strategies: tuple[str, ...]
    suppressed_strategies: tuple[str, ...]


def classify_market_situation(inp: MarketSituationInput) -> MarketSituationAssessment:
    """Classify multi-horizon market situation with continuous soft log-odds and <= 1.0x crash brake."""
    if inp.session_segment not in VALID_SESSIONS:
        raise ValueError(f"Invalid session_segment: {inp.session_segment!r}")
    rsp_z = _require_finite(inp.rsp_spy_trend_z, "rsp_spy_trend_z")
    yc_bps = _require_finite(inp.yield_curve_10y_2y_bps, "yield_curve_10y_2y_bps")
    hyg_z = _require_finite(inp.hyg_lqd_trend_z, "hyg_lqd_trend_z")
    iwm_z = _require_finite(inp.iwm_spy_trend_z, "iwm_spy_trend_z")
    st_corr = _require_finite(inp.spy_tlt_corr, "spy_tlt_corr")
    xly_z = _require_finite(inp.xly_xlp_trend_z, "xly_xlp_trend_z")
    breadth = _clamp(_require_finite(inp.breadth_score_0_100, "breadth_score_0_100"), 0.0, 100.0)
    if isinstance(inp.distribution_days_25d, bool) or not isinstance(inp.distribution_days_25d, int):
        raise ValueError("distribution_days_25d must be an integer")
    dist_days = max(0, inp.distribution_days_25d)
    vix = max(5.0, _require_finite(inp.vix_spot, "vix_spot"))
    vix9d = max(5.0, _require_finite(inp.vix_9d, "vix_9d"))
    vix3m = max(5.0, _require_finite(inp.vix_3m, "vix_3m"))
    uvxy_m = _require_finite(inp.uvxy_move_pct, "uvxy_move_pct")
    svix_m = _require_finite(inp.svix_move_pct, "svix_move_pct")
    spy_m = _require_finite(inp.spy_move_pct, "spy_move_pct")
    qqq_m = _require_finite(inp.qqq_move_pct, "qqq_move_pct")
    rv_ratio = max(0.25, _require_finite(inp.realized_vol_ratio, "realized_vol_ratio"))

    # 1. Structural Macro Regime (6-component cross-asset logic aligned with macro-regime-detector)
    risk_score = 0.25 * rsp_z + 0.15 * hyg_z + 0.15 * iwm_z + 0.10 * xly_z + (0.20 if yc_bps > 0 else -0.20)
    if hyg_z <= -0.75 and xly_z <= -0.50:
        macro_regime = "CONTRACTION"
    elif st_corr >= 0.45 and hyg_z < 0.25:
        macro_regime = "INFLATIONARY"
    elif rsp_z <= -0.50 and iwm_z <= -0.35 and hyg_z > -0.50:
        macro_regime = "CONCENTRATION"
    elif rsp_z >= 0.35 and iwm_z >= 0.25 and risk_score > 0.20:
        macro_regime = "BROADENING"
    else:
        macro_regime = "TRANSITIONAL"

    # 2. Tactical VIX Term Structure & Breadth Regime
    term_ratio = round(vix9d / vix3m, 6)
    if vix >= 28.0 or term_ratio >= 1.12 or uvxy_m >= 10.0:
        vix_regime = "CRISIS_SPIKE"
    elif vix >= 21.0 or term_ratio >= 1.02 or dist_days >= 5 or uvxy_m >= 5.0:
        vix_regime = "BACKWARDATION_STRESS"
    elif vix >= 16.0 or breadth < 45.0:
        vix_regime = "ELEVATED_TRANSITION"
    else:
        vix_regime = "CALM_CONTANGO"

    # 3. Continuous Soft VIX + Intraday Log-Odds (H107 philosophy: smooth prior, never brittle switch)
    # Positive log-odds => risk-on bullish tilt; negative => risk-off / defensive / short tilt.
    raw_lo = (
        0.03 * (breadth - 50.0)
        - 0.12 * (vix - 18.0)
        - 2.20 * (term_ratio - 0.95)
        - 0.10 * (uvxy_m - svix_m)
        + 0.35 * (0.5 * (spy_m + qqq_m))
        - 0.18 * max(0, dist_days - 3)
    )
    soft_log_odds = round(_clamp(raw_lo, -6.0, 6.0), 6)

    # 4. Cederburg-Safe Asymmetric Volatility Crash Brake (strictly <= 1.0x, never levers up in low vol)
    effective_vol = max(rv_ratio, vix / 18.0, 1.0 + max(0.0, uvxy_m) / 25.0)
    crash_brake_mult = round(
        compute_asymmetric_vol_crash_brake(
            realized_vol_ann_pct=effective_vol * 18.0,
            target_vol_ann_pct=18.0,
            min_scale=0.25,
        ),
        6,
    )

    # 5. Composite Market Situation
    if inp.session_segment == "OVERNIGHT_BOATS":
        composite = "OVERNIGHT_BOATS"
    elif macro_regime == "CONTRACTION" or (vix_regime == "CRISIS_SPIKE" and soft_log_odds <= -2.0):
        composite = "CONTRACTION_RISK_OFF"
    elif vix_regime in ("BACKWARDATION_STRESS", "CRISIS_SPIKE") or soft_log_odds <= -1.0:
        composite = "HIGH_VOL_STRESS"
    elif macro_regime == "CONCENTRATION":
        composite = "NARROW_CONCENTRATION"
    elif macro_regime == "BROADENING" and breadth >= 55.0 and soft_log_odds >= 0.0:
        composite = "BROADENING_BULL"
    else:
        composite = "TRANSITIONAL"

    # Regime stop/spread scaling (Conformal PID still caps hard stop <= 2.00%)
    if composite in ("HIGH_VOL_STRESS", "CONTRACTION_RISK_OFF"):
        stop_scale, spread_scale = 1.15, 0.85
    elif composite == "OVERNIGHT_BOATS":
        stop_scale, spread_scale = 1.05, 1.25
    elif composite == "NARROW_CONCENTRATION":
        stop_scale, spread_scale = 0.92, 0.90
    else:
        stop_scale, spread_scale = 1.00, 1.00

    short_enabled = inp.session_segment == "RTH_CORE" and composite in (
        "HIGH_VOL_STRESS",
        "CONTRACTION_RISK_OFF",
        "NARROW_CONCENTRATION",
        "TRANSITIONAL",
    )

    favored: list[str] = []
    suppressed: list[str] = []
    for sid, spec in SOTA_UPSTREAM_STRATEGY_LIBRARY.items():
        if inp.session_segment not in spec["allowed_sessions"] or composite in spec["suppressed_regimes"]:
            suppressed.append(sid)
        elif composite in spec["favored_regimes"]:
            favored.append(sid)

    return MarketSituationAssessment(
        session_segment=inp.session_segment,
        structural_macro_regime=macro_regime,
        tactical_vix_regime=vix_regime,
        composite_market_situation=composite,
        vix_term_ratio_9d_3m=term_ratio,
        soft_vix_log_odds=soft_log_odds,
        cederburg_crash_brake_mult=crash_brake_mult,
        regime_stop_scale=stop_scale,
        regime_spread_scale=spread_scale,
        short_overlay_enabled=short_enabled,
        favored_strategies=tuple(sorted(favored)),
        suppressed_strategies=tuple(sorted(suppressed)),
    )


def estimate_almgren_chriss_rt_cost_bps(
    *,
    spread_bps: float,
    daily_vol_bps: float,
    participation_ratio: float,
    impact_eta: float = 0.25,
    cost_multiplier: float = 1.0,
) -> float:
    """Almgren-Chriss round-trip slippage model: spread + 2 * eta * sigma * sqrt(Q / V)."""
    sp = max(0.5, _require_finite(spread_bps, "spread_bps"))
    vol = max(5.0, _require_finite(daily_vol_bps, "daily_vol_bps"))
    part = _clamp(_require_finite(participation_ratio, "participation_ratio"), 1e-6, 0.25)
    eta = max(0.0, _require_finite(impact_eta, "impact_eta"))
    mult = max(0.5, _require_finite(cost_multiplier, "cost_multiplier"))
    one_way = 0.5 * sp + eta * vol * math.sqrt(part)
    return round(2.0 * one_way * mult, 6)


def simulate_queue_parity_e2e_bar_series(
    *,
    symbol: str,
    strategy_id: str,
    bars: Sequence[Mapping[str, Any]],
    entry_indices: Sequence[int],
    direction: Literal["LONG", "SHORT"] = "LONG",
    stop_pct: float = 1.25,
    micro_trail_bps: float = 12.0,
    be_arm_bps: float = 20.0,
    be_lock_bps: float = 10.0,
    cadence: Literal["INTRADAY_5M", "DAILY_SWING", "COLLARED_BOATS_LIMIT_1000RPM"] = "INTRADAY_5M",
    max_hold_bars: int = 30,
    spread_bps: float = 4.0,
    daily_vol_bps: float = 180.0,
    participation_ratio: float = 0.002,
    study_wall_cutoff: date = STUDY_WALL_DATE,
) -> dict[str, Any]:
    """Deterministic `hftbacktest`-parity simulator (`STOP_CHECK_BEFORE_MFE_UPDATE`, `1x`/`2x` impact).

    - Hard-refuses any bar dated past `study_wall_cutoff` (`<= 2026-10-15`).
    - Hard-refuses quarantined strategies and `stop_pct > 2.00%` (`ADR-0022`).
    - Enforces `take_profit_cap_enabled = False` (uncapped right-tail Chandelier/lossless runner).
    - Disables tight `6-14 bps` micro-trail choking on `DAILY_SWING` strategies (`ADR-0034` / `ADR-0089`).
    """
    if is_post_ant_strategy_quarantined(strategy_id):
        raise ValueError(f"QUARANTINED_STRATEGY_BLOCKED: {strategy_id}")
    req_stop = _require_finite(stop_pct, "stop_pct")
    if req_stop < MIN_HARD_STOP_PCT or req_stop > ADR0022_MAX_STOP_PCT:
        raise ValueError(
            f"ADR-0022 violation: stop_pct={req_stop} outside [{MIN_HARD_STOP_PCT}, {ADR0022_MAX_STOP_PCT}]"
        )
    if direction not in ("LONG", "SHORT"):
        raise ValueError(f"direction must be LONG or SHORT (got {direction!r})")
    if max_hold_bars < 1:
        raise ValueError("max_hold_bars must be >= 1")

    for i, bar in enumerate(bars):
        b_date = bar.get("date")
        if b_date is not None:
            parsed = date.fromisoformat(str(b_date)[:10])
            if parsed > study_wall_cutoff:
                raise ValueError(
                    f"PREWALL_VIOLATION at bar {i}: {parsed.isoformat()} > {study_wall_cutoff.isoformat()}"
                )

    cost_1x_bps = max(
        8.0 if cadence != "INTRADAY_5M" or spread_bps >= 3.0 else 4.0,
        estimate_almgren_chriss_rt_cost_bps(
            spread_bps=spread_bps,
            daily_vol_bps=daily_vol_bps,
            participation_ratio=participation_ratio,
            cost_multiplier=1.0,
        ),
    )
    cost_2x_bps = round(2.0 * cost_1x_bps, 6)

    # Cadence-matched protection: DAILY_SWING disables tight intraday micro-trail choking
    use_micro_trail = cadence != "DAILY_SWING"
    eff_be_arm = max(float(be_arm_bps), 2.5 * cost_1x_bps) if use_micro_trail else 0.0
    eff_be_lock = max(float(be_lock_bps), 1.5 * cost_1x_bps) if use_micro_trail else 0.0
    eff_trail = max(float(micro_trail_bps), 1.8 * cost_1x_bps) if use_micro_trail else 0.0

    side_lit: Literal["long", "short"] = "long" if direction == "LONG" else "short"
    gross_list: list[float] = []
    net_1x_list: list[float] = []
    net_2x_list: list[float] = []
    green_to_red = 0
    last_exit_idx = -1

    for raw_idx in sorted(set(int(x) for x in entry_indices)):
        if raw_idx <= last_exit_idx or raw_idx < 0 or raw_idx + 1 >= len(bars):
            continue
        entry_bar = bars[raw_idx]
        entry_raw = _require_finite(entry_bar["close"], "bar.close")
        if entry_raw < 0.05:
            continue
        q_entry_d, q_stop_d, _ = quantize_nms_stop(entry_raw, req_stop, side=side_lit)
        entry_px = float(q_entry_d)
        stop_px = float(q_stop_d)
        best_px = entry_px
        was_green = False
        exit_px = entry_px
        exit_idx = min(len(bars) - 1, raw_idx + max_hold_bars)

        for j in range(raw_idx + 1, exit_idx + 1):
            hi = _require_finite(bars[j]["high"], "bar.high")
            lo = _require_finite(bars[j]["low"], "bar.low")
            cl = _require_finite(bars[j]["close"], "bar.close")

            # Strict hftbacktest / ADR-0030 invariant: STOP_CHECK_BEFORE_MFE_UPDATE (zero intrabar lookahead)
            if direction == "LONG":
                if lo <= stop_px:
                    exit_px = stop_px
                    exit_idx = j
                    break
                if hi > best_px:
                    best_px = hi
                mfe_bps = ((best_px - entry_px) / entry_px) * 10_000.0
                if use_micro_trail and mfe_bps >= eff_be_arm:
                    was_green = True
                    be_floor = entry_px * (1.0 + eff_be_lock / 10_000.0)
                    trail_px = best_px * (1.0 - eff_trail / 10_000.0)
                    stop_px = max(stop_px, be_floor, trail_px)
                elif not use_micro_trail and mfe_bps >= req_stop * 100.0:
                    was_green = True
                    stop_px = max(stop_px, entry_px * (1.0 + 0.0010), best_px * (1.0 - req_stop / 100.0))
                if j == exit_idx:
                    exit_px = cl
            else:
                if hi >= stop_px:
                    exit_px = stop_px
                    exit_idx = j
                    break
                if lo < best_px:
                    best_px = lo
                mfe_bps = ((entry_px - best_px) / entry_px) * 10_000.0
                if use_micro_trail and mfe_bps >= eff_be_arm:
                    was_green = True
                    be_ceil = entry_px * (1.0 - eff_be_lock / 10_000.0)
                    trail_px = best_px * (1.0 + eff_trail / 10_000.0)
                    stop_px = min(stop_px, be_ceil, trail_px)
                elif not use_micro_trail and mfe_bps >= req_stop * 100.0:
                    was_green = True
                    stop_px = min(stop_px, entry_px * (1.0 - 0.0010), best_px * (1.0 + req_stop / 100.0))
                if j == exit_idx:
                    exit_px = cl

        last_exit_idx = exit_idx
        gross_bps = (
            ((exit_px - entry_px) / entry_px) * 10_000.0
            if direction == "LONG"
            else ((entry_px - exit_px) / entry_px) * 10_000.0
        )
        n1 = gross_bps - cost_1x_bps
        n2 = gross_bps - cost_2x_bps
        if was_green and n1 < 0.0:
            green_to_red += 1
        gross_list.append(gross_bps)
        net_1x_list.append(n1)
        net_2x_list.append(n2)

    n_ev = len(net_2x_list)
    if n_ev == 0:
        return {
            "symbol": symbol.upper(),
            "strategy_id": strategy_id,
            "direction": direction,
            "cadence": cadence,
            "events_n": 0,
            "cost_1x_bps": round(cost_1x_bps, 4),
            "cost_2x_bps": round(cost_2x_bps, 4),
            "mean_gross_bps": 0.0,
            "mean_1x_bps": 0.0,
            "mean_2x_bps": 0.0,
            "win_rate_2x_pct": 0.0,
            "profit_factor_2x": 0.0,
            "max_dd_2x_pct": 0.0,
            "green_to_red_pct": 0.0,
            "survives_2x_cost": False,
            "take_profit_cap_enabled": False,
            "stop_check_before_mfe_update": True,
        }

    mean_gross = sum(gross_list) / n_ev
    mean_1x = sum(net_1x_list) / n_ev
    mean_2x = sum(net_2x_list) / n_ev
    wins_2x = sum(1 for r in net_2x_list if r > 0.0)
    pos_sum = sum(r for r in net_2x_list if r > 0.0)
    neg_sum = abs(sum(r for r in net_2x_list if r < 0.0))
    pf_2x = round(min(pos_sum / neg_sum, 99.0), 4) if neg_sum > 1e-9 else (99.0 if pos_sum > 0 else 0.0)

    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    for r_bps in net_2x_list:
        eq *= max(0.01, 1.0 + r_bps / 10_000.0)
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    return {
        "symbol": symbol.upper(),
        "strategy_id": strategy_id,
        "direction": direction,
        "cadence": cadence,
        "events_n": n_ev,
        "cost_1x_bps": round(cost_1x_bps, 4),
        "cost_2x_bps": round(cost_2x_bps, 4),
        "mean_gross_bps": round(mean_gross, 4),
        "mean_1x_bps": round(mean_1x, 4),
        "mean_2x_bps": round(mean_2x, 4),
        "win_rate_2x_pct": round(100.0 * wins_2x / n_ev, 2),
        "profit_factor_2x": pf_2x,
        "max_dd_2x_pct": round(100.0 * max_dd, 4),
        "green_to_red_pct": round(100.0 * green_to_red / n_ev, 2),
        "survives_2x_cost": mean_2x > 0.0,
        "take_profit_cap_enabled": False,
        "stop_check_before_mfe_update": True,
    }


@dataclass(frozen=True, slots=True)
class LiveTargetObservation:
    """Dimensionless PIT target observation from whole-market DF1..DF12 live scan."""

    symbol: str
    price: float
    move_pct: float
    rvol_tod: float
    ofi_pct: float
    spread_bps: float
    sip_inst_dark_pct: float = 0.0
    lit_iso_pct: float = 0.0
    atr_room_left_pct: float = 50.0
    zero_sell_score: float = 50.0
    early_potential_score: float = 50.0
    session_dvol_usd: float = 25_000_000.0
    has_sec_8k_catalyst: bool = False
    has_sec_90d_dilution: bool = False
    is_etb_shortable: bool = False
    is_ssr_active: bool = False
    is_boats_gap_fade_window: bool = False
    symbol_empirical_n: int = 0
    symbol_empirical_mean_2x_bps: float = 0.0
    symbol_empirical_se_bps: float = 25.0


@dataclass(frozen=True, slots=True)
class PerSymbolRegimeExecutionPlan:
    """Regime-conditioned, Hierarchically EB-shrunk, ADR-0022-compliant per-symbol strategy plan."""

    symbol: str
    peer_cluster: str
    eligible: bool
    rejection_reasons: tuple[str, ...]
    direction: str
    selected_strategy_id: str
    strategy_family: str
    cadence_protection_mode: str
    market_situation: str
    composite_opportunity_score: float
    eb_posterior_bps: float
    cluster_prior_bps: float
    eb_shrinkage_weight_ratio: float
    hedge_strategy_weight_ratio: float
    entry_limit_px: float
    stop_px: float
    quantized_stop_pct: float
    conformal_pid_stop_pct: float
    micro_be_arm_bps: float
    micro_be_lock_bps: float
    micro_trail_bps: float
    passive_ttl_sec: float
    max_spread_bps: float
    position_size_scale_mult: float
    take_profit_cap_enabled: bool
    adr0022_compliant: bool
    paper_venue_lane: str
    live_capital_refused: bool
    prewall_calibrated: bool


def apply_regime_conditioned_per_symbol_factors(
    *,
    targets: Sequence[LiveTargetObservation],
    market_situation: MarketSituationAssessment,
    per_symbol_prewall_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    cluster_priors_bps: Mapping[str, float] | None = None,
    strategy_hedge_prior: Mapping[str, float] | None = None,
    strategy_recent_rewards_bps: Mapping[str, float] | None = None,
    conformal_breach_rates: Mapping[str, float] | None = None,
    venue_lane: str = ":8787",
    eb_shrinkage_kappa: float = DEFAULT_EB_KAPPA,
) -> list[PerSymbolRegimeExecutionPlan]:
    """Apply pre-wall per-symbol execution physics + Hierarchical EB + Market-Situation routing."""
    sanctioned_lane = _validate_paper_venue_lane(venue_lane)
    kappa = _require_finite(eb_shrinkage_kappa, "eb_shrinkage_kappa")
    if kappa <= 0.0:
        raise ValueError(f"eb_shrinkage_kappa must be > 0 (got {kappa})")

    profiles = per_symbol_prewall_profiles or {}
    priors_map = dict(PEER_CLUSTER_DEFAULT_PRIORS_BPS)
    if cluster_priors_bps:
        for k, v in cluster_priors_bps.items():
            priors_map[str(k)] = _require_finite(v, f"cluster_priors_bps[{k}]")

    # Fixed-Share Hedge update across the canonical strategy library (non-stationary regime adaptation)
    strat_ids = sorted(SOTA_UPSTREAM_STRATEGY_LIBRARY.keys())
    init_weights = (
        {sid: float(strategy_hedge_prior.get(sid, 1.0 / len(strat_ids))) for sid in strat_ids}
        if strategy_hedge_prior
        else {sid: 1.0 / len(strat_ids) for sid in strat_ids}
    )
    rewards_map: dict[str, float] = {}
    for sid in strat_ids:
        base_r = float((strategy_recent_rewards_bps or {}).get(sid, 0.0))
        if sid in market_situation.favored_strategies:
            base_r += 6.0
        elif sid in market_situation.suppressed_strategies:
            base_r -= 12.0
        rewards_map[sid] = _clamp(base_r / 100.0, -1.0, 1.0)
    hedge_weights: dict[str, float] = update_fixed_share_hedge_weights(
        prior_weights=init_weights,
        realized_r_by_strategy=rewards_map,
        eta=0.35,
        switching_alpha=0.08,
    )

    # Hierarchical Empirical-Bayes pool across all observed targets (converted to dimensionless R units)
    eb_observations: list[StrategyClusterObservation] = []
    seen_ids: set[str] = set()
    for t in targets:
        sym = str(t.symbol or "").strip().upper()
        if not sym or sym in seen_ids:
            continue
        seen_ids.add(sym)
        cluster = infer_peer_cluster(
            sym,
            session_dvol_usd=_require_finite(t.session_dvol_usd, "session_dvol_usd"),
            has_catalyst=bool(t.has_sec_8k_catalyst),
        )
        n_obs = max(0, int(t.symbol_empirical_n))
        mean_obs = _require_finite(t.symbol_empirical_mean_2x_bps, "symbol_empirical_mean_2x_bps")
        se_obs = max(1.0, _require_finite(t.symbol_empirical_se_bps, "symbol_empirical_se_bps"))
        if n_obs > 0:
            eb_observations.append(
                StrategyClusterObservation(
                    strategy_id=sym,
                    cluster_id=cluster,
                    observed_mean_r=mean_obs / 100.0,
                    observed_se_r=se_obs / 100.0,
                    closed_roundtrips=n_obs,
                )
            )

    eb_pooled_map: dict[str, dict[str, Any]] = {}
    if eb_observations:
        eb_pooled_map = compute_hierarchical_eb_merit_pool(eb_observations)

    plans: list[PerSymbolRegimeExecutionPlan] = []
    for t in targets:
        sym = str(t.symbol or "").strip().upper()
        price = _require_finite(t.price, f"{sym}.price")
        move_pct = _require_finite(t.move_pct, f"{sym}.move_pct")
        rvol_tod = _require_finite(t.rvol_tod, f"{sym}.rvol_tod")
        ofi_pct = _require_finite(t.ofi_pct, f"{sym}.ofi_pct")
        spread_bps = _require_finite(t.spread_bps, f"{sym}.spread_bps")
        inst_dark = _require_finite(t.sip_inst_dark_pct, f"{sym}.sip_inst_dark_pct")
        lit_iso = _require_finite(t.lit_iso_pct, f"{sym}.lit_iso_pct")
        atr_room = _require_finite(t.atr_room_left_pct, f"{sym}.atr_room_left_pct")
        zero_sell = _require_finite(t.zero_sell_score, f"{sym}.zero_sell_score")
        early_pot = _require_finite(t.early_potential_score, f"{sym}.early_potential_score")
        dvol = _require_finite(t.session_dvol_usd, f"{sym}.session_dvol_usd")
        n_emp = max(0, int(t.symbol_empirical_n))
        mean_emp = _require_finite(t.symbol_empirical_mean_2x_bps, f"{sym}.symbol_empirical_mean_2x_bps")

        cluster = infer_peer_cluster(sym, session_dvol_usd=dvol, has_catalyst=bool(t.has_sec_8k_catalyst))
        prior_bps = float(priors_map.get(cluster, 0.5))

        # Scale-invariant Hierarchical Empirical-Bayes posterior (Jensen, Kelly & Pedersen 2023 JF)
        if n_emp > 0:
            shrink_w = n_emp / (n_emp + kappa)
            if sym in eb_pooled_map:
                hier_eb_bps = float(eb_pooled_map[sym]["hierarchical_eb_mean_r"]) * 100.0
                eb_post = 0.5 * (shrink_w * mean_emp + (1.0 - shrink_w) * prior_bps) + 0.5 * hier_eb_bps
            else:
                eb_post = shrink_w * mean_emp + (1.0 - shrink_w) * prior_bps
        else:
            shrink_w = 0.0
            eb_post = prior_bps

        # Direction resolution strictly aligned with OFI + regime + dilution/shortability
        rejections: list[str] = []
        if not _VALID_SYMBOL_RE.fullmatch(sym):
            rejections.append("INVALID_SYMBOL_FORMAT")
        if price < MIN_PRICE_USD_FILTER:
            rejections.append("PRICE_BELOW_MIN_2_50")
        if spread_bps <= 0.0:
            rejections.append("NON_POSITIVE_SPREAD_TELEMETRY")

        want_short = (
            ofi_pct <= -1.5
            and market_situation.short_overlay_enabled
            and (t.has_sec_90d_dilution or move_pct <= -2.0 or market_situation.soft_vix_log_odds <= -0.75)
        )
        direction: Literal["LONG", "SHORT"] = "SHORT" if want_short else "LONG"

        if direction == "LONG":
            if t.has_sec_90d_dilution and cluster not in ("INDEX_CORE_ETF", "VOL_HEDGE_ETF"):
                rejections.append("SEC_90D_DILUTION_LONG_BLOCK")
            if ofi_pct < 1.5 and zero_sell < 55.0:
                rejections.append("INSUFFICIENT_LONG_OFI_ALIGNMENT")
        else:
            if market_situation.session_segment == "OVERNIGHT_BOATS":
                rejections.append("ALPACA_OVERNIGHT_BOATS_SHORT_FORBIDDEN")
            if t.is_boats_gap_fade_window:
                rejections.append("ADR0032_TRIAL105_BLIND_BOATS_GAP_FADE_SHORT_BLOCK")
            if not t.is_etb_shortable:
                rejections.append("ETB_SHORT_LOCATE_UNVERIFIED")
            if ofi_pct > -1.5:
                rejections.append("INSUFFICIENT_SHORT_OFI_ALIGNMENT")

        # Select best regime-authorized strategy from SOTA_UPSTREAM_STRATEGY_LIBRARY
        best_sid = ""
        best_strat_score = -1e9
        for sid, spec in SOTA_UPSTREAM_STRATEGY_LIBRARY.items():
            if is_post_ant_strategy_quarantined(sid):
                continue
            if market_situation.session_segment not in spec["allowed_sessions"]:
                continue
            if spec["direction"] != "BIDIRECTIONAL" and spec["direction"] != direction:
                continue
            allowed_syms = spec.get("allowed_symbols")
            if allowed_syms is not None and sym not in allowed_syms:
                continue
            if sid in market_situation.suppressed_strategies:
                continue

            s_score = 40.0 * hedge_weights.get(sid, 0.05)
            if sid in market_situation.favored_strategies:
                s_score += 20.0
            if sid == "H08_FIRST_HAND_CATALYST_IGNITION" and t.has_sec_8k_catalyst:
                s_score += 25.0
            if sid == "BJZZ_SUBPENNY_INST_DARK_SWEEP" and inst_dark >= 30.0 and lit_iso >= 15.0:
                s_score += 18.0
            if sid == "H090_TIGHT_SPREAD_HIGH_OFI_PASSIVE_BID" and spread_bps <= 10.0 and ofi_pct >= 5.0:
                s_score += 16.0
            if sid == "H01_PRE_TRAVEL_POCKET_PIVOT" and 0.0 <= move_pct <= 5.5 and rvol_tod >= 2.2:
                s_score += 18.0
            if sid == "ZARATTINI_2025_NOISE_VWAP" and sym in ("SPY", "QQQ", "IWM", "SMH") and spread_bps <= 5.0:
                s_score += 22.0
            if sid == "VIX_SPIKE_AND_CRUSH_REGIME_SURFER" and sym in _VOL_HEDGE_ETF:
                s_score += 24.0
            if sid == "SEC_DILUTION_OVERHANG_ETB_SHORT" and direction == "SHORT" and t.has_sec_90d_dilution:
                s_score += 26.0
            if sid in ("S04_MINERVINI_VCP_QULLAMAGGIE", "S09_CONNORS_IBS_CONFORMAL_MR", "S12_PEAD_DRIFT_COIL_BREAKOUT"):
                s_score += 10.0  # MCS M* structural swing bonus
            if s_score > best_strat_score:
                best_strat_score = s_score
                best_sid = sid

        if not best_sid:
            best_sid = "SEC_DILUTION_OVERHANG_ETB_SHORT" if direction == "SHORT" else "BJZZ_SUBPENNY_INST_DARK_SWEEP"
            rejections.append("NO_REGIME_AUTHORIZED_STRATEGY")

        strat_spec = SOTA_UPSTREAM_STRATEGY_LIBRARY[best_sid]
        firewall_ok, firewall_reasons = check_retired_trap_firewall(tags=[best_sid])
        if not firewall_ok:
            rejections.append(f"RETIRED_TRAP_BLOCKED:{best_sid}:{'|'.join(firewall_reasons)}")

        # Load per-symbol pre-wall calibrated execution physics (ADR-0082 194-symbol manifest)
        sym_prof = profiles.get(sym, {})
        prewall_calibrated = bool(sym_prof)
        if market_situation.session_segment == "OVERNIGHT_BOATS":
            tuned = sym_prof.get("tuned_24x5_boats_near_lossless") or {}
        elif direction == "SHORT":
            tuned = sym_prof.get("tuned_1000rpm_short_near_lossless") or {}
        else:
            tuned = sym_prof.get("tuned_1000rpm_near_lossless") or {}

        max_strat_stop = min(ADR0022_MAX_STOP_PCT, float(strat_spec["max_stop_pct"]))
        base_stop = _clamp(
            float(tuned.get("initial_stop_pct", strat_spec["base_stop_pct"])) * market_situation.regime_stop_scale,
            MIN_HARD_STOP_PCT,
            max_strat_stop,
        )
        max_sp_cap = round(
            min(float(tuned.get("max_spread_bps", strat_spec["max_spread_bps"])), float(strat_spec["max_spread_bps"]))
            * market_situation.regime_spread_scale,
            2,
        )
        if spread_bps > max_sp_cap:
            rejections.append(f"SPREAD_EXCEEDS_COLLAR:{round(spread_bps, 2)}>{max_sp_cap}")

        # Conformal PID Online Stop Calibration strictly inside [0.05%, ADR-0022 2.00%]
        pid = ConformalPidStopCalibrator(
            target_miscoverage_alpha=0.10,
            quantile_multiplier=1.25,
            min_stop_pct=MIN_HARD_STOP_PCT,
            max_stop_pct=max_strat_stop,
        )
        breach_rate = float((conformal_breach_rates or {}).get(sym, 0.10))
        atr_proxy_pct = max(0.10, base_stop / 1.25)
        realized_mae_pct = base_stop * (1.15 if breach_rate > 0.15 else 0.70)
        pid_step = pid.update(realized_mae_pct=realized_mae_pct, atr_pct=atr_proxy_pct)
        conformal_stop_pct = float(pid_step["next_stop_pct"])

        side_lit: Literal["long", "short"] = "long" if direction == "LONG" else "short"
        if price >= 0.05:
            q_entry_d, q_stop_d, realized_stop_pct = quantize_nms_stop(price, conformal_stop_pct, side=side_lit)
            entry_limit_px = float(q_entry_d)
            stop_px = float(q_stop_d)
        else:
            entry_limit_px = price
            stop_px = price
            realized_stop_pct = conformal_stop_pct

        # Cadence-matched lossless trail (DAILY_SWING avoids 6-14 bps micro-trail choke)
        cadence = str(strat_spec["cadence"])
        if cadence == "DAILY_SWING":
            be_arm = 0.0
            be_lock = 0.0
            micro_trail = 0.0
            ttl_sec = 30.0
        else:
            be_arm = _clamp(float(tuned.get("micro_be_arm_bps", 10.0)), 6.0, 45.0)
            be_lock = _clamp(float(tuned.get("micro_be_lock_bps", 3.5)), 2.0, 25.0)
            micro_trail = _clamp(float(tuned.get("micro_trail_bps", 12.0)), 6.0, 35.0)
            ttl_sec = _clamp(float(tuned.get("passive_ttl_sec", 8.0)), 4.0, 25.0)

        # Additive [0, 100] cross-sectional opportunity score
        ofi_align = ofi_pct if direction == "LONG" else -ofi_pct
        score = (
            _clamp(ofi_align * 1.4, 0.0, 25.0)
            + _clamp(math.log1p(max(0.0, rvol_tod)) * 8.5, 0.0, 22.0)
            + _clamp(0.20 * early_pot + 0.15 * zero_sell, 0.0, 20.0)
            + _clamp(0.18 * inst_dark + 0.12 * lit_iso, 0.0, 13.0)
            + _clamp(0.10 * atr_room, 0.0, 10.0)
            + _clamp(5.0 + 0.5 * eb_post, 0.0, 10.0)
            - _clamp(0.45 * max(0.0, spread_bps - 6.0), 0.0, 15.0)
        )
        if t.has_sec_8k_catalyst and direction == "LONG":
            score += 6.0
        composite_score = round(_clamp(score, 0.0, 100.0), 2)

        # Position risk scale: strictly <= 1.0x via Cederburg crash brake & EB confidence
        eb_tilt = _clamp(0.85 + 0.02 * eb_post, 0.50, 1.00)
        pos_scale = round(_clamp(market_situation.cederburg_crash_brake_mult * eb_tilt, 0.15, 1.00), 4)

        plans.append(
            PerSymbolRegimeExecutionPlan(
                symbol=sym,
                peer_cluster=cluster,
                eligible=len(rejections) == 0,
                rejection_reasons=tuple(rejections),
                direction=direction,
                selected_strategy_id=best_sid,
                strategy_family=str(strat_spec["family"]),
                cadence_protection_mode=cadence,
                market_situation=market_situation.composite_market_situation,
                composite_opportunity_score=composite_score,
                eb_posterior_bps=round(eb_post, 4),
                cluster_prior_bps=round(prior_bps, 4),
                eb_shrinkage_weight_ratio=round(shrink_w, 4),
                hedge_strategy_weight_ratio=round(hedge_weights.get(best_sid, 0.0), 6),
                entry_limit_px=round(entry_limit_px, 4),
                stop_px=round(stop_px, 4),
                quantized_stop_pct=round(realized_stop_pct, 4),
                conformal_pid_stop_pct=round(conformal_stop_pct, 4),
                micro_be_arm_bps=round(be_arm, 2),
                micro_be_lock_bps=round(be_lock, 2),
                micro_trail_bps=round(micro_trail, 2),
                passive_ttl_sec=round(ttl_sec, 2),
                max_spread_bps=max_sp_cap,
                position_size_scale_mult=pos_scale,
                take_profit_cap_enabled=False,
                adr0022_compliant=(0.0 < realized_stop_pct <= ADR0022_MAX_STOP_PCT),
                paper_venue_lane=sanctioned_lane,
                live_capital_refused=True,
                prewall_calibrated=prewall_calibrated,
            )
        )

    plans.sort(key=lambda p: (not p.eligible, -p.composite_opportunity_score, p.symbol))
    return plans


def compute_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 over deterministic manifest fields (excluding volatile timestamps)."""
    base = {k: v for k, v in payload.items() if k not in ("manifest_sha256", "generated_at_utc")}
    canonical = json.dumps(base, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_upstream_lib_e2e_manifest(
    *,
    market_situation: MarketSituationAssessment,
    target_plans: Sequence[PerSymbolRegimeExecutionPlan],
    upstream_library_runs: Mapping[str, Any],
    prewall_sim_summary: Sequence[Mapping[str, Any]],
    cumulative_trials_n: int = 109,
    generated_at_utc: str = "",
    strict_full_resolution: bool = False,
) -> dict[str, Any]:
    """Assemble and SHA-256-seal the dimensionless E2E Upstream-Library & Regime Per-Symbol Manifest."""
    fr_audit = enforce_full_resolution_zero_fallback_gate(upstream_library_runs, require_all_engines=True)
    if strict_full_resolution and not fr_audit["full_resolution_verified"]:
        raise ValueError("FAIL_CLOSED_FULL_RESOLUTION_VIOLATION: " + "; ".join(fr_audit["violations"]))

    hlz_next = hlz_dynamic_t_hurdle(max(1, int(cumulative_trials_n) + 1))
    eligible_plans = [p for p in target_plans if p.eligible]
    any_2x_pass = any(bool(r.get("survives_2x_cost")) for r in prewall_sim_summary)

    payload: dict[str, Any] = {
        "schema": UPSTREAM_LIB_E2E_SCHEMA,
        "adr": UPSTREAM_LIB_E2E_ADR,
        "version": UPSTREAM_LIB_E2E_VERSION,
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "generated_at_utc": generated_at_utc or datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "governance": {
            "promotion_track": "PROTECTIVE_FILTER_OR_GOVERNANCE",
            "standalone_alpha_claim": False,
            "hlz_trial_consumed": False,
            "cumulative_trials_n": int(cumulative_trials_n),
            "next_hlz_t_hurdle": hlz_next,
            "adr0022_max_stop_pct": ADR0022_MAX_STOP_PCT,
            "take_profit_cap_enabled": False,
            "paper_only_live_refused": True,
            "forbidden_live_ports": [4001, 7496, 8796, 8798],
            "dimensionless_privacy_firewall": True,
            "full_resolution_only": True,
            "allow_silent_fallback": False,
        },
        "full_resolution_audit": fr_audit,
        "upstream_platforms": {
            tier: [dict(item) for item in items] for tier, items in SOTA_UPSTREAM_BACKTEST_PLATFORMS.items()
        },
        "strategy_library_count": len(SOTA_UPSTREAM_STRATEGY_LIBRARY),
        "strategy_library": {
            sid: {
                "family": spec["family"],
                "cadence": spec["cadence"],
                "direction": spec["direction"],
                "upstream_engines": list(spec["upstream_engines"]),
                "allowed_sessions": list(spec["allowed_sessions"]),
                "max_spread_bps": spec["max_spread_bps"],
                "max_stop_pct": spec["max_stop_pct"],
                "take_profit_cap_enabled": spec["take_profit_cap_enabled"],
                "citation": spec["citation"],
            }
            for sid, spec in SOTA_UPSTREAM_STRATEGY_LIBRARY.items()
        },
        "upstream_library_runs": dict(upstream_library_runs),
        "prewall_e2e_sim_summary": {
            "strategies_evaluated_n": len(prewall_sim_summary),
            "any_survives_2x_cost": any_2x_pass,
            "rows": [dict(r) for r in prewall_sim_summary],
        },
        "live_market_situation": asdict(market_situation),
        "live_target_routing": {
            "total_evaluated_n": len(target_plans),
            "eligible_n": len(eligible_plans),
            "blocked_n": len(target_plans) - len(eligible_plans),
            "long_eligible_n": sum(1 for p in eligible_plans if p.direction == "LONG"),
            "short_eligible_n": sum(1 for p in eligible_plans if p.direction == "SHORT"),
            "prewall_calibrated_n": sum(1 for p in target_plans if p.prewall_calibrated),
            "plans": [asdict(p) for p in target_plans],
        },
    }
    payload["manifest_sha256"] = compute_manifest_sha256(payload)
    # Fail-closed JSON round-trip check guaranteeing zero NaN/Inf
    json.dumps(payload, allow_nan=False)
    return payload


def render_upstream_lib_e2e_markdown(manifest: Mapping[str, Any]) -> str:
    """Render concise dimensionless Markdown audit receipt for the E2E Upstream-Library manifest."""
    gov = manifest["governance"]
    sit = manifest["live_market_situation"]
    routing = manifest["live_target_routing"]
    sim = manifest["prewall_e2e_sim_summary"]
    runs = manifest["upstream_library_runs"]

    lines: list[str] = [
        "# SOTA Upstream-Library Strategy E2E Sim & Regime-Conditioned Per-Symbol Manifest (`ADR-0035`)",
        "",
        (
            f"- **Schema:** `{manifest['schema']}` · **Version:** `{manifest['version']}` · "
            f"**Study Wall:** `<= {manifest['study_wall_date']}`"
        ),
        f"- **SHA-256 Digest:** `{manifest['manifest_sha256']}`",
        (
            f"- **Governance:** `{gov['promotion_track']}` · "
            f"`standalone_alpha_claim={gov['standalone_alpha_claim']}` · "
            f"`hlz_trial_consumed={gov['hlz_trial_consumed']}` · "
            f"Next HLZ `t*({gov['cumulative_trials_n'] + 1}) = {gov['next_hlz_t_hurdle']}`"
        ),
        (
            f"- **Safety Invariants:** `ADR-0022 <= {gov['adr0022_max_stop_pct']:.2f}%` · "
            f"`take_profit_cap_enabled={gov['take_profit_cap_enabled']}` · "
            f"`paper_only_live_refused={gov['paper_only_live_refused']}` "
            f"(live ports `{gov['forbidden_live_ports']}` refused)"
        ),
        (
            f"- **Full-Resolution Backbone (`ADR-0038`):** "
            f"`full_resolution_verified={manifest.get('full_resolution_audit', {}).get('full_resolution_verified')}` · "
            f"`allow_silent_fallback=False` · "
            f"`residual_fallbacks_count="
            f"{manifest.get('full_resolution_audit', {}).get('residual_fallbacks_count', 0)}` · "
            f"`engines_verified={manifest.get('full_resolution_audit', {}).get('engines_verified_n', 0)}/"
            f"{manifest.get('full_resolution_audit', {}).get('required_engines_n', 0)}`"
        ),
        "",
        "## 1. Live Multi-Horizon Market Situation",
        "",
        (
            "| Session | Macro Regime | Tactical VIX Regime | Composite Situation | "
            "VIX 9d/3m | Soft Log-Odds | Cederburg Brake | Short Overlay |"
        ),
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |",
        (
            f"| `{sit['session_segment']}` | `{sit['structural_macro_regime']}` | "
            f"`{sit['tactical_vix_regime']}` | `{sit['composite_market_situation']}` | "
            f"`{sit['vix_term_ratio_9d_3m']:.4f}` | `{sit['soft_vix_log_odds']:+.4f}` | "
            f"`{sit['cederburg_crash_brake_mult']:.4f}x` | `{sit['short_overlay_enabled']}` |"
        ),
        "",
        "## 2. Executed Pinned SOTA Upstream Libraries",
        "",
        "| Upstream Library | Executed | Version / Engine | Key Pre-Wall Verification Result |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for lib_name, info in sorted(runs.items()):
        if not isinstance(info, Mapping):
            continue
        lines.append(
            f"| `{lib_name}` | `{info.get('executed', False)}` | `{info.get('version', 'n/a')}` | "
            f"{info.get('summary', 'n/a')} |"
        )

    lines.extend(
        [
            "",
            "## 3. Real Pre-Wall (`<= 2026-10-15`) Queue-Parity E2E Simulation Summary",
            "",
            (
                "| Symbol | Strategy | Dir | Cadence | Events | 1x Cost bps | "
                "Mean 1x bps | Mean 2x bps | WR 2x % | PF 2x | Max DD 2x % |"
            ),
            "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sim.get("rows", [])[:25]:
        lines.append(
            f"| `{row['symbol']}` | `{row['strategy_id']}` | `{row['direction']}` | `{row['cadence']}` | "
            f"{row['events_n']} | {row['cost_1x_bps']:.2f} | {row['mean_1x_bps']:+.2f} | {row['mean_2x_bps']:+.2f} | "
            f"{row['win_rate_2x_pct']:.1f}% | {row['profit_factor_2x']:.2f} | {row['max_dd_2x_pct']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## 4. Live Regime-Conditioned Per-Symbol Factor & Strategy Routing",
            "",
            (
                f"- **Evaluated:** `{routing['total_evaluated_n']}` · **Eligible:** `{routing['eligible_n']}` "
                f"(Long `{routing['long_eligible_n']}`, Short `{routing['short_eligible_n']}`) · "
                f"**Blocked:** `{routing['blocked_n']}` · "
                f"**Pre-Wall Calibrated:** `{routing['prewall_calibrated_n']}`"
            ),
            "",
            (
                "| Symbol | Cluster | Eligible | Dir | Strategy | Score | "
                "EB Post bps | Stop % | Trail bps | Size Scale | Paper Lane | Reasons |"
            ),
            "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |",
        ]
    )
    for p in routing.get("plans", [])[:30]:
        reasons = ", ".join(p["rejection_reasons"]) if p["rejection_reasons"] else "OK"
        lines.append(
            f"| `{p['symbol']}` | `{p['peer_cluster']}` | `{p['eligible']}` | `{p['direction']}` | "
            f"`{p['selected_strategy_id']}` | {p['composite_opportunity_score']:.1f} | "
            f"{p['eb_posterior_bps']:+.2f} | {p['quantized_stop_pct']:.2f}% | "
            f"{p['micro_trail_bps']:.1f} | {p['position_size_scale_mult']:.2f}x | "
            f"`{p['paper_venue_lane']}` | `{reasons}` |"
        )
    lines.append("")
    return "\n".join(lines)


TIER0_FORBIDDEN_NON_SOTA_SIM_UPSTREAMS: frozenset[str] = frozenset(
    {
        "",
        "cpython-stdlib",
        "stdlib",
        "custom",
        "self-built",
        "self-invented",
        "homegrown",
        "ad-hoc",
    }
)

TIER0_PURGED_SELF_INVENTED_SIM_STRATEGIES: frozenset[str] = frozenset(
    {
        "WAVE2_HIGH_VOLUME_REENTRY",
        "BLIND_WAVE2_REENTRY",
        "RAPID_WAVE2_REENTRY_STRESS",
        "IBKR_P4_DISCIPLINED_WAVE2_PULLBACK",
        "FLOW_VELOCITY_CAPITAL_RECYCLE",
        "DEAD_FLOW_CAPITAL_RECYCLED_FOR_HIGH_VELOCITY_MOVER",
        "SOTA_DUAL_CLIMAX_STOP_AND_REVERSE",
        "ATOMIC_DUAL_CLIMAX_STOP_AND_REVERSE_FLIP",
        "SOTA_DUAL_EXTREME_BALANCE_BREAK_EARLY",
        "POST_VIX_REVERSAL",
        "POST_VIX_REVERSAL_MAJOR_GROWTH_THRUST",
        "VIX_REGIME_ETB_VWAP_FAIL_SHORT",
        "ADR0033_CHAMPION_SPIKE_SHORT",
        "S14_SOTA_EARLY_MOVER_BRACKET",
        "AD_HOC_EXT_SPREAD_CROSSING_COLLAR",
    }
)


def enforce_tier0_sota_source_only_sim_gate(
    *,
    strategy_id: str,
    upstream_repo: str,
    upstream_pin_sha: str,
    primary_literature: str,
) -> dict[str, Any]:
    """Fail-closed Tier-0 SOTA-Source-Only Sim gate (`ADR-0037`): ZERO self-built/self-invented strategies."""
    sid = str(strategy_id or "").strip().upper()
    repo = str(upstream_repo or "").strip()
    pin = str(upstream_pin_sha or "").strip()
    lit = str(primary_literature or "").strip()
    reasons: list[str] = []

    pinned_repos = {
        str(entry["repo"])
        for tier_entries in SOTA_UPSTREAM_BACKTEST_PLATFORMS.values()
        for entry in tier_entries
        if isinstance(entry, Mapping) and entry.get("repo")
    } | {"finra-sec-official-spec"}

    if not sid:
        reasons.append("TIER0_EMPTY_STRATEGY_ID")
    if sid in TIER0_PURGED_SELF_INVENTED_SIM_STRATEGIES or is_post_ant_strategy_quarantined(sid):
        reasons.append(f"TIER0_PURGED_SELF_INVENTED_OR_QUARANTINED:{sid}")
    if repo.lower() in TIER0_FORBIDDEN_NON_SOTA_SIM_UPSTREAMS:
        reasons.append(f"TIER0_FORBIDDEN_NON_SOTA_UPSTREAM:{repo}")
    elif repo not in pinned_repos:
        reasons.append(f"TIER0_UNPINNED_SIM_UPSTREAM_REPO:{repo}")
    if len(pin) < 6:
        reasons.append("TIER0_MISSING_UPSTREAM_PIN_SHA")
    if len(lit) < 20:
        reasons.append("TIER0_MISSING_PRIMARY_LITERATURE")

    allowed = len(reasons) == 0
    return {
        "schema": "sota.tier0-sim-sota-source-only-gate.v1",
        "adr": "ADR-0037",
        "strategy_id": sid,
        "allowed": allowed,
        "upstream_repo": repo,
        "upstream_pin_sha": pin,
        "primary_literature": lit,
        "rejection_reasons": reasons,
    }


FULL_RESOLUTION_RULE_SCHEMA = "sota.full-resolution-zero-fallback-rule.v1"
FULL_RESOLUTION_RULE_ADR = "ADR-0038"

REQUIRED_FULL_RESOLUTION_ENGINES: tuple[str, ...] = (
    "polars_duckdb_pandera_xnys",
    "nautilus_trader",
    "hftbacktest",
    "vectorbt",
    "statsmodels",
    "skfolio",
    "ml4t_diagnostic",
    "bashtage_arch",
    "rtx4090_gpu_cuda",
)

FORBIDDEN_RESIDUAL_OR_FALLBACK_RUN_KEYS: frozenset[str] = frozenset(
    {
        "hftbacktest_queue_parity_ref",
        "cpu_fallback",
        "python_bar_loop_fallback",
        "synthetic_smoke",
        "nautilus_boot_only",
        "vectorbt_nautilus",
        "econometric_battery",
    }
)

_FORBIDDEN_VERSION_TOKENS: frozenset[str] = frozenset(
    {"", "n/a", "none", "unavailable", "fallback", "stub", "mock", "synthetic"}
)


def _pos_int(val: Any) -> int:
    if isinstance(val, bool) or val is None:
        return 0
    try:
        iv = int(val)
    except (TypeError, ValueError):
        return 0
    return iv if iv > 0 else 0


def enforce_full_resolution_zero_fallback_gate(
    upstream_library_runs: Mapping[str, Any],
    *,
    require_all_engines: bool = True,
) -> dict[str, Any]:
    """Fail-closed Tier-0 Full-Resolution, Always-Full-Resolution, Zero-Residual, Zero-Fallback Gate (`ADR-0038`).

    Permanently blocks:
    - Empty `NautilusTrader` `BacktestEngine` boot checks (`iterations=0`, missing instruments/data/strategies/fills).
    - Pure-Python residual bar loops masquerading as `hftbacktest` (`hftbacktest_queue_parity_ref` or non-Rust engine).
    - Silent CPU / synthetic / degraded fallbacks (`fallback_used=True`, `synthetic=True`, `degraded=True`).
    - Missing GPU CUDA verification on `rtx4090_gpu_cuda` (`device` must start with `cuda`).
    - Any post-wall (`> 2026-10-15`) data contamination.
    """
    violations: list[str] = []
    residual_fallbacks_count = 0
    engines_verified: list[str] = []

    if not isinstance(upstream_library_runs, Mapping) or not upstream_library_runs:
        violations.append("FULL_RES_EMPTY_UPSTREAM_RUNS")
        return {
            "schema": FULL_RESOLUTION_RULE_SCHEMA,
            "adr": FULL_RESOLUTION_RULE_ADR,
            "full_resolution_verified": False,
            "allow_silent_fallback": False,
            "residual_fallbacks_count": 1,
            "engines_verified_n": 0,
            "required_engines_n": len(REQUIRED_FULL_RESOLUTION_ENGINES),
            "engines_verified": [],
            "violations": violations,
        }

    for key in sorted(upstream_library_runs.keys()):
        if key in FORBIDDEN_RESIDUAL_OR_FALLBACK_RUN_KEYS:
            residual_fallbacks_count += 1
            violations.append(f"FORBIDDEN_RESIDUAL_OR_FALLBACK_ENGINE:{key}")

    if require_all_engines:
        for req_eng in REQUIRED_FULL_RESOLUTION_ENGINES:
            if req_eng not in upstream_library_runs:
                violations.append(f"MISSING_REQUIRED_FULL_RES_ENGINE:{req_eng}")

    for eng_name, raw_entry in sorted(upstream_library_runs.items()):
        if eng_name in FORBIDDEN_RESIDUAL_OR_FALLBACK_RUN_KEYS:
            continue
        if not isinstance(raw_entry, Mapping):
            violations.append(f"MALFORMED_ENGINE_ENTRY:{eng_name}")
            continue

        entry_ok = True
        if raw_entry.get("executed") is not True:
            entry_ok = False
            violations.append(f"ENGINE_NOT_EXECUTED:{eng_name}")

        for flag_key in ("fallback_used", "synthetic", "degraded", "residual_loop_used"):
            if raw_entry.get(flag_key) is True:
                entry_ok = False
                residual_fallbacks_count += 1
                violations.append(f"FORBIDDEN_FALLBACK_OR_SYNTHETIC_FLAG:{eng_name}:{flag_key}")

        ver = str(raw_entry.get("version") or "").strip().lower()
        if ver in _FORBIDDEN_VERSION_TOKENS or "fallback" in ver or "err" in ver:
            entry_ok = False
            violations.append(f"INVALID_OR_FALLBACK_VERSION:{eng_name}:{ver}")

        summary_low = str(raw_entry.get("summary") or "").lower()
        if "iterations=0" in summary_low or "boot_error" in summary_low or "fallback" in summary_low:
            entry_ok = False
            residual_fallbacks_count += 1
            violations.append(f"FORBIDDEN_STUB_OR_FALLBACK_SUMMARY:{eng_name}")

        post_wall = raw_entry.get("post_wall_n")
        if isinstance(post_wall, int) and not isinstance(post_wall, bool) and post_wall > 0:
            entry_ok = False
            violations.append(f"POST_WALL_CONTAMINATION:{eng_name}:post_wall_n={post_wall}")

        # Engine-specific full-resolution execution physics verification
        if eng_name == "nautilus_trader":
            if _pos_int(raw_entry.get("iterations")) <= 0:
                entry_ok = False
                residual_fallbacks_count += 1
                violations.append("NAUTILUS_ZERO_ITERATIONS_BOOT_STUB")
            if _pos_int(raw_entry.get("instruments_n")) < 1:
                entry_ok = False
                violations.append("NAUTILUS_ZERO_INSTRUMENTS")
            if _pos_int(raw_entry.get("bars_or_ticks_loaded")) <= 0:
                entry_ok = False
                violations.append("NAUTILUS_ZERO_BARS_OR_TICKS_LOADED")
            if _pos_int(raw_entry.get("strategies_n")) < 1:
                entry_ok = False
                violations.append("NAUTILUS_ZERO_STRATEGIES")
            if _pos_int(raw_entry.get("orders_total")) <= 0 or _pos_int(raw_entry.get("fills_total")) <= 0:
                entry_ok = False
                violations.append("NAUTILUS_ZERO_ORDERS_OR_FILLS")

        elif eng_name == "hftbacktest":
            if str(raw_entry.get("engine_kind") or "") != "RUST_HFTBACKTEST_HASHMAP_DEPTH":
                entry_ok = False
                residual_fallbacks_count += 1
                violations.append("HFTBACKTEST_NON_RUST_OR_RESIDUAL_ENGINE")
            if str(raw_entry.get("queue_model") or "") != "RiskAdverseQueueModel":
                entry_ok = False
                violations.append("HFTBACKTEST_MISSING_RISK_ADVERSE_QUEUE_MODEL")
            if _pos_int(raw_entry.get("slices_replayed")) < 1 or _pos_int(raw_entry.get("events_replayed")) <= 0:
                entry_ok = False
                violations.append("HFTBACKTEST_ZERO_REAL_L1_EVENTS")

        elif eng_name == "polars_duckdb_pandera_xnys":
            if _pos_int(raw_entry.get("sessions_n")) < 60:
                entry_ok = False
                violations.append("LAKE_INSUFFICIENT_PREWALL_SESSIONS")
            if _pos_int(raw_entry.get("total_1m_bars_aggregated")) <= 0:
                entry_ok = False
                violations.append("LAKE_ZERO_1M_BARS")
            if raw_entry.get("non_xnys_n", 0) != 0:
                entry_ok = False
                violations.append("LAKE_NON_XNYS_SESSIONS_PRESENT")

        elif eng_name == "vectorbt":
            plateaus = raw_entry.get("plateau_results")
            if not isinstance(plateaus, Sequence) or isinstance(plateaus, (str, bytes)) or len(plateaus) < 1:
                entry_ok = False
                violations.append("VECTORBT_EMPTY_PLATEAU_RESULTS")

        elif eng_name == "skfolio":
            if _pos_int(raw_entry.get("cpcv_splits_n")) < 10:
                entry_ok = False
                violations.append("SKFOLIO_INSUFFICIENT_CPCV_SPLITS")

        elif eng_name == "ml4t_diagnostic":
            dsr_v = raw_entry.get("dsr_probability")
            wrc_v = raw_entry.get("whites_rc_p")
            if not (
                isinstance(dsr_v, (int, float))
                and not isinstance(dsr_v, bool)
                and math.isfinite(float(dsr_v))
                and isinstance(wrc_v, (int, float))
                and not isinstance(wrc_v, bool)
                and math.isfinite(float(wrc_v))
            ):
                entry_ok = False
                violations.append("ML4T_DIAGNOSTIC_NON_FINITE_METRICS")

        elif eng_name == "bashtage_arch":
            spa_v = raw_entry.get("spa_p_consistent")
            mcs_set = raw_entry.get("mcs_surviving_set")
            if not (
                isinstance(spa_v, (int, float))
                and not isinstance(spa_v, bool)
                and math.isfinite(float(spa_v))
                and isinstance(mcs_set, Sequence)
                and not isinstance(mcs_set, (str, bytes))
                and len(mcs_set) >= 1
            ):
                entry_ok = False
                violations.append("ARCH_INVALID_SPA_OR_MCS")

        elif eng_name == "rtx4090_gpu_cuda":
            dev = str(raw_entry.get("device") or "").lower()
            if not dev.startswith("cuda"):
                entry_ok = False
                residual_fallbacks_count += 1
                violations.append(f"GPU_CUDA_REQUIRED_NO_CPU_FALLBACK:{dev}")
            if _pos_int(raw_entry.get("n_boot")) < 1000:
                entry_ok = False
                violations.append("GPU_BOOTSTRAP_INSUFFICIENT_REPS")

        if entry_ok and eng_name in REQUIRED_FULL_RESOLUTION_ENGINES:
            engines_verified.append(eng_name)

    return {
        "schema": FULL_RESOLUTION_RULE_SCHEMA,
        "adr": FULL_RESOLUTION_RULE_ADR,
        "full_resolution_verified": len(violations) == 0,
        "allow_silent_fallback": False,
        "residual_fallbacks_count": residual_fallbacks_count,
        "engines_verified_n": len(engines_verified),
        "required_engines_n": len(REQUIRED_FULL_RESOLUTION_ENGINES),
        "engines_verified": sorted(engines_verified),
        "violations": violations,
    }


def _collect_call_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


def audit_python_source_for_full_resolution(source_text: str, rel_path: str) -> list[str]:
    """Static AST audit blocking empty `BacktestEngine` boot stubs (including embedded inline scripts)."""
    issues: list[str] = []
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return issues

    candidate_trees: list[tuple[str, ast.AST]] = [(rel_path, tree)]
    # Also inspect inline Python strings passed to subprocesses (e.g., `code = r\"\"\"...\"\"\"`)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "BacktestEngine(" in node.value:
            try:
                sub_tree = ast.parse(node.value)
                candidate_trees.append((f"{rel_path}<inline>", sub_tree))
            except SyntaxError:
                pass

    for label, sub in candidate_trees:
        calls = _collect_call_names(sub)
        if "BacktestEngine" in calls and "run" in calls:
            missing = [req for req in ("add_instrument", "add_data", "add_strategy") if req not in calls]
            if missing:
                issues.append(f"{label}: EMPTY_NAUTILUS_BACKTEST_ENGINE_STUB — BacktestEngine.run() missing {missing}")
    return issues
