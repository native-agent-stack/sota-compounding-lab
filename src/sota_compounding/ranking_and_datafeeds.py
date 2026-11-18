"""Canonical SOTA 7-Feed PIT DataFeed & 6-Pillar Trading Ranking Architecture (`sota-ranking-and-datafeeds.v2`).

Unites the SOTA quantitative research & microstructure skill contracts into deterministic,
pure-Python, Hypothesis-verifiable validators for open-core strategy & ranking governance:
1. 7-Feed Point-in-Time DataFeed Contract (`sota-research-feeds.v2`):
   - `alpaca_sip` (`04:00-20:00 ET` + `2026-12-06` `23/5` flag, `bs*100` CTA/UTP round-lot shares,
     `'F'` lit ISO vs `'I'` odd-lot, BJZZ 2021 institutional vs retail sub-penny `x=='D'`)
   - `alpaca_boats` (`20:00-04:00 ET` Sun-Fri, `<=900s` freshness, `<=8` sym / `<=8h` historical root isolation)
   - `alpaca_raw_vs_all_adjustment` (`RAW` for liquidity/spread/tick filters, `ALL` for returns & `RVOL_tod`)
   - `sec_edgar_acceptance_datetime` (`acceptanceDateTime <= as_of_utc`, `15m` half-life, `S-3`/`424B*` block)
   - `finra_regsho_cnmsshvol_t1` (`d_tag < target_tag` `T-1` short-vol + FINRA Rule 4560 `T+8` bd short-interest)
   - `corporate_actions_split_guard` (`XNYS` calendar + Stockbee `bar_age_sessions == 0` completed-bar gate)
   - `ibkr_paper_4002_l1` (loopback `ib_ids.PAPER_PORT` shortable/fee telemetry & `2700/300` msg/min reserve)
2. 6-Pillar Session-Aware SOTA Mover Ranking + First-Hand Incentive & `H090` Execution Tiers:
   - Separates true Gervais-Kaniel-Mingelgrin (2001) `TAPE_PRE_NEWS_IGNITION` (`no headline <= 8h`)
     from intraday `1800s..28800s` `TIER2_INTRADAY_POST_NEWS_DRIFT` (`ACTIVE_FOLLOW_THROUGH`).
   - Classifies Stockbee Day-1 Episodic Pivots, `4%` Momentum Bursts, and `H090` Tight-Spread (`<=10 bps`)
     × High-`OFI` (`>=25%`) Passive Queue Execution Tiers (`+20.84 bps` 1x, `5/5` CPCV paths).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

RANKING_AND_DATAFEEDS_SCHEMA = "sota-ranking-and-datafeeds.v2"
RESEARCH_FEEDS_SCHEMA = "sota-research-feeds.v2"
CTA_UTP_ROUND_LOT_SHARES = 100.0
CATALYST_HALF_LIFE_SECONDS = 900.0
PRENEWS_SILENCE_MIN_SECONDS = 28_800.0
ADR_0022_MAX_STOP_PCT = 2.0

CanonicalFeedId = Literal[
    "alpaca_sip",
    "alpaca_boats",
    "alpaca_raw_vs_all_adjustment",
    "sec_edgar_acceptance_datetime",
    "finra_regsho_cnmsshvol_t1",
    "corporate_actions_split_guard",
    "ibkr_paper_4002_l1",
]

CANONICAL_7_PIT_FEEDS: tuple[CanonicalFeedId, ...] = (
    "alpaca_sip",
    "alpaca_boats",
    "alpaca_raw_vs_all_adjustment",
    "sec_edgar_acceptance_datetime",
    "finra_regsho_cnmsshvol_t1",
    "corporate_actions_split_guard",
    "ibkr_paper_4002_l1",
)

BjzzDarkPoolClass = Literal[
    "LIT_EXCHANGE",
    "DARK_INSTITUTIONAL_ROUND_OR_MIDPOINT",
    "DARK_RETAIL_SUBPENNY_INTERNALIZATION",
]

ExecutionQualityTier = Literal[
    "TIER1_TIGHT_SPREAD_HIGH_OFI_PASSIVE",
    "TIER2_TWO_STAGE_HYBRID_QUEUE",
    "TIER3_WIDE_SPREAD_OR_ADVERSE_OFI_BLOCKED",
]


@dataclass(frozen=True, slots=True)
class PitFeedCheckResult:
    valid: bool
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FirstHandIncentiveScore:
    source_tier: str
    ep_category: str
    reaction_stage: str
    freshness_decay: float
    incentive_score: float
    is_first_hand_early: bool


@dataclass(frozen=True, slots=True)
class MoverRankingEvaluation:
    schema_version: str
    sota_score: float
    early_composite_rank: float
    large_vol_potential_score: float
    ucb_arbitrated_rank_score: float
    execution_tier: ExecutionQualityTier
    stockbee_momentum_burst_qualified: bool
    first_hand: FirstHandIncentiveScore


def _require_finite(name: str, val: float) -> float:
    if isinstance(val, bool) or not math.isfinite(val):
        raise ValueError(f"{name} must be finite IEEE-754 float, got {val!r}")
    return float(val)


def convert_sip_round_lots_to_shares(quote_lots: float) -> float:
    """Convert CTA/UTP consolidated SIP quote size in 100-share round lots into shares."""
    lots = _require_finite("quote_lots", quote_lots)
    if lots < 0.0:
        raise ValueError(f"quote_lots must be >= 0, got {lots}")
    return lots * CTA_UTP_ROUND_LOT_SHARES


def classify_bjzz_dark_pool_print(price: float, exchange: str) -> BjzzDarkPoolClass:
    """Classify FINRA ADF/TRF (`x == 'D'`) prints per Boehmer, Jones, Zhang & Zhang (2021, JF)."""
    px = _require_finite("price", price)
    if px <= 0.0:
        raise ValueError(f"price must be > 0, got {px}")
    if exchange.strip().upper() != "D":
        return "LIT_EXCHANGE"
    if px < 1.0:
        return "DARK_INSTITUTIONAL_ROUND_OR_MIDPOINT"
    cents_Scaled = round(px * 10_000.0) % 100
    if cents_Scaled in (0, 50):
        return "DARK_INSTITUTIONAL_ROUND_OR_MIDPOINT"
    return "DARK_RETAIL_SUBPENNY_INTERNALIZATION"


def validate_pit_datafeed_observation(
    *,
    feeds_present: tuple[str, ...] | list[str],
    sec_acceptance_delta_sec: float | None,
    finra_short_vol_lag_days: int,
    finra_short_interest_lag_bdays: int,
    boats_quote_age_sec: float | None = None,
    bar_age_sessions: int = 0,
) -> PitFeedCheckResult:
    """Fail-closed point-in-time verifier across all 7 `sota-research-feeds.v2` feeds."""
    violations: list[str] = []
    missing = [f for f in CANONICAL_7_PIT_FEEDS if f not in feeds_present]
    if missing:
        violations.append(f"MISSING_CANONICAL_FEEDS:{','.join(missing)}")
    if sec_acceptance_delta_sec is not None:
        d_sec = _require_finite("sec_acceptance_delta_sec", sec_acceptance_delta_sec)
        if d_sec < 0.0:
            violations.append("LOOKAHEAD_SEC_ACCEPTANCE_DATETIME")
    if finra_short_vol_lag_days < 1:
        violations.append("LOOKAHEAD_FINRA_CNMSSHVOL_REQUIRES_T_MINUS_1")
    if finra_short_interest_lag_bdays < 8:
        violations.append("LOOKAHEAD_FINRA_RULE_4560_REQUIRES_T_PLUS_8_BD")
    if boats_quote_age_sec is not None:
        b_age = _require_finite("boats_quote_age_sec", boats_quote_age_sec)
        if b_age < 0.0 or b_age > 900.0:
            violations.append("STALE_OR_FUTURE_BOATS_QUOTE_GT_900S")
    if bar_age_sessions < 0:
        violations.append("UNCOMPLETED_FUTURE_EXCHANGE_BAR")
    return PitFeedCheckResult(valid=(len(violations) == 0), violations=tuple(violations))


def evaluate_first_hand_incentive(
    *,
    catalyst_age_seconds: float | None,
    news_count_36h: int,
    is_sec_edgar_primary: bool,
    is_tier1_breaking_catalyst: bool,
    has_dilution_or_adverse_filing: bool,
    move_pct: float,
    vwap_dist_pct: float,
    room_left_pct: float,
    vol_zscore: float,
    rvol_tod: float,
    burst_5m_mult: float,
    ofi_pct: float,
) -> FirstHandIncentiveScore:
    """Score first-hand SEC/wire catalysts vs true pre-news volume ignition (Vega 2006 / Tetlock 2007)."""
    mv = _require_finite("move_pct", move_pct)
    vw = _require_finite("vwap_dist_pct", vwap_dist_pct)
    rl = _require_finite("room_left_pct", room_left_pct)
    vz = _require_finite("vol_zscore", vol_zscore)
    rv = _require_finite("rvol_tod", rvol_tod)
    b5 = _require_finite("burst_5m_mult", burst_5m_mult)
    ofi = _require_finite("ofi_pct", ofi_pct)
    age = _require_finite("catalyst_age_seconds", catalyst_age_seconds) if catalyst_age_seconds is not None else None

    if has_dilution_or_adverse_filing:
        return FirstHandIncentiveScore(
            source_tier="NEGATIVE_DILUTION_OR_ADVERSE",
            ep_category="DILUTION_OFFERING",
            reaction_stage="DILUTION_OR_ADVERSE_BLOCKED",
            freshness_decay=0.0,
            incentive_score=0.0,
            is_first_hand_early=False,
        )

    if age is not None:
        if age < 0.0:
            raise ValueError(f"catalyst_age_seconds cannot be negative (lookahead): {age}")
        freshness = round(max(0.10, 2.0 ** (-age / CATALYST_HALF_LIFE_SECONDS)) if age <= 129_600.0 else 0.0, 4)
    else:
        freshness = 0.0

    rehash_mult = 0.55 if (age is not None and age > 7_200.0 and news_count_36h >= 4) else 1.0

    if is_sec_edgar_primary and (age is None or age <= 1_800.0):
        source_tier = "TIER1_SEC_EDGAR_PRIMARY"
        ep_category = "SEC_8K_OR_13D_MATERIAL"
        authority = 1.00
    elif is_tier1_breaking_catalyst and age is not None and age <= 3_600.0:
        source_tier = "TIER1_BREAKING_CATALYST_WIRE"
        ep_category = "DAY1_EPISODIC_PIVOT"
        authority = 0.92
    elif age is not None and age <= 1_800.0:
        source_tier = "TIER2_FRESH_WIRE"
        ep_category = "GENERAL_CATALYST"
        authority = 0.72
    elif (age is None or (age > PRENEWS_SILENCE_MIN_SECONDS and news_count_36h <= 1)) and (
        (vz >= 2.6 or (rv >= 2.6 and b5 >= 1.8)) and ofi >= 15.0 and 0.30 <= mv <= 5.5
    ):
        # True Gervais-Kaniel-Mingelgrin (2001) pre-news volume shock requires >=8h news silence & strong OFI
        source_tier = "TAPE_PRE_NEWS_IGNITION"
        ep_category = "PRE_NEWS_VOLUME_IGNITION"
        authority = 0.76
    elif age is not None and age <= PRENEWS_SILENCE_MIN_SECONDS:
        source_tier = "TIER2_INTRADAY_POST_NEWS_DRIFT"
        ep_category = "POST_CATALYST_FOLLOW_THROUGH"
        authority = 0.66 if is_tier1_breaking_catalyst else 0.48
    elif age is not None:
        source_tier = "TIER3_STALE_OR_REHASH"
        ep_category = "STALE_MEDIA_REHASH"
        authority = 0.32
    else:
        source_tier = "NONE"
        ep_category = "NONE"
        authority = 0.0

    late_chase = (mv > 18.0 or vw > 3.0) or (age is not None and age > 7_200.0 and mv > 12.0 and vw > 2.2)
    is_early = (
        not late_chase
        and authority >= 0.70
        and (source_tier == "TAPE_PRE_NEWS_IGNITION" or (age is not None and age <= 1_200.0))
        and -0.50 <= mv <= 9.5
        and vw <= 2.15
        and rl >= 45.0
    )

    if late_chase and mv > 0.0:
        stage = "LATE_EXTENDED_CHASE"
        stage_mult = 0.50
    elif is_early:
        stage = "FIRST_HAND_EARLY"
        stage_mult = 1.12
    elif authority >= 0.48 and mv > 0.0:
        stage = "ACTIVE_FOLLOW_THROUGH"
        stage_mult = 0.90
    else:
        stage = "QUIET_OR_STALE"
        stage_mult = 0.75

    if source_tier == "TAPE_PRE_NEWS_IGNITION":
        raw_sc = min(92.0, 58.0 + min(vz, 6.0) * 4.0 + min(rv, 6.0) * 2.0)
    elif authority > 0.0:
        raw_sc = 100.0 * authority * (0.45 + 0.55 * freshness) * rehash_mult * stage_mult
    else:
        raw_sc = 0.0

    return FirstHandIncentiveScore(
        source_tier=source_tier,
        ep_category=ep_category,
        reaction_stage=stage,
        freshness_decay=freshness,
        incentive_score=round(min(100.0, max(0.0, raw_sc)), 1),
        is_first_hand_early=is_early,
    )


def classify_execution_quality_tier(*, spread_bps: float, ofi_pct: float) -> ExecutionQualityTier:
    """Classify execution tier per Trials #88/#89/#90 (`H090` `<=10 bps` spread + `>=25%` OFI)."""
    sp = _require_finite("spread_bps", spread_bps)
    ofi = _require_finite("ofi_pct", ofi_pct)
    if sp < 0.0:
        raise ValueError(f"spread_bps must be >= 0, got {sp}")
    if sp <= 10.0 and ofi >= 25.0:
        return "TIER1_TIGHT_SPREAD_HIGH_OFI_PASSIVE"
    if sp <= 35.0 and ofi >= 0.0:
        return "TIER2_TWO_STAGE_HYBRID_QUEUE"
    return "TIER3_WIDE_SPREAD_OR_ADVERSE_OFI_BLOCKED"


def evaluate_stockbee_momentum_burst(
    *,
    move_pct: float,
    dollar_change: float,
    rvol_tod: float,
    range_expansion_mult: float,
    close_location: float,
    stop_distance_pct: float,
    bar_age_sessions: int = 0,
) -> bool:
    """Evaluate Stockbee 4% / dollar Momentum Burst qualification under `ADR-0022 <= 2.00%` stop cap."""
    mv = _require_finite("move_pct", move_pct)
    dc = _require_finite("dollar_change", dollar_change)
    rv = _require_finite("rvol_tod", rvol_tod)
    rexp = _require_finite("range_expansion_mult", range_expansion_mult)
    cloc = _require_finite("close_location", close_location)
    stop_pct = _require_finite("stop_distance_pct", stop_distance_pct)
    if bar_age_sessions != 0:
        return False
    if not (0.0 < stop_pct <= ADR_0022_MAX_STOP_PCT):
        return False
    price_breakout = mv >= 4.0 or dc >= 0.90
    return bool(price_breakout and rv >= 2.0 and rexp >= 1.5 and 0.70 <= cloc <= 1.0)


def compute_sota_mover_ranking(
    *,
    pillars: dict[str, float],
    session_segment: str,
    boats_active: bool,
    early_potential_score: float,
    no_sell_pressure_score: float,
    sota_vol_rank_score: float,
    inst_dark_pct: float,
    session_dvol_usd: float,
    live_price: float,
    move_pct: float,
    dollar_change: float,
    rvol_tod: float,
    range_expansion_mult: float,
    close_location: float,
    spread_bps: float,
    ofi_pct: float,
    stop_distance_pct: float,
    strategy_confidence_0_1: float,
    ucb_priority_weight: float,
    first_hand: FirstHandIncentiveScore,
) -> MoverRankingEvaluation:
    """Compute the canonical 6-pillar SOTA mover score, 3 composite ranks, and `H090` execution tier."""
    req_pillars = ("move", "unusual_vol", "inst_dark", "iso_urgency", "overnight", "catalyst")
    p_clean: dict[str, float] = {}
    for k in req_pillars:
        v = _require_finite(f"pillars[{k}]", pillars[k])
        p_clean[k] = min(100.0, max(0.0, v))

    if session_segment == "OVERNIGHT_BOATS":
        move_w = 0.12 if boats_active else 0.06
        sota_score = round(
            move_w * p_clean["move"]
            + 0.22 * p_clean["unusual_vol"]
            + 0.15 * p_clean["inst_dark"]
            + 0.13 * p_clean["iso_urgency"]
            + 0.28 * p_clean["overnight"]
            + 0.10 * p_clean["catalyst"],
            2,
        )
    else:
        sota_score = round(
            0.22 * p_clean["move"]
            + 0.25 * p_clean["unusual_vol"]
            + 0.19 * p_clean["inst_dark"]
            + 0.16 * p_clean["iso_urgency"]
            + 0.10 * p_clean["overnight"]
            + 0.08 * p_clean["catalyst"],
            2,
        )

    exec_tier = classify_execution_quality_tier(spread_bps=spread_bps, ofi_pct=ofi_pct)
    h090_bonus = 4.0 if exec_tier == "TIER1_TIGHT_SPREAD_HIGH_OFI_PASSIVE" else 0.0
    fhi_bonus = 0.15 * first_hand.incentive_score if first_hand.reaction_stage == "FIRST_HAND_EARLY" else 0.0

    early_rank = round(
        min(
            100.0,
            max(
                0.0,
                0.45 * _require_finite("early_potential_score", early_potential_score)
                + 0.35 * _require_finite("no_sell_pressure_score", no_sell_pressure_score)
                + 0.20 * p_clean["inst_dark"]
                + fhi_bonus
                + h090_bonus,
            ),
        ),
        1,
    )

    dvol = _require_finite("session_dvol_usd", session_dvol_usd)
    px = _require_finite("live_price", live_price)
    mv = _require_finite("move_pct", move_pct)
    if dvol >= 20_000_000.0 and px >= 5.0 and mv > 0.0:
        large_vol = round(
            12.0 * math.log10(max(dvol / 1_000_000.0, 1.0))
            + 0.28 * _require_finite("sota_vol_rank_score", sota_vol_rank_score)
            + 0.24 * early_potential_score
            + 0.24 * no_sell_pressure_score
            + 0.24 * _require_finite("inst_dark_pct", inst_dark_pct),
            1,
        )
    else:
        large_vol = 0.0

    conf = min(1.0, max(0.0, _require_finite("strategy_confidence_0_1", strategy_confidence_0_1)))
    ucb_w = max(0.0, _require_finite("ucb_priority_weight", ucb_priority_weight))
    ucb_rank = round(conf * ucb_w * 100.0, 2)

    burst_ok = evaluate_stockbee_momentum_burst(
        move_pct=mv,
        dollar_change=dollar_change,
        rvol_tod=rvol_tod,
        range_expansion_mult=range_expansion_mult,
        close_location=close_location,
        stop_distance_pct=stop_distance_pct,
    )

    return MoverRankingEvaluation(
        schema_version=RANKING_AND_DATAFEEDS_SCHEMA,
        sota_score=sota_score,
        early_composite_rank=early_rank,
        large_vol_potential_score=large_vol,
        ucb_arbitrated_rank_score=ucb_rank,
        execution_tier=exec_tier,
        stockbee_momentum_burst_qualified=burst_ok,
        first_hand=first_hand,
    )
