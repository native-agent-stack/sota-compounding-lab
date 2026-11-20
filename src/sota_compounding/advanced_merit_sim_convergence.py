"""Advanced Sim-to-Merit Statistical Governance, Conformal PID Stops & Non-Stationary Allocation (`ADR-0029`).

Implements four primary-literature-grounded upgrades to the 5-stage `Sim -> Pre-Wall -> Paper -> Merit -> Broker`
pipeline in `sota-compounding-lab` (zero new runtime dependencies; strictly `PROTECTIVE_FILTER_OR_GOVERNANCE`,
`standalone_alpha_claim = False`, `hlz_trial_consumed = False`):

1. **Hansen (2005) Superior Predictive Ability (`SPA`) + Politis–Romano (1994) Stationary Block Bootstrap +
   Hansen–Lunde–Nason (2011) Model Confidence Set (`MCS`):**
   - Studentized sample-dependent recentering (`hat_mu_c`) eliminates White's Reality Check power loss from
     poor/irrelevant models, and benchmark-free sequential `MCS` elimination returns the exact surviving set
     `M*` at confidence `1 - alpha` to govern Paper Graduation rosters alongside `HLZ t*(N)`, `DSR`, `RAS`, `PBO`.
2. **Conformal PID / Adaptive Conformal Inference (`Gibbs & Candès 2021`; `Angelopoulos, Candès & Tibshirani 2023`):**
   - Distribution-free online calibration of intrabar MAE stop widths & strategy drawdown thresholds under
     arbitrary volatility regime shifts, strictly clamped inside `ADR-0022 <= 2.00%` (`TP OFF`).
3. **Full-Information Fixed-Share Hedge / MWU (`Freund & Schapire 1997`) + Hierarchical Empirical-Bayes Cluster
   Shrinkage (`Jensen, Kelly & Pedersen 2023 JF`) + Cederburg-Safe Asymmetric Volatility Crash Brake (`<= 1.0x`):**
   - Exploits full-information shadow-marked Paper Performance Manifests to outpace stationary `UCB1` on regime
     switches while shrinking noisy single-strategy cohorts toward their economic cluster prior (`tau_w^2`, `tau_b^2`)
     and capping exposure `<= 1.0x` under vol spikes (never levering `> 1.0x`, honoring `Cederburg et al. 2020 JFE`).
4. **Next-Wave Pre-Registration Strategy Blueprints (`H109..H114`, study-wall `<= 2026-10-15`):**
   - Machine-readable, `ADIA Lab #19`-compatible pre-registration registry for the 6 frontier 2021–2026 strategy
     families (`Overnight/Intraday BOATS Bifurcation`, `Residual Lead-Lag + EOD LETF Flow`, `Zero-New-Feed EDGAR
     Form 4 / XML 13D / Lazy Prices / ATM Overhang`, `Triple-Barrier Meta-Labeling`, `L2/L3 Integrated OFI + NOII`
     gated on future Databento DBN onboarding, and `Fixed-Share Hedge + MCS` meta-allocation).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE, hlz_dynamic_t_hurdle

ADVANCED_CONVERGENCE_SCHEMA = "sota.advanced-sim-merit-convergence.v1"
CURRENT_LEDGER_TRIALS_N = 104
NEXT_TRIAL_ID = 105
NEXT_HLZ_T_STAR = hlz_dynamic_t_hurdle(NEXT_TRIAL_ID)  # 4.2084
_STD_NORM = NormalDist(0.0, 1.0)


def _require_finite(val: float, name: str) -> float:
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite (got {val!r})")
    return f


def _stationary_bootstrap_indices(n: int, avg_block_len: float, rng: random.Random) -> list[int]:
    """Generate Politis & Romano (1994) stationary bootstrap index draw of length `n`."""
    if n <= 0:
        return []
    p = 1.0 / max(1.0, float(avg_block_len))
    idx = [rng.randrange(n)]
    for _ in range(1, n):
        if rng.random() < p:
            idx.append(rng.randrange(n))
        else:
            idx.append((idx[-1] + 1) % n)
    return idx


def _newey_west_long_run_var(series: Sequence[float], max_lag: int) -> float:
    """Bartlett-kernel HAC long-run variance estimator for autocorrelated daily performance differentials."""
    n = len(series)
    if n < 2:
        return 0.0
    mean = sum(series) / n
    centered = [x - mean for x in series]
    gamma0 = sum(x * x for x in centered) / n
    lrv = gamma0
    eff_lag = max(0, min(int(max_lag), n - 1))
    for lag in range(1, eff_lag + 1):
        w = 1.0 - lag / (eff_lag + 1.0)
        cov = sum(centered[t] * centered[t - lag] for t in range(lag, n)) / n
        lrv += 2.0 * w * cov
    return max(1e-12, float(lrv))


def evaluate_hansen_spa_and_mcs(
    strategy_returns_bps: Mapping[str, Sequence[float]],
    *,
    benchmark_returns_bps: Sequence[float] | None = None,
    alpha: float = 0.10,
    n_bootstrap: int = 400,
    avg_block_len: float = 5.0,
    seed: int = 20261015,
) -> dict[str, Any]:
    """Compute Hansen (2005) consistent SPA p-value & Hansen-Lunde-Nason (2011) Model Confidence Set (`M*`).

    Parameters
    ----------
    strategy_returns_bps:
        Mapping `{strategy_id: [daily_return_bps_t, ...]}` aligned across identical trading dates `t = 0..T-1`.
    benchmark_returns_bps:
        Optional benchmark return series of length `T` (defaults to `0.0` bps cash/hurdle).
    alpha:
        Significance level for `MCS` (`1 - alpha` confidence set, e.g. `0.10` -> `90%` MCS).
    n_bootstrap:
        Number of Politis–Romano stationary block bootstrap resamples (`>= 100`).
    avg_block_len:
        Mean geometric block length for stationary bootstrap (`>= 1.0`).
    seed:
        Deterministic PRNG seed for bit-exact reproducibility across Mac and Noesis.
    """
    if not strategy_returns_bps:
        raise ValueError("strategy_returns_bps must be non-empty")
    if not (0.001 <= alpha < 0.5):
        raise ValueError(f"alpha must be in [0.001, 0.5) (got {alpha})")
    names = sorted(strategy_returns_bps.keys())
    lengths = {len(strategy_returns_bps[k]) for k in names}
    if len(lengths) != 1:
        raise ValueError(f"All strategy return series must share identical length T (got {sorted(lengths)})")
    t_len = next(iter(lengths))
    if t_len < 10:
        raise ValueError(f"Require >= 10 aligned periods for stationary bootstrap (got {t_len})")

    bench = [0.0] * t_len if benchmark_returns_bps is None else [float(x) for x in benchmark_returns_bps]
    if len(bench) != t_len:
        raise ValueError(f"benchmark_returns_bps length {len(bench)} != T={t_len}")

    diffs: dict[str, list[float]] = {}
    for nm in names:
        vals = [_require_finite(x, f"{nm}[{i}]") for i, x in enumerate(strategy_returns_bps[nm])]
        diffs[nm] = [vals[i] - bench[i] for i in range(t_len)]

    nw_lag = max(1, int(round(avg_block_len)))
    means: dict[str, float] = {}
    se_map: dict[str, float] = {}
    t_stats: dict[str, float] = {}
    for nm in names:
        d = diffs[nm]
        mu = sum(d) / t_len
        omega = math.sqrt(_newey_west_long_run_var(d, nw_lag))
        se = max(1e-9, omega / math.sqrt(t_len))
        means[nm] = mu
        se_map[nm] = se
        t_stats[nm] = mu / se

    # Hansen (2005) consistent recentering threshold: -sqrt(2 * log(log(T)))
    # Models proven genuinely inferior (t_k <= -sqrt(2 log log T)) keep their negative sample mean
    # so they do not inflate the null maximum; borderline/superior models recenter at 0.0.
    ll_t = math.log(max(math.e, math.log(max(t_len, 3))))
    c_thresh = -math.sqrt(2.0 * ll_t)
    recenter_consistent: dict[str, float] = {nm: (means[nm] if t_stats[nm] <= c_thresh else 0.0) for nm in names}

    t_spa_obs = max(0.0, max(t_stats.values()))
    rng = random.Random(seed)
    n_boot = max(100, int(n_bootstrap))
    boot_draws = [_stationary_bootstrap_indices(t_len, avg_block_len, rng) for _ in range(n_boot)]

    # Consistent, Lower, and Upper SPA p-values (Hansen 2005 Eq. 9)
    exceed_c = 0
    exceed_l = 0
    exceed_u = 0
    for idx_draw in boot_draws:
        max_c = 0.0
        max_l = 0.0
        max_u = 0.0
        for nm in names:
            d = diffs[nm]
            b_mean = sum(d[j] for j in idx_draw) / t_len
            se = se_map[nm]
            z_u = (b_mean - means[nm]) / se
            z_c = (b_mean - means[nm] + min(0.0, recenter_consistent[nm])) / se
            z_l = (b_mean - means[nm] + min(0.0, means[nm])) / se
            if z_u > max_u:
                max_u = z_u
            if z_c > max_c:
                max_c = z_c
            if z_l > max_l:
                max_l = z_l
        if max_c >= t_spa_obs:
            exceed_c += 1
        if max_l >= t_spa_obs:
            exceed_l += 1
        if max_u >= t_spa_obs:
            exceed_u += 1

    spa_p_consistent = (exceed_c + 1.0) / (n_boot + 1.0)
    spa_p_lower = (exceed_l + 1.0) / (n_boot + 1.0)
    spa_p_upper = (exceed_u + 1.0) / (n_boot + 1.0)

    # Hansen-Lunde-Nason (2011) Model Confidence Set (M*) via max-t full sequential elimination
    # Computing the complete elimination path assigns exact HLN Definition 2 MCS p-values to every model.
    remaining = list(names)
    eliminated: list[dict[str, Any]] = []
    mcs_pvals: dict[str, float] = {}
    running_max_p = 0.0

    while len(remaining) > 1:
        k_cur = len(remaining)
        avg_series = [sum(diffs[nm][t] for nm in remaining) / k_cur for t in range(t_len)]
        rel_series: dict[str, list[float]] = {
            nm: [diffs[nm][t] - avg_series[t] for t in range(t_len)] for nm in remaining
        }
        rel_mean: dict[str, float] = {nm: sum(rel_series[nm]) / t_len for nm in remaining}
        rel_se: dict[str, float] = {
            nm: max(1e-9, math.sqrt(_newey_west_long_run_var(rel_series[nm], nw_lag)) / math.sqrt(t_len))
            for nm in remaining
        }
        # Loss-space max-t: negative relative return = worse performance (larger loss)
        loss_t: dict[str, float] = {nm: (-rel_mean[nm]) / rel_se[nm] for nm in remaining}
        worst_nm = max(remaining, key=lambda nm: (loss_t[nm], nm))
        t_max_obs = loss_t[worst_nm]

        exceed_mcs = 0
        for idx_draw in boot_draws:
            b_max_t = -math.inf
            for nm in remaining:
                rs = rel_series[nm]
                b_mu = sum(rs[j] for j in idx_draw) / t_len
                # Centered under null H0: E[rel_series] = 0
                bt = (-(b_mu - rel_mean[nm])) / rel_se[nm]
                if bt > b_max_t:
                    b_max_t = bt
            if b_max_t >= t_max_obs:
                exceed_mcs += 1

        step_p = (exceed_mcs + 1.0) / (n_boot + 1.0)
        running_max_p = max(running_max_p, step_p)
        mcs_pvals[worst_nm] = round(running_max_p, 6)
        remaining.remove(worst_nm)
        if running_max_p < alpha:
            eliminated.append({"strategy_id": worst_nm, "mcs_p_value": round(running_max_p, 6)})

    # Last surviving model always carries MCS p-value = 1.0 (HLN 2011 Definition 2)
    if remaining:
        mcs_pvals[remaining[0]] = 1.0
    surviving = [nm for nm in names if mcs_pvals[nm] >= alpha]

    return {
        "schema": "sota.spa-mcs-validation.v1",
        "periods_t": t_len,
        "n_strategies": len(names),
        "n_bootstrap": n_boot,
        "avg_block_len": float(avg_block_len),
        "alpha": float(alpha),
        "best_strategy_id": max(names, key=lambda nm: t_stats[nm]),
        "spa_t_stat": round(t_spa_obs, 4),
        "spa_p_lower": round(spa_p_lower, 6),
        "spa_p_consistent": round(spa_p_consistent, 6),
        "spa_p_upper": round(spa_p_upper, 6),
        "spa_passes_alpha": spa_p_consistent < alpha,
        "mcs_surviving_strategies": surviving,
        "mcs_eliminated": eliminated,
        "mcs_p_values": mcs_pvals,
        "per_strategy_t_stats": {nm: round(t_stats[nm], 4) for nm in names},
    }


@dataclass
class ConformalPidStopCalibrator:
    """Conformal PID / ACI online MAE stop calibrator strictly clamped inside `ADR-0022 <= 2.00%`.

    Implements Angelopoulos, Candès & Tibshirani (2023 NeurIPS) Conformal PID control over non-exchangeable
    intrabar Maximum Adverse Excursion (`MAE / ATR`) ratios with hard `ADR-0022` ceiling (`<= 2.00%`).
    """

    target_miscoverage_alpha: float = 0.10
    kp_learning_rate: float = 0.08
    ki_integral_gain: float = 0.25
    integral_decay: float = 0.90
    quantile_multiplier: float = 1.25
    min_stop_pct: float = 0.15
    max_stop_pct: float = ADR0022_MAX_STOP_PCT
    _base_quantile_multiplier: float = 1.25
    _integral_err: float = 0.0
    _steps: int = 0
    _miscoverages: int = 0

    def __post_init__(self) -> None:
        if not (0.01 <= self.target_miscoverage_alpha <= 0.40):
            raise ValueError("target_miscoverage_alpha must be in [0.01, 0.40]")
        if self.max_stop_pct > ADR0022_MAX_STOP_PCT:
            raise ValueError(f"ADR-0022 breach: max_stop_pct={self.max_stop_pct} > {ADR0022_MAX_STOP_PCT}%")
        if self.min_stop_pct <= 0.0 or self.min_stop_pct >= self.max_stop_pct:
            raise ValueError("min_stop_pct must lie strictly inside (0, max_stop_pct)")
        if not (0.5 <= self.integral_decay <= 1.0):
            raise ValueError("integral_decay must be in [0.5, 1.0]")
        self._base_quantile_multiplier = float(self.quantile_multiplier)

    def recommended_stop_pct(self, atr_pct: float) -> float:
        """Compute current distribution-free stop width (%) strictly inside `[min_stop_pct, 2.00%]`."""
        atr = max(0.05, _require_finite(atr_pct, "atr_pct"))
        raw = self.quantile_multiplier * atr
        return round(min(self.max_stop_pct, max(self.min_stop_pct, raw)), 4)

    def update(self, *, realized_mae_pct: float, atr_pct: float) -> dict[str, Any]:
        """Ingest one realized trade intrabar MAE (%) and update Conformal PID quantile online."""
        mae = max(0.0, _require_finite(realized_mae_pct, "realized_mae_pct"))
        atr = max(0.05, _require_finite(atr_pct, "atr_pct"))
        pred_stop = self.recommended_stop_pct(atr)
        score_ratio = mae / atr
        err = 1.0 if score_ratio > self.quantile_multiplier else 0.0

        self._steps += 1
        if err > 0.5:
            self._miscoverages += 1

        gap = err - self.target_miscoverage_alpha
        # Angelopoulos, Candes & Tibshirani (2023) Conformal PID:
        # P-state integrates proportional error; saturating integral boost is added directly to output
        # (never double-integrated into the P-state) with leaky anti-windup decay.
        self._base_quantile_multiplier = max(
            0.35, min(3.75, self._base_quantile_multiplier + self.kp_learning_rate * gap)
        )
        self._integral_err = max(-4.0, min(4.0, self.integral_decay * self._integral_err + gap))
        int_boost = self.ki_integral_gain * math.tanh(self._integral_err)
        self.quantile_multiplier = max(0.35, min(4.0, self._base_quantile_multiplier + int_boost))
        next_stop = self.recommended_stop_pct(atr)
        return {
            "step": self._steps,
            "previous_stop_pct": pred_stop,
            "realized_mae_pct": round(mae, 4),
            "stop_breached": bool(err > 0.5),
            "next_quantile_multiplier": round(self.quantile_multiplier, 4),
            "next_stop_pct": next_stop,
            "adr0022_compliant": next_stop <= ADR0022_MAX_STOP_PCT,
            "empirical_miscoverage_rate": round(self._miscoverages / self._steps, 4),
        }


def update_fixed_share_hedge_weights(
    prior_weights: Mapping[str, float],
    realized_r_by_strategy: Mapping[str, float],
    *,
    eta: float = 0.25,
    switching_alpha: float = 0.05,
    r_clip: float = 3.0,
) -> dict[str, float]:
    """Full-Information Fixed-Share Hedge / MWU weight update (`Freund & Schapire 1997` + Herbster-Warmuth).

    Strictly dominates partial-feedback `UCB1` on shadow-marked Paper Performance Manifests where all
    candidate strategies are observed daily; `switching_alpha` preserves adaptivity across regime breaks.
    """
    if not prior_weights:
        raise ValueError("prior_weights must be non-empty")
    if not (0.0 < eta <= 2.0):
        raise ValueError(f"eta must be in (0, 2.0] (got {eta})")
    if not (0.0 <= switching_alpha < 0.5):
        raise ValueError(f"switching_alpha must be in [0, 0.5) (got {switching_alpha})")

    keys = sorted(prior_weights.keys())
    k_cnt = len(keys)
    w_sum = sum(max(0.0, _require_finite(prior_weights[k], f"w[{k}]")) for k in keys)
    base = {k: (max(0.0, float(prior_weights[k])) / w_sum if w_sum > 0.0 else 1.0 / k_cnt) for k in keys}

    # Multiplicative exponential weights update on clipped realized R
    unnorm: dict[str, float] = {}
    for k in keys:
        r_val = _require_finite(realized_r_by_strategy.get(k, 0.0), f"r[{k}]")
        clipped = max(-r_clip, min(r_clip, r_val))
        unnorm[k] = base[k] * math.exp(eta * clipped)

    z = sum(unnorm.values())
    if z <= 0.0 or not math.isfinite(z):
        return {k: round(1.0 / k_cnt, 6) for k in keys}

    post = {k: unnorm[k] / z for k in keys}
    # Fixed-share pool mixing tracks non-stationary regime switches
    mixed = {k: (1.0 - switching_alpha) * post[k] + (switching_alpha / k_cnt) for k in keys}
    m_sum = sum(mixed.values())
    return {k: round(mixed[k] / m_sum, 6) for k in keys}


@dataclass(frozen=True, slots=True)
class StrategyClusterObservation:
    strategy_id: str
    cluster_id: str
    observed_mean_r: float
    observed_se_r: float
    closed_roundtrips: int


def compute_hierarchical_eb_merit_pool(
    observations: Sequence[StrategyClusterObservation],
    *,
    tau_between_clusters: float = 0.12,
    tau_within_cluster: float = 0.08,
) -> dict[str, dict[str, Any]]:
    """Jensen, Kelly & Pedersen (2023 JF) Hierarchical Empirical-Bayes partial pooling across strategy clusters.

    Shrinks noisy single-strategy paper means toward their economic cluster posterior `mu_c` and overall
    zero prior (`tau_b^2`, `tau_w^2`), preventing lucky small-sample paper streaks from false graduation.
    """
    if not observations:
        return {}
    tb2 = max(1e-6, _require_finite(tau_between_clusters, "tau_between_clusters") ** 2)
    tw2 = max(1e-6, _require_finite(tau_within_cluster, "tau_within_cluster") ** 2)

    by_cluster: dict[str, list[StrategyClusterObservation]] = {}
    for obs in observations:
        _require_finite(obs.observed_mean_r, f"{obs.strategy_id}.mean_r")
        _require_finite(obs.observed_se_r, f"{obs.strategy_id}.se_r")
        by_cluster.setdefault(obs.cluster_id, []).append(obs)

    out: dict[str, dict[str, Any]] = {}
    for cluster_id, rows in sorted(by_cluster.items()):
        # Precision-weighted cluster posterior mean mu_c ~ N(0, tb2)
        prec_sum = 1.0 / tb2
        num_sum = 0.0
        for r in rows:
            se2 = max(1e-4, float(r.observed_se_r) ** 2)
            eff_var = tw2 + se2
            prec_sum += 1.0 / eff_var
            num_sum += float(r.observed_mean_r) / eff_var
        cluster_post_mean = num_sum / prec_sum
        cluster_post_var = 1.0 / prec_sum

        for r in rows:
            se2 = max(1e-4, float(r.observed_se_r) ** 2)
            # Strategy posterior given cluster posterior mean
            w_obs = tw2 / (tw2 + se2)
            post_mean = w_obs * float(r.observed_mean_r) + (1.0 - w_obs) * cluster_post_mean
            post_var = (tw2 * se2) / (tw2 + se2) + ((1.0 - w_obs) ** 2) * cluster_post_var
            post_sd = math.sqrt(max(1e-9, post_var))
            prob_positive = 1.0 - _STD_NORM.cdf((0.0 - post_mean) / post_sd)
            out[r.strategy_id] = {
                "cluster_id": cluster_id,
                "closed_roundtrips": int(r.closed_roundtrips),
                "observed_mean_r": round(float(r.observed_mean_r), 4),
                "cluster_posterior_mean_r": round(cluster_post_mean, 4),
                "hierarchical_eb_mean_r": round(post_mean, 4),
                "hierarchical_eb_sd_r": round(post_sd, 4),
                "shrinkage_weight_on_obs": round(w_obs, 4),
                "posterior_prob_positive": round(prob_positive, 4),
            }
    return out


def compute_asymmetric_vol_crash_brake(
    realized_vol_ann_pct: float,
    *,
    target_vol_ann_pct: float = 18.0,
    min_scale: float = 0.25,
) -> float:
    """Cederburg-safe (`<= 1.0x`) asymmetric volatility crash brake (Barroso & Santa-Clara / Daniel & Moskowitz).

    Scales down exposure smoothly when lagged realized volatility exceeds `target_vol_ann_pct`, while NEVER
    levering above `1.0x` in quiet regimes (mitigating the `Cederburg et al. 2020 JFE` OOS leverage trap).
    """
    rv = max(1e-6, _require_finite(realized_vol_ann_pct, "realized_vol_ann_pct"))
    tv = max(1.0, _require_finite(target_vol_ann_pct, "target_vol_ann_pct"))
    floor = max(0.05, min(1.0, _require_finite(min_scale, "min_scale")))
    raw = tv / rv
    return round(min(1.0, max(floor, raw)), 4)


@dataclass(frozen=True, slots=True)
class NextWaveStrategyBlueprint:
    blueprint_id: str
    family: str
    primary_citations: tuple[str, ...]
    required_feeds: tuple[str, ...]
    datafeed_ready_now: bool
    mechanism_summary: str
    max_stop_pct: float
    expected_horizon: str
    adversarial_caveat: str


NEXT_WAVE_STRATEGY_BLUEPRINTS: tuple[NextWaveStrategyBlueprint, ...] = (
    NextWaveStrategyBlueprint(
        blueprint_id="H109_OVERNIGHT_BOATS_BIFURCATION_AND_RTH_VWAP_FADE",
        family="OVERNIGHT_INTRADAY_BIFURCATION",
        primary_citations=(
            "Lou, Polk & Skouras (2019 JFE) 10.1016/j.jfineco.2019.03.011",
            "Eaton, Shkilko & Werner (2025/2026) SSRN 5181159",
            "Lim (2026) SSRN 6610883",
        ),
        required_feeds=("DF1_SIP", "DF2_BOATS", "DF3_RAW_VS_ALL"),
        datafeed_ready_now=True,
        mechanism_summary=(
            "Restrict BOATS continuation strictly to macro ETFs, cross-listed ADRs & fresh EDGAR catalysts "
            "(<=28 bps spread); fade extended retail-only BOATS gaps failing RTH VWAP at 09:30-09:45 ET."
        ),
        max_stop_pct=1.60,
        expected_horizon="15m_to_1d",
        adversarial_caveat="BOATS spreads average ~7c wider; Dec 6 2026 23/5 SIP cutover changes overnight liquidity.",
    ),
    NextWaveStrategyBlueprint(
        blueprint_id="H110_RESIDUAL_DIRECTED_LEAD_LAG_AND_EOD_LETF_FLOW",
        family="CROSS_STOCK_RESIDUAL_AND_FLOW",
        primary_citations=(
            "Cartea, Cucuringu & Shi (2026) SSRN 6811380",
            "Baltussen, Da, Lammers & Martens (2021 JFE) 10.1016/j.jfineco.2021.04.029",
            "Barbon et al. SSRN 3925725",
        ),
        required_feeds=("DF1_SIP", "DF4_RVOL_TOD", "DF10_VIX_REGIME"),
        datafeed_ready_now=True,
        mechanism_summary=(
            "Strip SPY/Sector ETF beta to trade directed leader->follower residual catch-up plus mechanical "
            "15:30-16:00 ET LETF/dealer hedging flow on elevated-volatility days."
        ),
        max_stop_pct=1.40,
        expected_horizon="5m_to_30m",
        adversarial_caveat=(
            "High turnover decay; must survive 2x spread+impact stress & exclude positive-gamma pin days."
        ),
    ),
    NextWaveStrategyBlueprint(
        blueprint_id="H111_EDGAR_FORM4_CLUSTER_AND_XML_13D_ACCEPTANCE_PIT",
        family="ZERO_NEW_FEED_EDGAR_CATALYST",
        primary_citations=(
            "Cohen, Malloy & Pomorski (2012 JF) 10.1111/j.1540-6261.2012.01740.x",
            "Collin-Dufresne & Fos (2015 JF) 10.1111/jofi.12260",
            "SEC Final Rule 88 FR 76896 (2024 5-day XML 13D)",
        ),
        required_feeds=("DF1_SIP", "DF2_BOATS", "DF5_EDGARTOOLS_V620"),
        datafeed_ready_now=True,
        mechanism_summary=(
            "Key strictly on EDGAR acceptanceDateTime: >=2 distinct opportunistic C-suite Form 4 open-market "
            "buys (routine/10b5-1 stripped) and modernized XML SC 13D Item 4/Item 6 swap disclosures."
        ),
        max_stop_pct=1.80,
        expected_horizon="1d_to_5d",
        adversarial_caveat="Never key on trade date (late filings); 13D event-day pops absorb >=50% within minutes.",
    ),
    NextWaveStrategyBlueprint(
        blueprint_id="H112_LAZY_PRICES_FILING_DIFF_AND_ATM_OVERHANG_SHORT",
        family="SUPPLY_OVERHANG_AND_TEXT_DIFF",
        primary_citations=(
            "Cohen, Malloy & Nguyen (2020 JF) 10.1111/jofi.12885",
            "Billett, Floros & Garfinkel (2019 JFQA) 10.1017/S0022109018000893",
            "Lim, Schwert & Weisbach (2021 JFI) 10.1016/j.jfi.2019.100832",
            "Barardehi et al. (2025 Mgmt Sci) 10.1287/mnsc.2024.4987",
        ),
        required_feeds=("DF1_SIP", "DF5_EDGARTOOLS_V620", "DF7_REGSHO_T1", "DF9_IBKR_ETB_SSR"),
        datafeed_ready_now=True,
        mechanism_summary=(
            "Combine 10-K/10-Q Risk Factor cosine-diff changers with active S-3/424B5 ATM & PIPE EFFECT resale "
            "overhangs as a supply ceiling enhancing ETB Climax Reverse Shorts (strictly !RegSHO_Threshold)."
        ),
        max_stop_pct=1.80,
        expected_horizon="intraday_to_5d",
        adversarial_caveat="Violent squeeze & borrow-recall tail risk; strictly $0 ETB + Rule 201 SSR gated.",
    ),
    NextWaveStrategyBlueprint(
        blueprint_id="H113_TRIPLE_BARRIER_META_LABEL_AND_CONFORMAL_PID_STOP",
        family="META_LABELING_AND_CONFORMAL_RISK",
        primary_citations=(
            "Lopez de Prado (2023 CUP) 10.1017/9781009397315",
            "Angelopoulos, Candes & Tibshirani (2023 NeurIPS) arXiv:2307.16895",
            "Gibbs & Candes (2024 JMLR) arXiv:2208.08401",
        ),
        required_feeds=("DF1_SIP", "DF4_RVOL_TOD", "DF12_CAUSAL_TCA"),
        datafeed_ready_now=True,
        mechanism_summary=(
            "Wrap existing high-recall primary signals in a calibrated Triple-Barrier meta-classifier for sizing "
            "and Conformal PID MAE quantile stops clamped strictly inside ADR-0022 <= 2.00%."
        ),
        max_stop_pct=ADR0022_MAX_STOP_PCT,
        expected_horizon="intraday_multi_session",
        adversarial_caveat="Long-run average coverage only; does not eliminate single-bar gap risk (keep hard stop).",
    ),
    NextWaveStrategyBlueprint(
        blueprint_id="H114_MULTILEVEL_INTEGRATED_OFI_AND_NOII_AUCTION_DBN",
        family="L2_L3_MICROSTRUCTURE_EXPANSION",
        primary_citations=(
            "Cont, Cucuringu & Zhang (2023 QF) 10.1080/14697688.2023.2236159",
            "Kolm, Turiel & Westray (2023 Math Fin) 10.1111/mafi.12413",
            "Bogousslavsky & Muravyev (2023 JFM) 10.1016/j.finmar.2023.100852",
        ),
        required_feeds=("DATABENTO_DBN_MBP10", "EXCHANGE_NOII_IMBALANCE"),
        datafeed_ready_now=False,
        mechanism_summary=(
            "Lagged cross-asset 10-level Integrated OFI passive-queue tilt + 15:50-16:00 ET NOII closing "
            "auction liquidity provision, gated on future Databento DBN L2/L3 onboarding."
        ),
        max_stop_pct=1.00,
        expected_horizon="10s_to_10m",
        adversarial_caveat="Requires L2/L3 MBP-10 + NOII (unavailable on L1 SIP); must clear RiskAdverseQueueModel.",
    ),
)


def build_advanced_sim_merit_convergence_manifest() -> dict[str, Any]:
    """Emit deterministic SHA-256 sealed governance manifest for ADR-0029."""
    bps = [asdict(bp) for bp in NEXT_WAVE_STRATEGY_BLUEPRINTS]
    payload: dict[str, Any] = {
        "schema": ADVANCED_CONVERGENCE_SCHEMA,
        "adr": "ADR-0029",
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "promotion_track": "PROTECTIVE_FILTER_OR_GOVERNANCE",
        "standalone_alpha_claim": False,
        "hlz_trial_consumed": False,
        "completed_ledger_trials_n": CURRENT_LEDGER_TRIALS_N,
        "next_trial_id": NEXT_TRIAL_ID,
        "next_hlz_t_star": NEXT_HLZ_T_STAR,
        "max_stop_adr0022_pct": ADR0022_MAX_STOP_PCT,
        "upgraded_primitives": [
            "evaluate_hansen_spa_and_mcs",
            "ConformalPidStopCalibrator",
            "update_fixed_share_hedge_weights",
            "compute_hierarchical_eb_merit_pool",
            "compute_asymmetric_vol_crash_brake",
        ],
        "ready_now_blueprint_count": sum(1 for b in NEXT_WAVE_STRATEGY_BLUEPRINTS if b.datafeed_ready_now),
        "future_l2_l3_blueprint_count": sum(1 for b in NEXT_WAVE_STRATEGY_BLUEPRINTS if not b.datafeed_ready_now),
        "next_wave_blueprints": bps,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    payload["manifest_sha256"] = digest
    return payload
