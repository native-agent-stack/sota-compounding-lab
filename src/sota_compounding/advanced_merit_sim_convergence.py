"""Advanced Sim-to-Merit Statistical Governance, Conformal PID Stops & Non-Stationary Allocation (`ADR-0029`).

Implements four primary-literature-grounded upgrades to the 5-stage `Sim -> Pre-Wall -> Paper -> Merit -> Broker`
pipeline in `sota-compounding-lab` (zero new runtime dependencies; strictly `PROTECTIVE_FILTER_OR_GOVERNANCE`,
`standalone_alpha_claim = False`, `hlz_trial_consumed = False`):

1. **Hansen (2005) Superior Predictive Ability (`SPA`) + Politis–Romano (1994) Stationary Block Bootstrap +
   Hansen–Lunde–Nason (2011) Model Confidence Set (`MCS`):**
   - Pre-scaled overflow-safe Bartlett-kernel HAC long-run variance, exact constant-series zero-dispersion
     handling (zero ULP drift on decimals like `[0.22]*10`), exact Hansen (2005) consistent recentering
     threshold `-sqrt(2 * log(log(T)))`, benchmark-free zero-drift MCS contrasts, and unrounded `running_max_p`.
2. **Conformal PID / Adaptive Conformal Inference (`Gibbs & Candès 2021`; `Angelopoulos, Candès & Tibshirani 2023`):**
   - Floor-quantized stop widths guaranteed strictly inside `[min_stop_pct, max_stop_pct]` and `0 < stop <= 2.00%`
     after rounding, breach feedback tied directly to the issued stop, and dynamic attainable multiplier bounds.
3. **Full-Information Fixed-Share Hedge / MWU (`Freund & Schapire 1997`) + Hierarchical Empirical-Bayes Cluster
   Shrinkage (`Jensen, Kelly & Pedersen 2023 JF`) + Cederburg-Safe Asymmetric Volatility Crash Brake (`<= 1.0x`):**
   - Difference-first log-sum-exp stabilized Hedge (immune to `1e308` overflow and 6-decimal weight truncation),
     scale-normalized overflow-safe & floor-free Hierarchical EB (rejecting incompatible exact evidence and
     duplicate IDs), and fail-closed vol crash brake preserving non-zero floor.
4. **Next-Wave Pre-Registration Strategy Blueprints (`H109..H114`, study-wall `<= 2026-10-15`):**
   - Canonical `DF1..DF12` feed-aligned registry (`symbol_ranking_v310.py` + `ranking_and_datafeeds.py`).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from statistics import NormalDist
from typing import Any

from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE, hlz_dynamic_t_hurdle
from sota_compounding.ranking_and_datafeeds import CANONICAL_7_PIT_FEEDS
from sota_compounding.symbol_ranking_v310 import CANONICAL_12_RANKING_FEEDS

ADVANCED_CONVERGENCE_SCHEMA = "sota.advanced-sim-merit-convergence.v1"
CURRENT_LEDGER_TRIALS_N = 104
NEXT_TRIAL_ID = 105
NEXT_HLZ_T_STAR = hlz_dynamic_t_hurdle(NEXT_TRIAL_ID)  # 4.2084
_STD_NORM = NormalDist(0.0, 1.0)
_ZERO_VAR_T_DISPLAY_CAP = 1_000_000_000.0
_MIN_REPRESENTABLE_STOP_PCT = 0.0001

REGISTERED_PREWALL_READY_FEEDS: frozenset[str] = frozenset(
    {
        *CANONICAL_7_PIT_FEEDS,
        *CANONICAL_12_RANKING_FEEDS,
        "DF1_SIP",
        "DF2_BOATS",
        "DF3_RAW_VS_ALL",
        "DF4_RVOL_TOD",
        "DF5_EDGARTOOLS_V620",
        "DF7_REGSHO_T1",
        "DF9_IBKR_ETB_SSR",
        "DF10_OWN_HISTORY_TOD_ABNORMAL_RATE_AND_UCURVE",
        "DF12_QLIB_RANK_IC_AND_KAPPA40_POSTWALL_EB_FEEDBACK",
    }
)


def _require_finite(val: float, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite real number (got {val!r})")
    try:
        f = float(val)
    except OverflowError as exc:
        raise ValueError(f"{name} overflows IEEE-754 binary64 (got {val!r})") from exc
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite (got {val!r})")
    if isinstance(val, int) and int(f) != val:
        raise ValueError(f"{name} integer is not exactly representable in IEEE-754 binary64 (got {val!r})")
    return f


def _require_int(val: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be an integer (got {val!r})")
    if minimum is not None and val < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {val})")
    return val


def _exact_series_mean(series: Sequence[float]) -> float:
    """Compute finite mean preserving exact value on constant series (`[0.22]*10 -> 0.22`)."""
    n = len(series)
    if n == 0:
        return 0.0
    if min(series) == max(series):
        return float(series[0])
    return sum(x / n for x in series)


def _stationary_bootstrap_indices(n: int, avg_block_len: float, rng: random.Random) -> list[int]:
    """Generate Politis & Romano (1994) stationary bootstrap index draw of length `n`."""
    n_i = _require_int(n, "n", minimum=0)
    if n_i <= 0:
        return []
    blk = _require_finite(avg_block_len, "avg_block_len")
    if blk < 1.0:
        raise ValueError(f"avg_block_len must be >= 1.0 (got {avg_block_len!r})")
    p = 1.0 / blk
    idx = [rng.randrange(n_i)]
    for _ in range(1, n_i):
        if rng.random() < p:
            idx.append(rng.randrange(n_i))
        else:
            idx.append((idx[-1] + 1) % n_i)
    return idx


def _to_exact_dyadic_ints(series: Sequence[float]) -> tuple[list[int], int]:
    """Represent finite binary64 values exactly as integers `q_i` with `series[i] == q_i * 2^(-shift)`."""
    if not series:
        return [], 0
    ratios = [float(x).as_integer_ratio() for x in series]
    max_shift = max((den.bit_length() - 1 for _, den in ratios), default=0)
    ints = [num << (max_shift - (den.bit_length() - 1)) for num, den in ratios]
    return ints, max_shift


def _exact_hac_int_stats_from_q(
    q: list[int],
    shift: int,
    max_lag: int,
    *,
    indices: Sequence[int] | None = None,
) -> tuple[float, list[int], int, int, int, float]:
    """Compute bit-exact integer Bartlett HAC `(mean, q, sum_q, v_int, w_den, t_float)` from exact dyadic `q`.

    When `indices` (original chronological session indices) is supplied, lag-`l` autocovariance products are
    formed strictly over pairs `(orig_t, orig_t - l)` present in `indices`, never creating fictitious lag-1
    neighbors across held-out or purged CPCV gaps!
    """
    n = len(q)
    if n == 0:
        return 0.0, [], 0, 0, 1, 0.0
    sum_q = sum(q)
    if sum_q == 0:
        mu = 0.0
    else:
        sign_q = -1.0 if sum_q < 0 else 1.0
        abs_sq = abs(sum_q)
        b_len = abs_sq.bit_length()
        trim = max(0, b_len - 53)
        scaled_m = (abs_sq >> trim) if trim > 0 else abs_sq
        mu = sign_q * math.ldexp(float(scaled_m) / float(n), trim - shift)

    if min(q) == max(q):
        t_val = math.inf if sum_q > 0 else (-math.inf if sum_q < 0 else 0.0)
        return mu, q, sum_q, 0, 1, t_val

    c = [n * qi - sum_q for qi in q]
    if indices is None:
        eff_lag = max(0, min(int(max_lag), n - 1))
    else:
        if len(indices) != n:
            raise ValueError(f"indices length {len(indices)} != series length {n}")
        idx_span = max(indices) - min(indices)
        eff_lag = max(0, min(int(max_lag), idx_span))
    w_den = eff_lag + 1
    v_int = w_den * sum(ci * ci for ci in c)
    if indices is None:
        for lag in range(1, eff_lag + 1):
            cov_int = sum(c[t] * c[t - lag] for t in range(lag, n))
            v_int += 2 * (w_den - lag) * cov_int
    else:
        idx_pos = {orig_i: pos for pos, orig_i in enumerate(indices)}
        for lag in range(1, eff_lag + 1):
            cov_int = 0
            for pos_t, orig_t in enumerate(indices):
                prev_pos = idx_pos.get(orig_t - lag)
                if prev_pos is not None:
                    cov_int += c[pos_t] * c[prev_pos]
            v_int += 2 * (w_den - lag) * cov_int
    if v_int <= 0:
        v_int = 0
        t_val = math.inf if sum_q > 0 else (-math.inf if sum_q < 0 else 0.0)
        return mu, q, sum_q, 0, w_den, t_val

    with localcontext() as ctx:
        ctx.prec = 80
        d_t = (Decimal(sum_q) * Decimal(n) * Decimal(w_den).sqrt()) / Decimal(v_int).sqrt()
        t_val = float(d_t)
    return mu, q, sum_q, v_int, w_den, t_val


def _exact_hac_int_stats(
    series: Sequence[float],
    max_lag: int,
    *,
    indices: Sequence[int] | None = None,
) -> tuple[float, list[int], int, int, int, float]:
    """Compute bit-exact integer Bartlett HAC `(mean, q_ints, sum_q, v_int, w_den, t_float)` with zero ULP drift."""
    if not series:
        return 0.0, [], 0, 0, 1, 0.0
    q, shift = _to_exact_dyadic_ints(series)
    mu, q_out, sum_q, v_int, w_den, t_val = _exact_hac_int_stats_from_q(q, shift, max_lag, indices=indices)
    if min(q) == max(q):
        mu = float(series[0])
    return mu, q_out, sum_q, v_int, w_den, t_val


def _exact_t_cmp(num_a: int, v_a: int, num_b: int, v_b: int) -> int:
    """Bit-exact comparison of studentized ratios `(num_a / sqrt(v_a))` vs `(num_b / sqrt(v_b))` (same n & w_den).

    Returns `-1` if `t_a < t_b`, `0` if `t_a == t_b`, `+1` if `t_a > t_b`. Zero sqrt/division ULP drift!
    """

    def _tier(num: int, v: int) -> int:
        if v == 0:
            return 2 if num > 0 else (-2 if num < 0 else 0)
        return 1 if num > 0 else (-1 if num < 0 else 0)

    ta = _tier(num_a, v_a)
    tb = _tier(num_b, v_b)
    if ta != tb:
        return -1 if ta < tb else 1
    if ta in (-2, 0, 2):
        return 0
    lhs = num_a * num_a * v_b
    rhs = num_b * num_b * v_a
    if lhs == rhs:
        return 0
    if ta > 0:
        return 1 if lhs > rhs else -1
    return -1 if lhs > rhs else 1


def _exact_bartlett_hac_t_key(
    series: Sequence[float],
    max_lag: int,
    *,
    indices: Sequence[int] | None = None,
) -> tuple[int, int]:
    """Return exact `(sum_q, v_int)` for Bartlett HAC t-ranking (preserving original session distances if given)."""
    _, _, sum_q, v_int, _, _ = _exact_hac_int_stats(series, max_lag, indices=indices)
    return sum_q, v_int


def _newey_west_long_run_var(series: Sequence[float], max_lag: int) -> float:
    """Bartlett-kernel HAC long-run variance estimator for autocorrelated daily performance differentials."""
    n = len(series)
    if n < 2:
        return 0.0
    q, shift = _to_exact_dyadic_ints(series)
    if min(q) == max(q):
        return 0.0
    _, _, _, v_int, w_den, _ = _exact_hac_int_stats_from_q(q, shift, max_lag)
    if v_int <= 0:
        return 0.0
    with localcontext() as ctx:
        ctx.prec = 80
        d_lrv = Decimal(v_int) / (Decimal(w_den) * Decimal(n) ** 3)
        if shift > 0:
            d_lrv = d_lrv / (Decimal(2) ** (2 * shift))
        elif shift < 0:
            d_lrv = d_lrv * (Decimal(2) ** (-2 * shift))
        return float(d_lrv)


def _display_finite_t(t_val: float, *, inf_cap: float = _ZERO_VAR_T_DISPLAY_CAP) -> float:
    if math.isfinite(t_val):
        return round(t_val, 4)
    if t_val > 0.0:
        return inf_cap
    return -inf_cap


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

    Uses bit-exact dyadic integer arithmetic and exact cross-multiplied Bartlett variance comparisons so
    neither bootstrap ties (`[5,-3,1,1,1,1,1,0,2^-54,0]`) nor equal-SE complements (`B = const - A`) suffer
    even 1 ULP of floating-point drift across `[2^-1074, 1.797e308]`.
    """
    if not strategy_returns_bps:
        raise ValueError("strategy_returns_bps must be non-empty")
    alpha_f = _require_finite(alpha, "alpha")
    if not (0.001 <= alpha_f < 0.5):
        raise ValueError(f"alpha must be in [0.001, 0.5) (got {alpha})")
    n_boot = _require_int(n_bootstrap, "n_bootstrap", minimum=100)
    blk_len = _require_finite(avg_block_len, "avg_block_len")
    if blk_len < 1.0:
        raise ValueError(f"avg_block_len must be >= 1.0 (got {avg_block_len})")
    seed_i = _require_int(seed, "seed")

    names = sorted(strategy_returns_bps.keys())
    lengths = {len(strategy_returns_bps[k]) for k in names}
    if len(lengths) != 1:
        raise ValueError(f"All strategy return series must share identical length T (got {sorted(lengths)})")
    t_len = next(iter(lengths))
    if t_len < 10:
        raise ValueError(f"Require >= 10 aligned periods for stationary bootstrap (got {t_len})")
    if blk_len > t_len / 2.0:
        raise ValueError(f"avg_block_len={blk_len} exceeds stationary bootstrap bound T/2={t_len / 2.0}")

    if benchmark_returns_bps is None:
        bench = [0.0] * t_len
    else:
        if len(benchmark_returns_bps) != t_len:
            raise ValueError(f"benchmark_returns_bps length {len(benchmark_returns_bps)} != T={t_len}")
        bench = [_require_finite(x, f"benchmark[{i}]") for i, x in enumerate(benchmark_returns_bps)]

    orig: dict[str, list[float]] = {}
    diff_q_by_strat: dict[str, tuple[list[int], int]] = {}
    for nm in names:
        vals = [_require_finite(x, f"{nm}[{i}]") for i, x in enumerate(strategy_returns_bps[nm])]
        orig[nm] = vals
        for i in range(t_len):
            d_val = vals[i] - bench[i]
            if not math.isfinite(d_val):
                raise ValueError(f"Non-finite performance differential for {nm}[{i}]")
        # Subtract on a common exact dyadic integer grid BEFORE any float64 rounding (preserves `-3 - 2^-54`!)
        joint_q, joint_shift = _to_exact_dyadic_ints([*vals, *bench])
        diff_q_by_strat[nm] = ([joint_q[i] - joint_q[t_len + i] for i in range(t_len)], joint_shift)

    nw_lag = max(1, int(round(blk_len)))
    means: dict[str, float] = {}
    mean_fracs: dict[str, Fraction] = {}
    t_stats: dict[str, float] = {}
    q_series: dict[str, list[int]] = {}
    sum_q_map: dict[str, int] = {}
    v_int_map: dict[str, int] = {}
    for nm in names:
        q_d, sh_d = diff_q_by_strat[nm]
        mu, q_ints, s_q, v_i, _, tv = _exact_hac_int_stats_from_q(q_d, sh_d, nw_lag)
        if not math.isfinite(mu):
            raise ValueError(f"Non-finite mean differential for {nm}")
        means[nm] = mu
        mean_fracs[nm] = Fraction(s_q, t_len << sh_d) if sh_d >= 0 else Fraction(s_q << (-sh_d), t_len)
        t_stats[nm] = tv
        q_series[nm] = q_ints
        sum_q_map[nm] = s_q
        v_int_map[nm] = v_i

    # Exact Hansen (2005) Eq. 11 consistent recentering threshold: -sqrt(2 * log(log(T))) over T >= 10
    c_thresh = -math.sqrt(2.0 * math.log(math.log(float(t_len))))
    recenter_q: dict[str, int] = {nm: (sum_q_map[nm] if t_stats[nm] <= c_thresh else 0) for nm in names}

    # Exact observed max-t numerator & variance pair: max(0, max_i t_i)
    obs_best_num = 0
    obs_best_v = 1
    for nm in names:
        if _exact_t_cmp(sum_q_map[nm], v_int_map[nm], obs_best_num, obs_best_v) > 0:
            obs_best_num = sum_q_map[nm]
            obs_best_v = v_int_map[nm]
    t_spa_obs = max(0.0, max(t_stats.values()))

    rng = random.Random(seed_i)
    boot_draws = [_stationary_bootstrap_indices(t_len, blk_len, rng) for _ in range(n_boot)]

    # Consistent, Lower, and Upper SPA p-values (Hansen 2005 Eq. 9) via bit-exact integer comparisons
    exceed_c = 0
    exceed_l = 0
    exceed_u = 0
    for idx_draw in boot_draws:
        hit_c = _exact_t_cmp(0, 1, obs_best_num, obs_best_v) >= 0
        hit_l = hit_c
        hit_u = hit_c
        for nm in names:
            v_i = v_int_map[nm]
            if v_i == 0:
                b_dev = 0
            else:
                q_nm = q_series[nm]
                b_dev = sum(q_nm[j] for j in idx_draw) - sum_q_map[nm]
            if not hit_u and _exact_t_cmp(b_dev, v_i, obs_best_num, obs_best_v) >= 0:
                hit_u = True
            if not hit_c:
                num_c = b_dev + min(0, recenter_q[nm])
                if _exact_t_cmp(num_c, v_i, obs_best_num, obs_best_v) >= 0:
                    hit_c = True
            if not hit_l:
                num_l = b_dev + min(0, sum_q_map[nm])
                if _exact_t_cmp(num_l, v_i, obs_best_num, obs_best_v) >= 0:
                    hit_l = True
            if hit_u and hit_c and hit_l:
                break
        if hit_c:
            exceed_c += 1
        if hit_l:
            exceed_l += 1
        if hit_u:
            exceed_u += 1

    spa_p_consistent = (exceed_c + 1.0) / (n_boot + 1.0)
    spa_p_lower = (exceed_l + 1.0) / (n_boot + 1.0)
    spa_p_upper = (exceed_u + 1.0) / (n_boot + 1.0)

    # Hansen-Lunde-Nason (2011) Model Confidence Set (M*) via exact integer max-t sequential elimination.
    remaining = list(names)
    eliminated: list[dict[str, Any]] = []
    mcs_pvals_raw: dict[str, float] = {}
    running_max_p = 0.0

    # Pre-convert all original strategy series into a single shared exact dyadic integer grid
    all_ratios = {nm: [x.as_integer_ratio() for x in orig[nm]] for nm in names}
    global_shift = max(
        (den.bit_length() - 1 for rows_r in all_ratios.values() for _, den in rows_r),
        default=0,
    )
    orig_q: dict[str, list[int]] = {
        nm: [num << (global_shift - (den.bit_length() - 1)) for num, den in all_ratios[nm]] for nm in names
    }

    while len(remaining) > 1:
        k_cur = len(remaining)
        sum_by_t = [sum(orig_q[nm][t] for nm in remaining) for t in range(t_len)]
        # Exact integer relative loss contrast: d_i(t) = avg(t) - x_i(t) = (sum_by_t[t] - k_cur * orig_q[i][t]) / k_cur
        # Positive loss_q means strategy `nm` underperformed the peer average!
        eff_lag = max(0, min(nw_lag, t_len - 1))
        w_den = eff_lag + 1
        loss_q_series: dict[str, list[int]] = {}
        loss_sum_q: dict[str, int] = {}
        loss_v_int: dict[str, int] = {}
        for nm in remaining:
            row_l = [sum_by_t[t] - k_cur * orig_q[nm][t] for t in range(t_len)]
            s_l = sum(row_l)
            if min(row_l) == max(row_l):
                v_l = 0
            else:
                c_l = [t_len * x - s_l for x in row_l]
                v_l = w_den * sum(ci * ci for ci in c_l)
                for lag in range(1, eff_lag + 1):
                    v_l += 2 * (w_den - lag) * sum(c_l[t] * c_l[t - lag] for t in range(lag, t_len))
                v_l = max(0, v_l)
            loss_q_series[nm] = row_l
            loss_sum_q[nm] = s_l
            loss_v_int[nm] = v_l

        worst_nm = remaining[0]
        for nm in remaining[1:]:
            cmp_w = _exact_t_cmp(loss_sum_q[nm], loss_v_int[nm], loss_sum_q[worst_nm], loss_v_int[worst_nm])
            if cmp_w > 0 or (cmp_w == 0 and nm > worst_nm):
                worst_nm = nm
        obs_mcs_num = loss_sum_q[worst_nm]
        obs_mcs_v = loss_v_int[worst_nm]

        exceed_mcs = 0
        for idx_draw in boot_draws:
            hit_mcs = False
            for nm in remaining:
                v_l = loss_v_int[nm]
                if v_l == 0:
                    b_dev_l = 0
                else:
                    lq = loss_q_series[nm]
                    b_dev_l = sum(lq[j] for j in idx_draw) - loss_sum_q[nm]
                if _exact_t_cmp(b_dev_l, v_l, obs_mcs_num, obs_mcs_v) >= 0:
                    hit_mcs = True
                    break
            if hit_mcs:
                exceed_mcs += 1

        step_p = (exceed_mcs + 1.0) / (n_boot + 1.0)
        running_max_p = max(running_max_p, step_p)
        mcs_pvals_raw[worst_nm] = running_max_p
        remaining.remove(worst_nm)
        if running_max_p < alpha_f:
            eliminated.append({"strategy_id": worst_nm, "mcs_p_value": round(running_max_p, 6)})

    if remaining:
        mcs_pvals_raw[remaining[0]] = 1.0

    # Decide surviving vs eliminated strictly from unrounded `mcs_pvals_raw` so partition is exact and disjoint.
    surviving = [nm for nm in names if mcs_pvals_raw[nm] >= alpha_f]
    mcs_pvals_display = {nm: round(mcs_pvals_raw[nm], 6) for nm in names}
    best_sid = names[0]
    for nm in names[1:]:
        cmp_b = _exact_t_cmp(sum_q_map[nm], v_int_map[nm], sum_q_map[best_sid], v_int_map[best_sid])
        if cmp_b > 0 or (cmp_b == 0 and (mean_fracs[nm], nm) > (mean_fracs[best_sid], best_sid)):
            best_sid = nm

    # Ensure finite display cap for +inf zero-var t-stats strictly exceeds any finite t-stat in the batch!
    max_finite_t = max((abs(v) for v in t_stats.values() if math.isfinite(v)), default=0.0)
    scaled_t = max_finite_t * 10.0
    if max_finite_t > 0.0 and math.isfinite(scaled_t):
        inf_cap = max(_ZERO_VAR_T_DISPLAY_CAP, float(math.ceil(scaled_t)))
    elif max_finite_t > 0.0:
        inf_cap = sys.float_info.max
    else:
        inf_cap = _ZERO_VAR_T_DISPLAY_CAP

    return {
        "schema": "sota.spa-mcs-validation.v1",
        "periods_t": t_len,
        "n_strategies": len(names),
        "n_bootstrap": n_boot,
        "avg_block_len": blk_len,
        "alpha": alpha_f,
        "best_strategy_id": best_sid,
        "spa_t_stat": _display_finite_t(t_spa_obs, inf_cap=inf_cap),
        "spa_p_lower": round(spa_p_lower, 6),
        "spa_p_consistent": round(spa_p_consistent, 6),
        "spa_p_upper": round(spa_p_upper, 6),
        "spa_passes_alpha": spa_p_consistent < alpha_f,
        "mcs_surviving_strategies": surviving,
        "mcs_eliminated": eliminated,
        "mcs_p_values": mcs_pvals_display,
        "per_strategy_t_stats": {nm: _display_finite_t(t_stats[nm], inf_cap=inf_cap) for nm in names},
        "per_strategy_t_stats_raw": {
            nm: (_display_finite_t(t_stats[nm], inf_cap=inf_cap) if not math.isfinite(t_stats[nm]) else t_stats[nm])
            for nm in names
        },
    }


@dataclass
class ConformalPidStopCalibrator:
    """Conformal PID / ACI online MAE stop calibrator strictly clamped inside `[min_stop_pct, max_stop_pct]` <= 2%.

    Implements Angelopoulos, Candès & Tibshirani (2023 NeurIPS) Conformal PID control over non-exchangeable
    intrabar Maximum Adverse Excursion (`MAE`) with floor-quantized post-rounding compliance against BOTH
    configured `max_stop_pct` and `ADR-0022 (2.00%)`.
    """

    target_miscoverage_alpha: float = 0.10
    kp_learning_rate: float = 0.08
    ki_integral_gain: float = 0.25
    integral_decay: float = 0.90
    quantile_multiplier: float = 1.25
    min_stop_pct: float = 0.15
    max_stop_pct: float = ADR0022_MAX_STOP_PCT
    _base_quantile_multiplier: float | None = None
    _integral_err: float = 0.0
    _steps: int = 0
    _miscoverages: int = 0

    def _validate_bounds(self) -> tuple[float, float, float, float]:
        alpha = _require_finite(self.target_miscoverage_alpha, "target_miscoverage_alpha")
        if not (0.01 <= alpha <= 0.40):
            raise ValueError("target_miscoverage_alpha must be in [0.01, 0.40]")
        kp = _require_finite(self.kp_learning_rate, "kp_learning_rate")
        if kp <= 0.0:
            raise ValueError("kp_learning_rate must be > 0")
        ki = _require_finite(self.ki_integral_gain, "ki_integral_gain")
        if ki < 0.0:
            raise ValueError("ki_integral_gain must be >= 0")
        decay = _require_finite(self.integral_decay, "integral_decay")
        if not (0.5 <= decay <= 1.0):
            raise ValueError("integral_decay must be in [0.5, 1.0]")
        max_s = _require_finite(self.max_stop_pct, "max_stop_pct")
        min_s = _require_finite(self.min_stop_pct, "min_stop_pct")
        if max_s > ADR0022_MAX_STOP_PCT or max_s < _MIN_REPRESENTABLE_STOP_PCT:
            raise ValueError(
                f"ADR-0022 breach: max_stop_pct={self.max_stop_pct} outside "
                f"[{_MIN_REPRESENTABLE_STOP_PCT}, {ADR0022_MAX_STOP_PCT}]%"
            )
        if min_s < _MIN_REPRESENTABLE_STOP_PCT or min_s > max_s:
            raise ValueError(
                f"min_stop_pct={self.min_stop_pct} must lie in [{_MIN_REPRESENTABLE_STOP_PCT}, max_stop_pct]"
            )
        rep_min_i = math.ceil(min_s * 10_000.0)
        while (rep_min_i - 1) / 10_000.0 >= min_s:
            rep_min_i -= 1
        while rep_min_i / 10_000.0 < min_s:
            rep_min_i += 1
        rep_max_i = math.floor(max_s * 10_000.0)
        while (rep_max_i + 1) / 10_000.0 <= max_s:
            rep_max_i += 1
        while rep_max_i / 10_000.0 > max_s:
            rep_max_i -= 1
        rep_min = rep_min_i / 10_000.0
        rep_max = rep_max_i / 10_000.0
        if rep_min > rep_max or rep_max <= 0.0:
            raise ValueError(f"No representable 4-decimal stop exists inside [{min_s}, {max_s}]%")
        q_mult = _require_finite(self.quantile_multiplier, "quantile_multiplier")
        if q_mult <= 0.0:
            raise ValueError("quantile_multiplier must be > 0")
        base_raw = (
            self.quantile_multiplier if self._base_quantile_multiplier is None else self._base_quantile_multiplier
        )
        base_q = _require_finite(base_raw, "_base_quantile_multiplier")
        if base_q <= 0.0:
            raise ValueError("_base_quantile_multiplier must be > 0")
        _require_finite(self._integral_err, "_integral_err")
        steps_i = _require_int(self._steps, "_steps", minimum=0)
        misc_i = _require_int(self._miscoverages, "_miscoverages", minimum=0)
        if misc_i > steps_i:
            raise ValueError(f"_miscoverages={misc_i} cannot exceed _steps={steps_i}")
        return min_s, max_s, rep_min, rep_max

    def __post_init__(self) -> None:
        if self._base_quantile_multiplier is None:
            self._base_quantile_multiplier = _require_finite(self.quantile_multiplier, "quantile_multiplier")
        self._validate_bounds()

    def recommended_stop_pct(self, atr_pct: float) -> float:
        """Compute distribution-free stop width (%) strictly inside `[min_stop_pct, max_stop_pct]` after rounding."""
        min_s, max_s, rep_min, rep_max = self._validate_bounds()
        atr_in = _require_finite(atr_pct, "atr_pct")
        if atr_in <= 0.0:
            raise ValueError(f"atr_pct must be > 0 (got {atr_pct!r})")
        atr = max(0.05, atr_in)
        raw = self.quantile_multiplier * atr
        clamped = min(max_s, max(min_s, raw))
        # Floor-quantize downward so rounding can NEVER exceed configured `max_stop_pct` or `ADR0022_MAX_STOP_PCT`.
        issued = math.floor(clamped * 10_000.0 + 1e-12) / 10_000.0
        issued = min(rep_max, max(rep_min, issued))
        if not (min_s <= issued <= max_s and 0.0 < issued <= ADR0022_MAX_STOP_PCT):
            raise ValueError(f"Post-rounding stop {issued} breaches [{min_s}, {max_s}] / ADR-0022")
        return issued

    def update(self, *, realized_mae_pct: float, atr_pct: float) -> dict[str, Any]:
        """Ingest one realized trade intrabar MAE (%) and update Conformal PID quantile online."""
        min_s, max_s, _, _ = self._validate_bounds()
        mae = _require_finite(realized_mae_pct, "realized_mae_pct")
        if mae < 0.0:
            raise ValueError(f"realized_mae_pct must be >= 0 (got {realized_mae_pct!r})")
        atr_in = _require_finite(atr_pct, "atr_pct")
        if atr_in <= 0.0:
            raise ValueError(f"atr_pct must be > 0 (got {atr_pct!r})")
        atr = max(0.05, atr_in)

        pred_stop = self.recommended_stop_pct(atr)
        # Breach feedback MUST match the actual issued stop so clamped stops never flip sign of coverage error.
        breached = mae > pred_stop
        err = 1.0 if breached else 0.0

        self._steps += 1
        if breached:
            self._miscoverages += 1

        gap = err - self.target_miscoverage_alpha
        min_mult = min(0.35, min_s / atr)
        max_mult = max(4.0, (max_s / atr) + 0.25)
        base_cap = max(3.75, (max_s / atr) + 0.25)

        base_cur = (
            float(self.quantile_multiplier)
            if self._base_quantile_multiplier is None
            else float(self._base_quantile_multiplier)
        )
        self._base_quantile_multiplier = max(min_mult, min(base_cap, base_cur + self.kp_learning_rate * gap))
        self._integral_err = max(-4.0, min(4.0, self.integral_decay * self._integral_err + gap))
        int_boost = self.ki_integral_gain * math.tanh(self._integral_err)
        self.quantile_multiplier = max(min_mult, min(max_mult, self._base_quantile_multiplier + int_boost))
        next_stop = self.recommended_stop_pct(atr)
        return {
            "step": self._steps,
            "previous_stop_pct": pred_stop,
            "realized_mae_pct": round(mae, 4),
            "stop_breached": breached,
            "next_quantile_multiplier": round(self.quantile_multiplier, 4),
            "next_stop_pct": next_stop,
            "adr0022_compliant": min_s <= next_stop <= max_s and 0.0 < next_stop <= ADR0022_MAX_STOP_PCT,
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

    Requires exact matching strategy keys (rejecting both missing and extra marks), shifts clipped returns
    relative to the active maximum BEFORE multiplying by `eta` (immune to `1e308` overflow), and preserves
    full float64 simplex precision (`sum == 1.0`, never zeroing small fixed-share floors across iterative calls).
    """
    if not prior_weights:
        raise ValueError("prior_weights must be non-empty")
    eta_f = _require_finite(eta, "eta")
    if not (0.0 < eta_f <= 2.0):
        raise ValueError(f"eta must be in (0, 2.0] (got {eta})")
    alpha_f = _require_finite(switching_alpha, "switching_alpha")
    if not (0.0 <= alpha_f < 0.5):
        raise ValueError(f"switching_alpha must be in [0, 0.5) (got {switching_alpha})")
    clip_f = _require_finite(r_clip, "r_clip")
    if clip_f <= 0.0:
        raise ValueError(f"r_clip must be > 0 (got {r_clip})")

    keys = sorted(prior_weights.keys())
    if set(realized_r_by_strategy.keys()) != set(keys):
        missing = sorted(set(keys) - set(realized_r_by_strategy.keys()))
        extra = sorted(set(realized_r_by_strategy.keys()) - set(keys))
        raise ValueError(f"realized_r_by_strategy key mismatch (missing={missing}, extra={extra})")

    k_cnt = len(keys)
    clean_priors: dict[str, float] = {}
    for k in keys:
        pw = _require_finite(prior_weights[k], f"w[{k}]")
        if pw < 0.0:
            raise ValueError(f"prior weight w[{k}] must be >= 0 (got {pw})")
        clean_priors[k] = pw

    clipped_r: dict[str, float] = {}
    for k in keys:
        r_val = _require_finite(realized_r_by_strategy[k], f"r[{k}]")
        clipped_r[k] = max(-clip_f, min(clip_f, r_val))

    active_keys = [k for k in keys if clean_priors[k] > 0.0]
    if not active_keys:
        raise ValueError("prior_weights must contain at least one strictly positive weight")
    max_r = max(clipped_r[k] for k in active_keys)
    # Direct `math.log(prior)` avoids underflow on `(1e-300, 1e300)` ratios; subtract `r - max_r` FIRST
    # when finite to preserve precision on `(1e16, 1e16-2)`, falling back cleanly on `±1e308` overflow.
    log_scores: dict[str, float] = {}
    for k in active_keys:
        delta_r = clipped_r[k] - max_r
        if math.isfinite(delta_r):
            scaled = eta_f * delta_r
        else:
            er_k = eta_f * clipped_r[k]
            er_max = eta_f * max_r
            if math.isfinite(er_k) and math.isfinite(er_max) and math.isfinite(er_k - er_max):
                scaled = er_k - er_max
            else:
                half_diff = (clipped_r[k] * 0.5) - (max_r * 0.5)
                scaled = (2.0 * eta_f) * half_diff if math.isfinite(2.0 * eta_f) else (eta_f * half_diff) * 2.0
        log_scores[k] = math.log(clean_priors[k]) + scaled
    max_log = max(log_scores.values())
    shifted_exp = {
        k: (math.exp(log_scores[k] - max_log) if (k in log_scores and math.isfinite(log_scores[k] - max_log)) else 0.0)
        for k in keys
    }
    z = sum(shifted_exp.values())
    post = {k: shifted_exp[k] / z for k in keys} if z > 0.0 else {k: 1.0 / k_cnt for k in keys}

    mixed = {k: (1.0 - alpha_f) * post[k] + (alpha_f / k_cnt) for k in keys}
    m_sum = sum(mixed.values())
    return {k: mixed[k] / m_sum for k in keys}


def _dec_logaddexp(a: Decimal | None, b: Decimal | None) -> Decimal | None:
    if a is None:
        return b
    if b is None:
        return a
    m = a if a >= b else b
    diff = abs(a - b)
    if diff > Decimal(200):
        return m
    return m + (Decimal(1) + (-diff).exp()).ln()


def run_fixed_share_hedge_sequence(
    prior_weights: Mapping[str, float],
    reward_sequence: Sequence[Mapping[str, float]],
    *,
    eta: float = 0.25,
    switching_alpha: float = 0.05,
    r_clip: float = 3.0,
) -> dict[str, float]:
    """Run multi-step Fixed-Share Hedge in 80-digit `Decimal` log-weight space.

    Preserves exact recoverability across zero-share runs (`125*(+3,-3) + 125*(-3,+3) -> (0.5, 0.5)`),
    never loses intermediate rewards like `[1e16, 1, -1e16]` to float64 absorption, and never underflows
    subnormal `switching_alpha = 2^-1074` before taking `log`.
    """
    if not prior_weights:
        raise ValueError("prior_weights must be non-empty")
    eta_f = _require_finite(eta, "eta")
    if not (0.0 < eta_f <= 2.0):
        raise ValueError(f"eta must be in (0, 2.0] (got {eta})")
    alpha_f = _require_finite(switching_alpha, "switching_alpha")
    if not (0.0 <= alpha_f < 0.5):
        raise ValueError(f"switching_alpha must be in [0, 0.5) (got {switching_alpha})")
    clip_f = _require_finite(r_clip, "r_clip")
    if clip_f <= 0.0:
        raise ValueError(f"r_clip must be > 0 (got {r_clip})")

    keys = sorted(prior_weights.keys())
    k_cnt = len(keys)
    clean_priors: dict[str, float] = {}
    for k in keys:
        pw = _require_finite(prior_weights[k], f"w[{k}]")
        if pw < 0.0:
            raise ValueError(f"prior weight w[{k}] must be >= 0 (got {pw})")
        clean_priors[k] = pw
    active_keys = [k for k in keys if clean_priors[k] > 0.0]
    if not active_keys:
        raise ValueError("prior_weights must contain at least one strictly positive weight")

    with localcontext() as ctx:
        ctx.prec = 80
        d_eta = Decimal.from_float(eta_f)
        d_clip = Decimal.from_float(clip_f)
        d_neg_clip = -d_clip
        d_one = Decimal(1)

        log_w: dict[str, Decimal | None] = {
            k: (Decimal.from_float(clean_priors[k]).ln() if clean_priors[k] > 0.0 else None) for k in keys
        }
        log_z0: Decimal | None = None
        for lw in log_w.values():
            log_z0 = _dec_logaddexp(log_z0, lw)
        assert log_z0 is not None
        log_w = {k: (None if lw is None else lw - log_z0) for k, lw in log_w.items()}

        if alpha_f == 0.0:
            # Zero switching share: accumulate exact Fraction reward sums (immune even to `[2**400, 1, -2**400]`!)
            f_clip = Fraction(clip_f)
            f_neg_clip = -f_clip
            f_eta = Fraction(eta_f)
            cum_r: dict[str, Fraction] = {k: Fraction(0, 1) for k in active_keys}
            for step_idx, step_r in enumerate(reward_sequence):
                if set(step_r.keys()) != set(keys):
                    missing = sorted(set(keys) - set(step_r.keys()))
                    extra = sorted(set(step_r.keys()) - set(keys))
                    raise ValueError(f"reward_sequence[{step_idx}] key mismatch (missing={missing}, extra={extra})")
                for k in keys:
                    rv = _require_finite(step_r[k], f"r[{step_idx}][{k}]")
                    if k in cum_r:
                        fr = Fraction(rv)
                        cum_r[k] += max(f_neg_clip, min(f_clip, fr))
            max_cr = max(cum_r.values())
            for k in active_keys:
                cur_lw = log_w[k]
                assert cur_lw is not None
                f_delta = (cum_r[k] - max_cr) * f_eta
                log_w[k] = cur_lw + (Decimal(f_delta.numerator) / Decimal(f_delta.denominator))
        else:
            d_alpha = Decimal.from_float(alpha_f)
            d_log_1m_a = (d_one - d_alpha).ln()
            # Subtract ln(k_cnt) directly so subnormal `switching_alpha = 2^-1074` never underflows!
            d_log_a_share: Decimal | None = d_alpha.ln() - Decimal(k_cnt).ln()

            for step_idx, step_r in enumerate(reward_sequence):
                if set(step_r.keys()) != set(keys):
                    missing = sorted(set(keys) - set(step_r.keys()))
                    extra = sorted(set(step_r.keys()) - set(keys))
                    raise ValueError(f"reward_sequence[{step_idx}] key mismatch (missing={missing}, extra={extra})")
                clipped = {
                    k: max(
                        d_neg_clip, min(d_clip, Decimal.from_float(_require_finite(step_r[k], f"r[{step_idx}][{k}]")))
                    )
                    for k in keys
                }
                step_active = [k for k in keys if log_w[k] is not None]
                max_r = max(clipped[k] for k in step_active)
                for k in step_active:
                    cur_lw = log_w[k]
                    assert cur_lw is not None
                    log_w[k] = cur_lw + d_eta * (clipped[k] - max_r)
                log_z: Decimal | None = None
                for lw in log_w.values():
                    log_z = _dec_logaddexp(log_z, lw)
                assert log_z is not None
                log_w = {k: (None if lw is None else lw - log_z) for k, lw in log_w.items()}
                log_w = {
                    k: (
                        _dec_logaddexp(d_log_1m_a + lw, d_log_a_share)
                        if (lw := log_w[k]) is not None
                        else d_log_a_share
                    )
                    for k in keys
                }
                log_zm: Decimal | None = None
                for lw in log_w.values():
                    log_zm = _dec_logaddexp(log_zm, lw)
                assert log_zm is not None
                log_w = {k: (None if lw is None else lw - log_zm) for k, lw in log_w.items()}

        # Single-rounded 80-digit log-sum-exp normalization preserves representable subnormals (`2^-1074`)
        # even across multi-arm splits (e.g., 3-arm `[-744.4, 0, 0]`, avoiding double-rounding underflow!).
        log_z_final: Decimal | None = None
        for lw in log_w.values():
            log_z_final = _dec_logaddexp(log_z_final, lw)
        assert log_z_final is not None
        out_w: dict[str, float] = {}
        for k in keys:
            lw = log_w[k]
            if lw is None:
                out_w[k] = 0.0
            else:
                d_log_p = lw - log_z_final
                out_w[k] = float(d_log_p.exp()) if d_log_p > Decimal(-760) else 0.0
        return out_w


@dataclass(frozen=True, slots=True)
class StrategyClusterObservation:
    strategy_id: str
    cluster_id: str
    observed_mean_r: float
    observed_se_r: float
    closed_roundtrips: int
    exact_mean_r: Fraction | None = None
    exact_se_sq_r: Fraction | None = None


def compute_hierarchical_eb_merit_pool(
    observations: Sequence[StrategyClusterObservation],
    *,
    tau_between_clusters: float = 0.12,
    tau_within_cluster: float = 0.08,
) -> dict[str, dict[str, Any]]:
    """Jensen, Kelly & Pedersen (2023 JF) Hierarchical Empirical-Bayes partial pooling across strategy clusters.

    Uses exact `Fraction` cluster & posterior algebra (100% permutation-invariant across all scales and optional
    exact subnormal `Fraction` R-unit inputs), handles zero-variance limits analytically, and rejects duplicate
    strategy IDs or contradictory exact (`se=0`) observations.
    """
    if not observations:
        return {}
    tb = _require_finite(tau_between_clusters, "tau_between_clusters")
    tw = _require_finite(tau_within_cluster, "tau_within_cluster")
    if tb < 0.0:
        raise ValueError(f"tau_between_clusters must be >= 0 (got {tau_between_clusters})")
    if tw < 0.0:
        raise ValueError(f"tau_within_cluster must be >= 0 (got {tau_within_cluster})")

    seen_ids: set[str] = set()
    by_cluster: dict[str, list[StrategyClusterObservation]] = {}
    for obs in observations:
        if obs.strategy_id in seen_ids:
            raise ValueError(f"Duplicate strategy_id in EB pool: {obs.strategy_id!r}")
        seen_ids.add(obs.strategy_id)
        _require_finite(obs.observed_mean_r, f"{obs.strategy_id}.mean_r")
        se_v = _require_finite(obs.observed_se_r, f"{obs.strategy_id}.se_r")
        if se_v < 0.0:
            raise ValueError(f"{obs.strategy_id}.se_r must be >= 0 (got {obs.observed_se_r})")
        if obs.exact_mean_r is not None and not isinstance(obs.exact_mean_r, Fraction):
            raise ValueError(f"{obs.strategy_id}.exact_mean_r must be a Fraction or None")
        if obs.exact_se_sq_r is not None and (not isinstance(obs.exact_se_sq_r, Fraction) or obs.exact_se_sq_r < 0):
            raise ValueError(f"{obs.strategy_id}.exact_se_sq_r must be a non-negative Fraction or None")
        _require_int(obs.closed_roundtrips, f"{obs.strategy_id}.closed_roundtrips", minimum=0)
        by_cluster.setdefault(obs.cluster_id, []).append(obs)

    def _obs_exact_pair(row: StrategyClusterObservation) -> tuple[Fraction, Fraction]:
        f_m = row.exact_mean_r if row.exact_mean_r is not None else Fraction(float(row.observed_mean_r))
        f_s2 = row.exact_se_sq_r if row.exact_se_sq_r is not None else (Fraction(float(row.observed_se_r)) ** 2)
        return f_m, f_s2

    def _f2d(frac: Fraction) -> Decimal:
        return Decimal(frac.numerator) / Decimal(frac.denominator)

    out: dict[str, dict[str, Any]] = {}
    with localcontext() as ctx:
        ctx.prec = 80
        d_zero = Decimal(0)
        f_zero = Fraction(0, 1)
        f_one = Fraction(1, 1)
        f_tb = Fraction(tb)
        f_tw = Fraction(tw)
        f_tb2 = f_tb * f_tb
        f_tw2 = f_tw * f_tw

        for cluster_id, rows in sorted(by_cluster.items()):
            exact_pairs = {r.strategy_id: _obs_exact_pair(r) for r in rows}
            # Reject contradictory exact observations (`se^2 == 0`) under zero within-cluster or total variance
            exact_zero_rows = [r for r in rows if exact_pairs[r.strategy_id][1] == f_zero]
            if tw == 0.0 and exact_zero_rows:
                exact_means = [exact_pairs[r.strategy_id][0] for r in exact_zero_rows]
                if min(exact_means) != max(exact_means):
                    raise ValueError(
                        f"Incompatible exact observations (se=0) in cluster {cluster_id!r} under tau_within_cluster=0"
                    )
                if tb == 0.0 and exact_means[0] != f_zero:
                    raise ValueError(
                        f"Incompatible exact observation (se=0, mean={exact_means[0]}) against zero prior variance"
                    )

            # Exact `Fraction` cluster & posterior algebra is 100% permutation-invariant across ALL scales
            # (eliminating `[2**400, 3, -2**400]`, `[1e100, 3, -1e100]` & subnormal `2^-1074 / 100` underflow!).
            if tb == 0.0:
                f_cl_mean = f_zero
                f_cl_var = f_zero
            elif tw == 0.0 and exact_zero_rows:
                f_cl_mean = exact_pairs[exact_zero_rows[0].strategy_id][0]
                f_cl_var = f_zero
            else:
                f_prec_sum = f_one / f_tb2
                f_num_sum = f_zero
                for r in rows:
                    f_m_i, f_s2_i = exact_pairs[r.strategy_id]
                    f_prec_i = f_one / (f_tw2 + f_s2_i)
                    f_prec_sum += f_prec_i
                    f_num_sum += f_prec_i * f_m_i
                f_cl_mean = f_num_sum / f_prec_sum
                f_cl_var = f_one / f_prec_sum

            d_cl_mean = _f2d(f_cl_mean)
            c_vals: list[float] = []
            for r in rows:
                if r.exact_mean_r is not None:
                    em_f = float(_f2d(exact_pairs[r.strategy_id][0]))
                    c_vals.append(em_f if math.isfinite(em_f) else float(r.observed_mean_r))
                else:
                    c_vals.append(float(r.observed_mean_r))
            c_lo = min(0.0, *c_vals)
            c_hi = max(0.0, *c_vals)
            raw_cl_f = float(d_cl_mean)
            cluster_post_mean = (
                max(c_lo, min(c_hi, raw_cl_f)) if math.isfinite(raw_cl_f) else (c_hi if d_cl_mean > d_zero else c_lo)
            )

            for r in rows:
                f_obs, f_se2 = exact_pairs[r.strategy_id]
                if r.exact_mean_r is not None:
                    em_r = float(_f2d(f_obs))
                    obs_m = em_r if math.isfinite(em_r) else float(r.observed_mean_r)
                else:
                    obs_m = float(r.observed_mean_r)
                if tw == 0.0:
                    w_obs = 0.0
                    f_post_mean = f_cl_mean
                    f_post_var = f_cl_var
                elif f_se2 == f_zero:
                    w_obs = 1.0
                    f_post_mean = f_obs
                    f_post_var = f_zero
                else:
                    f_denom = f_tw2 + f_se2
                    f_w_obs = f_tw2 / f_denom
                    f_w_cl = f_se2 / f_denom
                    w_obs = float(_f2d(f_w_obs))
                    f_post_mean = (f_tw2 * f_obs + f_se2 * f_cl_mean) / f_denom
                    f_post_var = (f_tw2 * f_se2) / f_denom + (f_w_cl * f_w_cl) * f_cl_var

                d_post_mean = _f2d(f_post_mean)
                d_post_var = _f2d(f_post_var)
                d_post_sd = d_post_var.sqrt() if d_post_var > d_zero else d_zero
                if d_post_sd > d_zero:
                    d_z = d_post_mean / d_post_sd
                    if d_z > Decimal(40):
                        prob_positive = 1.0
                    elif d_z < Decimal(-40):
                        prob_positive = 0.0
                    else:
                        prob_positive = 1.0 - _STD_NORM.cdf(float(-d_z))
                else:
                    prob_positive = 1.0 if f_post_mean > f_zero else 0.0

                lo_m = min(obs_m, cluster_post_mean)
                hi_m = max(obs_m, cluster_post_mean)
                raw_pm_f = float(d_post_mean)
                post_mean = (
                    max(lo_m, min(hi_m, raw_pm_f))
                    if math.isfinite(raw_pm_f)
                    else (hi_m if f_post_mean > f_zero else lo_m)
                )
                post_sd = float(d_post_sd) if d_post_sd > d_zero else 0.0

                sd_display = post_sd if (0.0 < post_sd < 1e-3) else round(post_sd, 4)
                mean_display = post_mean if (0.0 < abs(post_mean) < 1e-3) else round(post_mean, 4)
                out[r.strategy_id] = {
                    "cluster_id": cluster_id,
                    "closed_roundtrips": int(r.closed_roundtrips),
                    "observed_mean_r": round(obs_m, 4),
                    "cluster_posterior_mean_r": (
                        cluster_post_mean if (0.0 < abs(cluster_post_mean) < 1e-3) else round(cluster_post_mean, 4)
                    ),
                    "hierarchical_eb_mean_r": mean_display,
                    "hierarchical_eb_sd_r": sd_display,
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

    Fails closed on non-positive or boolean volatility inputs and preserves non-zero `min_scale` without
    rounding positive small floors down to `0.0`.
    """
    rv = _require_finite(realized_vol_ann_pct, "realized_vol_ann_pct")
    if rv <= 0.0:
        raise ValueError(f"realized_vol_ann_pct must be > 0 (got {realized_vol_ann_pct!r})")
    tv = _require_finite(target_vol_ann_pct, "target_vol_ann_pct")
    if tv <= 0.0:
        raise ValueError(f"target_vol_ann_pct must be > 0 (got {target_vol_ann_pct!r})")
    floor = _require_finite(min_scale, "min_scale")
    if not (0.0 < floor <= 1.0):
        raise ValueError(f"min_scale must lie in (0, 1.0] (got {min_scale!r})")
    clamped = min(1.0, max(floor, tv / rv))
    if clamped < 1e-3:
        return clamped
    return min(1.0, max(floor, round(clamped, 4)))


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


def _feeds_ready_now(feeds: tuple[str, ...]) -> bool:
    return bool(feeds) and all(f in REGISTERED_PREWALL_READY_FEEDS for f in feeds)


NEXT_WAVE_STRATEGY_BLUEPRINTS: tuple[NextWaveStrategyBlueprint, ...] = (
    NextWaveStrategyBlueprint(
        blueprint_id="H109_OVERNIGHT_BOATS_BIFURCATION_AND_RTH_VWAP_FADE",
        family="OVERNIGHT_INTRADAY_BIFURCATION",
        primary_citations=(
            "Lou, Polk & Skouras (2019 JFE) 10.1016/j.jfineco.2019.03.011",
            "Eaton, Shkilko & Werner (2025/2026) SSRN 5181159",
            "Lim (2026) SSRN 6610883",
        ),
        required_feeds=("alpaca_sip", "alpaca_boats", "alpaca_raw_vs_all_adjustment"),
        datafeed_ready_now=_feeds_ready_now(("alpaca_sip", "alpaca_boats", "alpaca_raw_vs_all_adjustment")),
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
        required_feeds=(
            "alpaca_sip",
            "alpaca_raw_vs_all_adjustment",
            "DF10_OWN_HISTORY_TOD_ABNORMAL_RATE_AND_UCURVE",
        ),
        datafeed_ready_now=_feeds_ready_now(
            ("alpaca_sip", "alpaca_raw_vs_all_adjustment", "DF10_OWN_HISTORY_TOD_ABNORMAL_RATE_AND_UCURVE")
        ),
        mechanism_summary=(
            "Strip SPY/Sector ETF beta to trade directed leader->follower residual catch-up plus mechanical "
            "15:30-16:00 ET LETF/dealer hedging flow conditioned on own-history TOD U-curve vol dispersion."
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
        required_feeds=("alpaca_sip", "alpaca_boats", "sec_edgar_acceptance_datetime"),
        datafeed_ready_now=_feeds_ready_now(("alpaca_sip", "alpaca_boats", "sec_edgar_acceptance_datetime")),
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
        required_feeds=(
            "alpaca_sip",
            "sec_edgar_acceptance_datetime",
            "finra_regsho_cnmsshvol_t1",
            "ibkr_paper_4002_l1",
        ),
        datafeed_ready_now=_feeds_ready_now(
            ("alpaca_sip", "sec_edgar_acceptance_datetime", "finra_regsho_cnmsshvol_t1", "ibkr_paper_4002_l1")
        ),
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
        required_feeds=(
            "alpaca_sip",
            "alpaca_raw_vs_all_adjustment",
            "DF10_OWN_HISTORY_TOD_ABNORMAL_RATE_AND_UCURVE",
        ),
        datafeed_ready_now=_feeds_ready_now(
            ("alpaca_sip", "alpaca_raw_vs_all_adjustment", "DF10_OWN_HISTORY_TOD_ABNORMAL_RATE_AND_UCURVE")
        ),
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
        datafeed_ready_now=_feeds_ready_now(("DATABENTO_DBN_MBP10", "EXCHANGE_NOII_IMBALANCE")),
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
        "ready_now_blueprint_count": sum(1 for bp in NEXT_WAVE_STRATEGY_BLUEPRINTS if bp.datafeed_ready_now),
        "future_l2_l3_blueprint_count": sum(1 for bp in NEXT_WAVE_STRATEGY_BLUEPRINTS if not bp.datafeed_ready_now),
        "next_wave_blueprints": bps,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    payload["manifest_sha256"] = digest
    return payload
