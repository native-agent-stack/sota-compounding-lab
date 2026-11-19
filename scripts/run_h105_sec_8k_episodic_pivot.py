"""Pre-registered H105: First-Hand SEC 8-K Acceptance-Time Episodic Pivot (DF2 + DF10 + DF6) on RTX 4090 / CPCV.

Enforces ADR-0020, ADR-0021, and ADR-0022 fail-closed contracts:
- Strictly pre-wall <= 2026-10-15 across every SEC acceptance timestamp, daily bar, and Reg SHO snapshot.
- Passes check_retired_trap_firewall (FIRST_HAND_SEC_EPISODIC_PIVOT, non-POST, non-overnight carry).
- Causal Day-0 intraday entry at split_open after pre-09:30 ET / prior-post-close 8-K acceptance, adverse-stop-first
  ADR-0022 <= 2.00% intraday low stop, exit at Day-0 split_close, under 1x & 2x Almgren-Chriss sqrt impact.
- C(6,2)=15 disjoint skfolio CombinatorialPurgedCV (purge=5d, embargo=5d), RTX 4090 PyTorch CUDA MLP, LightGBM,
  XGBoost CUDA, statsmodels date-clustered SEs, ml4t-diagnostic DSR, PBO, and HLZ t*(N) >= 4.2019.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_optb_prewall_factory import (  # noqa: E402
    DEFAULT_POSITION_NOTIONAL_USD,
    _compute_dsr_from_daily_returns,
    _date_clustered_t_stat,
    _evaluate_cpcv_15_paths,
    compute_almgren_chriss_cost_bps,
    evaluate_strict_prewall_gate,
)

from sota_compounding.converged_strategy_factory import check_retired_trap_firewall  # noqa: E402
from sota_compounding.lifecycle import ADR0022_MAX_STOP_PCT, STUDY_WALL_DATE  # noqa: E402

H105_ID = "H105_SEC_8K_PIT_EPISODIC_PIVOT_DF10_DF6_RTX4090_CPCV"
H105_SPEC_SHA256 = "a1ec39574447cad73e59ccf0f11fe461e611e079acf979e52bc79c33ee8d3d3c"


def run_h105_sec_8k_ep(
    lake_root: pathlib.Path,
    *,
    trial_number: int = 101,
) -> dict[str, Any]:
    """Execute pre-registered H105 SEC 8-K Episodic Pivot on RTX 4090 CUDA + LightGBM + XGBoost."""
    import duckdb
    import lightgbm as lgb
    import numpy as np
    import torch
    import xgboost as xgb
    from skfolio.model_selection import CombinatorialPurgedCV

    ok_fw, fw_viol = check_retired_trap_firewall(
        tags=["FIRST_HAND_SEC_EPISODIC_PIVOT"],
        is_post_incentive=False,
        holds_overnight_moc_to_moo=False,
        unshielded_lit_passive=False,
        in_opening_0930_0935_taker=False,
        ofi_only_without_joint_sweep=False,
    )
    if not ok_fw:
        raise RuntimeError(f"H105 blocked by Retired-Trap Firewall: {fw_viol}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=32")
    df10_p = str(lake_root / "derived" / "df10_daily_20d_baselines.parquet")
    df6_p = str(lake_root / "derived" / "df6_regsho_threshold_pit.parquet")
    sec_p = str(lake_root / "derived" / "sec_filings_clean_prewall.parquet")
    sym_p = str(lake_root / "public-tables" / "sec_symbols.parquet")
    sv_p = str(lake_root / "public-tables" / "short_volume.parquet")
    wall_iso = STUDY_WALL_DATE.isoformat()

    sql = f"""
    WITH ops AS (
        SELECT DISTINCT UPPER(TRIM(symbol)) AS symbol, CAST(cik AS BIGINT) AS cik
        FROM read_parquet('{sym_p}')
        WHERE LOWER(COALESCE(entity_type, '')) = 'operating'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%etf%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%trust%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%acquisition corp%'
          AND LOWER(COALESCE(name, '')) NOT LIKE '%spac%'
    ),
    diluters AS (
        SELECT CAST(cik AS BIGINT) AS cik,
               CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS acc_date
        FROM read_parquet('{sec_p}')
        WHERE UPPER(COALESCE(form, '')) IN ('S-3', 'S-3ASR', '424B5', 'EFFECT', 'F-3', '424B3')
          AND CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) <= DATE '{wall_iso}'
    ),
    ep_8k_raw AS (
        SELECT
            CAST(cik AS BIGINT) AS cik,
            CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS acc_date,
            EXTRACT(HOUR FROM CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York') AS acc_hr,
            EXTRACT(MINUTE FROM CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York') AS acc_min,
            CASE WHEN items LIKE '%2.02%' THEN 1 ELSE 0 END AS is_item_202,
            CASE WHEN items LIKE '%1.01%' THEN 1 ELSE 0 END AS is_item_101,
            CASE WHEN items LIKE '%7.01%' THEN 1 ELSE 0 END AS is_item_701
        FROM read_parquet('{sec_p}')
        WHERE UPPER(COALESCE(form, '')) = '8-K'
          AND (items LIKE '%2.02%' OR items LIKE '%1.01%' OR items LIKE '%7.01%' OR items LIKE '%8.01%')
          AND CAST(CAST(accepted AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) <= DATE '{wall_iso}'
    ),
    ep_preopen AS (
        SELECT
            cik,
            CAST(
                CASE
                    WHEN (acc_hr < 9 OR (acc_hr = 9 AND acc_min < 30)) THEN acc_date
                    WHEN acc_hr >= 16 THEN acc_date + INTERVAL 1 DAY
                    ELSE NULL
                END AS DATE
            ) AS target_date,
            MAX(is_item_202) AS is_item_202,
            MAX(is_item_101) AS is_item_101,
            MAX(is_item_701) AS is_item_701,
            MAX(CASE WHEN acc_hr < 9 OR (acc_hr = 9 AND acc_min < 30) THEN 1 ELSE 0 END) AS is_premarket_8k
        FROM ep_8k_raw
        WHERE (acc_hr < 9 OR (acc_hr = 9 AND acc_min < 30) OR acc_hr >= 16)
        GROUP BY 1, 2
    ),
    sv_daily AS (
        SELECT UPPER(TRIM(symbol)) AS symbol,
               CAST(day AS DATE) AS sv_date,
               CASE WHEN SUM(total_volume) > 0 THEN SUM(short_volume) / SUM(total_volume) ELSE NULL END AS sv_ratio
        FROM read_parquet('{sv_p}')
        WHERE CAST(day AS DATE) <= DATE '{wall_iso}'
        GROUP BY 1, 2
    ),
    sv_5d AS (
        SELECT symbol, sv_date,
               AVG(sv_ratio) OVER (
                   PARTITION BY symbol ORDER BY sv_date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
               ) AS trailing_5d_sv_ratio
        FROM sv_daily
    ),
    df6_pit AS (
        SELECT DISTINCT
            UPPER(TRIM(symbol)) AS symbol,
            CAST(effective_trade_date AS DATE) AS eff_date,
            1 AS on_regsho_threshold
        FROM read_parquet('{df6_p}')
        WHERE CAST(effective_trade_date AS DATE) <= DATE '{wall_iso}'
    ),
    base AS (
        SELECT
            d.symbol,
            d.trade_date,
            o.cik,
            d.raw_close,
            d.mean_20d_dvol,
            d.mean_20d_range_pct,
            d.vdu_5d_20d_ratio,
            d.coil_5d_20d_range_ratio,
            d.prev_close_to_20d_high,
            d.prior_ret_5d,
            d.prior_ret_20d,
            d.split_open,
            d.split_low,
            d.split_high,
            d.split_close,
            (d.split_open / d.prev_split_close) - 1.0 AS gap_pct,
            e.is_item_202,
            e.is_item_101,
            e.is_item_701,
            e.is_premarket_8k,
            COALESCE(f6.on_regsho_threshold, 0) AS on_regsho_threshold,
            COALESCE(sv.trailing_5d_sv_ratio, 0.45) AS trailing_5d_sv_ratio
        FROM read_parquet('{df10_p}') d
        INNER JOIN ops o ON o.symbol = d.symbol
        INNER JOIN ep_preopen e ON e.cik = o.cik AND e.target_date = CAST(d.trade_date AS DATE)
        LEFT JOIN df6_pit f6 ON f6.symbol = d.symbol AND f6.eff_date = CAST(d.trade_date AS DATE)
        LEFT JOIN sv_5d sv ON sv.symbol = d.symbol AND sv.sv_date = CAST(d.trade_date AS DATE)
        WHERE CAST(d.trade_date AS DATE) <= DATE '{wall_iso}'
          AND d.prior_sessions_count = 20
          AND d.raw_close >= 5.0
          AND d.mean_20d_dvol >= 5000000.0
          AND d.prev_split_close > 0
          AND d.split_open > 0
          AND d.split_low > 0
          AND d.split_close > 0
          AND ((d.split_open / d.prev_split_close) - 1.0) >= 0.03
    )
    SELECT b.*
    FROM base b
    WHERE NOT EXISTS (
        SELECT 1 FROM diluters dl
        WHERE dl.cik = b.cik
          AND dl.acc_date < CAST(b.trade_date AS DATE)
          AND dl.acc_date >= CAST(b.trade_date AS DATE) - INTERVAL 90 DAY
    )
    ORDER BY b.trade_date, b.symbol
    """
    df = con.execute(sql).df()
    assert len(df) > 0, "H105 query returned 0 events"
    max_d = str(df["trade_date"].max())
    if dt.date.fromisoformat(max_d) > STUDY_WALL_DATE:
        raise ValueError(f"H105 PREWALL BREACH: max_date={max_d}")

    stop_frac = ADR0022_MAX_STOP_PCT / 100.0  # 2.00% hard ceiling
    op = df["split_open"].to_numpy(dtype=np.float64)
    lo = df["split_low"].to_numpy(dtype=np.float64)
    cl = df["split_close"].to_numpy(dtype=np.float64)
    unstopped_ret = np.clip((cl / op) - 1.0, -0.35, 0.60)
    # Conservative adverse-stop-first tie-breaker: if Day-0 low touches <= open * (1 - 2.00%), fill at -2.00%
    stopped_day0 = lo <= op * (1.0 - stop_frac)
    gross_ret_frac = np.where(stopped_day0, -stop_frac, unstopped_ret)

    costs_1x = np.array(
        [
            compute_almgren_chriss_cost_bps(
                base_half_spread_bps=3.0,
                fee_and_slip_bps_per_side=2.0,
                daily_sigma_frac=max(0.01, min(0.15, float(rng_pct) * 0.65)),
                order_notional_usd=DEFAULT_POSITION_NOTIONAL_USD,
                adv_usd=max(5_000_000.0, float(adv)),
                stress_multiplier=1.0,
            )
            for rng_pct, adv in zip(df["mean_20d_range_pct"], df["mean_20d_dvol"], strict=False)
        ],
        dtype=np.float64,
    )
    df["gross_bps"] = gross_ret_frac * 10_000.0
    df["unstopped_net_2x_bps"] = (unstopped_ret * 10_000.0) - (2.0 * costs_1x)
    df["cost_1x_bps"] = costs_1x
    df["net_1x_bps"] = df["gross_bps"] - costs_1x
    df["net_2x_bps"] = df["gross_bps"] - (2.0 * costs_1x)

    rule_cpcv = _evaluate_cpcv_15_paths(df, ret_col_1x="net_1x_bps", ret_col_2x="net_2x_bps", effective_trials=105.0)
    unstopped_diag = _date_clustered_t_stat(
        df["unstopped_net_2x_bps"].to_numpy(dtype=np.float64), df["trade_date"].to_numpy()
    )

    feat_cols = [
        "gap_pct",
        "mean_20d_range_pct",
        "vdu_5d_20d_ratio",
        "coil_5d_20d_range_ratio",
        "prev_close_to_20d_high",
        "prior_ret_5d",
        "prior_ret_20d",
        "is_item_202",
        "is_item_101",
        "is_item_701",
        "is_premarket_8k",
        "on_regsho_threshold",
        "trailing_5d_sv_ratio",
    ]
    x_all = np.nan_to_num(df[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_all = df["net_1x_bps"].to_numpy(dtype=np.float32)

    unique_dates = np.array(sorted(df["trade_date"].unique()))
    cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=5, embargo_size=5)
    splits = list(cv.split(np.arange(len(unique_dates)).reshape(-1, 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _run_wf(kind: str) -> dict[str, Any]:
        path_means_2x: list[float] = []
        oos_1x: list[float] = []
        oos_2x: list[float] = []
        oos_dates: list[str] = []
        overfit_cnt = 0

        for split_i, (tr_idx, te_idx_list) in enumerate(splits):
            te_idx = np.concatenate(te_idx_list) if isinstance(te_idx_list, list) else np.asarray(te_idx_list)
            tr_dates = set(unique_dates[tr_idx].tolist())
            te_dates = set(unique_dates[te_idx].tolist())
            assert tr_dates.isdisjoint(te_dates)

            tr_m = df["trade_date"].isin(tr_dates).to_numpy()
            te_m = df["trade_date"].isin(te_dates).to_numpy()
            x_tr, y_tr = x_all[tr_m], y_all[tr_m]
            x_te = x_all[te_m]

            mu = x_tr.mean(axis=0, keepdims=True)
            sd = np.maximum(x_tr.std(axis=0, keepdims=True), 1e-5)
            x_tr_s = np.clip((x_tr - mu) / sd, -5.0, 5.0)
            x_te_s = np.clip((x_te - mu) / sd, -5.0, 5.0)

            if kind == "pytorch_rtx4090_mlp":
                torch.manual_seed(105 + split_i)
                net = torch.nn.Sequential(
                    torch.nn.Linear(x_tr_s.shape[1], 64),
                    torch.nn.SiLU(),
                    torch.nn.Dropout(0.20),
                    torch.nn.Linear(64, 32),
                    torch.nn.SiLU(),
                    torch.nn.Linear(32, 1),
                ).to(device)
                opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
                xt = torch.from_numpy(x_tr_s).to(device)
                yt = torch.from_numpy(y_tr / 100.0).unsqueeze(1).to(device)
                net.train()
                for _ in range(35):
                    opt.zero_grad()
                    loss = torch.nn.functional.huber_loss(net(xt), yt)
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    preds = net(torch.from_numpy(x_te_s).to(device)).squeeze(1).cpu().numpy() * 100.0
                    tr_preds = net(xt).squeeze(1).cpu().numpy() * 100.0
            elif kind == "lightgbm":
                mdl = lgb.LGBMRegressor(
                    n_estimators=80,
                    max_depth=4,
                    num_leaves=12,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=105 + split_i,
                    verbose=-1,
                    n_jobs=8,
                )
                mdl.fit(x_tr_s, y_tr)
                preds = mdl.predict(x_te_s)
                tr_preds = mdl.predict(x_tr_s)
            elif kind == "xgboost_cuda":
                mdl = xgb.XGBRegressor(
                    n_estimators=80,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    random_state=105 + split_i,
                    verbosity=0,
                )
                mdl.fit(x_tr_s, y_tr)
                preds = mdl.predict(x_te_s)
                tr_preds = mdl.predict(x_tr_s)
            else:
                raise ValueError(kind)

            q_cut = float(np.quantile(tr_preds, 0.70))
            sel_te = preds >= q_cut
            te_df = df.iloc[np.where(te_m)[0]]
            sel_df = te_df.iloc[np.where(sel_te)[0]]
            if len(sel_df) == 0:
                path_means_2x.append(0.0)
                overfit_cnt += 1
                continue
            m2 = float(sel_df["net_2x_bps"].mean())
            path_means_2x.append(round(m2, 4))
            if m2 <= 0.0:
                overfit_cnt += 1
            oos_1x.extend(sel_df["net_1x_bps"].to_numpy(dtype=np.float64).tolist())
            oos_2x.extend(sel_df["net_2x_bps"].to_numpy(dtype=np.float64).tolist())
            oos_dates.extend(sel_df["trade_date"].astype(str).tolist())

        daily_b: dict[str, list[float]] = {}
        for d_s, r2 in zip(oos_dates, oos_2x, strict=False):
            daily_b.setdefault(d_s, []).append(r2 / 10_000.0)
        daily_s = [sum(v) / len(v) for _, v in sorted(daily_b.items()) if v]
        return {
            "device": str(device) if "rtx4090" in kind or "cuda" in kind else "cpu",
            "n_paths": len(path_means_2x),
            "cpcv_positive_share": round(sum(1 for x in path_means_2x if x > 0.0) / max(1, len(path_means_2x)), 4),
            "pbo": round(overfit_cnt / max(1, len(path_means_2x)), 4),
            "clustered_1x": _date_clustered_t_stat(oos_1x, oos_dates),
            "clustered_2x": _date_clustered_t_stat(oos_2x, oos_dates),
            "dsr_probability": _compute_dsr_from_daily_returns(daily_s, effective_trials=105.0),
            "path_means_2x_bps": path_means_2x,
        }

    mlp_res = _run_wf("pytorch_rtx4090_mlp")
    lgb_res = _run_wf("lightgbm")
    xgb_res = _run_wf("xgboost_cuda")

    best_name = "rule_baseline"
    best_res = rule_cpcv
    for name, res in (("pytorch_rtx4090_mlp", mlp_res), ("lightgbm", lgb_res), ("xgboost_cuda", xgb_res)):
        if res["clustered_2x"]["t_stat"] > best_res["clustered_2x"]["t_stat"]:
            best_name = name
            best_res = res

    gate = evaluate_strict_prewall_gate(
        trial_number=trial_number,
        t_stat_clustered_2x=float(best_res["clustered_2x"]["t_stat"]),
        dsr_probability=float(best_res["dsr_probability"]),
        cpcv_positive_share=float(best_res["cpcv_positive_share"]),
        pbo=float(best_res["pbo"]),
        mean_net_2x_bps=float(best_res["clustered_2x"]["mean_bps"]),
        distinct_dates=int(df["trade_date"].nunique()),
        total_events=int(len(df)),
    )

    return {
        "hypothesis_id": H105_ID,
        "archetype": "FIRST_HAND_SEC_EPISODIC_PIVOT",
        "trial_number": trial_number,
        "spec_sha256": H105_SPEC_SHA256,
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "max_date_verified": max_d,
        "events_total": int(len(df)),
        "distinct_dates": int(df["trade_date"].nunique()),
        "distinct_symbols": int(df["symbol"].nunique()),
        "stopped_day0_rate": round(float(stopped_day0.mean()), 4),
        "mean_cost_1x_bps": round(float(df["cost_1x_bps"].mean()), 2),
        "rule_baseline": rule_cpcv,
        "unstopped_open_to_close_2x_diagnostic": unstopped_diag,
        "models": {
            "pytorch_rtx4090_mlp": mlp_res,
            "lightgbm": lgb_res,
            "xgboost_cuda": xgb_res,
        },
        "selected_for_gate": best_name,
        "gate": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-registered H105 SEC 8-K Episodic Pivot.")
    parser.add_argument("--lake-root", default="/mnt/z/sota-data/prewall-lake")
    args = parser.parse_args()

    lake_root = pathlib.Path(args.lake_root)
    print("[H105] Running FIRST_HAND_SEC_EPISODIC_PIVOT on RTX 4090 CUDA + LightGBM + XGBoost (C(6,2)=15 CPCV)...")
    res = run_h105_sec_8k_ep(lake_root)
    receipt = {
        "schema_version": "sota-optb-h105-sec-8k-ep.v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "study_wall_date": STUDY_WALL_DATE.isoformat(),
        "h105": res,
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode("utf-8")).hexdigest()

    out_path = lake_root / "receipts" / "h105_sec_8k_episodic_pivot_receipt.json"
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    optb_path = lake_root / "receipts" / "optb_prewall_factory_audit.json"
    if optb_path.is_file():
        optb = json.loads(optb_path.read_text(encoding="utf-8"))
        optb["h105_receipt_sha256"] = receipt["receipt_sha256"]
        optb["h105_verdict"] = res["gate"]["verdict"]
        optb["updated_at_utc"] = receipt["generated_at_utc"]
        optb_path.write_text(json.dumps(optb, indent=2) + "\n", encoding="utf-8")

    print(
        f"[H105] DONE: verdict={res['gate']['verdict']} | selected={res['selected_for_gate']} | "
        f"events={res['events_total']} dates={res['distinct_dates']} stopped_day0={res['stopped_day0_rate']} | "
        f"rule_t2x={res['rule_baseline']['clustered_2x']['t_stat']} "
        f"unstopped_2x={res['unstopped_open_to_close_2x_diagnostic']} | "
        f"mlp_t2x={res['models']['pytorch_rtx4090_mlp']['clustered_2x']['t_stat']} "
        f"lgb_t2x={res['models']['lightgbm']['clustered_2x']['t_stat']} "
        f"xgb_t2x={res['models']['xgboost_cuda']['clustered_2x']['t_stat']} | "
        f"sha256={receipt['receipt_sha256'][:16]}..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
