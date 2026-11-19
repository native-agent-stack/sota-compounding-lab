#!/usr/bin/env python3
"""Empirical SOTA GPU vs CPU Benchmark on Noesis-Carbon + RTX 4090."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
import time
import urllib.request
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sota_compounding.gpu_accelerator import (  # noqa: E402
    configure_rtx4090_torch_backend,
    gpu_block_bootstrap_sharpe_dsr,
    train_batched_cpcv_mlp_gpu,
)


def _bench_parquet_scan(df10_path: pathlib.Path, *, engine: str) -> dict[str, Any]:
    import polars as pl

    if not df10_path.is_file():
        return {"available": False, "path": str(df10_path)}
    t0 = time.perf_counter()
    q = (
        pl.scan_parquet(str(df10_path))
        .filter(pl.col("prior_sessions_count") == 20)
        .filter(pl.col("raw_close") >= 5.0)
        .group_by("symbol")
        .agg(
            [
                pl.len().alias("n_days"),
                pl.col("vdu_5d_20d_ratio").mean().alias("mean_vdu"),
                pl.col("coil_5d_20d_range_ratio").mean().alias("mean_coil"),
            ]
        )
    )
    out = q.collect(engine=engine)
    ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {"available": True, "engine": engine, "rows_out": out.height, "elapsed_ms": ms}


def _bench_serial_vs_batched_cpcv(n_rows: int = 6400, n_feat: int = 13, k_folds: int = 15) -> dict[str, Any]:
    import numpy as np
    import torch

    rng = np.random.default_rng(105)
    x_all = rng.normal(0.0, 1.0, size=(n_rows, n_feat)).astype(np.float32)
    y_all = (x_all[:, 0] * 12.0 - x_all[:, 1] * 6.0 + rng.normal(0.0, 25.0, size=n_rows)).astype(np.float32)

    folds = np.array_split(np.arange(n_rows), 6)
    tr_masks: list[Any] = []
    te_masks: list[Any] = []
    for i in range(6):
        for j in range(i + 1, 6):
            te_idx = np.concatenate([folds[i], folds[j]])
            te_m = np.zeros(n_rows, dtype=bool)
            te_m[te_idx] = True
            tr_masks.append(~te_m)
            te_masks.append(te_m)
    tr_masks = tr_masks[:k_folds]
    te_masks = te_masks[:k_folds]

    # 1. Serial CPU baseline (matching original fold loop)
    t_cpu0 = time.perf_counter()
    for s_i in range(k_folds):
        torch.manual_seed(105 + s_i)
        net = torch.nn.Sequential(
            torch.nn.Linear(n_feat, 64),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(64, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        )
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        xt = torch.from_numpy(x_all[tr_masks[s_i]])
        yt = torch.from_numpy(y_all[tr_masks[s_i]] / 100.0).unsqueeze(1)
        net.train()
        for _ in range(35):
            opt.zero_grad()
            loss = torch.nn.functional.huber_loss(net(xt), yt)
            loss.backward()
            opt.step()
    cpu_serial_ms = round((time.perf_counter() - t_cpu0) * 1000.0, 2)

    # 2. Warm up + measure fused batched RTX 4090 TF32/BF16 kernel
    _ = train_batched_cpcv_mlp_gpu(x_all, y_all, tr_masks, te_masks, epochs=5, seed=105)
    batched = train_batched_cpcv_mlp_gpu(x_all, y_all, tr_masks, te_masks, epochs=35, seed=105, use_bf16=True)
    gpu_batched_ms = float(batched["elapsed_ms"])
    return {
        "shape": [n_rows, n_feat, k_folds],
        "cpu_serial_ms": cpu_serial_ms,
        "gpu_batched_ms": gpu_batched_ms,
        "device": batched["device"],
        "amp_bf16": batched["amp_bf16"],
        "speedup_x": round(cpu_serial_ms / max(0.01, gpu_batched_ms), 2),
    }


def _bench_sidecar_http(url: str, n_symbols: int = 1000) -> dict[str, Any]:
    rows = [
        {
            "symbol": f"SYM{i:04d}",
            "gap_pct": 0.02 + (i % 15) * 0.005,
            "rvol": 1.5 + (i % 20) * 0.25,
            "velocity_bps": 10.0 + (i % 30) * 1.2,
            "ofi": -0.5 + (i % 10) * 0.12,
            "catalyst_score": (i % 5) * 0.25,
        }
        for i in range(n_symbols)
    ]
    body = json.dumps({"rows": rows}).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/gpu/score_batch",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rtt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "ok": bool(payload.get("ok")),
            "n_symbols": n_symbols,
            "roundtrip_ms": rtt_ms,
            "gpu_compute_ms": payload.get("gpu_compute_ms"),
            "device": payload.get("device"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Noesis RTX 4090 GPU vs CPU SOTA stack.")
    parser.add_argument("--canon-lake", default="/mnt/z/sota-data/prewall-lake")
    parser.add_argument("--ext4-lake", default=str(pathlib.Path.home() / "sota-data-ext4" / "prewall-lake"))
    parser.add_argument("--shm-lake", default="/dev/shm/prewall-lake")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8794")
    parser.add_argument(
        "--out",
        default="/mnt/z/sota-data/prewall-lake/receipts/noesis_rtx4090_sota_tune_benchmark.json",
    )
    args = parser.parse_args()

    hw = configure_rtx4090_torch_backend()
    df10_rel = pathlib.Path("derived") / "df10_daily_20d_baselines.parquet"

    parquet_bench = {
        "9p_cpu": _bench_parquet_scan(pathlib.Path(args.canon_lake) / df10_rel, engine="cpu"),
        "ext4_cpu": _bench_parquet_scan(pathlib.Path(args.ext4_lake) / df10_rel, engine="cpu"),
        "shm_cpu": _bench_parquet_scan(pathlib.Path(args.shm_lake) / df10_rel, engine="cpu"),
        "shm_cudf_gpu": _bench_parquet_scan(pathlib.Path(args.shm_lake) / df10_rel, engine="gpu")
        if hw["backends"].get("cudf_polars")
        else {"available": False},
    }

    cpcv_bench = _bench_serial_vs_batched_cpcv()

    import random

    rng = random.Random(42)
    synth_rets = [rng.gauss(0.0012, 0.015) for _ in range(500)]
    boot_cpu = gpu_block_bootstrap_sharpe_dsr(synth_rets, n_boot=10_000, prefer_cuda=False)
    boot_gpu = gpu_block_bootstrap_sharpe_dsr(synth_rets, n_boot=10_000, prefer_cuda=True)

    sidecar_bench = _bench_sidecar_http(args.sidecar_url, n_symbols=1000)

    receipt: dict[str, Any] = {
        "schema_version": "noesis-rtx4090-sota-tune-benchmark.v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "hardware": hw,
        "parquet_scan_df10": parquet_bench,
        "cpcv_15fold_mlp": cpcv_bench,
        "block_bootstrap_10k": {
            "cpu_ms": boot_cpu["gpu_compute_ms"],
            "gpu_ms": boot_gpu["gpu_compute_ms"],
            "speedup_x": round(float(boot_cpu["gpu_compute_ms"]) / max(0.01, float(boot_gpu["gpu_compute_ms"])), 2),
            "gpu_device": boot_gpu["device"],
        },
        "sidecar_1000_symbols": sidecar_bench,
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode("utf-8")).hexdigest()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
