"""RTX 4090 GPU Compute Accelerator, Zero-9P Fast-Lake Resolver & Batched CPCV Engine.

Provides hardware-accelerated PyTorch Ada TF32/BF16 kernels, batched C(6,2)=15 CPCV MLP
training, GPU circular block bootstrap, cross-sectional mover feature scoring, and
manifest-verified RAM-disk (`/dev/shm`) / native `ext4` fast-tier lake resolution.
Falls back cleanly to pure Python / NumPy / CPU when CUDA is unavailable.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
import statistics
import time
from typing import Any

GPU_SIDECAR_SCHEMA = "sota-gpu-scorer.v1"

DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "gap_pct": 0.22,
    "rvol": 0.28,
    "velocity_bps": 0.25,
    "ofi": 0.15,
    "catalyst_score": 0.10,
}


def _import_optional(name: str) -> Any | None:
    try:
        return __import__(name)
    except Exception:
        return None


def configure_rtx4090_torch_backend() -> dict[str, Any]:
    """Enable Ada Lovelace TF32/cudnn optimizations and report available GPU stack."""
    torch = _import_optional("torch")
    info: dict[str, Any] = {
        "torch_available": torch is not None,
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": None,
        "cuda_version": None,
        "sm_count": 0,
        "vram_total_mb": 0.0,
        "vram_used_mb": 0.0,
        "bf16_supported": False,
        "tf32_enabled": False,
        "backends": {
            "torch_cuda": False,
            "triton": _import_optional("triton") is not None,
            "cupy": _import_optional("cupy") is not None,
            "cudf_polars": _import_optional("cudf_polars") is not None,
            "xgboost": _import_optional("xgboost") is not None,
            "onnxruntime_gpu": False,
        },
    }
    ort = _import_optional("onnxruntime")
    if ort is not None:
        try:
            providers = ort.get_available_providers()
            info["backends"]["onnxruntime_gpu"] = "CUDAExecutionProvider" in providers
        except Exception:
            pass

    if torch is None:
        return info

    info["cuda_version"] = getattr(torch.version, "cuda", None)
    if not torch.cuda.is_available():
        return info

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        info.update(
            {
                "cuda_available": True,
                "device": "cuda:0",
                "gpu_name": props.name,
                "sm_count": int(props.multi_processor_count),
                "compute_capability": f"{props.major}.{props.minor}",
                "vram_total_mb": round(total_b / (1024 * 1024), 1),
                "vram_used_mb": round((total_b - free_b) / (1024 * 1024), 1),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "tf32_enabled": True,
            }
        )
        info["backends"]["torch_cuda"] = True
    except Exception:
        info["cuda_available"] = True
        info["device"] = "cuda:0"
        info["backends"]["torch_cuda"] = True
        info["tf32_enabled"] = True
    return info


def _manifest_file_signatures(lake_root: pathlib.Path) -> dict[str, Any] | None:
    manifest = lake_root / "PREWALL_LAKE_MANIFEST.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
    except Exception:
        return None
    req_files = (
        lake_root / "derived" / "df10_daily_20d_baselines.parquet",
        lake_root / "derived" / "df6_regsho_threshold_pit.parquet",
        lake_root / "derived" / "sec_filings_clean_prewall.parquet",
    )
    sizes = {p.name: p.stat().st_size for p in req_files if p.is_file()}
    if len(sizes) != len(req_files):
        return None
    return {"manifest": data, "sizes": sizes}


def resolve_fast_prewall_lake_root(
    canonical_root: pathlib.Path,
    *,
    shm_mirror: pathlib.Path | None = pathlib.Path("/dev/shm/prewall-lake"),
    ext4_mirror: pathlib.Path | None = None,
) -> tuple[pathlib.Path, str]:
    """Select the fastest verified pre-wall lake tier (`/dev/shm` -> native `ext4` -> `canonical_root`)."""
    canon_sig = _manifest_file_signatures(canonical_root)
    if ext4_mirror is None:
        ext4_mirror = pathlib.Path.home() / "sota-data-ext4" / "prewall-lake"

    candidates: list[tuple[pathlib.Path | None, str]] = [
        (shm_mirror, "ramdisk_shm"),
        (ext4_mirror, "native_ext4"),
    ]
    for cand_path, tier in candidates:
        if cand_path is None:
            continue
        try:
            if cand_path.resolve() == canonical_root.resolve():
                continue
        except Exception:
            pass
        cand_sig = _manifest_file_signatures(cand_path)
        if cand_sig is None:
            continue
        if canon_sig is not None:
            if cand_sig["manifest"] != canon_sig["manifest"] or cand_sig["sizes"] != canon_sig["sizes"]:
                continue
        return cand_path, tier

    return canonical_root, "canonical"


def gpu_score_mover_feature_matrix(
    rows: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    """Compute cross-sectional robust Z-scores and 0..100 alpha score across a mover batch."""
    t0 = time.perf_counter()
    if not rows:
        return {"ok": True, "device": "none", "gpu_compute_ms": 0.0, "scores": []}

    eff_w = dict(DEFAULT_SCORE_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in eff_w and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                eff_w[k] = float(v)
    feat_keys = tuple(eff_w.keys())
    w_list = [eff_w[k] for k in feat_keys]
    symbols = [str(r.get("symbol", f"SYM{i}")).upper().strip() for i, r in enumerate(rows)]

    matrix: list[list[float]] = []
    for r in rows:
        vec: list[float] = []
        for k in feat_keys:
            raw = r.get(k, 0.0)
            val = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 0.0
            vec.append(val if math.isfinite(val) else 0.0)
        matrix.append(vec)

    torch = _import_optional("torch")
    if torch is not None:
        use_cuda = bool(prefer_cuda and torch.cuda.is_available())
        device = torch.device("cuda:0" if use_cuda else "cpu")
        x = torch.tensor(matrix, dtype=torch.float32, device=device)
        w = torch.tensor(w_list, dtype=torch.float32, device=device)
        mu = x.mean(dim=0, keepdim=True)
        sd = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
        z = torch.clamp((x - mu) / sd, -4.0, 4.0)
        raw_score = (z @ w).squeeze(-1) if z.ndim > 1 else torch.zeros(len(rows), device=device)
        # Logistic squash onto 0..100 for bounded dimensionless ranking parity
        alpha_0_100 = 100.0 * torch.sigmoid(raw_score * 1.25)
        ranks = torch.argsort(torch.argsort(alpha_0_100, descending=True)) + 1
        if use_cuda:
            torch.cuda.synchronize()
        alpha_vals = alpha_0_100.detach().cpu().tolist()
        raw_vals = raw_score.detach().cpu().tolist()
        rank_vals = ranks.detach().cpu().tolist()
        dev_str = str(device)
    else:
        n_rows = len(matrix)
        n_cols = len(feat_keys)
        cols = [[matrix[i][j] for i in range(n_rows)] for j in range(n_cols)]
        means = [sum(c) / max(1, n_rows) for c in cols]
        stds = [max(1e-5, math.sqrt(sum((v - means[j]) ** 2 for v in cols[j]) / max(1, n_rows))) for j in range(n_cols)]
        raw_vals = []
        for i in range(n_rows):
            acc = 0.0
            for j in range(n_cols):
                z_val = max(-4.0, min(4.0, (matrix[i][j] - means[j]) / stds[j]))
                acc += z_val * w_list[j]
            raw_vals.append(acc)
        alpha_vals = [100.0 / (1.0 + math.exp(-max(-20.0, min(20.0, rv * 1.25)))) for rv in raw_vals]
        order = sorted(range(n_rows), key=lambda idx: alpha_vals[idx], reverse=True)
        rank_vals = [0] * n_rows
        for pos, idx in enumerate(order, start=1):
            rank_vals[idx] = pos
        dev_str = "python-cpu"

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    scored = [
        {
            "symbol": symbols[i],
            "rank": int(rank_vals[i]),
            "raw_z_score": round(float(raw_vals[i]), 5),
            "gpu_alpha_score": round(float(alpha_vals[i]), 3),
        }
        for i in range(len(symbols))
    ]
    scored.sort(key=lambda item: item["rank"])
    return {
        "ok": True,
        "schema_version": GPU_SIDECAR_SCHEMA,
        "device": dev_str,
        "count": len(scored),
        "gpu_compute_ms": elapsed_ms,
        "scores": scored,
    }


def gpu_block_bootstrap_sharpe_dsr(
    returns: list[float] | Any,
    *,
    n_boot: int = 10_000,
    block_len: int = 5,
    annualization: float = 252.0,
    seed: int = 42,
    prefer_cuda: bool = True,
) -> dict[str, Any]:
    """Run vectorized circular block bootstrap for annualized Sharpe CI and positivity probability."""
    t0 = time.perf_counter()
    clean_vals = [
        float(v) for v in returns if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
    ]
    n_obs = len(clean_vals)
    if n_obs < 5:
        raise ValueError("Need at least 5 finite return observations for block bootstrap")
    n_boot = max(100, int(n_boot))
    block_len = max(1, min(n_obs, int(block_len)))
    n_blocks = math.ceil(n_obs / block_len)
    ann_scale = math.sqrt(float(annualization))

    torch = _import_optional("torch")
    if torch is not None:
        use_cuda = bool(prefer_cuda and torch.cuda.is_available())
        device = torch.device("cuda:0" if use_cuda else "cpu")
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        r_t = torch.tensor(clean_vals, dtype=torch.float32, device=device)
        # Circular wrap so block indices can start at any day in [0, n_obs)
        r_wrap = torch.cat([r_t, r_t[: block_len - 1]], dim=0)
        starts = torch.randint(0, n_obs, (n_boot, n_blocks), generator=gen).to(device)
        offsets = torch.arange(block_len, device=device)
        idx = (starts.unsqueeze(-1) + offsets.view(1, 1, -1)).reshape(n_boot, -1)[:, :n_obs]
        samples = r_wrap[idx]
        means = samples.mean(dim=1)
        stds = samples.std(dim=1, unbiased=True).clamp_min(1e-8)
        sharpes = (means / stds) * ann_scale
        q = torch.quantile(
            sharpes,
            torch.tensor([0.025, 0.50, 0.975], dtype=torch.float32, device=device),
        )
        p_pos = float((sharpes > 0.0).float().mean().item())
        mean_sr = float(sharpes.mean().item())
        if use_cuda:
            torch.cuda.synchronize()
        ci_lo, med_sr, ci_hi = (float(x) for x in q.detach().cpu().tolist())
        dev_str = str(device)
    else:
        rng = random.Random(int(seed))
        sharpes_py: list[float] = []
        r_wrap_py = clean_vals + clean_vals[: block_len - 1]
        for _ in range(n_boot):
            path: list[float] = []
            for _b in range(n_blocks):
                st = rng.randrange(0, n_obs)
                path.extend(r_wrap_py[st : st + block_len])
            path = path[:n_obs]
            m = sum(path) / n_obs
            sd = max(1e-8, statistics.pstdev(path))
            sharpes_py.append((m / sd) * ann_scale)
        sharpes_py.sort()
        ci_lo = sharpes_py[int(0.025 * (n_boot - 1))]
        med_sr = sharpes_py[int(0.50 * (n_boot - 1))]
        ci_hi = sharpes_py[int(0.975 * (n_boot - 1))]
        mean_sr = sum(sharpes_py) / n_boot
        p_pos = sum(1 for s in sharpes_py if s > 0.0) / n_boot
        dev_str = "python-cpu"

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "ok": True,
        "device": dev_str,
        "n_obs": n_obs,
        "n_boot": n_boot,
        "block_len": block_len,
        "sharpe_mean": round(mean_sr, 4),
        "sharpe_median": round(med_sr, 4),
        "sharpe_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "prob_sharpe_gt_zero": round(p_pos, 4),
        "gpu_compute_ms": elapsed_ms,
    }


def train_batched_cpcv_mlp_gpu(
    x_all: Any,
    y_all: Any,
    split_train_masks: list[Any],
    split_test_masks: list[Any],
    *,
    hidden1: int = 64,
    hidden2: int = 32,
    epochs: int = 35,
    lr: float = 3e-3,
    weight_decay: float = 1e-2,
    dropout: float = 0.20,
    seed: int = 42,
    use_bf16: bool = True,
) -> dict[str, Any]:
    """Train all K CPCV folds simultaneously on RTX 4090 using batched `bmm` + Ada TF32/BF16."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    t0 = time.perf_counter()
    x_np = np.asarray(x_all, dtype=np.float32)
    y_np = np.asarray(y_all, dtype=np.float32).reshape(-1)
    n_rows, n_feat = x_np.shape
    k_folds = len(split_train_masks)
    if k_folds == 0 or len(split_test_masks) != k_folds:
        raise ValueError("split_train_masks and split_test_masks must be non-empty and equal length")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        configure_rtx4090_torch_backend()

    x_folds = np.empty((k_folds, n_rows, n_feat), dtype=np.float32)
    tr_masks = np.empty((k_folds, n_rows), dtype=np.float32)
    te_masks = np.empty((k_folds, n_rows), dtype=np.float32)

    for k in range(k_folds):
        tm = np.asarray(split_train_masks[k], dtype=bool)
        em = np.asarray(split_test_masks[k], dtype=bool)
        x_tr = x_np[tm]
        mu = x_tr.mean(axis=0, keepdims=True)
        sd = np.maximum(x_tr.std(axis=0, keepdims=True), 1e-5)
        x_folds[k] = np.clip((x_np - mu) / sd, -5.0, 5.0)
        tr_masks[k] = tm.astype(np.float32)
        te_masks[k] = em.astype(np.float32)

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    w1 = (torch.randn((k_folds, n_feat, hidden1), generator=gen) * math.sqrt(2.0 / max(1, n_feat))).to(device)
    b1 = torch.zeros((k_folds, 1, hidden1), device=device)
    w2 = (torch.randn((k_folds, hidden1, hidden2), generator=gen) * math.sqrt(2.0 / max(1, hidden1))).to(device)
    b2 = torch.zeros((k_folds, 1, hidden2), device=device)
    w3 = (torch.randn((k_folds, hidden2, 1), generator=gen) * math.sqrt(1.0 / max(1, hidden2))).to(device)
    b3 = torch.zeros((k_folds, 1, 1), device=device)
    for param in (w1, b1, w2, b2, w3, b3):
        param.requires_grad_(True)

    xb = torch.from_numpy(x_folds).to(device)
    yb = torch.from_numpy(y_np / 100.0).view(1, n_rows, 1).expand(k_folds, n_rows, 1).to(device)
    tr_m_t = torch.from_numpy(tr_masks).view(k_folds, n_rows, 1).to(device)
    tr_denom = tr_m_t.sum(dim=1, keepdim=True).clamp_min(1.0)

    opt = torch.optim.AdamW([w1, b1, w2, b2, w3, b3], lr=lr, weight_decay=weight_decay)
    amp_enabled = bool(use_cuda and use_bf16 and torch.cuda.is_bf16_supported())

    for _ in range(max(1, int(epochs))):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            h1 = F.silu(torch.bmm(xb, w1) + b1)
            if dropout > 0.0:
                h1 = F.dropout(h1, p=dropout, training=True)
            h2 = F.silu(torch.bmm(h1, w2) + b2)
            pred = torch.bmm(h2, w3) + b3
            per_elem = F.huber_loss(pred, yb, reduction="none")
            fold_loss = ((per_elem * tr_m_t) / tr_denom).sum(dim=1).sum()
        fold_loss.backward()
        opt.step()

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            h1_e = F.silu(torch.bmm(xb, w1) + b1)
            h2_e = F.silu(torch.bmm(h1_e, w2) + b2)
            preds_all = (torch.bmm(h2_e, w3) + b3).squeeze(-1).float() * 100.0
    if use_cuda:
        torch.cuda.synchronize()
    preds_np = preds_all.detach().cpu().numpy()
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "device": str(device),
        "amp_bf16": amp_enabled,
        "k_folds": k_folds,
        "n_rows": n_rows,
        "elapsed_ms": elapsed_ms,
        "preds_by_fold": preds_np,
    }
