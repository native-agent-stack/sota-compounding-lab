# ADR-0022: Noesis RTX 4090 GPU Accelerator, RAM-Disk / ext4 Fast Pre-Wall Lake Tier & Mirrored Scoring Sidecar

- **Status:** Accepted
- **Date:** 2026-11-19
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `src/sota_compounding/gpu_accelerator.py`, `src/sota_compounding/gpu_scoring_sidecar.py`, `scripts/sync_prewall_lake_fast_tier.py`, `tests/test_gpu_accelerator_and_sidecar.py`

## Context

On Stationary Workstation `Noesis` (`Threadripper PRO 5975WX 32c/64t`, `128 GB RAM`, `NVIDIA GeForce RTX 4090 24 GB`, `Noesis-Carbon` WSL2):
1. All pre-wall Parquet reads (`df10_daily_20d_baselines.parquet`, `sec_filings_clean_prewall.parquet`, `short_volume.parquet`) crossed the WSL2 9P filesystem bridge (`/mnt/z`) on every sweep.
2. Walk-forward `C(6,2)=15` CPCV MLP training launched 15 sequential single-model Python loops without Ada Lovelace `TF32` / `BF16` fusion, leaving RTX 4090 CUDA compute near `0%`.
3. `Noesis-Carbon` lacked a Python `3.12` RAPIDS `cudf-polars` / `CuPy` / `ONNX Runtime GPU` stack and had no resident loopback inference sidecar reachable from Win32 `sota-mover-ranker` (`:8787`).

## Decision

1. **GPU Accelerator Module (`src/sota_compounding/gpu_accelerator.py`):**
   - `configure_rtx4090_torch_backend()` enables Ada `TF32` (`allow_tf32=True`, `set_float32_matmul_precision("high")`) and probes installed GPU backends (`torch_cuda`, `triton`, `cudf_polars`, `cupy`, `xgboost`, `onnxruntime_gpu`).
   - `train_batched_cpcv_mlp_gpu()` executes all `K` (`15`) CPCV folds simultaneously via fused `torch.bmm` + `BF16` AMP with strict per-fold train-only standardization.
   - `gpu_block_bootstrap_sharpe_dsr()` and `gpu_score_mover_feature_matrix()` run vectorized circular block bootstraps and cross-sectional mover scoring on CUDA (with pure Python/CPU fallback when CUDA is absent).
2. **Manifest-Verified Zero-9P Fast-Lake Tier (`resolve_fast_prewall_lake_root`, `scripts/sync_prewall_lake_fast_tier.py`):**
   - Mirrors the `2.1 GB` pre-wall lake from `/mnt/z/sota-data/prewall-lake` to native `ext4` (`~/sota-data-ext4/prewall-lake`) and `/dev/shm/prewall-lake` with `PREWALL_LAKE_MANIFEST.json` + derived Parquet SHA-256 verification, failing back cleanly to canonical `/mnt/z` on any mismatch.
3. **Resident RTX 4090 Scoring Sidecar (`src/sota_compounding/gpu_scoring_sidecar.py`, `scripts/start_noesis_gpu_sidecar.sh`):**
   - Serves `/api/gpu/health`, `/api/gpu/score_batch`, and `/api/gpu/bootstrap_cpcv` on `127.0.0.1:8794` over WSL2 `networkingMode=mirrored` loopback, pre-warming CUDA context on startup.

## Verification

- `python scripts/architecture_catalog.py --check` (`22/22` MADR 4.0 ADRs), `ruff check .`, `ruff format --check .`, `pytest -q`, and `gitleaks`.
- Cross-OS empirical benchmark in `Noesis-Carbon` `.venv-gpu-py312` (`cudf-polars`, `torch 2.15+cu132`, `cupy`, `onnxruntime-gpu`) and live Win32 -> `:8794` health + 500-symbol batch scoring probe.
