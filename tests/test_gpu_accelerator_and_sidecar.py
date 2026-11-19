"""Tests for ADR-0022: RTX 4090 GPU Accelerator, Fast-Tier Lake Resolver & Scoring Sidecar."""

from __future__ import annotations

import json
import pathlib
import random
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from sota_compounding import (
    configure_rtx4090_torch_backend,
    gpu_block_bootstrap_sharpe_dsr,
    gpu_score_mover_feature_matrix,
    resolve_fast_prewall_lake_root,
)
from sota_compounding.gpu_scoring_sidecar import build_sidecar_state, make_sidecar_handler


def _make_fake_lake(root: pathlib.Path, *, manifest_tag: str = "v1", df10_body: bytes = b"df10") -> None:
    (root / "derived").mkdir(parents=True, exist_ok=True)
    (root / "PREWALL_LAKE_MANIFEST.json").write_text(json.dumps({"tag": manifest_tag}), encoding="utf-8")
    (root / "derived" / "df10_daily_20d_baselines.parquet").write_bytes(df10_body)
    (root / "derived" / "df6_regsho_threshold_pit.parquet").write_bytes(b"df6")
    (root / "derived" / "sec_filings_clean_prewall.parquet").write_bytes(b"sec")


def test_configure_rtx4090_torch_backend_is_safe_and_complete() -> None:
    info = configure_rtx4090_torch_backend()
    assert isinstance(info, dict)
    assert "cuda_available" in info
    assert "backends" in info and isinstance(info["backends"], dict)


def test_resolve_fast_prewall_lake_root_prefers_durable_ext4_then_shm_then_canonical(
    tmp_path: pathlib.Path,
) -> None:
    canon = tmp_path / "canon"
    ext4 = tmp_path / "ext4"
    shm = tmp_path / "shm"
    _make_fake_lake(canon, manifest_tag="lock-1", df10_body=b"exact-bytes")
    _make_fake_lake(ext4, manifest_tag="lock-1", df10_body=b"exact-bytes")
    _make_fake_lake(shm, manifest_tag="lock-1", df10_body=b"exact-bytes")

    # Default prefers durable native_ext4 (Z: VHDX) to avoid redundant /dev/shm RAM duplication
    chosen, tier = resolve_fast_prewall_lake_root(canon, shm_mirror=shm, ext4_mirror=ext4)
    assert chosen == ext4 and tier == "native_ext4"

    # Explicit opt-in prefers ramdisk_shm when requested
    chosen_shm, tier_shm = resolve_fast_prewall_lake_root(canon, shm_mirror=shm, ext4_mirror=ext4, prefer_shm=True)
    assert chosen_shm == shm and tier_shm == "ramdisk_shm"

    # Mismatched ext4 manifest/size MUST fail over safely to verified shm
    _make_fake_lake(ext4, manifest_tag="stale", df10_body=b"exact-bytes")
    chosen2, tier2 = resolve_fast_prewall_lake_root(canon, shm_mirror=shm, ext4_mirror=ext4)
    assert chosen2 == shm and tier2 == "ramdisk_shm"

    # Mismatched shm MUST fail back safely to canonical
    _make_fake_lake(shm, manifest_tag="lock-1", df10_body=b"corrupt")
    chosen3, tier3 = resolve_fast_prewall_lake_root(canon, shm_mirror=shm, ext4_mirror=ext4)
    assert chosen3 == canon and tier3 == "canonical"


def test_gpu_score_mover_feature_matrix_orders_ranks_deterministically() -> None:
    rows = [
        {"symbol": "AAA", "gap_pct": 0.01, "rvol": 1.1, "velocity_bps": 5.0, "ofi": -0.2, "catalyst_score": 0.0},
        {"symbol": "ZZZ", "gap_pct": 0.12, "rvol": 6.5, "velocity_bps": 45.0, "ofi": 0.8, "catalyst_score": 1.0},
        {"symbol": "MMM", "gap_pct": 0.04, "rvol": 2.4, "velocity_bps": 18.0, "ofi": 0.1, "catalyst_score": 0.5},
    ]
    res = gpu_score_mover_feature_matrix(rows, prefer_cuda=False)
    assert res["ok"] is True
    assert res["count"] == 3
    assert [r["symbol"] for r in res["scores"]] == ["ZZZ", "MMM", "AAA"]
    assert all(0.0 <= r["gpu_alpha_score"] <= 100.0 for r in res["scores"])

    single = gpu_score_mover_feature_matrix([rows[0]], prefer_cuda=True)
    assert single["ok"] is True and single["count"] == 1 and single["scores"][0]["rank"] == 1


def test_gpu_block_bootstrap_sharpe_dsr_is_seeded_and_validates_inputs() -> None:
    rng = random.Random(7)
    rets = [rng.gauss(0.002, 0.01) for _ in range(80)]
    b1 = gpu_block_bootstrap_sharpe_dsr(rets, n_boot=300, block_len=4, seed=11, prefer_cuda=False)
    b2 = gpu_block_bootstrap_sharpe_dsr(rets, n_boot=300, block_len=4, seed=11, prefer_cuda=False)
    assert b1["sharpe_mean"] == b2["sharpe_mean"]
    assert b1["sharpe_ci95"][0] <= b1["sharpe_median"] <= b1["sharpe_ci95"][1]
    assert 0.0 <= b1["prob_sharpe_gt_zero"] <= 1.0

    with pytest.raises(ValueError):
        gpu_block_bootstrap_sharpe_dsr([0.01, 0.02], n_boot=200)


def test_gpu_scoring_sidecar_http_endpoints(tmp_path: pathlib.Path) -> None:
    canon = tmp_path / "canon"
    _make_fake_lake(canon)
    state = build_sidecar_state(canon)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_sidecar_handler(state))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        with urllib.request.urlopen(f"{base}/api/gpu/health", timeout=2.0) as r:
            health = json.loads(r.read().decode("utf-8"))
        assert health["ok"] is True
        assert health["schema_version"] == "sota-gpu-scorer.v1"

        req_score = urllib.request.Request(
            f"{base}/api/gpu/score_batch",
            data=json.dumps(
                {
                    "rows": [
                        {"symbol": "NVDA", "gap_pct": 0.05, "rvol": 4.2, "velocity_bps": 28.0, "ofi": 0.6},
                        {"symbol": "SPY", "gap_pct": 0.01, "rvol": 1.1, "velocity_bps": 4.0, "ofi": 0.0},
                    ]
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_score, timeout=2.0) as r:
            scored = json.loads(r.read().decode("utf-8"))
        assert scored["ok"] is True
        assert scored["scores"][0]["symbol"] == "NVDA"

        req_boot = urllib.request.Request(
            f"{base}/api/gpu/bootstrap_cpcv",
            data=json.dumps({"returns": [0.01, -0.005, 0.008, 0.004, 0.012, -0.002], "n_boot": 250}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_boot, timeout=2.0) as r:
            boot = json.loads(r.read().decode("utf-8"))
        assert boot["ok"] is True
        assert boot["n_boot"] == 250
    finally:
        srv.shutdown()
        srv.server_close()
