"""Resident RTX 4090 Loopback Scoring & Bootstrap Sidecar (`127.0.0.1:8794`).

Bridges Win32 `sota-mover-ranker` (`:8787`) and `Noesis-Carbon` WSL2 CUDA (`RTX 4090`)
over `networkingMode=mirrored` loopback with pre-warmed CUDA context and Ada TF32/BF16 kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sota_compounding.gpu_accelerator import (
    GPU_SIDECAR_SCHEMA,
    configure_rtx4090_torch_backend,
    gpu_block_bootstrap_sharpe_dsr,
    gpu_score_mover_feature_matrix,
    resolve_fast_prewall_lake_root,
)
from sota_compounding.upstream_lib_strategy_e2e import (
    LiveTargetObservation,
    MarketSituationInput,
    apply_regime_conditioned_per_symbol_factors,
    classify_market_situation,
    compute_manifest_sha256,
    enforce_full_resolution_zero_fallback_gate,
)

DEFAULT_SIDECAR_HOST = "127.0.0.1"
DEFAULT_SIDECAR_PORT = 8794
_DEFAULT_E2E_MANIFEST_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "docs" / "SOTA_UPSTREAM_LIB_E2E_REGIME_PERSYMBOL_MANIFEST.json"
)


def build_sidecar_state(canonical_lake: pathlib.Path) -> dict[str, Any]:
    hw = configure_rtx4090_torch_backend()
    lake_path, tier = resolve_fast_prewall_lake_root(canonical_lake)
    # Pre-warm CUDA context so first live scoring request stays <5ms
    warm = gpu_score_mover_feature_matrix(
        [{"symbol": "WARM", "gap_pct": 0.02, "rvol": 2.0, "velocity_bps": 15.0, "ofi": 0.3, "catalyst_score": 0.5}]
    )
    return {
        "started_mono": time.monotonic(),
        "hardware": hw,
        "fast_lake_root": str(lake_path),
        "fast_lake_tier": tier,
        "warmup_ms": warm.get("gpu_compute_ms", 0.0),
    }


def make_sidecar_handler(state: dict[str, Any]) -> type[BaseHTTPRequestHandler]:
    class _GpuSidecarHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            del format, args

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            route = self.path.split("?", 1)[0].rstrip("/")
            if route in ("/api/gpu/health", "/health"):
                hw = state["hardware"]
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "schema_version": GPU_SIDECAR_SCHEMA,
                        "device": hw.get("device", "cpu"),
                        "gpu_name": hw.get("gpu_name"),
                        "cuda_available": hw.get("cuda_available", False),
                        "tf32_enabled": hw.get("tf32_enabled", False),
                        "bf16_supported": hw.get("bf16_supported", False),
                        "vram_total_mb": hw.get("vram_total_mb", 0.0),
                        "vram_used_mb": hw.get("vram_used_mb", 0.0),
                        "backends": hw.get("backends", {}),
                        "fast_lake_tier": state.get("fast_lake_tier", "canonical"),
                        "warmup_ms": state.get("warmup_ms", 0.0),
                        "uptime_sec": round(time.monotonic() - float(state["started_mono"]), 2),
                        "adr0035_routes": [
                            "/api/upstream_lib_e2e/manifest",
                            "/api/upstream_lib_e2e/route_targets",
                        ],
                        "adr0038_full_resolution_routes": [
                            "/api/full_resolution/status",
                            "/api/full_resolution/manifest",
                        ],
                        "allow_silent_fallback": False,
                    },
                )
                return
            if route in ("/api/full_resolution/status", "/api/full_resolution/manifest"):
                manifest_path = pathlib.Path(state.get("e2e_manifest_path", _DEFAULT_E2E_MANIFEST_PATH))
                if not manifest_path.is_file():
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {
                            "ok": False,
                            "full_resolution_verified": False,
                            "allow_silent_fallback": False,
                            "error": "FAIL_CLOSED_FULL_RESOLUTION_MANIFEST_MISSING",
                        },
                    )
                    return
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {
                            "ok": False,
                            "full_resolution_verified": False,
                            "allow_silent_fallback": False,
                            "error": f"FAIL_CLOSED_FULL_RESOLUTION_MANIFEST_CORRUPT: {exc}",
                        },
                    )
                    return
                expected_sha = compute_manifest_sha256(data) if isinstance(data, dict) else ""
                sha_ok = isinstance(data, dict) and data.get("manifest_sha256") == expected_sha
                audit = enforce_full_resolution_zero_fallback_gate(
                    data.get("upstream_library_runs") if isinstance(data, dict) else {},
                    require_all_engines=True,
                )
                hw = state["hardware"]
                if not sha_ok or not audit["full_resolution_verified"]:
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {
                            "ok": False,
                            "full_resolution_verified": False,
                            "allow_silent_fallback": False,
                            "sha256_verified": sha_ok,
                            "audit": audit,
                            "error": "FAIL_CLOSED_FULL_RESOLUTION_GATE_REJECTED",
                        },
                    )
                    return
                if route == "/api/full_resolution/status":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "adr": "ADR-0038",
                            "full_resolution_verified": True,
                            "allow_silent_fallback": False,
                            "sha256_verified": True,
                            "manifest_sha256": expected_sha,
                            "device": hw.get("device", "cpu"),
                            "cuda_available": bool(hw.get("cuda_available", False)),
                            "audit": audit,
                        },
                    )
                    return
                self._send_json(HTTPStatus.OK, data)
                return
            if route == "/api/upstream_lib_e2e/manifest":
                manifest_path = pathlib.Path(state.get("e2e_manifest_path", _DEFAULT_E2E_MANIFEST_PATH))
                if not manifest_path.is_file():
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {"ok": False, "error": "e2e_manifest_not_generated"},
                    )
                    return
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"ok": False, "error": f"e2e_manifest_read_error: {exc}"},
                    )
                    return
                self._send_json(HTTPStatus.OK, data)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                if content_len <= 0 or content_len > 16 * 1024 * 1024:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid_content_length"})
                    return
                body = json.loads(self.rfile.read(content_len).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object")
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"bad_json: {exc}"})
                return

            route = self.path.rstrip("/")
            if route == "/api/gpu/score_batch":
                rows = body.get("rows")
                weights = body.get("weights")
                require_gpu = bool(
                    body.get("require_gpu", os.environ.get("SOTA_REQUIRE_FULL_RESOLUTION_GPU", "0") == "1")
                )
                if not isinstance(rows, list):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "rows must be a list"})
                    return
                res = gpu_score_mover_feature_matrix(
                    rows,
                    weights=weights if isinstance(weights, dict) else None,
                )
                if require_gpu and rows and not str(res.get("device", "")).startswith("cuda"):
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {
                            "ok": False,
                            "allow_silent_fallback": False,
                            "device": res.get("device"),
                            "error": "FAIL_CLOSED_NO_CPU_FALLBACK: CUDA required in full-resolution mode",
                        },
                    )
                    return
                self._send_json(HTTPStatus.OK, res)
                return

            if route == "/api/gpu/bootstrap_cpcv":
                returns = body.get("returns")
                require_gpu = bool(
                    body.get("require_gpu", os.environ.get("SOTA_REQUIRE_FULL_RESOLUTION_GPU", "0") == "1")
                )
                if not isinstance(returns, list):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "returns must be a list"})
                    return
                try:
                    res = gpu_block_bootstrap_sharpe_dsr(
                        returns,
                        n_boot=int(body.get("n_boot", 10_000)),
                        block_len=int(body.get("block_len", 5)),
                        annualization=float(body.get("annualization", 252.0)),
                        seed=int(body.get("seed", 42)),
                    )
                    if require_gpu and not str(res.get("device", "")).startswith("cuda"):
                        self._send_json(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            {
                                "ok": False,
                                "allow_silent_fallback": False,
                                "device": res.get("device"),
                                "error": "FAIL_CLOSED_NO_CPU_FALLBACK: CUDA required in full-resolution mode",
                            },
                        )
                        return
                    self._send_json(HTTPStatus.OK, res)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return

            if route == "/api/upstream_lib_e2e/route_targets":
                raw_sit = body.get("market_situation") or {}
                raw_targets = body.get("targets")
                venue_lane = str(body.get("venue_lane", ":8787"))
                if not isinstance(raw_sit, dict) or not isinstance(raw_targets, list):
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"ok": False, "error": "market_situation must be object and targets must be list"},
                    )
                    return
                try:
                    sit_input = MarketSituationInput(
                        **{k: v for k, v in raw_sit.items() if k in MarketSituationInput.__dataclass_fields__}
                    )
                    assessment = classify_market_situation(sit_input)
                    obs_list = [
                        LiveTargetObservation(
                            **{k: v for k, v in item.items() if k in LiveTargetObservation.__dataclass_fields__}
                        )
                        for item in raw_targets
                        if isinstance(item, dict)
                    ]
                    plans = apply_regime_conditioned_per_symbol_factors(
                        targets=obs_list,
                        market_situation=assessment,
                        per_symbol_prewall_profiles=body.get("per_symbol_prewall_profiles")
                        if isinstance(body.get("per_symbol_prewall_profiles"), dict)
                        else None,
                        venue_lane=venue_lane,
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "market_situation": asdict(assessment),
                            "eligible_n": sum(1 for p in plans if p.eligible),
                            "plans": [asdict(p) for p in plans],
                        },
                    )
                except ValueError as exc:
                    status_code = (
                        HTTPStatus.FORBIDDEN
                        if "FAIL_CLOSED_LIVE_CAPITAL_REFUSAL" in str(exc)
                        else HTTPStatus.BAD_REQUEST
                    )
                    self._send_json(status_code, {"ok": False, "error": str(exc)})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

    return _GpuSidecarHandler


def create_gpu_sidecar_server(
    *,
    host: str = DEFAULT_SIDECAR_HOST,
    port: int = DEFAULT_SIDECAR_PORT,
    canonical_lake: pathlib.Path = pathlib.Path("/mnt/z/sota-data/prewall-lake"),
) -> ThreadingHTTPServer:
    state = build_sidecar_state(canonical_lake)
    handler = make_sidecar_handler(state)
    return ThreadingHTTPServer((host, int(port)), handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Noesis RTX 4090 GPU Scoring Sidecar (:8794).")
    parser.add_argument("--host", default=os.environ.get("SOTA_GPU_SIDECAR_HOST", DEFAULT_SIDECAR_HOST))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SOTA_GPU_SIDECAR_PORT", str(DEFAULT_SIDECAR_PORT)))
    )
    parser.add_argument("--lake-root", default="/mnt/z/sota-data/prewall-lake")
    args = parser.parse_args()

    srv = create_gpu_sidecar_server(
        host=args.host,
        port=args.port,
        canonical_lake=pathlib.Path(args.lake_root),
    )
    print(f"[gpu-sidecar] Listening on http://{args.host}:{srv.server_port} (ctrl-c to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
