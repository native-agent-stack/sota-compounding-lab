#!/usr/bin/env python3
"""Mirror the canonical pre-wall Parquet lake to native WSL2 ext4 and `/dev/shm` RAM disk.

Avoids WSL2 9P filesystem bridge latency (`/mnt/z`) by maintaining SHA-256 verified
copies of `PREWALL_LAKE_MANIFEST.json`, `derived/`, and `public-tables/` on native
`ext4` (`~/sota-data-ext4/prewall-lake`) and `/dev/shm/prewall-lake`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
from typing import Any

SYNC_SUBDIRS = ("derived", "public-tables")


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _copy_if_stale(src: pathlib.Path, dst: pathlib.Path) -> tuple[bool, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_st = src.stat()
    if dst.is_file():
        dst_st = dst.stat()
        if dst_st.st_size == src_st.st_size and int(dst_st.st_mtime) >= int(src_st.st_mtime):
            return False, src_st.st_size
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return True, src_st.st_size


def sync_lake_tier(src_root: pathlib.Path, dst_root: pathlib.Path, *, verify_sha: bool = True) -> dict[str, Any]:
    manifest_src = src_root / "PREWALL_LAKE_MANIFEST.json"
    if not manifest_src.is_file():
        raise FileNotFoundError(f"Missing manifest at {manifest_src}")

    dst_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    total_bytes = 0
    verified_hashes: dict[str, str] = {}

    did_copy, sz = _copy_if_stale(manifest_src, dst_root / "PREWALL_LAKE_MANIFEST.json")
    copied += int(did_copy)
    skipped += int(not did_copy)
    total_bytes += sz

    for sub in SYNC_SUBDIRS:
        sub_src = src_root / sub
        if not sub_src.is_dir():
            continue
        for item in sorted(sub_src.rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(src_root)
            dst_item = dst_root / rel
            did_c, item_sz = _copy_if_stale(item, dst_item)
            copied += int(did_c)
            skipped += int(not did_c)
            total_bytes += item_sz
            if verify_sha and item.name in (
                "df10_daily_20d_baselines.parquet",
                "df6_regsho_threshold_pit.parquet",
                "sec_filings_clean_prewall.parquet",
                "df8_short_interest_clean_prewall.parquet",
            ):
                src_h = _sha256_file(item)
                dst_h = _sha256_file(dst_item)
                if src_h != dst_h:
                    raise RuntimeError(f"SHA-256 mismatch syncing {rel}: {src_h} != {dst_h}")
                verified_hashes[str(rel)] = dst_h

    return {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "files_copied": copied,
        "files_unchanged": skipped,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "verified_sha256": verified_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirror pre-wall Parquet lake to ext4 and /dev/shm fast tiers.")
    parser.add_argument("--src", default="/mnt/z/sota-data/prewall-lake")
    parser.add_argument("--ext4-dst", default=str(pathlib.Path.home() / "sota-data-ext4" / "prewall-lake"))
    parser.add_argument("--shm-dst", default="/dev/shm/prewall-lake")
    parser.add_argument("--to-ext4", action="store_true", help="Mirror to native ext4 tier")
    parser.add_argument("--to-shm", action="store_true", help="Mirror to /dev/shm RAM disk tier")
    parser.add_argument("--no-sha", action="store_true", help="Skip SHA-256 verification on derived parquets")
    args = parser.parse_args()

    src_root = pathlib.Path(args.src)
    if not args.to_ext4 and not args.to_shm:
        args.to_ext4 = True

    results: dict[str, Any] = {}
    if args.to_ext4:
        results["native_ext4"] = sync_lake_tier(src_root, pathlib.Path(args.ext4_dst), verify_sha=not args.no_sha)
    if args.to_shm:
        shm_src = pathlib.Path(args.ext4_dst) if args.to_ext4 else src_root
        results["ramdisk_shm"] = sync_lake_tier(shm_src, pathlib.Path(args.shm_dst), verify_sha=not args.no_sha)

    print(json.dumps({"ok": True, "tiers": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
