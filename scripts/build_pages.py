#!/usr/bin/env python3
"""Assemble the static GitHub Pages bundle (`_site/`) with SHA-256 verified WASM/WebGL vendor assets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent.parent

# Pinned vendor artifacts verified by cryptographic SHA-256 digest at build time.
VENDOR_SPECS = [
    (
        "mujoco.min.js",
        "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@4.0.1/mujoco.min.js",
        "2de9477f52661f0eb9a453a2e6df3eb750f897cd84d00f605494cc83f95315ca",
    ),
    (
        "mujoco.wasm",
        "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@4.0.1/mujoco.wasm",
        "904645316cfe464162473b063550e9c0f5c15e06fba77a7b9dd6fdd1c47b3f7c",
    ),
    (
        "three.module.min.js",
        "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js",
        "08fd7545d13d2c7fb65ab691530a802dafefd638596501854f267d0fb13c39e7",
    ),
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_verified(url: str, expected_sha256: str, cache_dir: Path | None) -> bytes:
    fname = url.rsplit("/", 1)[-1]
    if cache_dir is not None:
        cached = cache_dir / fname
        if cached.is_file():
            raw = cached.read_bytes()
            if _sha256_bytes(raw) == expected_sha256:
                return raw

    req = urllib.request.Request(url, headers={"User-Agent": "microduck-harness-pages-builder/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()

    actual = _sha256_bytes(raw)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / fname).write_bytes(raw)
    return raw


def build_site(out_dir: Path, cache_dir: Path | None = None) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy static application files from `site/`
    site_src = ROOT / "site"
    for item in site_src.iterdir():
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)

    # 2. Copy referenced MicroDuck MJCF + STL meshes
    mj_src = ROOT / "assets" / "microduck"
    mj_dst = out_dir / "assets" / "microduck"
    (mj_dst / "assets").mkdir(parents=True, exist_ok=True)

    for fname in ("scene.xml", "robot_groundcontact.xml", "LICENSE"):
        src_f = mj_src / fname
        if src_f.is_file():
            shutil.copy2(src_f, mj_dst / fname)

    robot_xml = (mj_src / "robot_groundcontact.xml").read_text(encoding="utf-8")
    mesh_files = sorted(set(re.findall(r'file="([^"]+)"', robot_xml)))
    for mname in mesh_files:
        shutil.copy2(mj_src / "assets" / mname, mj_dst / "assets" / mname)

    # 3. Download and SHA-256 verify self-hosted vendor modules (`script-src 'self'`)
    vendor_dst = out_dir / "vendor"
    vendor_dst.mkdir(parents=True, exist_ok=True)
    for fname, url, sha256_hex in VENDOR_SPECS:
        payload = _fetch_verified(url, sha256_hex, cache_dir)
        (vendor_dst / fname).write_bytes(payload)
        print(f"  verified {fname} ({len(payload) / 1024:.0f} KB, sha256={sha256_hex[:12]}...)")

    # 4. Prevent Jekyll processing on GitHub Pages
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built static GitHub Pages bundle at {out_dir} ({len(mesh_files)} STL meshes).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static GitHub Pages site bundle.")
    parser.add_argument("--out", default=str(ROOT / "_site"), help="Output directory (default: _site).")
    parser.add_argument("--cache-dir", default=None, help="Optional local cache directory for vendor downloads.")
    args = parser.parse_args()

    cache = Path(args.cache_dir) if args.cache_dir else None
    build_site(Path(args.out), cache_dir=cache)


if __name__ == "__main__":
    main()
