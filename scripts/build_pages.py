#!/usr/bin/env python3
"""Assemble the static GitHub Pages bundle (`_site/`) from official SHA-256 verified npm tarballs."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent

# Immutable official npm registry tarballs + SHA-256 digests of both the tarballs and extracted files.
NPM_TARBALLS = [
    (
        "mujoco-4.0.1.tgz",
        "https://registry.npmjs.org/@mujoco/mujoco/-/mujoco-4.0.1.tgz",
        "66c6f3e1c9274d68509477a044d84ce0be9e659cc3deebe45b267919913bd554",
        [
            ("package/mujoco.js", "mujoco.js", "6ab204c4c51d0a5f68a5c96e56f607314551b7b1e2770603d84c7a1318e70070"),
            ("package/mujoco.wasm", "mujoco.wasm", "904645316cfe464162473b063550e9c0f5c15e06fba77a7b9dd6fdd1c47b3f7c"),
        ],
    ),
    (
        "three-0.170.0.tgz",
        "https://registry.npmjs.org/three/-/three-0.170.0.tgz",
        "4a608a355dcaba72e0e5383cdc814303f5b6060b43c238cdf6932dceb699238d",
        [
            (
                "package/build/three.module.min.js",
                "three.module.min.js",
                "08fd7545d13d2c7fb65ab691530a802dafefd638596501854f267d0fb13c39e7",
            ),
        ],
    ),
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download_tarball(fname: str, url: str, expected_sha256: str, cache_dir: Path | None) -> bytes:
    if cache_dir is not None:
        cached = cache_dir / fname
        if cached.is_file():
            raw = cached.read_bytes()
            if _sha256_bytes(raw) == expected_sha256:
                return raw

    proc = subprocess.run(["curl", "-fsSL", "--retry", "3", url], check=True, capture_output=True)
    raw = proc.stdout
    actual = _sha256_bytes(raw)
    if actual != expected_sha256:
        raise RuntimeError(f"Tarball SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}")

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

    # 3. Extract and SHA-256 verify self-hosted vendor modules from official npm tarballs
    vendor_dst = out_dir / "vendor"
    vendor_dst.mkdir(parents=True, exist_ok=True)
    for tgz_name, url, tgz_sha256, members in NPM_TARBALLS:
        tgz_bytes = _download_tarball(tgz_name, url, tgz_sha256, cache_dir)
        with tarfile.open(fileobj=io.BytesIO(tgz_bytes), mode="r:gz") as tf:
            for archive_member, out_name, file_sha256 in members:
                extracted = tf.extractfile(tf.getmember(archive_member))
                if extracted is None:
                    raise RuntimeError(f"Missing {archive_member} in {tgz_name}")
                payload = extracted.read()
                actual_sha = _sha256_bytes(payload)
                if actual_sha != file_sha256:
                    raise RuntimeError(f"SHA-256 mismatch for {out_name}: expected {file_sha256}, got {actual_sha}")
                (vendor_dst / out_name).write_bytes(payload)
                print(f"  verified {out_name} ({len(payload) / 1024:.0f} KB, sha256={file_sha256[:12]}...)")

    # 4. Prevent Jekyll processing on GitHub Pages
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built static GitHub Pages bundle at {out_dir} ({len(mesh_files)} STL meshes).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static GitHub Pages site bundle.")
    parser.add_argument("--out", default=str(ROOT / "_site"), help="Output directory (default: _site).")
    parser.add_argument("--cache-dir", default=None, help="Optional local cache directory for npm tarballs.")
    args = parser.parse_args()

    cache = Path(args.cache_dir) if args.cache_dir else None
    build_site(Path(args.out), cache_dir=cache)


if __name__ == "__main__":
    main()
