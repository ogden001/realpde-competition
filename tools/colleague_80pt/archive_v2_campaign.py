#!/usr/bin/env python3
"""Archive lightweight review evidence from the three-arm campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ALLOWED_SUFFIXES = {".json", ".csv", ".md", ".jsonl", ".png"}
ALLOWED_LOG_SUFFIX = ".train.review.log"
BLOCKED_PARTS = {"checkpoints", "__pycache__"}
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allowed(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in BLOCKED_PARTS for part in rel.parts):
        return False
    if path.suffix in ALLOWED_SUFFIXES:
        return True
    return path.name.endswith(ALLOWED_LOG_SUFFIX)


def archive(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)

    files = [path for path in source.rglob("*") if path.is_file() and allowed(path, source)]
    destination.mkdir(parents=True)
    manifest_rows = []
    total = 0
    for src in files:
        rel = src.relative_to(source)
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        total += size
        manifest_rows.append({
            "path": str(rel),
            "bytes": size,
            "sha256": sha256(dst),
        })
    if total > MAX_ARCHIVE_BYTES:
        shutil.rmtree(destination)
        raise RuntimeError(
            f"lightweight archive exceeds {MAX_ARCHIVE_BYTES} bytes: {total}"
        )

    result = {
        "source": str(source),
        "destination": str(destination),
        "files": len(manifest_rows),
        "bytes": total,
        "max_bytes": MAX_ARCHIVE_BYTES,
        "excluded_checkpoints": True,
        "excluded_raw_logs": True,
        "items": manifest_rows,
    }
    (destination / "ARCHIVE_MANIFEST.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.run_root, args.dest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
