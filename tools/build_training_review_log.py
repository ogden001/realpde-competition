#!/usr/bin/env python3
"""Build a compact, deterministic training log for Sol review.

Small logs are copied in full. Large logs keep the beginning/end, all key
training events, and evenly sampled TRAIN lines (or ordinary lines when no
TRAIN lines exist). A metadata JSON is written next to the review log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

DEFAULT_EVENT_RE = re.compile(
    r"(?:^|\b)(?:EVAL|CHECKPOINT|SAVE|SAVED|RESUME|RESUMED|BEST|"
    r"WARNING|WARN|ERROR|EXCEPTION|TRACEBACK|OOM|OUT OF MEMORY|"
    r"NAN|INF|ABORT|STOP|FAILED|FAILURE)(?:\b|:|=)",
    re.IGNORECASE,
)
TRAIN_RE = re.compile(r"(?:^|\b)TRAIN(?:\b|:|=)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced(indices: list[int], limit: int) -> list[int]:
    if limit <= 0 or not indices:
        return []
    if len(indices) <= limit:
        return indices
    if limit == 1:
        return [indices[len(indices) // 2]]
    positions = [round(i * (len(indices) - 1) / (limit - 1)) for i in range(limit)]
    return [indices[pos] for pos in positions]


def build_review(
    lines: list[str],
    *,
    full_copy_max_lines: int,
    full_copy_max_bytes: int,
    head_lines: int,
    tail_lines: int,
    sample_lines: int,
) -> tuple[list[str], dict]:
    raw_bytes = sum(len(line.encode("utf-8")) for line in lines)
    if len(lines) <= full_copy_max_lines and raw_bytes <= full_copy_max_bytes:
        return lines, {
            "selection_mode": "full",
            "event_line_count": sum(bool(DEFAULT_EVENT_RE.search(line)) for line in lines),
            "sampled_line_count": 0,
        }

    n = len(lines)
    keep: set[int] = set(range(min(head_lines, n)))
    keep.update(range(max(0, n - tail_lines), n))

    event_indices = [i for i, line in enumerate(lines) if DEFAULT_EVENT_RE.search(line)]
    keep.update(event_indices)

    train_indices = [i for i, line in enumerate(lines) if TRAIN_RE.search(line) and i not in keep]
    sample_source = train_indices
    sample_source_name = "train"
    if not sample_source:
        sample_source = [i for i in range(n) if i not in keep]
        sample_source_name = "ordinary"

    sampled = evenly_spaced(sample_source, sample_lines)
    keep.update(sampled)

    selected = sorted(keep)
    return [lines[i] for i in selected], {
        "selection_mode": f"compressed_{sample_source_name}",
        "event_line_count": len(event_indices),
        "sampled_line_count": len(sampled),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-copy-max-lines", type=int, default=1000)
    parser.add_argument("--full-copy-max-kib", type=int, default=200)
    parser.add_argument("--head-lines", type=int, default=20)
    parser.add_argument("--tail-lines", type=int, default=20)
    parser.add_argument("--sample-lines", type=int, default=40)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must differ from --input")

    raw = args.input.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines(keepends=True)
    review_lines, selection = build_review(
        lines,
        full_copy_max_lines=args.full_copy_max_lines,
        full_copy_max_bytes=args.full_copy_max_kib * 1024,
        head_lines=args.head_lines,
        tail_lines=args.tail_lines,
        sample_lines=args.sample_lines,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(review_lines), encoding="utf-8")

    raw_size = args.input.stat().st_size
    review_size = args.output.stat().st_size
    meta = {
        "raw_log_path": str(args.input),
        "review_log_path": str(args.output),
        "raw_sha256": sha256(args.input),
        "raw_line_count": len(lines),
        "raw_size_bytes": raw_size,
        "review_line_count": len(review_lines),
        "review_size_bytes": review_size,
        "full_copy_max_lines": args.full_copy_max_lines,
        "full_copy_max_kib": args.full_copy_max_kib,
        "head_lines": args.head_lines,
        "tail_lines": args.tail_lines,
        "sample_lines_requested": args.sample_lines,
        **selection,
    }
    meta_path = Path(str(args.output) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
