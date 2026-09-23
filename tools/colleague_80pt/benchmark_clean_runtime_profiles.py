#!/usr/bin/env python3
"""Disposable runtime profiling for the clean RealPDE research protocol.

Compare batch=8 vs batch=16 using training-step throughput and VRAM only.
The recommendation is evidence for future 24G-GPU experiment families and
never overrides REALPDE_CLEAN_BASELINE_V1, which remains batch=8.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from batch_profiles import select_profile

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_UPDATES = 220
BENCHMARK_WARMUP = 20
BATCHES = (8, 16)
PREFETCH_FACTOR = 4


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_command(
    *,
    stage: str,
    python: str,
    real_root: Path,
    split_manifest: Path,
    checkpoint: Path,
    model_root: Path,
    out_dir: Path,
    workers: int,
    batch_size: int,
) -> list[str]:
    if stage == "cno":
        return [
            python, "-u", "-B", str(SCRIPT_DIR / "train_clean_baseline_cno.py"),
            "--real-root", str(real_root),
            "--split-manifest", str(split_manifest),
            "--init-checkpoint", str(checkpoint),
            "--realpdebench-root", str(model_root),
            "--out-dir", str(out_dir),
            "--updates", str(BENCHMARK_UPDATES),
            "--eval-interval", str(BENCHMARK_UPDATES),
            "--batch-size", str(batch_size),
            "--test-batch-size", "32",
            "--workers", str(workers),
            "--preload-to-ram",
            "--prefetch-factor", str(PREFETCH_FACTOR),
            "--lr", "0.0001",
            "--seed", "41",
            "--benchmark-mode",
            "--benchmark-warmup", str(BENCHMARK_WARMUP),
        ]
    if stage == "residual":
        return [
            python, "-u", "-B", str(SCRIPT_DIR / "residual_multi.py"),
            "--real-root", str(real_root),
            "--split-manifest", str(split_manifest),
            "--checkpoint", str(checkpoint),
            "--realpdebench-root", str(model_root),
            "--base-model", "cno",
            "--out-dir", str(out_dir),
            "--updates", str(BENCHMARK_UPDATES),
            "--eval-interval", str(BENCHMARK_UPDATES),
            "--batch-size", str(batch_size),
            "--test-batch-size", "32",
            "--workers", str(workers),
            "--preload-to-ram",
            "--prefetch-factor", str(PREFETCH_FACTOR),
            "--lr", "0.0002",
            "--weight-decay", "0.00001",
            "--hidden", "96",
            "--blocks", "2",
            "--dropout", "0",
            "--max-delta", "0.04",
            "--stride", "1",
            "--eval-stride", "20",
            "--train-window-mode", "fixed",
            "--train-alpha", "1.0",
            "--eval-alphas", "1.0",
            "--point", "1.0",
            "--mse", "0.05",
            "--tke", "0.06",
            "--temporal", "0.03",
            "--grad", "0.015",
            "--p-zero", "0.01",
            "--residual-mse", "0.25",
            "--delta-penalty", "0.02",
            "--clip-grad", "1.0",
            "--bound-abs", "0.0075",
            "--bound-rel", "0.0075",
            "--selection-metric", "point_score",
            "--max-eval-batches", "1",
            "--seed", "41",
            "--benchmark-mode",
            "--benchmark-warmup", str(BENCHMARK_WARMUP),
        ]
    raise ValueError(f"unsupported stage: {stage}")


def run_profile(command: list[str], out_dir: Path, log_path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "command": command,
        "out_dir": str(out_dir),
        "log": str(log_path),
    }
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record["returncode"] = completed.returncode
    runtime_path = out_dir / "runtime.json"
    record["success"] = completed.returncode == 0 and runtime_path.is_file()
    if record["success"]:
        record.update(json.loads(runtime_path.read_text(encoding="utf-8")))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cno", "residual"), required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.out_root.exists():
        raise FileExistsError(args.out_root)
    for path in (args.real_root, args.split_manifest, args.checkpoint, args.model_root):
        if not path.exists():
            raise FileNotFoundError(path)
    args.out_root.mkdir(parents=True)

    results: dict[str, object] = {
        "kind": "clean_runtime_profile",
        "stage": args.stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_updates": BENCHMARK_UPDATES,
        "benchmark_warmup": BENCHMARK_WARMUP,
        "selection_uses_validation_metrics": False,
        "baseline_auto_override": False,
        "profiles": {},
    }
    records: dict[str, dict[str, object]] = {}
    for batch in BATCHES:
        name = f"b{batch}"
        out_dir = args.out_root / name
        command = build_command(
            stage=args.stage,
            python=sys.executable,
            real_root=args.real_root,
            split_manifest=args.split_manifest,
            checkpoint=args.checkpoint,
            model_root=args.model_root,
            out_dir=out_dir,
            workers=args.workers,
            batch_size=batch,
        )
        record = run_profile(command, out_dir, args.out_root / f"{name}.log")
        records[name] = record
        results["profiles"][name] = record
        write_json(args.out_root / "benchmark_results.json", results)

    if not bool(records["b8"].get("success")):
        raise RuntimeError("batch=8 reference runtime benchmark failed")
    decision = select_profile(records["b8"], records["b16"])
    recommendation = {
        "stage": args.stage,
        "recommended_future_profile": decision["selected"],
        "decision": decision,
        "baseline_auto_override": False,
        "note": (
            "Recommendation is for future experiment families only. "
            "REALPDE_CLEAN_BASELINE_V1 remains batch=8."
        ),
    }
    results["decision"] = decision
    write_json(args.out_root / "benchmark_results.json", results)
    write_json(args.out_root / "recommendation.json", recommendation)
    print(json.dumps(recommendation, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
