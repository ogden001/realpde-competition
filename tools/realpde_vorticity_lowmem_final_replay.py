#!/usr/bin/env python3
"""Independent official-v9 replay for the low-memory final checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def arm_output_dir(root: Path, arm: str) -> Path:
    output = root / arm
    output.mkdir(parents=True, exist_ok=True)
    return output


def configure_cap(device: torch.device, gib: float) -> None:
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction((gib * 1024 ** 3) / total, torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def replay_one(arm: str, checkpoint: Path, manifest: Path, kit_root: Path, out: Path,
               micro_batch: int, workers: int, seed: int, memory_cap_gib: float) -> dict:
    import realpde_loss_official_v9 as core
    import realpde_mf01 as direct

    _, train_paths = core.read_manifest(manifest, "train")
    _, dev_paths = core.read_manifest(manifest, "dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("low-memory replay requires CUDA")
    configure_cap(device, memory_cap_gib)
    builder, _ = direct.build_features(train_paths, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("iteration", -1)) != 12000:
        raise ValueError(f"unexpected replay checkpoint iteration: {checkpoint}")
    model = direct.cno_direct(kit_root, len(builder.feature_names), device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    args = argparse.Namespace(batch_size=micro_batch, workers=workers, max_windows=None,
                              seed=seed, kit_root=kit_root)
    ds, loader = core.loader(dev_paths, args, shuffle=False)
    pred_all, target_all = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _, _ in loader:
            pred = direct.forward(model, builder, x.to(device, non_blocking=True)).clone()
            pred[..., 2] = 0.0
            pred_all.append(pred.cpu().numpy().astype(np.float32))
            target_all.append(y.numpy().astype(np.float32))
    pred, target = np.concatenate(pred_all), np.concatenate(target_all)
    if not np.isfinite(pred).all() or float(np.abs(pred[..., 2]).max()) != 0.0:
        raise FloatingPointError(f"invalid replay prediction for {arm}")
    result = core.score_bundle(kit_root, pred, target, 0.0, arm_output_dir(out, arm))
    result.update({"arm": arm, "update": 12000, "checkpoint": str(checkpoint),
                   "checkpoint_sha256": core.sha256(checkpoint),
                   "peak_gpu_memory_reserved": int(torch.cuda.max_memory_reserved())})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--c0-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--memory-cap-gib", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.seed != 20260901 or args.micro_batch < 1:
        raise ValueError("replay requires seed 20260901 and positive micro-batch")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [replay_one("C0", args.c0_checkpoint, args.manifest, args.kit_root, args.out_dir,
                          args.micro_batch, args.workers, args.seed, args.memory_cap_gib),
               replay_one("V1", args.v1_checkpoint, args.manifest, args.kit_root, args.out_dir,
                          args.micro_batch, args.workers, args.seed, args.memory_cap_gib)]
    checks = []
    for item in results:
        arm_dir = args.first_root / ("C0-LOWMEM" if item["arm"] == "C0" else "V1-LOWMEM")
        first = json.loads((arm_dir / "eval_12000" / "scores.json").read_text())
        delta = {name: item["raw_errors"][name] - first["raw_errors"][name] for name in ("rel_l2", "tke", "mvpe")}
        checks.append({"arm": item["arm"], "first_eval_raw_errors": first["raw_errors"],
                       "replay_raw_errors": item["raw_errors"], "absolute_delta": delta,
                       "pass": all(abs(value) <= 2e-5 for value in delta.values())})
    result = {"status": "REVIEW_REQUIRED", "checks": checks, "all_pass": all(item["pass"] for item in checks)}
    (args.out_dir / "replay_check.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    with (args.out_dir / "replay_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "rel_l2", "tke", "mvpe"])
        writer.writeheader()
        for item in results:
            writer.writerow({"arm": item["arm"], **item["raw_errors"]})
    if not result["all_pass"]:
        raise RuntimeError("final replay mismatch")


if __name__ == "__main__":
    main()
