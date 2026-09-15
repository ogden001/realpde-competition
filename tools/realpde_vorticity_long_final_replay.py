#!/usr/bin/env python3
"""Independent C0@15000/V1@15000 official-v9 scorer replay."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import realpde_loss_official_v9 as core
import realpde_mf01 as direct


def replay_one(arm: str, checkpoint: Path, manifest: Path, kit_root: Path, out: Path,
               batch_size: int, workers: int, seed: int) -> dict:
    _, train_paths = core.read_manifest(manifest, "train")
    _, dev_paths = core.read_manifest(manifest, "dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder, _ = direct.build_features(train_paths, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("iteration", -1)) != 15000:
        raise ValueError(f"unexpected replay checkpoint iteration: {checkpoint}")
    model = direct.cno_direct(kit_root, len(builder.feature_names), device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    args = argparse.Namespace(batch_size=batch_size, workers=workers, max_windows=None, seed=seed, kit_root=kit_root)
    ds, loader = core.loader(dev_paths, args, shuffle=False)
    pred_all, target_all = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _, _ in loader:
            pred = direct.forward(model, builder, x.to(device, non_blocking=True)).clone()
            pred[..., 2] = 0.0
            pred_all.append(pred.cpu().numpy().astype(np.float32)); target_all.append(y.numpy().astype(np.float32))
    pred, target = np.concatenate(pred_all), np.concatenate(target_all)
    if not np.isfinite(pred).all() or float(np.abs(pred[..., 2]).max()) != 0.0:
        raise FloatingPointError(f"invalid replay prediction for {arm}")
    result = core.score_bundle(kit_root, pred, target, 0.0, out / arm)
    result.update({"arm": arm, "update": 15000, "checkpoint": str(checkpoint), "checkpoint_sha256": core.sha256(checkpoint)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--c0-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.batch_size != 8 or args.seed != 20260901:
        raise ValueError("replay must use batch=8 and seed=20260901")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay = [replay_one("C0", args.c0_checkpoint, args.manifest, args.kit_root, args.out_dir,
                         args.batch_size, args.workers, args.seed),
              replay_one("V1", args.v1_checkpoint, args.manifest, args.kit_root, args.out_dir,
                         args.batch_size, args.workers, args.seed)]
    first = {}
    for arm in ("C0-LONG", "V1-LONG"):
        rows = list(csv.DictReader((args.first_root / arm / "aggregate_metrics.csv").open(newline="")))
        first[arm[:2]] = {name: float(next(row for row in rows if int(row["update"]) == 15000)[name])
                          for name in ("rel_l2", "tke", "mvpe")}
    checks = []
    for item in replay:
        arm = item["arm"]
        values = item["raw_errors"]
        source = first[arm]
        delta = {name: values[name] - source[name] for name in source}
        checks.append({"arm": arm, "first_eval_raw_errors": source, "replay_raw_errors": values,
                       "absolute_delta": delta, "pass": all(abs(x) <= 2e-5 for x in delta.values())})
    result = {"status": "REVIEW_REQUIRED", "checks": checks,
              "all_pass": all(item["pass"] for item in checks)}
    (args.out_dir / "replay_check.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    with (args.out_dir / "replay_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "rel_l2", "tke", "mvpe"]); writer.writeheader()
        for item in replay:
            writer.writerow({"arm": item["arm"], **item["raw_errors"]})
    if not result["all_pass"]:
        raise RuntimeError("final replay mismatch")


if __name__ == "__main__":
    main()
