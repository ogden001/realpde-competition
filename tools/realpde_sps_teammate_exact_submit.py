#!/usr/bin/env python3
"""Exact teammate SPS recipe on the frozen full SOTA-V2 point predictor.

This is a submission-oriented SPS-only run:
- point predictor is frozen full SOTA-V2 @53582;
- uncertainty head trains on the project's frozen 50 Train trajectories;
- checkpoint and floor/mult are selected on the project's frozen 16 Dev trajectories;
- teammate 35-channel features/head, masked Gaussian NLL, 2000-update budget,
  200-update evaluation cadence, and 28-row SPS calibration grid are copied;
- no point-model training or modification is allowed.

The 16 Dev trajectories were seen by the full point predictor because @53582 is
the all-released refit. They are held out from uncertainty-head training only.
This runner never accesses locked-final data and never submits Codabench.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import realpde_sota_v2_adaptive as legacy
import realpde_sps_teammate_final as teammate
from sps_teammate_uncertainty_runtime import masked_gaussian_nll_from_log_std

EXPERIMENT = "SPS-TEAMMATE-EXACT-SUBMIT-01"
SEED = 41
MAX_UPDATES = 2000
EVAL_INTERVAL = 200
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-5
SIGMA0 = 0.02
HEAD_SCOPE = "full53582_train50_teammate35_exact"
RECIPE = "teammate35_exact"
READY_GATE = "SPS_TEAMMATE_EXACT_READY"


def _dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _select_best_checkpoint(evals: list[dict[str, object]]) -> dict[str, object]:
    if not evals:
        raise ValueError("checkpoint evals are empty")
    return max(
        evals,
        key=lambda row: (
            float(row["best"]["sps"]),
            -float(row["best"]["mean_width_uv"]),
        ),
    )


def _train_exact(
    *,
    model,
    builder,
    train_paths: list[Path],
    dev_paths: list[Path],
    batch_size: int,
    workers: int,
    device: torch.device,
    scoring,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = teammate._canonical_dataset(train_paths)
    if len(train_ds) != legacy.EXPECTED_TRAIN_WINDOWS:
        raise ValueError(
            f"frozen 50-Train windows {len(train_ds)} != {legacy.EXPECTED_TRAIN_WINDOWS}"
        )

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    head = teammate._init_head(device)
    # _init_head uses the same h32/b2/drop0 architecture and sigma0=0.02 bias.
    if teammate.SIGMA0 != SIGMA0:
        raise ValueError("teammate sigma0 drifted from exact recipe")

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=HEAD_LR,
        weight_decay=HEAD_WEIGHT_DECAY,
    )
    iterator = iter(loader)
    losses: list[dict[str, float]] = []
    evals: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    started = time.monotonic()

    head.train()
    for update in range(1, MAX_UPDATES + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _ = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            prediction = legacy._forward(model, builder, x)

        log_std = head(x, prediction)
        loss = masked_gaussian_nll_from_log_std(
            y[..., :2],
            prediction[..., :2],
            log_std,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite exact teammate SPS loss @ {update}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if update == 1 or update % 100 == 0 or update == MAX_UPDATES:
            losses.append({"update": update, "loss": float(loss.detach().cpu())})

        if update % EVAL_INTERVAL == 0:
            head.eval()
            pred_dev, sigma_dev, target_dev = teammate._collect(
                model,
                builder,
                head,
                dev_paths,
                batch_size=batch_size,
                workers=workers,
                device=device,
            )
            grid, best = teammate._calibrate(pred_dev, sigma_dev, target_dev, scoring)
            record: dict[str, object] = {
                "iteration": update,
                "best": best,
                "grid": grid,
            }
            evals.append(record)

            selected = _select_best_checkpoint(evals)
            if int(selected["iteration"]) == update:
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in head.state_dict().items()
                }
            head.train()

    if best_state is None:
        raise RuntimeError("no exact teammate SPS checkpoint selected")

    selected = _select_best_checkpoint(evals)
    head.load_state_dict(best_state, strict=True)
    head.eval()
    return head, losses, evals, selected, time.monotonic() - started, len(train_ds)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    if teammate.sha256(args.manifest) != legacy.EXPECTED_MANIFEST_SHA:
        raise ValueError("frozen 50/16 manifest SHA mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    sys.path.insert(0, str(args.kit_root.resolve()))
    import scoring

    train_paths, dev_paths = legacy._split_paths(args.manifest, args.data_root)
    if len(train_paths) != 50 or len(dev_paths) != 16:
        raise ValueError("frozen manifest is not 50 Train / 16 Dev")

    _, full_config, builder, model, full_sha = teammate._load_full_backbone(
        args.full_checkpoint,
        args.kit_root,
        device,
    )
    if full_sha != teammate.EXPECTED_FULL_CHECKPOINT_SHA:
        raise ValueError("full SOTA checkpoint SHA mismatch")

    # Capture the frozen point prediction before SPS-head training.
    pred_before, _, target_before = teammate._collect(
        model,
        builder,
        teammate._init_head(device).eval(),
        dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    raw_point_errors = legacy._raw_metrics(pred_before, target_before, scoring)

    head, losses, evals, selected, elapsed, train_windows = _train_exact(
        model=model,
        builder=builder,
        train_paths=train_paths,
        dev_paths=dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        scoring=scoring,
    )

    pred_after, sigma_after, target_after = teammate._collect(
        model,
        builder,
        head,
        dev_paths,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    if not np.array_equal(target_before, target_after):
        raise RuntimeError("Dev targets changed or reordered")
    parity = float(np.max(np.abs(pred_after - pred_before)))
    if parity > teammate.MAX_PREDICTION_PARITY:
        raise RuntimeError(f"point prediction parity failed: {parity}")

    final_grid, final_best = teammate._calibrate(
        pred_after,
        sigma_after,
        target_after,
        scoring,
    )
    selected_best = selected["best"]
    if (
        int(selected["iteration"]) not in range(EVAL_INTERVAL, MAX_UPDATES + 1, EVAL_INTERVAL)
        or abs(float(final_best["sps"]) - float(selected_best["sps"])) > 1e-9
        or float(final_best["floor"]) != float(selected_best["floor"])
        or float(final_best["mult"]) != float(selected_best["mult"])
    ):
        raise RuntimeError("restored best checkpoint does not reproduce selected calibration")

    head_path = args.out_dir / "teammate_exact_full53582_train50_head.pth"
    metadata = {
        "head_scope": HEAD_SCOPE,
        "recipe": RECIPE,
        "seed": SEED,
        "max_updates": MAX_UPDATES,
        "eval_interval": EVAL_INTERVAL,
        "selected_updates": int(selected["iteration"]),
        "lr": HEAD_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "sigma0": SIGMA0,
        "loss": "masked_gaussian_nll_nonzero_uv",
        "architecture": {
            "in_channels": 35,
            "hidden": 32,
            "blocks": 2,
            "dropout": 0.0,
            "include_pressure": True,
        },
        "backbone_checkpoint_iteration": teammate.EXPECTED_FULL_ITERATION,
        "backbone_checkpoint_sha256": full_sha,
        "manifest_sha256": legacy.EXPECTED_MANIFEST_SHA,
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "train_windows": train_windows,
        "dev_windows": legacy.EXPECTED_DEV_WINDOWS,
        "window_mode": "fixed",
        "feature_config": vars(full_config),
        "bound_floor": float(final_best["floor"]),
        "bound_mult": float(final_best["mult"]),
        "calibration_source": "frozen project 16-Dev / real SPS / 28-row floor-mult grid",
        "point_predictor_note": (
            "Full @53582 point predictor is frozen and saw all released trajectories; "
            "Dev is held out from uncertainty-head training only."
        ),
    }
    torch.save(
        {
            "head_state_dict": {
                name: value.detach().cpu()
                for name, value in head.state_dict().items()
            },
            "metadata": metadata,
        },
        head_path,
    )

    _dump_json(args.out_dir / "checkpoint_evals.json", evals)
    _dump_json(args.out_dir / "selected_calibration_grid.json", final_grid)
    _dump_json(
        args.out_dir / "head_training_summary.json",
        {
            "head": str(head_path),
            "head_sha256": teammate.sha256(head_path),
            "elapsed_seconds": elapsed,
            "train_windows": train_windows,
            "loss_curve": losses,
            "selected_iteration": int(selected["iteration"]),
        },
    )

    result = {
        "status": "REVIEW_REQUIRED",
        "experiment": EXPERIMENT,
        "gate": READY_GATE,
        "goal": "Copy teammate SPS recipe onto frozen full SOTA-V2 @53582 for submission.",
        "best": final_best,
        "selected_iteration": int(selected["iteration"]),
        "prediction_parity_max_abs": parity,
        "raw_point_errors": raw_point_errors,
        "head": str(head_path),
        "head_sha256": teammate.sha256(head_path),
        "full_checkpoint_sha256": full_sha,
        "manifest_sha256": teammate.sha256(args.manifest),
        "scorer_sha256": teammate.sha256(args.kit_root / "scoring.py"),
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "train_windows": train_windows,
        "dev_windows": legacy.EXPECTED_DEV_WINDOWS,
        "recipe": {
            "features": "teammate exact 35-channel future features",
            "architecture": "h32/b2/drop0/include_pressure",
            "loss": "masked Gaussian NLL over non-zero u/v",
            "updates": MAX_UPDATES,
            "eval_interval": EVAL_INTERVAL,
            "lr": HEAD_LR,
            "weight_decay": HEAD_WEIGHT_DECAY,
            "sigma0": SIGMA0,
            "seed": SEED,
            "calibration_grid": [
                {"floor": float(floor), "mult": float(mult)}
                for floor, mult in legacy.FIXED_GRID
            ],
        },
        "locked_final_accessed": False,
        "codabench_accessed": False,
    }
    _dump_json(args.out_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
