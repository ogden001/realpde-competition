#!/usr/bin/env python3
"""Exact teammate SPS recipe on the frozen Full Joint@6500 point predictor.

Scientific semantics intentionally match the uploaded teammate package
submission_all81_probe_fp16_20260915:
- usable released trajectories: all except 7575_0.h5;
- trajectory split: seed 41, 20% validation => 65 train / 16 validation;
- canonical fixed windows: 20 in / 20 out / stride 20 / 2x spatial subsample;
- uncertainty head: teammate 35-channel features, h32/b2/drop0;
- head input: Past20 + pre-corrector base prediction;
- supervised error: frozen final post-corrector point prediction;
- masked Gaussian NLL on measured u/v;
- AdamW, lr=1e-3, weight_decay=1e-5, batch=8, 2000 updates;
- evaluate every 200 updates;
- calibration: the teammate 28-row floor x multiplier grid, same multiplier for u/v;
- no rel-width term, no channel-specific multipliers, no width cap, no AoA selection.

The Full Joint point predictor is frozen.  Its outputs are cached once in RAM so
head training does not repeatedly execute the expensive CNO/corrector.  The
cache is float32, so this is an engineering acceleration only and does not
change the SPS recipe.

No locked-final/private/Codabench access or automatic submission exists here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import realpde_sps_teammate_final as teammate_cal
import train_sota_merge_sps as joint_sps
from colleague_80pt.realpde_h5_feature_adapter_train import (
    BAD_TRAIN_FILES,
    H5WindowDataset,
    list_h5,
    split_paths,
)
from sps_teammate_uncertainty_runtime import (
    TeammateUncertaintyHead,
    masked_gaussian_nll_from_log_std,
    sigma_from_log_std,
)

STATUS = "REVIEW_REQUIRED"
PROTOCOL = "REALPDE_FULL_JOINT_TEAMMATE_SPS_EXACT_V1"
RECIPE = "teammate35_exact_full_joint"

SEED = 41
VAL_FRACTION = 0.20
UPDATES = 2_000
EVAL_INTERVAL = 200
BATCH = 8
LR = 1e-3
WEIGHT_DECAY = 1e-5
SIGMA0 = 0.02

EXPECTED_JOINT_PROTOCOL = "REALPDE_SOTA_MERGE_FULL_JOINT_V1"
EXPECTED_JOINT_UPDATE = 6_500
EXPECTED_BACKBONE_SHA256 = "e965958385004c0fa17e34d4b6cb0aea518cfa0b6181c7e79d3ec82eb931b305"
EXPECTED_CORRECTOR_SHA256 = "0abd4029b2e55225127b18e10fbcd6d47e8ce40fbeac8607de3d28c2a36908d9"

EXPECTED_USABLE_TRAJECTORIES = 81
EXPECTED_TRAIN_TRAJECTORIES = 65
EXPECTED_VAL_TRAJECTORIES = 16
EXPECTED_TRAIN_WINDOWS = 2_701
EXPECTED_VAL_WINDOWS = 640

FIXED_GRID = tuple(
    (float(floor), float(mult))
    for floor in (0.0, 0.0025, 0.005, 0.0075)
    for mult in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
)


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def canonical_dataset(paths: list[Path]) -> H5WindowDataset:
    return H5WindowDataset(
        paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="fixed",
    )


def teammate_split(data_root: Path) -> tuple[list[Path], list[Path]]:
    paths = list_h5(data_root, BAD_TRAIN_FILES)
    if len(paths) != EXPECTED_USABLE_TRAJECTORIES:
        raise ValueError(
            f"exact teammate recipe expects {EXPECTED_USABLE_TRAJECTORIES} usable "
            f"trajectories after excluding {sorted(BAD_TRAIN_FILES)}, got {len(paths)}"
        )
    train_paths, val_paths = split_paths(paths, VAL_FRACTION, SEED)
    if (len(train_paths), len(val_paths)) != (
        EXPECTED_TRAIN_TRAJECTORIES,
        EXPECTED_VAL_TRAJECTORIES,
    ):
        raise ValueError(
            f"exact teammate split drift: {len(train_paths)}/{len(val_paths)} "
            f"!= {EXPECTED_TRAIN_TRAJECTORIES}/{EXPECTED_VAL_TRAJECTORIES}"
        )
    return train_paths, val_paths


def init_head(device: torch.device) -> TeammateUncertaintyHead:
    head = TeammateUncertaintyHead(
        hidden=32,
        blocks=2,
        dropout=0.0,
        include_pressure=True,
    ).to(device)
    torch.nn.init.zeros_(head.net[-1].weight)
    torch.nn.init.constant_(head.net[-1].bias, math.log(SIGMA0))
    return head


@torch.no_grad()
def cache_frozen_split(
    *,
    backbone,
    builder,
    corrector,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    workers: int,
    label: str,
) -> dict[str, np.ndarray]:
    ds = canonical_dataset(paths)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    shape = (len(ds), 20, 32, 64, 2)
    arrays = {
        "x": np.empty(shape, dtype=np.float32),
        "base": np.empty(shape, dtype=np.float32),
        "final": np.empty(shape, dtype=np.float32),
        "target": np.empty(shape, dtype=np.float32),
    }

    backbone.eval()
    corrector.eval()
    offset = 0
    started = time.monotonic()
    for batch_index, batch in enumerate(loader, start=1):
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        base, final = joint_sps.point_forward(backbone, builder, corrector, x)
        n = int(x.shape[0])
        arrays["x"][offset : offset + n] = x[..., :2].cpu().numpy()
        arrays["base"][offset : offset + n] = base[..., :2].cpu().numpy()
        arrays["final"][offset : offset + n] = final[..., :2].cpu().numpy()
        arrays["target"][offset : offset + n] = y[..., :2].numpy()
        offset += n
        if batch_index % 20 == 0 or offset == len(ds):
            print(
                f"[TEAMMATE-SPS] cache {label}: {offset}/{len(ds)} "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )
    if offset != len(ds):
        raise RuntimeError(f"cache {label} incomplete: {offset} != {len(ds)}")
    return arrays


def uv_to_three(array_uv: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(array_uv, dtype=np.float32),
            np.zeros(array_uv.shape[:-1] + (1,), dtype=np.float32),
        ],
        axis=-1,
    )


@torch.no_grad()
def predict_sigma_cached(
    head: TeammateUncertaintyHead,
    cache: dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    head.eval()
    n = int(cache["x"].shape[0])
    sigma = np.empty(cache["x"].shape, dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = torch.from_numpy(cache["x"][start:stop]).to(device, non_blocking=True)
        base = torch.from_numpy(cache["base"][start:stop]).to(
            device, non_blocking=True
        )
        sigma[start:stop] = (
            sigma_from_log_std(head(x, base)).cpu().numpy().astype(np.float32)
        )
    return sigma


def calibrate_exact(
    *,
    head: TeammateUncertaintyHead,
    cache: dict[str, np.ndarray],
    device: torch.device,
    scoring,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    prediction = uv_to_three(cache["final"])
    target = uv_to_three(cache["target"])
    sigma = predict_sigma_cached(head, cache, device=device)
    rows, best = teammate_cal._calibrate(prediction, sigma, target, scoring)
    got_grid = {(float(row["floor"]), float(row["mult"])) for row in rows}
    if got_grid != set(FIXED_GRID) or len(rows) != len(FIXED_GRID):
        raise RuntimeError("teammate 28-row calibration grid drifted")
    return rows, best


def select_best(evals: list[dict[str, object]]) -> dict[str, object]:
    if not evals:
        raise ValueError("no SPS evaluations")
    return max(
        evals,
        key=lambda row: (
            float(row["best"]["sps"]),
            -float(row["best"]["mean_width_uv"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--corrector-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_paths, val_paths = teammate_split(args.data_root)
    train_ds = canonical_dataset(train_paths)
    val_ds = canonical_dataset(val_paths)
    if len(train_ds) != EXPECTED_TRAIN_WINDOWS:
        raise ValueError(
            f"teammate train windows drift: {len(train_ds)} != "
            f"{EXPECTED_TRAIN_WINDOWS}"
        )
    if len(val_ds) != EXPECTED_VAL_WINDOWS:
        raise ValueError(
            f"teammate val windows drift: {len(val_ds)} != {EXPECTED_VAL_WINDOWS}"
        )

    (
        _bp,
        _cp,
        feature_cfg,
        builder,
        backbone,
        corrector,
        backbone_sha,
        corrector_sha,
    ) = joint_sps.load_joint_pair(
        args.backbone_checkpoint,
        args.corrector_checkpoint,
        args.kit_root,
        device,
        target_joint_update=EXPECTED_JOINT_UPDATE,
        expected_backbone_sha256=EXPECTED_BACKBONE_SHA256,
        expected_corrector_sha256=EXPECTED_CORRECTOR_SHA256,
        expected_joint_protocol=EXPECTED_JOINT_PROTOCOL,
    )

    sys.path.insert(0, str(args.kit_root.resolve()))
    import scoring

    cache_started = time.monotonic()
    train_cache = cache_frozen_split(
        backbone=backbone,
        builder=builder,
        corrector=corrector,
        paths=train_paths,
        device=device,
        batch_size=args.cache_batch_size,
        workers=args.workers,
        label="train65",
    )
    val_cache = cache_frozen_split(
        backbone=backbone,
        builder=builder,
        corrector=corrector,
        paths=val_paths,
        device=device,
        batch_size=args.cache_batch_size,
        workers=args.workers,
        label="val16",
    )
    cache_seconds = time.monotonic() - cache_started

    # Point-model GPU memory is no longer needed after the frozen cache exists.
    del backbone, corrector, builder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_tensor_ds = TensorDataset(
        torch.from_numpy(train_cache["x"]),
        torch.from_numpy(train_cache["base"]),
        torch.from_numpy(train_cache["final"]),
        torch.from_numpy(train_cache["target"]),
    )
    train_loader = DataLoader(
        train_tensor_ds,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Re-seed immediately before exact head initialization/training.  Cache
    # generation must not consume the scientific training RNG stream.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    head = init_head(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    iterator = iter(train_loader)
    losses: list[dict[str, float]] = []
    evals: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    started = time.monotonic()

    head.train()
    for update in range(1, UPDATES + 1):
        try:
            x_uv, base_uv, final_uv, target_uv = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x_uv, base_uv, final_uv, target_uv = next(iterator)

        x_uv = x_uv.to(device, non_blocking=True)
        base_uv = base_uv.to(device, non_blocking=True)
        final_uv = final_uv.to(device, non_blocking=True)
        target_uv = target_uv.to(device, non_blocking=True)

        log_std = head(x_uv, base_uv)
        loss = masked_gaussian_nll_from_log_std(
            target_uv,
            final_uv,
            log_std,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite teammate SPS loss @ {update}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if update == 1 or update % 100 == 0 or update == UPDATES:
            row = {"update": update, "loss": float(loss.detach().cpu())}
            losses.append(row)
            print(
                f"[TEAMMATE-SPS] update {update}/{UPDATES} "
                f"loss={row['loss']:.6f}",
                flush=True,
            )

        if update % EVAL_INTERVAL == 0:
            head.eval()
            grid, best = calibrate_exact(
                head=head,
                cache=val_cache,
                device=device,
                scoring=scoring,
            )
            record = {
                "iteration": update,
                "best": best,
                "grid": grid,
            }
            evals.append(record)
            selected = select_best(evals)
            if int(selected["iteration"]) == update:
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in head.state_dict().items()
                }
            print(
                f"[TEAMMATE-SPS] eval @{update}: "
                f"SPS={float(best['sps']):.6f} "
                f"floor={float(best['floor']):.4f} "
                f"mult={float(best['mult']):.2f} "
                f"coverage={float(best['coverage']):.6f}",
                flush=True,
            )
            dump_json(args.out_dir / "checkpoint_evals.json", evals)
            head.train()

    if best_state is None:
        raise RuntimeError("no exact teammate SPS checkpoint selected")

    selected = select_best(evals)
    head.load_state_dict(best_state, strict=True)
    head.eval()
    final_grid, final_best = calibrate_exact(
        head=head,
        cache=val_cache,
        device=device,
        scoring=scoring,
    )
    if (
        int(selected["iteration"]) % EVAL_INTERVAL != 0
        or abs(float(final_best["sps"]) - float(selected["best"]["sps"])) > 1e-9
        or float(final_best["floor"]) != float(selected["best"]["floor"])
        or float(final_best["mult"]) != float(selected["best"]["mult"])
    ):
        raise RuntimeError("restored teammate SPS checkpoint/calibration mismatch")

    head_path = args.out_dir / "teammate_sps_head_best.pth"
    metadata = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "recipe": RECIPE,
        "seed": SEED,
        "selected_updates": int(selected["iteration"]),
        "max_updates": UPDATES,
        "eval_interval": EVAL_INTERVAL,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH,
        "sigma0": SIGMA0,
        "loss": "masked_gaussian_nll_nonzero_uv",
        "architecture": {
            "in_channels": 35,
            "hidden": 32,
            "blocks": 2,
            "dropout": 0.0,
            "include_pressure": True,
        },
        "calibration": {
            "floor": float(final_best["floor"]),
            "mult": float(final_best["mult"]),
            "grid": "teammate_exact_28_row_floor_x_mult",
            "same_multiplier_u_v": True,
            "rel_term": 0.0,
        },
        "point_model": {
            "protocol": EXPECTED_JOINT_PROTOCOL,
            "joint_update": EXPECTED_JOINT_UPDATE,
            "backbone_sha256": backbone_sha,
            "corrector_sha256": corrector_sha,
            "frozen": True,
        },
        "split": {
            "usable_trajectories": EXPECTED_USABLE_TRAJECTORIES,
            "excluded": sorted(BAD_TRAIN_FILES),
            "seed": SEED,
            "val_fraction": VAL_FRACTION,
            "train_trajectories": len(train_paths),
            "val_trajectories": len(val_paths),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "stride": 20,
            "sub_sample": 2,
        },
        "feature_config": vars(feature_cfg),
        "head_observes": "Past20 + pre-corrector base via teammate exact 35-channel features",
        "error_target": "post-corrector + spatial-TKE-projection final point prediction",
    }
    torch.save(
        {
            "head_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in head.state_dict().items()
            },
            "metadata": metadata,
        },
        head_path,
    )

    dump_json(args.out_dir / "selected_calibration_grid.json", final_grid)
    summary = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "recipe": RECIPE,
        "point_model_frozen": True,
        "optimizer_updates_point_model": 0,
        "backbone_sha256": backbone_sha,
        "corrector_sha256": corrector_sha,
        "selected_iteration": int(selected["iteration"]),
        "best": final_best,
        "head": str(head_path),
        "head_sha256": joint_sps.strong.sha256(head_path),
        "train_trajectories": len(train_paths),
        "val_trajectories": len(val_paths),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "cache_seconds": cache_seconds,
        "head_training_seconds": time.monotonic() - started,
        "loss_curve": losses,
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "auto_submit": False,
    }
    dump_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "DONE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
