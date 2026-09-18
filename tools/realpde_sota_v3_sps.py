#!/usr/bin/env python3
"""SOTA-V3 SPS stage aligned to the teammate base->corrector structure.

Frozen inference semantics:
- uncertainty features/head observe the frozen base backbone prediction;
- point prediction is base -> residual corrector -> spatial_tke_map projection;
- intervals are centered on that final corrected prediction.

The package audit confirms the head-input/base vs final-center split, but did
not contain the teammate head-training source. V3 therefore preregisters one
reconstruction choice: masked Gaussian NLL uses the final corrected prediction
as the error center, because sigma must describe the final submitted point
prediction. No SPS-only sweep is performed.
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

import realpde_residual_corrector_projection as projection
import realpde_sota_v2_full as full
import realpde_sota_v2_integrated as base
import realpde_sps_teammate_final as teammate
import realpde_sps_teammate_exact_submit as exact
import realpde_sota_v3_residual as residual
import realpde_teammate_residual_transfer as transfer
from sps_teammate_uncertainty_runtime import (
    TeammateUncertaintyHead,
    masked_gaussian_nll_from_log_std,
    sigma_from_log_std,
)

SEED = exact.SEED
MAX_UPDATES = exact.MAX_UPDATES
EVAL_INTERVAL = exact.EVAL_INTERVAL
HEAD_LR = exact.HEAD_LR
HEAD_WEIGHT_DECAY = exact.HEAD_WEIGHT_DECAY
SIGMA0 = exact.SIGMA0


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _prepare_out(out_dir: Path, resume: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not resume:
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _init_head(device: torch.device) -> TeammateUncertaintyHead:
    head = TeammateUncertaintyHead(
        hidden=32, blocks=2, dropout=0.0, include_pressure=True
    ).to(device)
    torch.nn.init.zeros_(head.net[-1].weight)
    torch.nn.init.constant_(
        head.net[-1].bias, math.log(SIGMA0)
    )
    return head


def _load_stack(
    backbone_checkpoint: Path,
    corrector_checkpoint: Path,
    kit_root: Path,
    device: torch.device,
):
    (
        backbone_payload, config, builder, backbone
    ) = residual._load_backbone(
        backbone_checkpoint, kit_root, device
    )
    corrector, corrector_payload = residual._load_corrector(
        corrector_checkpoint,
        device,
        base.sha256(backbone_checkpoint),
    )
    return (
        backbone_payload,
        corrector_payload,
        config,
        builder,
        backbone,
        corrector,
    )


def _final_prediction(
    x, *, backbone, builder, corrector
):
    base_prediction = base.forward_mf(
        backbone, builder, x
    )
    delta = transfer._delta_from_corrector(
        corrector, x, base_prediction
    )
    corrected = transfer.apply_scaled_correction(
        base_prediction, delta, 1.0
    )
    final_prediction = (
        projection.spatial_tke_map_projection(
            base_prediction, corrected
        )
    )
    return base_prediction, final_prediction


def _dataset(paths: list[Path]):
    return teammate._canonical_dataset(paths)


@torch.no_grad()
def _collect(
    *, paths: list[Path], backbone, builder,
    corrector, head, batch_size: int,
    workers: int, device: torch.device,
):
    loader = torch.utils.data.DataLoader(
        _dataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )
    finals, bases, sigmas, targets = [], [], [], []
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        base_prediction, final_prediction = (
            _final_prediction(
                x,
                backbone=backbone,
                builder=builder,
                corrector=corrector,
            )
        )
        sigma = sigma_from_log_std(
            head(x, base_prediction)
        )
        bases.append(
            base_prediction.cpu().numpy().astype(np.float32)
        )
        finals.append(
            final_prediction.cpu().numpy().astype(np.float32)
        )
        sigmas.append(
            sigma.cpu().numpy().astype(np.float32)
        )
        targets.append(
            y.numpy().astype(np.float32)
        )
    return (
        np.concatenate(bases),
        np.concatenate(finals),
        np.concatenate(sigmas),
        np.concatenate(targets),
    )


def _calibrate(
    final_prediction: np.ndarray,
    sigma: np.ndarray,
    target: np.ndarray,
    scoring,
):
    return teammate._calibrate(
        final_prediction, sigma, target, scoring
    )


def _select_best(evals: list[dict]) -> dict:
    return exact._select_best_checkpoint(evals)


def _train_dev(
    *, train_paths, dev_paths, backbone, builder,
    corrector, batch_size, workers, device, scoring,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    train_ds = _dataset(train_paths)
    if len(train_ds) != 2052:
        raise ValueError(
            f"Train50 canonical windows {len(train_ds)} != 2052"
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
    head = _init_head(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=HEAD_LR,
        weight_decay=HEAD_WEIGHT_DECAY,
    )
    iterator = iter(loader)
    losses, evals = [], []
    best_state = None
    started = time.monotonic()

    for update in range(1, MAX_UPDATES + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _ = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            base_prediction, final_prediction = (
                _final_prediction(
                    x,
                    backbone=backbone,
                    builder=builder,
                    corrector=corrector,
                )
            )

        log_std = head(x, base_prediction)
        loss = masked_gaussian_nll_from_log_std(
            y[..., :2],
            final_prediction[..., :2],
            log_std,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite V3 SPS loss @ {update}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if update == 1 or update % 100 == 0:
            losses.append({
                "update": update,
                "loss": float(loss.detach().cpu()),
            })

        if update % EVAL_INTERVAL == 0:
            head.eval()
            (
                base_dev,
                final_dev,
                sigma_dev,
                target_dev,
            ) = _collect(
                paths=dev_paths,
                backbone=backbone,
                builder=builder,
                corrector=corrector,
                head=head,
                batch_size=batch_size,
                workers=workers,
                device=device,
            )
            grid, best = _calibrate(
                final_dev, sigma_dev, target_dev, scoring
            )
            record = {
                "iteration": update,
                "best": best,
                "grid": grid,
                "base_final_mean_abs_delta": float(
                    np.mean(np.abs(
                        final_dev[..., :2]
                        - base_dev[..., :2]
                    ))
                ),
            }
            evals.append(record)
            selected = _select_best(evals)
            if int(selected["iteration"]) == update:
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in head.state_dict().items()
                }
            head.train()

    if best_state is None:
        raise RuntimeError(
            "no V3 SPS checkpoint selected"
        )
    selected = _select_best(evals)
    head.load_state_dict(best_state, strict=True)
    head.eval()
    (
        _,
        final_dev,
        sigma_dev,
        target_dev,
    ) = _collect(
        paths=dev_paths,
        backbone=backbone,
        builder=builder,
        corrector=corrector,
        head=head,
        batch_size=batch_size,
        workers=workers,
        device=device,
    )
    grid, best = _calibrate(
        final_dev, sigma_dev, target_dev, scoring
    )
    if abs(
        float(best["sps"])
        - float(selected["best"]["sps"])
    ) > 1e-9:
        raise RuntimeError(
            "restored V3 SPS best checkpoint mismatch"
        )
    return (
        head,
        losses,
        evals,
        selected,
        grid,
        best,
        time.monotonic() - started,
        len(train_ds),
    )


def _train_full(
    *, paths, backbone, builder, corrector,
    selected_updates, batch_size, workers, device,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    dataset = _dataset(paths)
    if len(dataset) != full.EXPECTED_CANONICAL_WINDOWS:
        raise ValueError(
            f"full canonical windows {len(dataset)} "
            f"!= {full.EXPECTED_CANONICAL_WINDOWS}"
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    head = _init_head(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=HEAD_LR,
        weight_decay=HEAD_WEIGHT_DECAY,
    )
    iterator = iter(loader)
    losses = []
    started = time.monotonic()

    for update in range(1, selected_updates + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _ = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            base_prediction, final_prediction = (
                _final_prediction(
                    x,
                    backbone=backbone,
                    builder=builder,
                    corrector=corrector,
                )
            )

        loss = masked_gaussian_nll_from_log_std(
            y[..., :2],
            final_prediction[..., :2],
            head(x, base_prediction),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite full V3 SPS loss @ {update}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (
            update == 1
            or update % 100 == 0
            or update == selected_updates
        ):
            losses.append({
                "update": update,
                "loss": float(loss.detach().cpu()),
            })

    head.eval()
    return (
        head,
        losses,
        time.monotonic() - started,
        len(dataset),
    )


def run(args: argparse.Namespace) -> dict:
    _prepare_out(args.out_dir, args.resume)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    sys.path.insert(0, str(args.kit_root.resolve()))
    import scoring

    (
        backbone_payload,
        corrector_payload,
        config,
        builder,
        backbone,
        corrector,
    ) = _load_stack(
        args.backbone_checkpoint,
        args.corrector_checkpoint,
        args.kit_root,
        device,
    )
    common_meta = {
        "recipe": (
            "teammate35_exact_on_V3_base_features_"
            "centered_on_projected_final"
        ),
        "head_input": "base backbone prediction",
        "interval_center": (
            "residual-corrected spatial_tke_map "
            "final prediction"
        ),
        "training_loss": (
            "masked Gaussian NLL on final corrected "
            "prediction error"
        ),
        "training_loss_evidence": (
            "V3 preregistered reconstruction; teammate "
            "package training source unavailable"
        ),
        "architecture": {
            "in_channels": 35,
            "hidden": 32,
            "blocks": 2,
            "dropout": 0.0,
            "include_pressure": True,
        },
        "seed": SEED,
        "lr": HEAD_LR,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "sigma0": SIGMA0,
        "backbone_iteration": int(
            backbone_payload.get("iteration", -1)
        ),
        "backbone_sha256": base.sha256(
            args.backbone_checkpoint
        ),
        "corrector_sha256": base.sha256(
            args.corrector_checkpoint
        ),
        "corrector_backbone_sha256": (
            corrector_payload.get("backbone_sha256")
        ),
        "projection": "spatial_tke_map",
        "locked_final_accessed": False,
        "codabench_accessed": False,
    }

    if args.mode == "dev":
        if (
            args.manifest is None
            or base.sha256(args.manifest) != base.MANIFEST_SHA
        ):
            raise ValueError(
                "dev mode requires frozen 50/16 manifest"
            )
        train_paths = base.split_paths(
            args.manifest, "train", args.data_root
        )
        dev_paths = base.split_paths(
            args.manifest, "dev", args.data_root
        )
        (
            head,
            losses,
            evals,
            selected,
            grid,
            best,
            elapsed,
            train_windows,
        ) = _train_dev(
            train_paths=train_paths,
            dev_paths=dev_paths,
            backbone=backbone,
            builder=builder,
            corrector=corrector,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            scoring=scoring,
        )
        selected_updates = int(
            selected["iteration"]
        )
        metadata = {
            **common_meta,
            "mode": "dev_50_16",
            "head_scope": "v3_train50_dev16",
            "max_updates": MAX_UPDATES,
            "eval_interval": EVAL_INTERVAL,
            "selected_updates": selected_updates,
            "train_trajectories": 50,
            "dev_trajectories": 16,
            "train_windows": train_windows,
            "dev_windows": 659,
            "bound_floor": float(best["floor"]),
            "bound_mult": float(best["mult"]),
            "feature_config": vars(config),
        }
        head_path = (
            args.out_dir / "v3_teammate35_dev_head.pth"
        )
        torch.save({
            "head_state_dict": {
                k: v.detach().cpu()
                for k, v in head.state_dict().items()
            },
            "metadata": metadata,
        }, head_path)
        dump(
            args.out_dir / "checkpoint_evals.json",
            evals,
        )
        dump(
            args.out_dir / "selected_calibration_grid.json",
            grid,
        )
        dump(
            args.out_dir / "head_training_summary.json",
            {
                "head": str(head_path),
                "head_sha256": base.sha256(head_path),
                "elapsed_seconds": elapsed,
                "loss_curve": losses,
                "selected_iteration": selected_updates,
            },
        )
        result = {
            "status": "REVIEW_REQUIRED",
            "mode": "dev",
            "selected_iteration": selected_updates,
            "best": best,
            "head": str(head_path),
            "head_sha256": base.sha256(head_path),
            "backbone_sha256": common_meta[
                "backbone_sha256"
            ],
            "corrector_sha256": common_meta[
                "corrector_sha256"
            ],
        }
    else:
        if (
            args.selected_updates is None
            or not (
                EVAL_INTERVAL
                <= args.selected_updates
                <= MAX_UPDATES
            )
            or args.selected_updates % EVAL_INTERVAL
        ):
            raise ValueError(
                "full mode requires Dev-selected "
                "200-step checkpoint in [200,2000]"
            )
        if (
            float(args.bound_floor),
            float(args.bound_mult),
        ) not in exact.legacy.FIXED_GRID:
            raise ValueError(
                "bounds must come from frozen 28-row grid"
            )

        paths = full.released_paths(args.data_root)
        (
            head,
            losses,
            elapsed,
            train_windows,
        ) = _train_full(
            paths=paths,
            backbone=backbone,
            builder=builder,
            corrector=corrector,
            selected_updates=args.selected_updates,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
        )
        metadata = {
            **common_meta,
            "mode": "full_82",
            "head_scope": "v3_full82",
            "selected_updates": int(
                args.selected_updates
            ),
            "train_trajectories": len(paths),
            "train_windows": train_windows,
            "bound_floor": float(args.bound_floor),
            "bound_mult": float(args.bound_mult),
            "calibration_source": (
                "frozen Dev16 before full-data refit"
            ),
            "feature_config": vars(config),
        }
        head_path = (
            args.out_dir / "v3_teammate35_full_head.pth"
        )
        torch.save({
            "head_state_dict": {
                k: v.detach().cpu()
                for k, v in head.state_dict().items()
            },
            "metadata": metadata,
        }, head_path)
        dump(
            args.out_dir / "head_training_summary.json",
            {
                "head": str(head_path),
                "head_sha256": base.sha256(head_path),
                "elapsed_seconds": elapsed,
                "loss_curve": losses,
                "selected_iteration": int(
                    args.selected_updates
                ),
            },
        )
        result = {
            "status": "REVIEW_REQUIRED",
            "mode": "full",
            "selected_iteration": int(
                args.selected_updates
            ),
            "bounds": {
                "floor": float(args.bound_floor),
                "mult": float(args.bound_mult),
            },
            "head": str(head_path),
            "head_sha256": base.sha256(head_path),
            "backbone_sha256": common_meta[
                "backbone_sha256"
            ],
            "corrector_sha256": common_meta[
                "corrector_sha256"
            ],
        }

    dump(args.out_dir / "run_metadata.json", metadata)
    dump(args.out_dir / "summary.json", result)
    dump(args.out_dir / "status.json", {
        "state": "DONE",
        "status": "REVIEW_REQUIRED",
    })
    print(json.dumps(
        result, indent=2, sort_keys=True, default=str
    ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("dev", "full"), required=True
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--data-root", type=Path, required=True
    )
    parser.add_argument(
        "--kit-root", type=Path, required=True
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--corrector-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True
    )
    parser.add_argument("--selected-updates", type=int)
    parser.add_argument("--bound-floor", type=float)
    parser.add_argument("--bound-mult", type=float)
    parser.add_argument(
        "--batch-size", type=int, default=8
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--resume", action="store_true"
    )
    parser.add_argument(
        "--require-cuda", action="store_true"
    )
    args = parser.parse_args()
    if args.mode == "full" and (
        args.bound_floor is None
        or args.bound_mult is None
    ):
        parser.error(
            "full mode requires --bound-floor and --bound-mult"
        )
    run(args)


if __name__ == "__main__":
    main()
