#!/usr/bin/env python3
"""SOTA-V3 backbone training with conservative AoA augmentation.

V3 keeps the validated SOTA-V2 model/loss recipe and changes only the training
input distribution: with probability 0.5 per sample, the u/v vectors of the
whole Past20+Future20 pair are rotated by one shared angle sampled uniformly
from [-2 deg, +2 deg]. Pressure is unchanged. Dev/evaluation data are never
augmented.

Modes:
- dev: frozen 50/16 protocol, evaluate frozen milestones, auto-select one
  reference checkpoint from updates >= 30k using a preregistered normalized
  three-metric objective.
- full: all 82 released PIV trajectories, no label-based model selection; the
  selected dev reference update is mapped by Dense-All epoch exposure.

Both modes checkpoint runner state and support --resume. No locked-final,
private data, SPS, package, or Codabench path exists here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import realpde_sota_v2_full as full
import realpde_sota_v2_integrated as base

AOA_MAX_DEG = 2.0
AOA_PROBABILITY = 0.5
SELECT_MIN_UPDATE = 30_000
REGISTERED_BASELINE = {
    "rel_l2": 0.09993461519479752,
    "tke": 0.4692927300930023,
    "mvpe": 0.07577798515558243,
}
RECOVERY_EVERY = 1_000
EFFECTIVE_BATCH = 8


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, values: list[dict]) -> None:
    if not values:
        return
    fields = list(dict.fromkeys(
        key for row in values for key in row
    ))
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(values)


def rotate_uv(
    flow: Tensor, angles_deg: Tensor | float
) -> Tensor:
    """Rotate u/v vectors per sample; keep pressure/extra channels."""
    if flow.ndim != 5 or flow.shape[-1] < 2:
        raise ValueError(
            "flow must be [B,T,H,W,C>=2]"
        )
    angles = torch.as_tensor(
        angles_deg,
        dtype=flow.dtype,
        device=flow.device,
    )
    if angles.ndim == 0:
        angles = angles.repeat(flow.shape[0])
    if (
        angles.ndim != 1
        or angles.shape[0] != flow.shape[0]
    ):
        raise ValueError(
            "angles_deg must be scalar or [B]"
        )
    radians = angles * (math.pi / 180.0)
    c = torch.cos(radians).view(-1, 1, 1, 1)
    s = torch.sin(radians).view(-1, 1, 1, 1)
    u, v = flow[..., 0], flow[..., 1]
    out = flow.clone()
    out[..., 0] = c * u - s * v
    out[..., 1] = s * u + c * v
    return out


def sample_aoa_angles(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_deg: float = AOA_MAX_DEG,
    probability: float = AOA_PROBABILITY,
) -> Tensor:
    if (
        batch_size < 1
        or max_deg < 0
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError(
            "invalid AoA augmentation parameters"
        )
    active = (
        torch.rand(batch_size, device=device)
        < probability
    )
    angles = (
        2.0
        * torch.rand(
            batch_size,
            device=device,
            dtype=dtype,
        )
        - 1.0
    ) * float(max_deg)
    return torch.where(
        active, angles, torch.zeros_like(angles)
    )


def augment_pair(
    x: Tensor,
    y: Tensor,
    *,
    max_deg: float = AOA_MAX_DEG,
    probability: float = AOA_PROBABILITY,
) -> tuple[Tensor, Tensor, Tensor]:
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Past/Future batch sizes differ"
        )
    angles = sample_aoa_angles(
        x.shape[0],
        device=x.device,
        dtype=x.dtype,
        max_deg=max_deg,
        probability=probability,
    )
    return (
        rotate_uv(x, angles),
        rotate_uv(y, angles),
        angles,
    )


def selection_objective(
    row: dict[str, float]
) -> float:
    """Lower is better; equal normalized weight across physical errors."""
    values = []
    for key in ("rel_l2", "tke", "mvpe"):
        value = float(row[key])
        if (
            not math.isfinite(value)
            or value < 0
        ):
            return float("inf")
        values.append(
            value / REGISTERED_BASELINE[key]
        )
    return float(sum(values))


def select_dev_checkpoint(
    history: list[dict]
) -> dict:
    candidates = [
        row for row in history
        if int(row["update"]) >= SELECT_MIN_UPDATE
    ]
    if not candidates:
        raise ValueError(
            "no eligible Dev checkpoint at/after 30k"
        )
    selected = min(
        candidates,
        key=lambda row: (
            selection_objective(row),
            int(row["update"]),
        ),
    )
    return {
        **selected,
        "selection_objective": (
            selection_objective(selected)
        ),
    }


def _save_runner_state(
    path: Path,
    *,
    model,
    optimizer,
    update: int,
    epoch: int,
    batches_in_epoch: int,
    history: list[dict],
    feature_config,
    metadata: dict,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "iteration": int(update),
        "epoch": int(epoch),
        "batches_in_epoch": int(
            batches_in_epoch
        ),
        "history": history,
        "feature_set": "P0-A",
        "feature_config": vars(feature_config),
        "loss_weights": base.N2,
        "lambda_vort": base.LAMBDA_VORT,
        "extra_rel_stage_b": base.EXTRA_REL,
        "metadata": metadata,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = (
            torch.cuda.get_rng_state_all()
        )
    torch.save(payload, path)


def _restore_runner_state(
    path: Path, *, model, optimizer
) -> dict:
    payload = torch.load(
        path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(
        payload["model_state_dict"], strict=True
    )
    optimizer.load_state_dict(
        payload["optimizer_state_dict"]
    )
    torch.set_rng_state(
        payload["torch_rng_state"]
    )
    np.random.set_state(
        payload["numpy_rng_state"]
    )
    if (
        torch.cuda.is_available()
        and "cuda_rng_state_all" in payload
    ):
        torch.cuda.set_rng_state_all(
            payload["cuda_rng_state_all"]
        )
    return payload


def _resume_iterator(
    loader, sampler, *, epoch: int,
    batches_in_epoch: int,
):
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(batches_in_epoch):
        try:
            next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                "saved batches_in_epoch exceeds "
                "loader length"
            ) from exc
    return iterator


def _validate_common(
    args: argparse.Namespace
) -> None:
    if (
        args.micro_batch
        * args.accumulation_steps
        != EFFECTIVE_BATCH
    ):
        raise ValueError(
            "effective batch must equal 8"
        )
    if (
        args.aoa_max_deg != AOA_MAX_DEG
        or args.aoa_probability
        != AOA_PROBABILITY
    ):
        raise ValueError(
            "SOTA-V3 freezes AoA augmentation "
            "at +/-2deg, p=0.5"
        )
    if args.seed != base.SEED:
        raise ValueError(
            "SOTA-V3 uses frozen backbone seed"
        )
    if (
        base.sha256(args.checkpoint)
        != full.WARM_START_SHA
    ):
        raise ValueError(
            "warm-start checkpoint SHA mismatch"
        )
    if (
        base.sha256(
            args.kit_root / "scoring.py"
        )
        != full.SCORER_SHA
    ):
        raise ValueError(
            "official scorer SHA mismatch"
        )


def _prepare_out(
    out_dir: Path, *, resume: bool
) -> None:
    if (
        out_dir.exists()
        and any(out_dir.iterdir())
        and not resume
    ):
        raise FileExistsError(out_dir)
    out_dir.mkdir(
        parents=True, exist_ok=True
    )


def _preflight(
    model, builder, payload, train_paths,
    args, device, out_dir: Path,
) -> None:
    loader_args = Namespace(
        eval_batch_size=2, workers=0
    )
    probe_ds, probe_loader = base.dev_loader(
        train_paths[:1], loader_args
    )
    probe_x, probe_y, _, _ = next(
        iter(probe_loader)
    )
    probe_x = probe_x.to(device)
    probe_y = probe_y.to(device)
    aug_x, aug_y, angles = augment_pair(
        probe_x,
        probe_y,
        max_deg=args.aoa_max_deg,
        probability=args.aoa_probability,
    )
    pred = base.forward_mf(
        model, builder, aug_x
    )
    loss, parts = base.integrated_loss(
        pred, aug_y, 1
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(
            "non-finite V3 preflight loss"
        )
    loss.backward()
    grad = max(
        float(p.grad.abs().max())
        for p in model.parameters()
        if p.grad is not None
    )
    model.zero_grad(set_to_none=True)

    direct = base.build_direct(
        args.kit_root,
        builder,
        payload,
        device,
    )
    parity_pred = base.forward_mf(
        model, builder, probe_x
    )
    direct_pred = base.forward_direct(
        direct, builder, probe_x
    )
    parity = float(
        (
            parity_pred[..., :2]
            - direct_pred[..., :2]
        ).abs().max()
    )
    result = {
        "passed": bool(
            torch.isfinite(pred).all()
            and grad > 0
            and parity <= args.parity_tolerance
        ),
        "direct_to_mf_uv_max_abs_diff": parity,
        "gradient_max_abs": grad,
        "probe_angles_deg": [
            float(v)
            for v in angles.detach().cpu()
        ],
        "loss": float(loss.detach().cpu()),
        "loss_parts": {
            k: float(v.detach().cpu())
            for k, v in parts.items()
        },
        "probe_windows": len(probe_ds),
    }
    dump(
        out_dir / "preflight.json", result
    )
    if not result["passed"]:
        raise RuntimeError(
            "SOTA-V3 backbone preflight failed"
        )


def _train_loop(
    *,
    args,
    mode: str,
    train_paths: list[Path],
    model,
    builder,
    feature_config,
    optimizer,
    final_update: int,
    stage_a_end: int,
    milestones: tuple[int, ...],
    metadata: dict,
    dev_paths: list[Path] | None,
    out_dir: Path,
) -> list[dict]:
    train_ds, sampler, loader = (
        base.dense_loader(train_paths, args)
    )
    state_path = (
        out_dir
        / "checkpoints"
        / "runner_latest.pth"
    )
    state_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    history: list[dict] = []
    start_update = 1
    epoch = 0
    batches_in_epoch = 0
    current_lr = base.STAGE_A_LR

    if args.resume and state_path.is_file():
        payload = _restore_runner_state(
            state_path,
            model=model,
            optimizer=optimizer,
        )
        start_update = (
            int(payload["iteration"]) + 1
        )
        epoch = int(
            payload.get("epoch", 0)
        )
        batches_in_epoch = int(
            payload.get(
                "batches_in_epoch", 0
            )
        )
        history = list(
            payload.get("history", [])
        )
        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

    iterator = _resume_iterator(
        loader,
        sampler,
        epoch=epoch,
        batches_in_epoch=batches_in_epoch,
    )
    started = time.monotonic()

    for update in range(
        start_update, final_update + 1
    ):
        if update <= stage_a_end:
            cfg = {
                "stage": "A",
                "lr": base.STAGE_A_LR,
                "extra_rel": 0.0,
            }
        else:
            cfg = {
                "stage": "B",
                "lr": base.STAGE_B_LR,
                "extra_rel": base.EXTRA_REL,
            }

        if float(cfg["lr"]) != current_lr:
            current_lr = float(cfg["lr"])
            for group in optimizer.param_groups:
                group["lr"] = current_lr

        optimizer.zero_grad(
            set_to_none=True
        )
        totals = {
            k: 0.0
            for k in (*base.N2, "vorticity")
        }
        angle_abs_sum = 0.0
        angle_count = 0

        for _ in range(
            args.accumulation_steps
        ):
            try:
                x, y, _, _ = next(iterator)
                batches_in_epoch += 1
            except StopIteration:
                epoch += 1
                batches_in_epoch = 0
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                x, y, _, _ = next(iterator)
                batches_in_epoch = 1

            x = x.to(
                args.device, non_blocking=True
            )
            y = y.to(
                args.device, non_blocking=True
            )
            x, y, angles = augment_pair(
                x,
                y,
                max_deg=args.aoa_max_deg,
                probability=args.aoa_probability,
            )
            angle_abs_sum += float(
                angles.abs().sum().detach().cpu()
            )
            angle_count += int(
                angles.numel()
            )

            model.train()
            pred = base.forward_mf(
                model, builder, x
            )
            parts = base.core.loss_parts(
                pred, y
            )
            parts["vorticity"] = (
                base.vorticity(
                    pred, dx=1.0, dy=1.0
                )
                - base.vorticity(
                    y, dx=1.0, dy=1.0
                )
            ).square().mean()
            loss = sum(
                base.N2[k] * parts[k]
                for k in base.N2
            )
            loss = (
                loss
                + base.LAMBDA_VORT
                * parts["vorticity"]
                + float(cfg["extra_rel"])
                * parts["rel"]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite backbone loss "
                    f"@ {update}"
                )
            (
                loss
                / args.accumulation_steps
            ).backward()
            for key in totals:
                totals[key] += (
                    float(
                        parts[key]
                        .detach()
                        .cpu()
                    )
                    / args.accumulation_steps
                )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()

        if update % args.log_every == 0:
            print(json.dumps({
                "mode": mode,
                "update": update,
                "stage": cfg["stage"],
                "lr": current_lr,
                "mean_abs_aug_angle_deg": (
                    angle_abs_sum
                    / max(angle_count, 1)
                ),
                **totals,
            }), flush=True)

        if (
            mode == "dev"
            and update in milestones
        ):
            assert dev_paths is not None
            ev = base.evaluate(
                model,
                builder,
                dev_paths,
                args,
                args.device,
                out_dir
                / f"eval_{update:05d}",
                update,
            )
            row = {
                "update": update,
                "stage": cfg["stage"],
                "lr": current_lr,
                **ev["raw_errors"],
                "mean_t_neural_s": (
                    ev["mean_t_neural_s"]
                ),
                "selection_objective": (
                    selection_objective(
                        ev["raw_errors"]
                    )
                ),
                "elapsed_seconds": (
                    time.monotonic() - started
                ),
            }
            history.append(row)
            write_rows(
                out_dir
                / "aggregate_metrics.csv",
                history,
            )
            ckpt = (
                out_dir
                / "checkpoints"
                / f"model_update_{update:05d}.pth"
            )
            _save_runner_state(
                ckpt,
                model=model,
                optimizer=optimizer,
                update=update,
                epoch=epoch,
                batches_in_epoch=(
                    batches_in_epoch
                ),
                history=history,
                feature_config=feature_config,
                metadata=metadata,
            )

        if (
            update % args.recovery_every == 0
            or update == final_update
        ):
            _save_runner_state(
                state_path,
                model=model,
                optimizer=optimizer,
                update=update,
                epoch=epoch,
                batches_in_epoch=(
                    batches_in_epoch
                ),
                history=history,
                feature_config=feature_config,
                metadata=metadata,
            )

    return history


def run_dev(
    args: argparse.Namespace
) -> dict:
    _validate_common(args)
    _prepare_out(
        args.out_dir, resume=args.resume
    )
    if (
        base.sha256(args.manifest)
        != base.MANIFEST_SHA
    ):
        raise ValueError(
            "frozen 50/16 manifest SHA mismatch"
        )

    train_paths = base.split_paths(
        args.manifest,
        "train",
        args.data_root,
    )
    dev_paths = base.split_paths(
        args.manifest,
        "dev",
        args.data_root,
    )
    if (
        len(train_paths),
        len(dev_paths),
    ) != (50, 16):
        raise ValueError(
            "expected frozen 50/16 split"
        )

    args.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    if (
        args.require_cuda
        and args.device.type != "cuda"
    ):
        raise RuntimeError("CUDA required")

    base.core.set_seed(args.seed)
    builder, feature_config = (
        base.build_features(
            train_paths, args.device
        )
    )
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = base.MF01CNO(
        args.kit_root,
        len(builder.feature_names),
        args.device,
    )
    base.init_mf_from_direct(
        model,
        payload,
        len(builder.feature_names),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base.STAGE_A_LR,
    )

    metadata = {
        "recipe": (
            "SOTA-V2 + AoA-vector-rotation"
        ),
        "mode": "frozen-50/16-dev",
        "aoa_augmentation": {
            "max_deg": AOA_MAX_DEG,
            "probability": AOA_PROBABILITY,
            "distribution": (
                "per-sample Uniform[-2,+2] "
                "with p=0.5 else 0"
            ),
            "pair_semantics": (
                "same angle for Past20 and "
                "Future20; u/v only; "
                "pressure unchanged"
            ),
            "train_only": True,
        },
        "selection": {
            "eligible_update_min": (
                SELECT_MIN_UPDATE
            ),
            "objective": (
                "sum(rel/base_rel, "
                "tke/base_tke, "
                "mvpe/base_mvpe)"
            ),
            "baseline": REGISTERED_BASELINE,
        },
        "locked_final_accessed": False,
        "codabench_accessed": False,
    }
    dump(
        args.out_dir
        / "run_metadata.json",
        metadata,
    )

    if (
        not args.resume
        or not (
            args.out_dir
            / "checkpoints"
            / "runner_latest.pth"
        ).is_file()
    ):
        _preflight(
            model,
            builder,
            payload,
            train_paths,
            args,
            args.device,
            args.out_dir,
        )

    history = _train_loop(
        args=args,
        mode="dev",
        train_paths=train_paths,
        model=model,
        builder=builder,
        feature_config=feature_config,
        optimizer=optimizer,
        final_update=base.FINAL_UPDATE,
        stage_a_end=base.STAGE_A_END,
        milestones=base.MILESTONES,
        metadata=metadata,
        dev_paths=dev_paths,
        out_dir=args.out_dir,
    )

    selected = select_dev_checkpoint(
        history
    )
    selected_update = int(
        selected["update"]
    )
    selected_path = (
        args.out_dir
        / "checkpoints"
        / f"model_update_{selected_update:05d}.pth"
    )
    if not selected_path.is_file():
        raise FileNotFoundError(
            selected_path
        )

    result = {
        "status": "REVIEW_REQUIRED",
        "mode": "dev",
        "selected_reference_update": (
            selected_update
        ),
        "selected_checkpoint": str(
            selected_path
        ),
        "selected_checkpoint_sha256": (
            base.sha256(selected_path)
        ),
        "selection": selected,
        "history": history,
    }
    dump(
        args.out_dir / "selection.json",
        result,
    )
    dump(
        args.out_dir / "status.json",
        {
            "state": "DONE",
            "status": "REVIEW_REQUIRED",
        },
    )
    return result


def run_full(
    args: argparse.Namespace
) -> dict:
    _validate_common(args)
    _prepare_out(
        args.out_dir, resume=args.resume
    )
    if (
        args.reference_update
        not in base.MILESTONES
        or args.reference_update
        < SELECT_MIN_UPDATE
    ):
        raise ValueError(
            "reference update must be an "
            "eligible frozen Dev milestone"
        )

    train_paths = full.released_paths(
        args.data_root
    )
    args.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    if (
        args.require_cuda
        and args.device.type != "cuda"
    ):
        raise RuntimeError("CUDA required")

    base.core.set_seed(args.seed)
    builder, feature_config = (
        base.build_features(
            train_paths, args.device
        )
    )
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = base.MF01CNO(
        args.kit_root,
        len(builder.feature_names),
        args.device,
    )
    base.init_mf_from_direct(
        model,
        payload,
        len(builder.feature_names),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base.STAGE_A_LR,
    )

    final_update = (
        full.map_reference_update(
            args.reference_update
        )
    )
    stage_a_end = (
        full.map_reference_update(
            base.STAGE_A_END
        )
    )

    metadata = {
        "recipe": (
            "SOTA-V2 + AoA-vector-rotation"
        ),
        "mode": (
            "all-82-released-PIV-refit"
        ),
        "reference_update": (
            args.reference_update
        ),
        "mapped_final_update": (
            final_update
        ),
        "mapped_stage_a_end": (
            stage_a_end
        ),
        "aoa_augmentation": {
            "max_deg": AOA_MAX_DEG,
            "probability": AOA_PROBABILITY,
            "distribution": (
                "per-sample Uniform[-2,+2] "
                "with p=0.5 else 0"
            ),
            "pair_semantics": (
                "same angle for Past20 and "
                "Future20; u/v only; "
                "pressure unchanged"
            ),
            "train_only": True,
        },
        "checkpoint_selection": (
            "frozen from Dev before "
            "full-data refit"
        ),
        "locked_final_accessed": False,
        "codabench_accessed": False,
    }
    dump(
        args.out_dir
        / "run_metadata.json",
        metadata,
    )

    if (
        not args.resume
        or not (
            args.out_dir
            / "checkpoints"
            / "runner_latest.pth"
        ).is_file()
    ):
        _preflight(
            model,
            builder,
            payload,
            train_paths,
            args,
            args.device,
            args.out_dir,
        )

    _train_loop(
        args=args,
        mode="full",
        train_paths=train_paths,
        model=model,
        builder=builder,
        feature_config=feature_config,
        optimizer=optimizer,
        final_update=final_update,
        stage_a_end=stage_a_end,
        milestones=(final_update,),
        metadata=metadata,
        dev_paths=None,
        out_dir=args.out_dir,
    )

    latest = (
        args.out_dir
        / "checkpoints"
        / "runner_latest.pth"
    )
    final_path = (
        args.out_dir
        / "checkpoints"
        / f"model_update_{final_update:05d}.pth"
    )
    final_payload = torch.load(
        latest,
        map_location="cpu",
        weights_only=False,
    )
    torch.save(
        final_payload, final_path
    )
    result = {
        "status": "REVIEW_REQUIRED",
        "mode": "full",
        "reference_update": (
            args.reference_update
        ),
        "final_update": final_update,
        "checkpoint": str(final_path),
        "checkpoint_sha256": (
            base.sha256(final_path)
        ),
        "released_trajectories": len(
            train_paths
        ),
    }
    dump(
        args.out_dir / "summary.json",
        result,
    )
    dump(
        args.out_dir / "status.json",
        {
            "state": "DONE",
            "status": "REVIEW_REQUIRED",
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("dev", "full"),
        required=True,
    )
    parser.add_argument(
        "--manifest", type=Path
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--kit-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reference-update", type=int
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=base.SEED,
    )
    parser.add_argument(
        "--aoa-max-deg",
        type=float,
        default=AOA_MAX_DEG,
    )
    parser.add_argument(
        "--aoa-probability",
        type=float,
        default=AOA_PROBABILITY,
    )
    parser.add_argument(
        "--recovery-every",
        type=int,
        default=RECOVERY_EVERY,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--resume", action="store_true"
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
    )
    args = parser.parse_args()

    if args.mode == "dev":
        if args.manifest is None:
            parser.error(
                "--manifest is required in dev mode"
            )
        result = run_dev(args)
    else:
        if args.reference_update is None:
            parser.error(
                "--reference-update is "
                "required in full mode"
            )
        result = run_full(args)

    print(json.dumps(
        result,
        indent=2,
        sort_keys=True,
        default=str,
    ))


if __name__ == "__main__":
    main()
