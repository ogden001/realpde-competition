#!/usr/bin/env python3
"""Matched late-horizon diagnosis + backbone continuation screen for RealPDE Track 1.

Experiment 0 (no training):
  Replay four predictors on the exact same colleague Dev16 / stride20 windows:
    1) colleague CNO backbone
    2) colleague current80 residual model
    3) audited SOTA-V2 strong backbone @53582
    4) Arm-B strong-backbone + residual final
  Write common post-train diagnostics and a compact horizon fingerprint table.

Experiment 1 (matched training):
  Start Control and Ramp from the exact same audited SOTA-V2 checkpoint and
  optimizer state. Continue the backbone for 5k updates on all 82 released PIV
  trajectories with identical dense-all sampling.

  Control keeps the historical Stage-B objective.
  Ramp changes only the MSE term from uniform horizon weighting to a normalized
  linear 1.0 -> 2.0 horizon ramp. Because the ramp is normalized to mean 1,
  the average MSE loss scale is preserved.

No residual training, uncertainty training, locked-final/private access, or
Codabench access occurs here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
COLLEAGUE_TOOLS = ROOT / "tools" / "colleague_80pt"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(COLLEAGUE_TOOLS))

import realpde_sota_v2_full as full  # noqa: E402
import realpde_sota_v2_integrated as sota  # noqa: E402
import realpde_sota_v3_residual as sota_v3  # noqa: E402
from post_train_diagnostics import horizon_rows, write_post_train_diagnostics  # noqa: E402
from residual_multi import (  # noqa: E402
    CorrectorConfig,
    ResidualCorrectionModel,
    ResidualCorrector3D,
    load_frozen_base,
    load_residual_checkpoint,
)
from realpde_h5_feature_adapter_train import (  # noqa: E402
    H5WindowDataset,
    measured_channels,
    mvpe_rel_l2_per_sample,
    paths_from_split_manifest,
    rel_l2_per_sample,
    tke_rel_l2_per_sample,
)

SEED = 20260901
UPDATES = 5_000
EVAL_INTERVAL = 1_000
EVAL_STEPS = (0, 1_000, 2_000, 3_000, 4_000, 5_000)
MICRO_BATCH = 4
ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH = MICRO_BATCH * ACCUMULATION_STEPS
LR = 3e-6
RAMP_START = 1.0
RAMP_END = 2.0
STRONG_BACKBONE_SHA = "cc732555859cb00b3ea1af31415632c087af1fc60abd4f4056263e6f02ffa307"
COLLEAGUE_BASE_SHA = "ff28aaf0114d57e320e5ea9cf9945d874a7482f22e3f0aeaed5ebbe286573f8a"
CURRENT80_RESIDUAL_SHA = "909fdc7f8a6a42335e4ea4ce7471fc9a507135e9a29a4927bc7b7318be9c85b2"
ARM_B_FINAL_SHA = "fde7f3c20e1e42f93f616741a3c6e1a68d4e38c4dc002f3f4875f9f2c5b18762"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty csv: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def normalized_linear_horizon_weights(
    out_steps: int,
    *,
    start: float = RAMP_START,
    end: float = RAMP_END,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if out_steps < 2:
        raise ValueError("out_steps must be >= 2")
    if start <= 0 or end <= 0:
        raise ValueError("horizon weights must be positive")
    weights = torch.linspace(start, end, out_steps, device=device, dtype=dtype)
    return weights / weights.mean()


def horizon_weighted_velocity_mse(
    pred: Tensor,
    target: Tensor,
    *,
    start: float = RAMP_START,
    end: float = RAMP_END,
) -> Tensor:
    if pred.shape != target.shape or pred.ndim != 5 or pred.shape[-1] < 2:
        raise ValueError("expected matched [B,T,H,W,C>=2]")
    error2 = (pred[..., :2] - target[..., :2]).square()
    per_horizon = error2.mean(dim=(0, 2, 3, 4))
    weights = normalized_linear_horizon_weights(
        pred.shape[1],
        start=start,
        end=end,
        device=pred.device,
        dtype=pred.dtype,
    )
    return torch.sum(weights * per_horizon) / weights.sum()


def historical_stage_b_loss(
    pred: Tensor,
    target: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    parts = sota.core.loss_parts(pred, target)
    parts["vorticity"] = (
        sota.vorticity(pred, dx=1.0, dy=1.0)
        - sota.vorticity(target, dx=1.0, dy=1.0)
    ).square().mean()
    total = sum(full.N2[name] * parts[name] for name in full.N2)
    total = total + full.LAMBDA_VORT * parts["vorticity"]
    total = total + full.EXTRA_REL * parts["rel"]
    return total, parts


def continuation_loss(
    pred: Tensor,
    target: Tensor,
    *,
    mode: str,
) -> tuple[Tensor, dict[str, Tensor]]:
    control, parts = historical_stage_b_loss(pred, target)
    if mode == "control":
        parts = dict(parts)
        parts["uniform_velocity_mse"] = parts["mse"]
        parts["ramped_velocity_mse"] = parts["mse"]
        parts["horizon_ramp_delta"] = parts["mse"].new_zeros(())
        return control, parts
    if mode != "ramp":
        raise ValueError(f"unknown continuation mode: {mode}")

    ramped_mse = horizon_weighted_velocity_mse(pred, target)
    if abs(float(full.N2["mse"]) - 1.0) > 1e-12:
        raise RuntimeError("late-horizon screen assumes historical N2 mse weight == 1")
    total = control + (ramped_mse - parts["mse"])
    parts = dict(parts)
    parts["uniform_velocity_mse"] = parts["mse"]
    parts["ramped_velocity_mse"] = ramped_mse
    parts["horizon_ramp_delta"] = ramped_mse - parts["mse"]
    return total, parts


def _plain_predictor(base_model: torch.nn.Module):
    @torch.no_grad()
    def predict(x: Tensor) -> tuple[Tensor, Tensor | None]:
        pred = base_model(x)
        return pred, None
    return predict


def load_residual_with_base(
    *,
    residual_checkpoint: Path,
    base_kind: str,
    base_checkpoint: Path,
    kit_root: Path,
    device: torch.device,
) -> ResidualCorrectionModel:
    payload = torch.load(residual_checkpoint, map_location="cpu", weights_only=False)
    raw_config = payload.get("corrector_config")
    if not isinstance(raw_config, dict):
        raise KeyError(f"{residual_checkpoint} lacks corrector_config")
    base_model = load_frozen_base(base_kind, base_checkpoint, kit_root, device)
    model = ResidualCorrectionModel(
        base_model,
        ResidualCorrector3D(CorrectorConfig(**raw_config)),
    ).to(device)
    load_residual_checkpoint(model, residual_checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _residual_predictor(model: ResidualCorrectionModel):
    @torch.no_grad()
    def predict(x: Tensor) -> tuple[Tensor, Tensor | None]:
        base = model.base_predict(x)
        delta = model.predict_delta(x, base)
        return model.combine(base, delta, 1.0), base
    return predict


def collect_predictions(
    predictor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    has_base = False
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred, base = predictor(x)
        predictions.append(pred.detach().cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
        if base is not None:
            has_base = True
            bases.append(base.detach().cpu().numpy().astype(np.float32))
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(bases, axis=0) if has_base else None,
    )


def aggregate_raw_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    channels = measured_channels(target)
    return {
        "rel_l2_raw": float(np.mean(rel_l2_per_sample(pred, target, channels))),
        "tke_raw": float(np.mean(tke_rel_l2_per_sample(pred, target, channels))),
        "mvpe_raw": float(np.mean(mvpe_rel_l2_per_sample(pred, target))),
    }


def tail_fingerprint(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    by_h = {int(row["horizon"]): float(row["frame_rel_l2"]) for row in rows}
    if set(by_h) != set(range(1, 21)):
        raise ValueError("fingerprint requires exactly horizons 1..20")
    early = float(np.mean([by_h[h] for h in range(1, 18)]))
    return {
        "f17_rel": by_h[17],
        "f18_rel": by_h[18],
        "f19_rel": by_h[19],
        "f20_rel": by_h[20],
        "f20_over_mean_f1_f17": by_h[20] / max(early, 1e-12),
        "f20_over_f18": by_h[20] / max(by_h[18], 1e-12),
        "f19_over_f18": by_h[19] / max(by_h[18], 1e-12),
    }


def evaluate_prediction_bundle(
    *,
    label: str,
    pred: np.ndarray,
    target: np.ndarray,
    trajectories: list[str],
    starts: list[int],
    out_dir: Path,
    base_pred: np.ndarray | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = write_post_train_diagnostics(
        out_dir=out_dir / "diagnostics",
        experiment=label,
        prediction=pred,
        target=target,
        trajectories=trajectories,
        starts=starts,
        base_prediction=base_pred,
    )
    hrows = horizon_rows(
        pred,
        target,
        experiment=label,
        base_prediction=base_pred,
    )
    raw = aggregate_raw_metrics(pred, target)
    result = {
        "label": label,
        **raw,
        **tail_fingerprint(hrows),
        "windows": int(pred.shape[0]),
        "trajectories": len(set(trajectories)),
        "mean_fluctuation": diagnostics["mean_fluctuation"],
    }
    dump(out_dir / "summary.json", result)
    return result


def make_dev_loader(
    *,
    real_root: Path,
    split_manifest: Path,
    batch_size: int,
    workers: int,
) -> tuple[H5WindowDataset, DataLoader]:
    _, val_paths = paths_from_split_manifest(
        real_root,
        split_manifest,
        allow_train_dev_overlap=True,
    )
    dataset = H5WindowDataset(
        val_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    if len(dataset) != 640:
        raise ValueError(f"expected historical colleague Dev16 = 640 windows, got {len(dataset)}")
    return dataset, loader


def run_fingerprint(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dev_dataset: H5WindowDataset,
    dev_loader: DataLoader,
) -> dict[str, object]:
    out = args.out_root / "experiment0_fingerprint"
    out.mkdir(parents=True, exist_ok=False)
    sha_manifest = {
        "colleague_base": verify_sha(args.colleague_base, COLLEAGUE_BASE_SHA, "colleague base"),
        "current80_residual": verify_sha(
            args.current80_residual, CURRENT80_RESIDUAL_SHA, "current80 residual"
        ),
        "strong_backbone": verify_sha(
            args.strong_backbone, STRONG_BACKBONE_SHA, "strong backbone"
        ),
        "arm_b_final": verify_sha(args.arm_b_final, ARM_B_FINAL_SHA, "Arm B final"),
    }
    dump(out / "checkpoint_sha256.json", sha_manifest)

    colleague_base = load_frozen_base("cno", args.colleague_base, args.kit_root, device)
    current80 = load_residual_with_base(
        residual_checkpoint=args.current80_residual,
        base_kind="cno",
        base_checkpoint=args.colleague_base,
        kit_root=args.kit_root,
        device=device,
    )
    strong_base = load_frozen_base(
        "sota_v2_mf", args.strong_backbone, args.kit_root, device
    )
    arm_b = load_residual_with_base(
        residual_checkpoint=args.arm_b_final,
        base_kind="sota_v2_mf",
        base_checkpoint=args.strong_backbone,
        kit_root=args.kit_root,
        device=device,
    )

    predictors = [
        ("colleague_backbone", _plain_predictor(colleague_base)),
        ("current80", _residual_predictor(current80)),
        ("strong_backbone", _plain_predictor(strong_base)),
        ("arm_b", _residual_predictor(arm_b)),
    ]
    trajectories = [ref.path.name for ref in dev_dataset.refs]
    starts = [int(ref.start) for ref in dev_dataset.refs]
    summaries = []
    target_reference: np.ndarray | None = None
    for label, predictor in predictors:
        pred, target, base_pred = collect_predictions(predictor, dev_loader, device)
        if target_reference is None:
            target_reference = target
        elif not np.array_equal(target_reference, target):
            raise RuntimeError("fingerprint models did not see identical target windows")
        summaries.append(
            evaluate_prediction_bundle(
                label=label,
                pred=pred,
                target=target,
                trajectories=trajectories,
                starts=starts,
                out_dir=out / label,
                base_pred=base_pred,
            )
        )

    write_csv(out / "fingerprint_summary.csv", summaries)
    by_label = {row["label"]: row for row in summaries}
    result = {
        "status": "COMPLETE",
        "windows": len(dev_dataset),
        "checkpoint_sha256": sha_manifest,
        "strong_vs_colleague_tail_amplification_ratio": (
            float(by_label["strong_backbone"]["f20_over_f18"])
            / max(float(by_label["colleague_backbone"]["f20_over_f18"]), 1e-12)
        ),
        "arm_b_vs_current80_tail_amplification_ratio": (
            float(by_label["arm_b"]["f20_over_f18"])
            / max(float(by_label["current80"]["f20_over_f18"]), 1e-12)
        ),
        "models": by_label,
    }
    dump(out / "fingerprint_result.json", result)
    del colleague_base, current80, strong_base, arm_b
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_trainable_strong_backbone(
    checkpoint: Path,
    kit_root: Path,
    device: torch.device,
):
    payload, config, builder, model = sota_v3._load_backbone(checkpoint, kit_root, device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return payload, config, builder, model


@torch.no_grad()
def evaluate_backbone(
    *,
    model: torch.nn.Module,
    builder,
    loader: DataLoader,
    device: torch.device,
    dataset: H5WindowDataset,
    label: str,
    out_dir: Path,
) -> dict[str, object]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred = sota.forward_mf(model, builder, x)
        predictions.append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    pred = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    trajectories = [ref.path.name for ref in dataset.refs]
    starts = [int(ref.start) for ref in dataset.refs]
    return evaluate_prediction_bundle(
        label=label,
        pred=pred,
        target=target,
        trajectories=trajectories,
        starts=starts,
        out_dir=out_dir,
    )


def train_continuation_arm(
    *,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    train_paths: list[Path],
    dev_dataset: H5WindowDataset,
    dev_loader: DataLoader,
) -> dict[str, object]:
    if mode not in {"control", "ramp"}:
        raise ValueError(mode)
    arm_dir = args.out_root / "experiment1_backbone_continuation" / mode
    arm_dir.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    payload, config, builder, model = load_trainable_strong_backbone(
        args.strong_backbone,
        args.kit_root,
        device,
    )
    if int(payload.get("iteration", -1)) != 53582:
        raise ValueError(f"expected strong backbone iteration 53582, got {payload.get('iteration')}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    if "optimizer_state_dict" not in payload:
        raise KeyError("strong backbone checkpoint lacks optimizer_state_dict")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for group in optimizer.param_groups:
        group["lr"] = LR

    loader_args = argparse.Namespace(
        seed=SEED,
        micro_batch=MICRO_BATCH,
        accumulation_steps=ACCUMULATION_STEPS,
        workers=args.workers,
    )
    train_ds, sampler, train_loader = full.dense_loader(
        train_paths, loader_args, sota, torch
    )
    if len(train_ds) != full.FULL_DENSE_WINDOWS:
        raise ValueError(
            f"expected {full.FULL_DENSE_WINDOWS} dense windows, got {len(train_ds)}"
        )
    consumed_samples = UPDATES * EFFECTIVE_BATCH
    selected = list(getattr(sampler, "_selected_indices"))
    if consumed_samples >= len(selected):
        raise RuntimeError(
            "5k screen unexpectedly crosses a Dense-All epoch boundary; "
            "sampling parity contract must be reviewed"
        )
    sampler_prefix = np.asarray(selected[:consumed_samples], dtype=np.int64)
    sampler_prefix_sha256 = hashlib.sha256(sampler_prefix.tobytes()).hexdigest()

    run_config = {
        "mode": mode,
        "scientific_variable": (
            "none; exact historical Stage-B objective"
            if mode == "control"
            else "replace uniform N2 velocity MSE with mean-1 normalized linear horizon ramp"
        ),
        "strong_backbone_sha256": STRONG_BACKBONE_SHA,
        "start_iteration": 53582,
        "updates": UPDATES,
        "eval_interval": EVAL_INTERVAL,
        "eval_steps": list(EVAL_STEPS),
        "seed": SEED,
        "train_trajectories": len(train_paths),
        "dense_windows": len(train_ds),
        "micro_batch": MICRO_BATCH,
        "accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch": EFFECTIVE_BATCH,
        "optimizer": "AdamW restored from checkpoint",
        "lr": LR,
        "weight_decay": optimizer.param_groups[0].get("weight_decay"),
        "sampler": "DenseAllWindowSampler reset identically at continuation epoch 0",
        "samples_consumed": consumed_samples,
        "sampler_prefix_sha256": sampler_prefix_sha256,
        "base_loss": {
            "n2": full.N2,
            "lambda_vort": full.LAMBDA_VORT,
            "extra_rel": full.EXTRA_REL,
        },
        "horizon_ramp": {
            "enabled": mode == "ramp",
            "start": RAMP_START,
            "end": RAMP_END,
            "normalization": "divide by mean; mean weight == 1",
            "applied_to": "velocity MSE term only",
        },
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "uncertainty_head_trained": False,
        "residual_trained": False,
    }
    dump(arm_dir / "run_config.json", run_config)

    progress: list[dict[str, object]] = []
    initial = evaluate_backbone(
        model=model,
        builder=builder,
        loader=dev_loader,
        device=device,
        dataset=dev_dataset,
        label=f"{mode}_step_00000",
        out_dir=arm_dir / "eval_step_00000",
    )
    progress.append({"step": 0, **initial})
    write_csv(arm_dir / "progress.csv", progress)

    iterator = iter(train_loader)
    epoch = 0
    started = time.monotonic()
    accum: dict[str, float] = {}
    accum_count = 0
    checkpoints = arm_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    for step in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        step_parts: dict[str, float] = {}
        for _ in range(ACCUMULATION_STEPS):
            try:
                x, y, _, _ = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                x, y, _, _ = next(iterator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            model.train()
            pred = sota.forward_mf(model, builder, x)
            loss, parts = continuation_loss(pred, y, mode=mode)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{mode}: nonfinite loss @ step {step}")
            (loss / ACCUMULATION_STEPS).backward()
            for name, value in parts.items():
                step_parts[name] = step_parts.get(name, 0.0) + (
                    float(value.detach()) / ACCUMULATION_STEPS
                )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_parts["loss"] = float(loss.detach())
        for name, value in step_parts.items():
            accum[name] = accum.get(name, 0.0) + value
        accum_count += 1

        if step % 100 == 0:
            row = {
                "step": step,
                "mode": mode,
                "lr": optimizer.param_groups[0]["lr"],
                **{name: value / max(accum_count, 1) for name, value in accum.items()},
            }
            with (arm_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            accum = {}
            accum_count = 0

        if step % EVAL_INTERVAL == 0:
            evaluation = evaluate_backbone(
                model=model,
                builder=builder,
                loader=dev_loader,
                device=device,
                dataset=dev_dataset,
                label=f"{mode}_step_{step:05d}",
                out_dir=arm_dir / f"eval_step_{step:05d}",
            )
            progress.append({"step": step, **evaluation})
            write_csv(arm_dir / "progress.csv", progress)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": 53582 + step,
                    "continuation_step": step,
                    "feature_set": "P0-A",
                    "feature_config": vars(config),
                    "loss_weights": full.N2,
                    "lambda_vort": full.LAMBDA_VORT,
                    "extra_rel_stage_b": full.EXTRA_REL,
                    "late_horizon_mode": mode,
                    "horizon_ramp": run_config["horizon_ramp"],
                    "source_checkpoint_sha256": STRONG_BACKBONE_SHA,
                },
                checkpoints / f"model_step_{step:05d}.pth",
            )

    result = {
        "status": "COMPLETE",
        "mode": mode,
        "wall_seconds": time.monotonic() - started,
        "dense_epochs": UPDATES * EFFECTIVE_BATCH / len(train_ds),
        "initial": progress[0],
        "final": progress[-1],
        "progress": progress,
    }
    dump(arm_dir / "result.json", result)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def compare_continuations(
    control: dict[str, object],
    ramp: dict[str, object],
    out_dir: Path,
) -> dict[str, object]:
    control_progress = {int(row["step"]): row for row in control["progress"]}
    ramp_progress = {int(row["step"]): row for row in ramp["progress"]}
    if set(control_progress) != set(ramp_progress):
        raise RuntimeError("Control/Ramp evaluation steps differ")
    for metric in (
        "rel_l2_raw",
        "tke_raw",
        "mvpe_raw",
        "f18_rel",
        "f19_rel",
        "f20_rel",
        "f20_over_mean_f1_f17",
        "f20_over_f18",
    ):
        if not np.isclose(
            float(control_progress[0][metric]),
            float(ramp_progress[0][metric]),
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(f"Control/Ramp initialization parity failed for {metric}")
    rows = []
    for step in sorted(control_progress):
        c = control_progress[step]
        r = ramp_progress[step]
        row: dict[str, object] = {"step": step}
        for metric in (
            "rel_l2_raw",
            "tke_raw",
            "mvpe_raw",
            "f18_rel",
            "f19_rel",
            "f20_rel",
            "f20_over_mean_f1_f17",
            "f20_over_f18",
        ):
            cv = float(c[metric])
            rv = float(r[metric])
            row[f"control_{metric}"] = cv
            row[f"ramp_{metric}"] = rv
            row[f"ramp_vs_control_{metric}_pct"] = 100.0 * (rv / max(cv, 1e-12) - 1.0)
        rows.append(row)
    write_csv(out_dir / "matched_comparison.csv", rows)
    final = rows[-1]
    result = {
        "primary_comparison": "Ramp@5000 vs Control@5000",
        "negative_percentage_is_error_improvement": True,
        "final_step": final,
        "review_questions": [
            "Does Ramp selectively reduce F19/F20 and tail amplification?",
            "Are aggregate Rel-L2 and MVPE protected?",
            "Does TKE remain stable rather than trading amplitude for point accuracy?",
            "Is the effect already visible by 1k-3k and sustained through 5k?",
        ],
        "automatic_go_no_go": False,
    }
    dump(out_dir / "comparison_result.json", result)
    return result


def run(args: argparse.Namespace) -> None:
    if args.out_root.exists():
        raise FileExistsError(f"out_root exists: {args.out_root}")
    args.out_root.mkdir(parents=True)

    verify_sha(args.strong_backbone, STRONG_BACKBONE_SHA, "strong backbone")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    train_paths = full.released_paths(args.real_root)
    dev_dataset, dev_loader = make_dev_loader(
        real_root=args.real_root,
        split_manifest=args.split_manifest,
        batch_size=args.eval_batch_size,
        workers=args.workers,
    )
    campaign_config = {
        "experiment0": "four-model same-window horizon fingerprint replay",
        "experiment1": "matched Control vs normalized 1->2 late-horizon MSE ramp",
        "strong_backbone_sha256": STRONG_BACKBONE_SHA,
        "colleague_base_sha256": COLLEAGUE_BASE_SHA,
        "current80_residual_sha256": CURRENT80_RESIDUAL_SHA,
        "arm_b_final_sha256": ARM_B_FINAL_SHA,
        "train_trajectories": len(train_paths),
        "dev_windows": len(dev_dataset),
        "dev_overlap_note": (
            "Dev16 follows colleague campaign protocol and overlaps all-released training; "
            "this campaign is a mechanism screen, not a clean generalization estimate."
        ),
        "locked_final_accessed": False,
        "codabench_accessed": False,
    }
    dump(args.out_root / "campaign_config.json", campaign_config)

    fingerprint = run_fingerprint(
        args,
        device=device,
        dev_dataset=dev_dataset,
        dev_loader=dev_loader,
    )
    continuation_root = args.out_root / "experiment1_backbone_continuation"
    continuation_root.mkdir(parents=True, exist_ok=False)
    control = train_continuation_arm(
        mode="control",
        args=args,
        device=device,
        train_paths=train_paths,
        dev_dataset=dev_dataset,
        dev_loader=dev_loader,
    )
    ramp = train_continuation_arm(
        mode="ramp",
        args=args,
        device=device,
        train_paths=train_paths,
        dev_dataset=dev_dataset,
        dev_loader=dev_loader,
    )
    comparison = compare_continuations(control, ramp, continuation_root)
    dump(
        args.out_root / "campaign_result.json",
        {
            "status": "REVIEW_REQUIRED",
            "fingerprint": fingerprint,
            "continuation_comparison": comparison,
            "automatic_long_followup_started": False,
            "codabench_accessed": False,
            "locked_final_accessed": False,
        },
    )
    (args.out_root / "DONE").touch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--colleague-base", type=Path, required=True)
    parser.add_argument("--current80-residual", type=Path, required=True)
    parser.add_argument("--strong-backbone", type=Path, required=True)
    parser.add_argument("--arm-b-final", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--require-cuda", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
