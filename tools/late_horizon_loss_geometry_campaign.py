#!/usr/bin/env python3
"""Loss-geometry diagnosis and one-shot repair screen for the RealPDE tail cliff.

Scientific question:
Does the strong SOTA-V2 backbone retain a Future19/Future20 cliff because its
late-stage optimization is dominated by absolute velocity MSE geometry, and
can an E1-style MSE-to-Rel-L2 replacement repair that cliff without changing
the model, data, optimizer state, sampling, TKE/MVPE/vorticity terms, residual,
or uncertainty stack?

The script deliberately has two separate modes.

1. diagnose: no optimizer step. On a deterministic, trajectory-diverse subset
   of colleague Dev16 windows, compare parameter-gradient alignment of the
   historical Stage-B objective and the Rel-dominant candidate objective with
   a tail-only F19/F20 Rel-L2 objective.

2. train: exactly one 5k candidate continuation from the audited strong
   backbone @53582. The only scientific change is replacing the historical
   N2 velocity-MSE term (weight 1.0) with one additional full-Future20 Rel-L2
   term (weight 1.0). Existing TKE, small Rel, MVPE, vorticity, extra Stage-B
   Rel, optimizer state, LR, seed, Dense-All sampling, effective batch, and
   evaluation protocol are preserved.

There is no automatic transition from diagnose to train. Human/Sol review is
required between the two modes. No residual training, uncertainty training,
locked-final/private access, submission packaging, or Codabench access occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import late_horizon_backbone_campaign as late  # noqa: E402
import realpde_loss_official_v9 as core  # noqa: E402
import realpde_sota_v2_full as full  # noqa: E402
import realpde_sota_v2_integrated as sota  # noqa: E402

SEED = 20260901
STRONG_BACKBONE_SHA = late.STRONG_BACKBONE_SHA
START_ITERATION = 53_582
UPDATES = 5_000
EVAL_INTERVAL = 1_000
EVAL_STEPS = (0, 1_000, 2_000, 3_000, 4_000, 5_000)
MICRO_BATCH = 4
ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH = MICRO_BATCH * ACCUMULATION_STEPS
LR = 3e-6
PROBE_TRAJECTORIES = 8
PROBE_BATCH_SIZE = 2

METRICS = (
    "rel_l2_raw",
    "tke_raw",
    "mvpe_raw",
    "f18_rel",
    "f19_rel",
    "f20_rel",
    "f20_over_mean_f1_f17",
    "f20_over_f18",
)


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


def sha256(path: Path) -> str:
    return late.sha256(path)


def rel_dominant_stage_b_loss(
    pred: Tensor,
    target: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Replace only the historical unit-weight velocity MSE with unit Rel-L2."""
    control, raw_parts = late.historical_stage_b_loss(pred, target)
    if abs(float(full.N2["mse"]) - 1.0) > 1e-12:
        raise RuntimeError("candidate assumes historical N2 mse weight == 1.0")

    candidate = control - raw_parts["mse"] + raw_parts["rel"]
    parts = dict(raw_parts)
    parts["historical_total"] = control
    parts["replacement_rel"] = raw_parts["rel"]
    parts["rel_dominant_delta"] = raw_parts["rel"] - raw_parts["mse"]
    parts["candidate_total"] = candidate
    return candidate, parts


def tail_rel_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Tail-only diagnostic objective over Future19/Future20 velocity."""
    if pred.shape != target.shape or pred.ndim != 5 or pred.shape[1] != 20:
        raise ValueError("expected matched [B,20,H,W,C]")
    return core.rel_loss(pred[:, 18:20, ..., :2], target[:, 18:20, ..., :2])


def first18_rel_loss(pred: Tensor, target: Tensor) -> Tensor:
    if pred.shape != target.shape or pred.ndim != 5 or pred.shape[1] != 20:
        raise ValueError("expected matched [B,20,H,W,C]")
    return core.rel_loss(pred[:, :18, ..., :2], target[:, :18, ..., :2])


def _grads(
    loss: Tensor,
    params: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[Tensor | None, ...]:
    return torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )


def gradient_norm(grads: Iterable[Tensor | None]) -> float:
    value = None
    for grad in grads:
        if grad is None:
            continue
        term = grad.detach().double().square().sum()
        value = term if value is None else value + term
    return float(torch.sqrt(value).cpu()) if value is not None else 0.0


def gradient_cosine(
    left: Iterable[Tensor | None],
    right: Iterable[Tensor | None],
) -> float:
    dot = None
    ln = None
    rn = None
    for a, b in zip(left, right):
        if a is None or b is None:
            continue
        aa = a.detach().double()
        bb = b.detach().double()
        d = (aa * bb).sum()
        la = aa.square().sum()
        rb = bb.square().sum()
        dot = d if dot is None else dot + d
        ln = la if ln is None else ln + la
        rn = rb if rn is None else rn + rb
    if dot is None or ln is None or rn is None:
        return float("nan")
    denom = torch.sqrt(ln * rn).clamp_min(1e-30)
    return float((dot / denom).cpu())


def select_probe_indices(refs: Sequence[object], count: int) -> list[int]:
    """Pick the first window from distinct trajectories, deterministically."""
    selected: list[int] = []
    seen: set[str] = set()
    for index, ref in enumerate(refs):
        path = getattr(ref, "path", None)
        name = Path(path).name if path is not None else str(path)
        if name in seen:
            continue
        seen.add(name)
        selected.append(index)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"need {count} distinct trajectories, found {len(selected)}")
    return selected


def diagnostic_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("empty diagnostic rows")

    def mean(key: str) -> float:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        return float(values.mean())

    control_tail = mean("control_tail_cosine")
    candidate_tail = mean("candidate_tail_cosine")
    mse_tail = mean("mse_tail_cosine")
    rel_tail = mean("rel_tail_cosine")
    candidate_wins = sum(
        float(row["candidate_tail_cosine"]) > float(row["control_tail_cosine"])
        for row in rows
    )
    rel_wins = sum(
        float(row["rel_tail_cosine"]) > float(row["mse_tail_cosine"])
        for row in rows
    )
    return {
        "status": "REVIEW_REQUIRED",
        "batches": len(rows),
        "control_tail_cosine_mean": control_tail,
        "candidate_tail_cosine_mean": candidate_tail,
        "candidate_minus_control_tail_cosine": candidate_tail - control_tail,
        "mse_tail_cosine_mean": mse_tail,
        "rel_tail_cosine_mean": rel_tail,
        "rel_minus_mse_tail_cosine": rel_tail - mse_tail,
        "candidate_tail_alignment_wins": candidate_wins,
        "rel_tail_alignment_wins": rel_wins,
        "automatic_train_started": False,
        "interpretation_rule": (
            "Support requires the Rel-dominant candidate to align more strongly "
            "with the F19/F20 Rel objective than the historical control on a "
            "consistent majority of probe batches. No automatic threshold "
            "launches training."
        ),
    }


def run_diagnosis(args: argparse.Namespace) -> dict[str, object]:
    out = args.out_root / "diagnosis"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    late.verify_sha(args.strong_backbone, STRONG_BACKBONE_SHA, "strong backbone")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    dev_dataset, _ = late.make_dev_loader(
        real_root=args.real_root,
        split_manifest=args.split_manifest,
        batch_size=args.eval_batch_size,
        workers=args.workers,
    )
    probe_indices = select_probe_indices(dev_dataset.refs, PROBE_TRAJECTORIES)
    probe_loader = DataLoader(
        Subset(dev_dataset, probe_indices),
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    _, _, builder, model = late.load_trainable_strong_backbone(
        args.strong_backbone,
        args.kit_root,
        device,
    )
    model.eval()
    params = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)

    probe_manifest = []
    for index in probe_indices:
        ref = dev_dataset.refs[index]
        probe_manifest.append(
            {
                "dataset_index": index,
                "trajectory": ref.path.name,
                "start": int(ref.start),
            }
        )
    write_csv(out / "probe_windows.csv", probe_manifest)

    rows: list[dict[str, object]] = []
    for batch_id, (x, y) in enumerate(probe_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        pred = sota.forward_mf(model, builder, x)

        control, control_parts = late.historical_stage_b_loss(pred, y)
        candidate, _ = rel_dominant_stage_b_loss(pred, y)
        tail = tail_rel_loss(pred, y)
        first18 = first18_rel_loss(pred, y)

        g_tail = _grads(tail, params, retain_graph=True)
        g_control = _grads(control, params, retain_graph=True)
        g_candidate = _grads(candidate, params, retain_graph=True)
        g_mse = _grads(control_parts["mse"], params, retain_graph=True)
        g_rel = _grads(control_parts["rel"], params, retain_graph=False)

        rows.append(
            {
                "batch_id": batch_id,
                "batch_size": int(x.shape[0]),
                "historical_total": float(control.detach().cpu()),
                "candidate_total": float(candidate.detach().cpu()),
                "mse": float(control_parts["mse"].detach().cpu()),
                "rel": float(control_parts["rel"].detach().cpu()),
                "tail_rel_f19_f20": float(tail.detach().cpu()),
                "first18_rel": float(first18.detach().cpu()),
                "tail_over_first18_rel": float(tail.detach().cpu())
                / max(float(first18.detach().cpu()), 1e-12),
                "tail_grad_norm": gradient_norm(g_tail),
                "control_grad_norm": gradient_norm(g_control),
                "candidate_grad_norm": gradient_norm(g_candidate),
                "mse_grad_norm": gradient_norm(g_mse),
                "rel_grad_norm": gradient_norm(g_rel),
                "control_tail_cosine": gradient_cosine(g_control, g_tail),
                "candidate_tail_cosine": gradient_cosine(g_candidate, g_tail),
                "mse_tail_cosine": gradient_cosine(g_mse, g_tail),
                "rel_tail_cosine": gradient_cosine(g_rel, g_tail),
            }
        )
        del pred, g_tail, g_control, g_candidate, g_mse, g_rel

    write_csv(out / "gradient_alignment.csv", rows)
    summary = diagnostic_summary(rows)
    summary.update(
        {
            "strong_backbone_sha256": STRONG_BACKBONE_SHA,
            "start_iteration": START_ITERATION,
            "probe_trajectories": PROBE_TRAJECTORIES,
            "probe_batch_size": PROBE_BATCH_SIZE,
            "scientific_variable": (
                "diagnostic only: compare historical Stage-B vs exact "
                "MSE-to-Rel-L2 replacement gradient alignment to F19/F20 Rel-L2"
            ),
            "locked_final_accessed": False,
            "codabench_accessed": False,
            "optimizer_step_performed": False,
        }
    )
    dump(out / "diagnosis_result.json", summary)
    (out / "DIAGNOSIS_DONE").touch()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def load_frozen_control(
    result_path: Path,
    config_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETE" or result.get("mode") != "control":
        raise ValueError("frozen control result is not completed control arm")
    if config.get("mode") != "control":
        raise ValueError("frozen control config mode mismatch")
    if config.get("strong_backbone_sha256") != STRONG_BACKBONE_SHA:
        raise ValueError("frozen control strong-backbone SHA mismatch")
    if int(config.get("updates", -1)) != UPDATES:
        raise ValueError("frozen control update budget mismatch")
    if int(config.get("effective_batch", -1)) != EFFECTIVE_BATCH:
        raise ValueError("frozen control effective batch mismatch")
    if float(config.get("lr", -1)) != LR:
        raise ValueError("frozen control LR mismatch")
    return result, config


def assert_step0_parity(
    candidate: dict[str, object],
    control: dict[str, object],
    *,
    atol: float = 1e-10,
) -> None:
    control0 = {int(row["step"]): row for row in control["progress"]}[0]
    for metric in METRICS:
        cv = float(control0[metric])
        rv = float(candidate[metric])
        if not np.isclose(cv, rv, rtol=0.0, atol=atol):
            raise RuntimeError(
                f"candidate/frozen-control step0 parity failed for {metric}: "
                f"{rv} != {cv}"
            )


def compare_candidate_to_control(
    control: dict[str, object],
    candidate_progress: list[dict[str, object]],
    out_dir: Path,
) -> dict[str, object]:
    control_progress = {int(row["step"]): row for row in control["progress"]}
    candidate = {int(row["step"]): row for row in candidate_progress}
    if set(control_progress) != set(candidate):
        raise RuntimeError("candidate/frozen-control evaluation steps differ")

    rows: list[dict[str, object]] = []
    for step in sorted(control_progress):
        c = control_progress[step]
        r = candidate[step]
        row: dict[str, object] = {"step": step}
        for metric in METRICS:
            cv = float(c[metric])
            rv = float(r[metric])
            row[f"control_{metric}"] = cv
            row[f"candidate_{metric}"] = rv
            row[f"candidate_vs_control_{metric}_pct"] = (
                100.0 * (rv / max(cv, 1e-12) - 1.0)
            )
        rows.append(row)

    write_csv(out_dir / "matched_comparison.csv", rows)
    final = rows[-1]
    result = {
        "status": "REVIEW_REQUIRED",
        "primary_comparison": "Rel-dominant@5000 vs frozen historical Control@5000",
        "negative_percentage_is_error_improvement": True,
        "final_step": final,
        "pre_registered_review_targets": {
            "tail": "F19/F20 and F20/F18 should improve materially, not merely reshuffle error",
            "aggregate_rel": "must not worsen versus frozen Control",
            "tke_mvpe": "should remain protected; no >2% regression is considered clean",
            "trajectory": "review by-trajectory diagnostics before any production merge",
        },
        "automatic_go_no_go": False,
        "automatic_long_followup_started": False,
    }
    dump(out_dir / "comparison_result.json", result)
    return result


def run_candidate(args: argparse.Namespace) -> dict[str, object]:
    out = args.out_root / "candidate_rel_dominant"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    late.verify_sha(args.strong_backbone, STRONG_BACKBONE_SHA, "strong backbone")
    control, control_config = load_frozen_control(
        args.frozen_control_result,
        args.frozen_control_config,
    )

    diagnosis_path = args.out_root / "diagnosis" / "diagnosis_result.json"
    if not diagnosis_path.is_file():
        raise FileNotFoundError(
            "diagnosis_result.json missing; REVIEW_REQUIRED diagnosis must be "
            "completed and reviewed first"
        )
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if diagnosis.get("status") != "REVIEW_REQUIRED":
        raise ValueError("unexpected diagnosis status")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    train_paths = full.released_paths(args.real_root)
    dev_dataset, dev_loader = late.make_dev_loader(
        real_root=args.real_root,
        split_manifest=args.split_manifest,
        batch_size=args.eval_batch_size,
        workers=args.workers,
    )
    payload, config, builder, model = late.load_trainable_strong_backbone(
        args.strong_backbone,
        args.kit_root,
        device,
    )
    if int(payload.get("iteration", -1)) != START_ITERATION:
        raise ValueError(
            f"expected strong backbone iteration {START_ITERATION}, "
            f"got {payload.get('iteration')}"
        )
    if "optimizer_state_dict" not in payload:
        raise KeyError("strong backbone checkpoint lacks optimizer_state_dict")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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
        raise RuntimeError("candidate unexpectedly crosses a Dense-All epoch boundary")
    sampler_prefix = np.asarray(selected[:consumed_samples], dtype=np.int64)
    sampler_prefix_sha = hashlib.sha256(sampler_prefix.tobytes()).hexdigest()
    if sampler_prefix_sha != control_config.get("sampler_prefix_sha256"):
        raise RuntimeError(
            "candidate sampler prefix differs from frozen historical Control"
        )

    run_config = {
        "mode": "rel_dominant",
        "scientific_variable": (
            "replace only the historical unit-weight N2 velocity MSE term "
            "with one unit-weight full-Future20 Rel-L2 term"
        ),
        "strong_backbone_sha256": STRONG_BACKBONE_SHA,
        "start_iteration": START_ITERATION,
        "updates": UPDATES,
        "eval_interval": EVAL_INTERVAL,
        "eval_steps": list(EVAL_STEPS),
        "seed": SEED,
        "train_trajectories": len(train_paths),
        "dense_windows": len(train_ds),
        "micro_batch": MICRO_BATCH,
        "accumulation_steps": ACCUMULATION_STEPS,
        "effective_batch": EFFECTIVE_BATCH,
        "optimizer": "AdamW restored from strong-backbone checkpoint",
        "lr": LR,
        "weight_decay": optimizer.param_groups[0].get("weight_decay"),
        "sampler": "DenseAllWindowSampler reset identically at continuation epoch 0",
        "samples_consumed": consumed_samples,
        "sampler_prefix_sha256": sampler_prefix_sha,
        "frozen_control_sampler_prefix_sha256": control_config.get(
            "sampler_prefix_sha256"
        ),
        "historical_base_loss": {
            "n2": full.N2,
            "lambda_vort": full.LAMBDA_VORT,
            "extra_rel": full.EXTRA_REL,
        },
        "candidate_change": {
            "remove": "1.0 * velocity MSE",
            "add": "1.0 * full-Future20 velocity Rel-L2",
            "unchanged": [
                "TKE",
                "existing small Rel",
                "MVPE",
                "vorticity",
                "Stage-B extra Rel",
                "model/feature architecture",
                "optimizer state",
                "LR",
                "sampling order",
                "effective batch",
            ],
        },
        "diagnosis_result_sha256": sha256(diagnosis_path),
        "locked_final_accessed": False,
        "codabench_accessed": False,
        "residual_trained": False,
        "uncertainty_head_trained": False,
        "automatic_followup": False,
    }
    dump(out / "run_config.json", run_config)

    progress: list[dict[str, object]] = []
    initial = late.evaluate_backbone(
        model=model,
        builder=builder,
        loader=dev_loader,
        device=device,
        dataset=dev_dataset,
        label="rel_dominant_step_00000",
        out_dir=out / "eval_step_00000",
    )
    initial_row = {"step": 0, **initial}
    assert_step0_parity(initial_row, control)
    progress.append(initial_row)
    write_csv(out / "progress.csv", progress)

    iterator = iter(train_loader)
    epoch = 0
    started = time.monotonic()
    accum: dict[str, float] = {}
    accum_count = 0
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    for step in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
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
            loss, parts = rel_dominant_stage_b_loss(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite candidate loss @ step {step}")
            (loss / ACCUMULATION_STEPS).backward()
            last_loss = loss
            for name, value in parts.items():
                accum[name] = accum.get(name, 0.0) + (
                    float(value.detach()) / ACCUMULATION_STEPS
                )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        accum["loss"] = accum.get("loss", 0.0) + float(last_loss.detach())
        accum_count += 1

        if step % 100 == 0:
            row = {
                "step": step,
                "mode": "rel_dominant",
                "lr": optimizer.param_groups[0]["lr"],
                **{
                    name: value / max(accum_count, 1)
                    for name, value in accum.items()
                },
            }
            with (out / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            accum = {}
            accum_count = 0

        if step % EVAL_INTERVAL == 0:
            evaluation = late.evaluate_backbone(
                model=model,
                builder=builder,
                loader=dev_loader,
                device=device,
                dataset=dev_dataset,
                label=f"rel_dominant_step_{step:05d}",
                out_dir=out / f"eval_step_{step:05d}",
            )
            progress.append({"step": step, **evaluation})
            write_csv(out / "progress.csv", progress)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iteration": START_ITERATION + step,
                    "continuation_step": step,
                    "feature_set": "P0-A",
                    "feature_config": vars(config),
                    "loss_weights": full.N2,
                    "lambda_vort": full.LAMBDA_VORT,
                    "extra_rel_stage_b": full.EXTRA_REL,
                    "loss_geometry_mode": "rel_dominant",
                    "source_checkpoint_sha256": STRONG_BACKBONE_SHA,
                    "scientific_variable": run_config["scientific_variable"],
                },
                checkpoints / f"model_step_{step:05d}.pth",
            )

    comparison = compare_candidate_to_control(control, progress, out)
    result = {
        "status": "REVIEW_REQUIRED",
        "mode": "rel_dominant",
        "wall_seconds": time.monotonic() - started,
        "dense_epochs": UPDATES * EFFECTIVE_BATCH / len(train_ds),
        "initial": progress[0],
        "final": progress[-1],
        "progress": progress,
        "comparison": comparison,
        "automatic_long_followup_started": False,
        "residual_training_started": False,
        "uncertainty_training_started": False,
        "codabench_accessed": False,
        "locked_final_accessed": False,
    }
    dump(out / "result.json", result)
    (out / "TRAIN_DONE").touch()

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--strong-backbone", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--require-cuda", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    diagnose = sub.add_parser("diagnose")
    add_common_args(diagnose)

    train = sub.add_parser("train")
    add_common_args(train)
    train.add_argument("--frozen-control-result", type=Path, required=True)
    train.add_argument("--frozen-control-config", type=Path, required=True)

    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "diagnose":
        run_diagnosis(args)
    elif args.mode == "train":
        run_candidate(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
