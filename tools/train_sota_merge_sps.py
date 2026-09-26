#!/usr/bin/env python3
"""Matched SPS optimization for frozen REALPDE_SOTA_MERGE_JOINT_V1 @6k.

This runner is intentionally narrow:
- point predictor is the already-selected Joint@6k backbone + corrector pair;
- point predictor parameters are frozen and never enter an optimizer;
- uncertainty training reuses the validated Clean Exp3 recipe exactly;
- Seen-Dev selects the uncertainty checkpoint/calibration;
- AoA10 holdout is touched only once, after the Seen-Dev gate is GO;
- no locked-final/private/Codabench/submission path exists here.

Scientific SPS recipe preserved from Clean Exp3:
  h64 / 2 blocks / dropout 0 / Past20 + pre-residual base features
  target = post-residual final absolute error
  Train51 stride 5 / Seen-Dev12 stride 20
  5000 uncertainty-head updates / batch 16 / AdamW 1e-3
  warmup + cosine / log-MAE objective
  frozen static grid + frozen 300-combination adaptive grid
  select max Seen-Dev SPS subject to mean-width <= 1.20x static width
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import realpde_sota_merge_joint_runtime as joint
import realpde_sota_v2_integrated as strong
import train_clean_residual_aware_sps as exp3
from colleague_80pt.train_head_fast import Head3D, HeadConfig, initialize_coupled
from realpde_adaptive_probe import ResidualCorrector3D, feature_config_from_checkpoint
from realpde_mf01 import MF01CNO
from realpde_p0_data import H5WindowDataset
from realpde_p0_features import P0FeatureBuilder

STATUS = "REVIEW_REQUIRED"
PROTOCOL = "REALPDE_SOTA_MERGE_SPS_V1"

# Final point predictor chosen by Sol review of Joint V1.
TARGET_JOINT_UPDATE = 6_000
EXPECTED_BACKBONE_SHA256 = "ca9d3efbcbe10a5b0190875c6bc749386002678e33b86d7ecda0fba55c306d86"
EXPECTED_CORRECTOR_SHA256 = "a98b4eb048e193a6337d267fe54ead8645d77ddcfaad18321f0f6c07c76ebec8"

# Frozen Clean Exp3 recipe.
SEED = 41
UPDATES = 5_000
EVAL_EVERY = 500
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP = 200
EPS = 1e-6


# Calibration is pure post-processing over frozen NumPy arrays. The original
# Exp3 implementation evaluates the fixed calibration grid serially; on the
# 16-core execution host that leaves most CPU cores idle. Keep the exact Exp3
# score functions and frozen grids, but distribute independent grid points
# across spawned CPU processes using shared memory.
_CALIBRATION_ARRAYS: dict[str, np.ndarray] = {}
_CALIBRATION_SHMS: list[shared_memory.SharedMemory] = []


def _init_calibration_worker(
    specs: list[tuple[str, str, tuple[int, ...], str]],
) -> None:
    global _CALIBRATION_ARRAYS, _CALIBRATION_SHMS
    _CALIBRATION_ARRAYS = {}
    _CALIBRATION_SHMS = []
    for key, shm_name, shape, dtype_str in specs:
        shm = shared_memory.SharedMemory(name=shm_name)
        array = np.ndarray(
            tuple(shape),
            dtype=np.dtype(dtype_str),
            buffer=shm.buf,
        )
        array.setflags(write=False)
        _CALIBRATION_SHMS.append(shm)
        _CALIBRATION_ARRAYS[key] = array


def _adaptive_calibration_worker(
    params: tuple[float, float, float, float],
) -> dict[str, float]:
    floor, mult_u, mult_v, rel = params
    score = exp3.adaptive_score(
        _CALIBRATION_ARRAYS["pred"],
        _CALIBRATION_ARRAYS["target"],
        _CALIBRATION_ARRAYS["sigma"],
        floor,
        mult_u,
        mult_v,
        rel,
    )
    return {
        "floor": floor,
        "mult_u": mult_u,
        "mult_v": mult_v,
        "rel": rel,
        **score,
    }


def _static_calibration_worker(
    params: tuple[float, float],
) -> dict[str, float]:
    abs_w, rel_w = params
    score = exp3.static_score(
        _CALIBRATION_ARRAYS["pred"],
        _CALIBRATION_ARRAYS["target"],
        abs_w,
        rel_w,
    )
    return {"abs": abs_w, "rel": rel_w, **score}


def _parallel_calibration_rows(
    arrays: dict[str, np.ndarray],
    params: list[tuple],
    worker,
    workers: int,
) -> list[dict[str, float]]:
    """Run independent calibration points in spawned processes.

    Shared memory avoids serializing the large Seen-Dev tensors once per grid
    point. Spawn is deliberate because the parent already initialized CUDA;
    forking a CUDA-initialized process is unsafe.
    """
    if workers < 1:
        raise ValueError("calibration workers must be >= 1")

    shm_handles: list[shared_memory.SharedMemory] = []
    specs: list[tuple[str, str, tuple[int, ...], str]] = []
    try:
        for key, value in arrays.items():
            contiguous = np.ascontiguousarray(value)
            shm = shared_memory.SharedMemory(
                create=True,
                size=contiguous.nbytes,
            )
            shared_view = np.ndarray(
                contiguous.shape,
                dtype=contiguous.dtype,
                buffer=shm.buf,
            )
            shared_view[...] = contiguous
            shm_handles.append(shm)
            specs.append(
                (
                    key,
                    shm.name,
                    tuple(contiguous.shape),
                    contiguous.dtype.str,
                )
            )

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(params)),
            mp_context=ctx,
            initializer=_init_calibration_worker,
            initargs=(specs,),
        ) as pool:
            return list(pool.map(worker, params, chunksize=1))
    finally:
        for shm in shm_handles:
            shm.close()
            shm.unlink()


def calibrate_static_parallel(
    pred: np.ndarray,
    target: np.ndarray,
    workers: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if workers == 1:
        return exp3.calibrate_static(pred, target)
    params = [
        (abs_w, rel_w)
        for abs_w in exp3.STATIC_ABS
        for rel_w in exp3.STATIC_REL
    ]
    rows = _parallel_calibration_rows(
        {"pred": pred, "target": target},
        params,
        _static_calibration_worker,
        workers,
    )
    rows.sort(key=lambda row: -float(row["sps"]))
    return rows, rows[0]


def calibrate_adaptive_parallel(
    pred: np.ndarray,
    target: np.ndarray,
    sigma: np.ndarray,
    workers: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if workers == 1:
        return exp3.calibrate_adaptive(pred, target, sigma)
    params = [
        (floor, mult_u, mult_v, rel)
        for floor in exp3.FLOORS
        for mult_u in exp3.MULTS
        for mult_v in exp3.MULTS
        for rel in exp3.RELS
    ]
    rows = _parallel_calibration_rows(
        {"pred": pred, "target": target, "sigma": sigma},
        params,
        _adaptive_calibration_worker,
        workers,
    )
    rows.sort(key=lambda row: -float(row["sps"]))
    return rows, rows[0]


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def canonical_manifest_paths(manifest: Path, root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Resolve and hard-audit the canonical 51/12/18 split from one manifest."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    def resolve(split: str) -> list[Path]:
        rows = payload.get(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"manifest missing non-empty {split!r}")
        names = [str(row["file"] if isinstance(row, dict) else row) for row in rows]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate filenames in manifest split {split}")
        paths = [root / name for name in names]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing {split} files: {missing[:5]}")
        return paths

    train = resolve("train")
    dev = resolve("dev")
    holdout = resolve("holdout")

    if (len(train), len(dev), len(holdout)) != (51, 12, 18):
        raise ValueError(
            f"expected canonical 51/12/18 split, got {len(train)}/{len(dev)}/{len(holdout)}"
        )

    train_names = {p.name for p in train}
    dev_names = {p.name for p in dev}
    holdout_names = {p.name for p in holdout}
    if train_names & dev_names:
        raise ValueError("Train51 overlaps Seen-Dev12")
    if holdout_names & (train_names | dev_names):
        raise ValueError("AoA10 holdout overlaps Train51/Seen-Dev12")

    # Hard role check: all and only AoA=10 trajectories belong to holdout.
    if any(name.endswith("_10.h5") for name in train_names | dev_names):
        raise ValueError("AoA=10 trajectory leaked into train/dev")
    if any(not name.endswith("_10.h5") for name in holdout_names):
        raise ValueError("holdout contains a non-AoA10 trajectory")

    return train, dev, holdout


def fixed_dataset(paths: list[Path], stride: int) -> H5WindowDataset:
    return H5WindowDataset(
        paths,
        in_steps=20,
        out_steps=20,
        stride=stride,
        sub_sample=2,
        include_pressure=False,
        window_mode="fixed",
    )


def load_joint_pair(
    backbone_path: Path,
    corrector_path: Path,
    kit_root: Path,
    device: torch.device,
):
    """Load exactly the reviewed Joint@6k pair and freeze it."""
    for path in (backbone_path, corrector_path, kit_root):
        joint.merge.assert_safe_path(path)

    backbone_sha = strong.sha256(backbone_path)
    corrector_sha = strong.sha256(corrector_path)
    if backbone_sha != EXPECTED_BACKBONE_SHA256:
        raise RuntimeError(
            f"Joint@6k backbone SHA mismatch: got {backbone_sha}, "
            f"expected {EXPECTED_BACKBONE_SHA256}"
        )
    if corrector_sha != EXPECTED_CORRECTOR_SHA256:
        raise RuntimeError(
            f"Joint@6k corrector SHA mismatch: got {corrector_sha}, "
            f"expected {EXPECTED_CORRECTOR_SHA256}"
        )

    bp = torch.load(backbone_path, map_location="cpu", weights_only=False)
    cp = torch.load(corrector_path, map_location="cpu", weights_only=False)

    if bp.get("protocol") != joint.PROTOCOL or cp.get("protocol") != joint.PROTOCOL:
        raise ValueError("checkpoint pair is not REALPDE_SOTA_MERGE_JOINT_V1")
    if int(bp.get("joint_update", -1)) != TARGET_JOINT_UPDATE:
        raise ValueError(
            f"backbone must be Joint@{TARGET_JOINT_UPDATE}, "
            f"got @{bp.get('joint_update')}"
        )
    if int(cp.get("joint_update", -1)) != TARGET_JOINT_UPDATE:
        raise ValueError(
            f"corrector must be Joint@{TARGET_JOINT_UPDATE}, "
            f"got @{cp.get('joint_update')}"
        )
    if cp.get("backbone_sha256") != backbone_sha:
        raise ValueError("corrector is not bound to the supplied Joint@6k backbone")

    cfg = feature_config_from_checkpoint(bp)
    builder = P0FeatureBuilder(cfg).to(device)
    backbone = MF01CNO(kit_root, len(builder.feature_names), device)
    backbone.load_state_dict(bp["model_state_dict"], strict=True)

    arch = cp.get("architecture", {})
    expected_arch = {
        "in_channels": 42,
        "hidden": 64,
        "blocks": 2,
        "max_delta": 0.04,
    }
    for key, expected in expected_arch.items():
        actual = arch.get(key, expected)
        if actual != expected:
            raise ValueError(
                f"corrector architecture mismatch for {key}: {actual} != {expected}"
            )

    corrector = ResidualCorrector3D(
        in_channels=expected_arch["in_channels"],
        hidden=expected_arch["hidden"],
        blocks=expected_arch["blocks"],
        max_delta=expected_arch["max_delta"],
    ).to(device)
    corrector.load_state_dict(cp["corrector_state_dict"], strict=True)

    backbone.eval()
    corrector.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in corrector.parameters():
        parameter.requires_grad_(False)

    return bp, cp, cfg, builder, backbone, corrector, backbone_sha, corrector_sha


@torch.no_grad()
def point_forward(backbone, builder, corrector, x):
    """Exact frozen Joint V1 point-prediction path."""
    return joint.forward_final(backbone, builder, corrector, x)


@torch.no_grad()
def collect(backbone, builder, corrector, head, paths, device, workers):
    ds = fixed_dataset(paths, 20)
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    preds, sigmas, targets = [], [], []
    backbone.eval()
    corrector.eval()
    head.eval()
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        base, final = point_forward(backbone, builder, corrector, x)
        log_std = head(x, base)
        preds.append(final.cpu().numpy().astype(np.float32))
        sigmas.append(torch.exp(log_std).cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))

    return (
        ds,
        np.concatenate(preds),
        np.concatenate(sigmas),
        np.concatenate(targets),
    )


def save_head_snapshot(
    path: Path,
    *,
    head: Head3D,
    cfg: HeadConfig,
    step: int,
    backbone_sha: str,
    corrector_sha: str,
) -> None:
    torch.save(
        {
            "head_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            },
            "head_config": cfg.__dict__,
            "step": int(step),
            "protocol": PROTOCOL,
            "joint_update": TARGET_JOINT_UPDATE,
            "backbone_sha256": backbone_sha,
            "corrector_sha256": corrector_sha,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="canonical clean manifest containing train/dev/holdout = 51/12/18",
    )
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--corrector-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--calibration-workers",
        type=int,
        default=12,
        help="CPU processes for exact static/adaptive calibration grids",
    )
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

    train_paths, dev_paths, aoa10_paths = canonical_manifest_paths(
        args.manifest, args.data_root
    )

    (
        bp,
        cp,
        _cfg,
        builder,
        backbone,
        corrector,
        backbone_sha,
        corrector_sha,
    ) = load_joint_pair(
        args.backbone_checkpoint,
        args.corrector_checkpoint,
        args.kit_root,
        device,
    )

    # Validated Exp3 semantics: uncertainty observes Past20 + pre-residual base,
    # while supervision is absolute error of the final post-corrector prediction.
    hcfg = HeadConfig(
        hidden=64,
        blocks=2,
        dropout=0.0,
        include_pressure=True,
        history_context=False,
        sigma0=0.02,
        min_sigma=1e-4,
        max_sigma=1.0,
        include_delta=False,
    )
    head = Head3D(hcfg).to(device)
    initialize_coupled(head, seed=SEED)

    # HARD: optimizer owns uncertainty-head parameters only.
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / WARMUP)
        * (0.5 * (1 + np.cos(np.pi * min(1.0, step / UPDATES)))),
    )

    train_ds = fixed_dataset(train_paths, 5)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        generator=generator,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    iterator = iter(loader)

    # Static SPS reference is frozen from Seen-Dev before head training.
    _, pred_before, _, target_before = collect(
        backbone,
        builder,
        corrector,
        head,
        dev_paths,
        device,
        args.workers,
    )
    static_started = time.monotonic()
    print(
        f"[SPS] static calibration start: workers={args.calibration_workers}",
        flush=True,
    )
    static_grid, static_best = calibrate_static_parallel(
        pred_before,
        target_before,
        args.calibration_workers,
    )
    print(
        "[SPS] static calibration done: "
        f"{time.monotonic() - static_started:.1f}s, "
        f"SPS={float(static_best['sps']):.6f}",
        flush=True,
    )
    dump(args.out_dir / "static_calibration_grid.json", static_grid)

    best_record = None
    best_state = None
    evals = []
    losses = []
    started = time.monotonic()

    for step in range(1, UPDATES + 1):
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y, _, _ = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # HARD: point predictor forward is no-grad and frozen.
        with torch.no_grad():
            base, final = point_forward(backbone, builder, corrector, x)

        log_std = head(x, base)
        err = (final[..., :2] - y[..., :2]).abs()
        mask = (y[..., :2] != 0.0).to(log_std.dtype)

        # Exact validated Exp3 masked log-MAE objective.
        loss = (
            (log_std - torch.log(err + EPS)).abs() * mask
        ).sum() / mask.sum().clamp_min(1.0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite SPS loss @{step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not np.isfinite(float(grad_norm)):
            raise FloatingPointError(f"nonfinite SPS head gradient @{step}")
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 100 == 0:
            loss_record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "head_grad_norm_preclip": float(grad_norm),
            }
            losses.append(loss_record)
            if step % 100 == 0:
                print(
                    f"[SPS] train step {step}/{UPDATES}: "
                    f"loss={loss_record['loss']:.6f}",
                    flush=True,
                )

        if step % EVAL_EVERY == 0:
            eval_started = time.monotonic()
            print(
                f"[SPS] milestone {step}: Seen-Dev collect start",
                flush=True,
            )
            _, pred_dev, sigma_dev, target_dev = collect(
                backbone,
                builder,
                corrector,
                head,
                dev_paths,
                device,
                args.workers,
            )
            collect_seconds = time.monotonic() - eval_started
            calibration_started = time.monotonic()
            print(
                f"[SPS] milestone {step}: adaptive calibration start "
                f"(300 combos, workers={args.calibration_workers})",
                flush=True,
            )
            grid, unconstrained = calibrate_adaptive_parallel(
                pred_dev,
                target_dev,
                sigma_dev,
                args.calibration_workers,
            )
            calibration_seconds = time.monotonic() - calibration_started
            chosen = exp3.select_adaptive_under_width_cap(
                grid,
                static_best,
            )
            record = {
                "step": step,
                "best": chosen,
                "unconstrained_best": unconstrained,
                "width_cap": (
                    exp3.MAX_DEV_WIDTH_RATIO
                    * float(static_best["mean_width_uv"])
                ),
                "collect_wall_seconds": collect_seconds,
                "calibration_wall_seconds": calibration_seconds,
                "milestone_wall_seconds": time.monotonic() - eval_started,
            }
            evals.append(record)

            dump(
                args.out_dir / f"calibration_grid_{step:05d}.json",
                grid,
            )
            save_head_snapshot(
                args.out_dir / f"head_step_{step:05d}.pth",
                head=head,
                cfg=hcfg,
                step=step,
                backbone_sha=backbone_sha,
                corrector_sha=corrector_sha,
            )
            dump(
                args.out_dir / "training_progress.json",
                {"losses": losses, "evals": evals},
            )
            print(
                f"[SPS] milestone {step} done: "
                f"SPS={float(chosen['sps']):.6f}, "
                f"width={float(chosen['mean_width_uv']):.6f}, "
                f"collect={collect_seconds:.1f}s, "
                f"calibration={calibration_seconds:.1f}s",
                flush=True,
            )

            if (
                best_record is None
                or float(chosen["sps"]) > float(best_record["best"]["sps"])
            ):
                best_record = record
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                }

    if best_record is None or best_state is None:
        raise RuntimeError("no SPS checkpoint selected")

    head.load_state_dict(best_state, strict=True)
    head.eval()

    _, pred_after, sigma_after, target_after = collect(
        backbone,
        builder,
        corrector,
        head,
        dev_paths,
        device,
        args.workers,
    )

    # The SPS head must never alter point predictions.
    parity = float(np.max(np.abs(pred_after - pred_before)))
    selected = best_record["best"]
    candidate_dev = exp3.adaptive_score(
        pred_after,
        target_after,
        sigma_after,
        float(selected["floor"]),
        float(selected["mult_u"]),
        float(selected["mult_v"]),
        float(selected["rel"]),
    )
    dev_gain = float(candidate_dev["sps"] - static_best["sps"])
    width_ratio = float(
        candidate_dev["mean_width_uv"]
        / max(float(static_best["mean_width_uv"]), 1e-12)
    )

    checks = {
        "sps_gain_ge_min": dev_gain >= exp3.MIN_DEV_SPS_GAIN,
        "width_ratio_le_max": (
            width_ratio <= exp3.MAX_DEV_WIDTH_RATIO + 1e-12
        ),
        "point_parity_le_tol": parity <= exp3.POINT_PARITY_TOL,
    }
    gate = "GO" if all(checks.values()) else "NO_GO"

    summary = {
        "status": STATUS,
        "protocol": PROTOCOL,
        "joint_update": int(bp["joint_update"]),
        "backbone_sha256": backbone_sha,
        "corrector_sha256": corrector_sha,
        "point_model_frozen": True,
        "optimizer_updates_point_model": 0,
        "uncertainty_optimizer_updates": UPDATES,
        "recipe": (
            "validated Clean Exp3: h64/b2/dropout0/logmae/"
            "Past20+pre-residual-base/final-error target"
        ),
        "train_stride": 5,
        "dev_stride": 20,
        "batch": BATCH,
        "lr": LR,
        "calibration_workers": args.calibration_workers,
        "warmup": WARMUP,
        "selected_step": int(best_record["step"]),
        "selected_calibration": selected,
        "static_seen_dev": static_best,
        "adaptive_seen_dev": candidate_dev,
        "seen_dev_sps_gain": dev_gain,
        "seen_dev_width_ratio": width_ratio,
        "point_prediction_parity_max_abs": parity,
        "checks": checks,
        "gate": gate,
        "training_wall_seconds": time.monotonic() - started,
        "selection_split": "seen_dev",
        "aoa10_accessed": False,
        "aoa10_used_for_selection": False,
        "aoa10_recalibrated": False,
        "locked_final_accessed": False,
        "private_accessed": False,
        "codabench_accessed": False,
    }

    # HARD: one-shot AoA10 audit only after Seen-Dev gate GO.
    if gate == "GO":
        _, pred_aoa10, sigma_aoa10, target_aoa10 = collect(
            backbone,
            builder,
            corrector,
            head,
            aoa10_paths,
            device,
            args.workers,
        )
        static_aoa10 = exp3.static_score(
            pred_aoa10,
            target_aoa10,
            float(static_best["abs"]),
            float(static_best["rel"]),
        )
        adaptive_aoa10 = exp3.adaptive_score(
            pred_aoa10,
            target_aoa10,
            sigma_aoa10,
            float(selected["floor"]),
            float(selected["mult_u"]),
            float(selected["mult_v"]),
            float(selected["rel"]),
        )
        summary.update(
            {
                "aoa10_accessed": True,
                "static_aoa10": static_aoa10,
                "adaptive_aoa10": adaptive_aoa10,
                "aoa10_sps_gain": float(
                    adaptive_aoa10["sps"] - static_aoa10["sps"]
                ),
                "aoa10_coverage_gain": float(
                    adaptive_aoa10["coverage"] - static_aoa10["coverage"]
                ),
            }
        )

    torch.save(
        {
            "head_state_dict": best_state,
            "head_config": hcfg.__dict__,
            "selected_step": int(best_record["step"]),
            "calibration": selected,
            "protocol": PROTOCOL,
            "joint_update": TARGET_JOINT_UPDATE,
            "backbone_sha256": backbone_sha,
            "corrector_sha256": corrector_sha,
        },
        args.out_dir / "head_best.pth",
    )

    dump(
        args.out_dir / "training_progress.json",
        {"losses": losses, "evals": evals},
    )
    dump(args.out_dir / "summary.json", summary)
    (args.out_dir / "DONE").touch()

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
