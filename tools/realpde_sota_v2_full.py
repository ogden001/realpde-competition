#!/usr/bin/env python3
"""Frozen SOTA-V2 full-data competition refit.

Scientific recipe is inherited from realpde_sota_v2_integrated:
Dense-All + P0-A + MF-CNO + N2 + fixed vorticity supervision, followed by
low-LR Stage B with one extra Rel-L2 term. This runner changes only the data
scope and maps the validated 50-train update schedule by Dense-epoch exposure.

Primary checkpoint selection is frozen from Dev before this refit: the Dev
sweet spot at reference update 32,500 maps to full-data update 53,582.
There is no held-out model selection inside this all-82-trajectory refit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

SEED = 20260901
EFFECTIVE_BATCH = 8
REFERENCE_DENSE_WINDOWS = 40_488
FULL_DENSE_WINDOWS = 66_755
EXPECTED_TRAJECTORIES = 82
EXPECTED_CANONICAL_WINDOWS = 3_383
EXPECTED_FEATURES = 20
WARM_START_SHA = "82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61"
SCORER_SHA = "a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39"

REFERENCE_UPDATES_PER_EPOCH = REFERENCE_DENSE_WINDOWS // EFFECTIVE_BATCH
FULL_UPDATES_PER_EPOCH = FULL_DENSE_WINDOWS // EFFECTIVE_BATCH

N2 = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}
LAMBDA_VORT = 15.5385751724
EXTRA_REL = 0.027514
STAGE_A_LR, STAGE_B_LR = 1e-5, 3e-6


def map_reference_update(reference_update: int) -> int:
    if reference_update < 1:
        raise ValueError("reference update must be positive")
    return int(round(reference_update * FULL_UPDATES_PER_EPOCH / REFERENCE_UPDATES_PER_EPOCH))


STAGE_A_END = map_reference_update(30_000)
FINAL_UPDATE = map_reference_update(32_500)
REFERENCE_MILESTONES = (7_500, 15_000, 20_000, 25_000, 30_000, 31_000, 32_500)
MILESTONES = tuple(map_reference_update(update) for update in REFERENCE_MILESTONES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def rows(path: Path, values: list[dict]) -> None:
    if not values:
        return
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def released_paths(data_root: Path) -> list[Path]:
    paths = sorted(data_root.glob("*.h5"))
    if len(paths) != EXPECTED_TRAJECTORIES:
        raise ValueError(f"expected exactly {EXPECTED_TRAJECTORIES} released PIV trajectories, got {len(paths)}")
    return paths


def stage_config(update: int) -> dict[str, float | str]:
    if not 1 <= update <= FINAL_UPDATE:
        raise ValueError(f"update must be in [1,{FINAL_UPDATE}]")
    if update <= STAGE_A_END:
        return {"stage": "A", "lr": STAGE_A_LR, "extra_rel": 0.0}
    return {"stage": "B", "lr": STAGE_B_LR, "extra_rel": EXTRA_REL}


def validate_protocol(args: argparse.Namespace) -> None:
    if args.seed != SEED:
        raise ValueError("seed differs from frozen full-data protocol")
    if args.final_update != FINAL_UPDATE or tuple(args.milestones) != MILESTONES:
        raise ValueError("final update/milestones differ from frozen Dense-epoch mapping")
    if args.micro_batch * args.accumulation_steps != EFFECTIVE_BATCH:
        raise ValueError("effective batch must equal 8")
    if min(args.micro_batch, args.accumulation_steps, args.workers + 1) < 1:
        raise ValueError("invalid batch/worker configuration")


def integrated_loss(pred, target, update: int, base):
    parts = base.core.loss_parts(pred, target)
    parts["vorticity"] = (
        base.vorticity(pred, dx=1.0, dy=1.0) - base.vorticity(target, dx=1.0, dy=1.0)
    ).square().mean()
    cfg = stage_config(update)
    total = sum(N2[name] * parts[name] for name in N2)
    total = total + LAMBDA_VORT * parts["vorticity"] + float(cfg["extra_rel"]) * parts["rel"]
    return total, parts


def dense_loader(paths, args, base, torch):
    dataset = base.H5WindowDataset(
        paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="dense_all",
    )
    sampler = base.DenseAllWindowSampler(dataset, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.micro_batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )
    return dataset, sampler, loader


def save_checkpoint(path: Path, model, optimizer, update: int, config, metadata: dict) -> None:
    import torch

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": update,
            "feature_set": "P0-A",
            "feature_config": vars(config),
            "loss_weights": N2,
            "lambda_vort": LAMBDA_VORT,
            "extra_rel_stage_b": EXTRA_REL,
            "metadata": metadata,
        },
        path,
    )


def run(args: argparse.Namespace) -> None:
    validate_protocol(args)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import realpde_sota_v2_integrated as base

    if not (args.kit_root / "scoring.py").is_file():
        raise FileNotFoundError(args.kit_root / "scoring.py")
    checkpoint_sha = sha256(args.checkpoint)
    scorer_sha = sha256(args.kit_root / "scoring.py")
    if checkpoint_sha != WARM_START_SHA:
        raise ValueError(f"official sim_real_ft checkpoint SHA mismatch: {checkpoint_sha}")
    if scorer_sha != SCORER_SHA:
        raise ValueError(f"official scorer SHA mismatch: {scorer_sha}")

    base.core.set_seed(args.seed)
    train_paths = released_paths(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    builder, config = base.build_features(train_paths, device)
    if len(builder.feature_names) != EXPECTED_FEATURES:
        raise ValueError(f"P0-A feature count {len(builder.feature_names)} != {EXPECTED_FEATURES}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = base.MF01CNO(args.kit_root, len(builder.feature_names), device)
    base.init_mf_from_direct(model, payload, len(builder.feature_names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE_A_LR)

    train_ds, sampler, loader = dense_loader(train_paths, args, base, torch)
    canonical = base.H5WindowDataset(
        train_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="fixed",
    )
    if len(canonical) != EXPECTED_CANONICAL_WINDOWS or len(train_ds) != FULL_DENSE_WINDOWS:
        raise ValueError(f"full-data window audit mismatch: canonical={len(canonical)} dense={len(train_ds)}")

    metadata = {
        "recipe": "Dense-All+P0-A+MF-CNO+N2+Vorticity+StageB-extra-Rel",
        "mode": "all-82-released-PIV competition refit",
        "seed": args.seed,
        "checkpoint_sha256": checkpoint_sha,
        "scorer_sha256": scorer_sha,
        "feature_config": vars(config),
        "p0a_spacing_mode": "historical_online_sota",
        "n2": N2,
        "lambda_vort": LAMBDA_VORT,
        "stage_a": {"end": STAGE_A_END, "reference_update": 30_000, "lr": STAGE_A_LR},
        "stage_b": {"end": FINAL_UPDATE, "reference_update": 32_500, "lr": STAGE_B_LR, "extra_rel": EXTRA_REL},
        "reference_dense_windows": REFERENCE_DENSE_WINDOWS,
        "full_dense_windows": FULL_DENSE_WINDOWS,
        "reference_updates_per_dense_epoch": REFERENCE_UPDATES_PER_EPOCH,
        "full_updates_per_dense_epoch": FULL_UPDATES_PER_EPOCH,
        "update_scale": FULL_UPDATES_PER_EPOCH / REFERENCE_UPDATES_PER_EPOCH,
        "released_trajectories": len(train_paths),
        "canonical_windows": len(canonical),
        "micro_batch": args.micro_batch,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch": EFFECTIVE_BATCH,
        "milestones": list(args.milestones),
        "reference_milestones": list(REFERENCE_MILESTONES),
        "primary_checkpoint_policy": "Dev-selected reference 32.5k mapped by Dense epochs before full-data refit",
        "private_test_accessed": False,
        "codabench_accessed": False,
        "sps_accessed": False,
    }
    dump(args.out_dir / "run_metadata.json", metadata)

    probe_x, probe_y, _, _ = next(iter(loader))
    probe_x, probe_y = probe_x.to(device), probe_y.to(device)
    mf_pred = base.forward_mf(model, builder, probe_x)
    direct = base.build_direct(args.kit_root, builder, payload, device)
    direct_pred = base.forward_direct(direct, builder, probe_x)
    parity = float((mf_pred[..., :2] - direct_pred[..., :2]).abs().max())
    if parity > args.parity_tolerance:
        raise RuntimeError(f"Direct->MF parity failed: {parity}")
    loss, parts = integrated_loss(mf_pred, probe_y, 1, base)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    grad = max(float(value.abs().max()) for value in gradients)
    optimizer.zero_grad(set_to_none=True)
    preflight = {
        "passed": bool(torch.isfinite(mf_pred).all() and torch.isfinite(loss) and grad > 0),
        "direct_to_mf_uv_max_abs_diff": parity,
        "pressure_max_abs": float(mf_pred[..., 2].abs().max()),
        "loss": float(loss.detach()),
        "loss_parts": {name: float(value.detach()) for name, value in parts.items()},
        "gradient_max_abs": grad,
        "released_trajectories": len(train_paths),
        "canonical_windows": len(canonical),
        "dense_windows": len(train_ds),
        "feature_count": len(builder.feature_names),
        "stage_a_end": STAGE_A_END,
        "final_update": FINAL_UPDATE,
    }
    dump(args.out_dir / "preflight.json", preflight)
    if not preflight["passed"] or preflight["pressure_max_abs"] != 0.0:
        raise RuntimeError("preflight failed")
    if args.preflight_only:
        return
    del direct, direct_pred, mf_pred, probe_x, probe_y

    checkpoints = args.out_dir / "checkpoints"
    checkpoints.mkdir()
    history: list[dict] = []
    iterator = iter(loader)
    epoch = 0
    started = time.monotonic()
    current_lr = STAGE_A_LR

    for update in range(1, args.final_update + 1):
        cfg = stage_config(update)
        if float(cfg["lr"]) != current_lr:
            current_lr = float(cfg["lr"])
            for group in optimizer.param_groups:
                group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        total_parts = {name: 0.0 for name in (*N2, "vorticity")}
        for _ in range(args.accumulation_steps):
            try:
                x, y, _, _ = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                x, y, _, _ = next(iterator)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            model.train()
            pred = base.forward_mf(model, builder, x)
            step_loss, step_parts = integrated_loss(pred, y, update, base)
            if not torch.isfinite(step_loss):
                raise FloatingPointError(f"nonfinite loss @ {update}")
            (step_loss / args.accumulation_steps).backward()
            for name in total_parts:
                total_parts[name] += float(step_parts[name].detach()) / args.accumulation_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if update % args.log_every == 0:
            print(json.dumps({"update": update, "stage": cfg["stage"], "lr": current_lr, **total_parts}), flush=True)

        if update in args.milestones:
            reference_update = min(
                REFERENCE_MILESTONES,
                key=lambda candidate: abs(map_reference_update(candidate) - update),
            )
            row = {
                "update": update,
                "reference_update": reference_update,
                "stage": cfg["stage"],
                "lr": current_lr,
                **total_parts,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            rows(args.out_dir / "training_milestones.csv", history)
            save_checkpoint(checkpoints / f"model_update_{update:05d}.pth", model, optimizer, update, config, metadata)
            save_checkpoint(checkpoints / "model_latest.pth", model, optimizer, update, config, metadata)
            dump(args.out_dir / "status.json", {"state": "RUNNING", "update": update, "reference_update": reference_update})
            print(json.dumps(row), flush=True)

    runtime = {
        "training_wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_allocated": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "peak_gpu_memory_reserved": int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0,
        "dense_epochs_completed": FINAL_UPDATE / FULL_UPDATES_PER_EPOCH,
    }
    dump(args.out_dir / "runtime.json", runtime)
    dump(args.out_dir / "status.json", {"state": "DONE", "update": FINAL_UPDATE, "reference_update": 32_500})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--final-update", type=int, default=FINAL_UPDATE)
    parser.add_argument("--milestones", type=int, nargs="+", default=list(MILESTONES))
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
