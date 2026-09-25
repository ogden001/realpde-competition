#!/usr/bin/env python3
"""Long-running SOTA-merge backbone trainer for clean and full RealPDE data.

Stage A:
- every legal dense temporal window (stride 1)
- exhaustive P00/P01/P10/P11 real spatial views
- Strong Backbone: P0-A + MF-CNO + N2 + vorticity
- LR 1e-5, FP32, effective batch 8
- open-ended checkpoint/resume

Stage B:
- fork from any immutable Stage-A checkpoint
- P00-only dense windows
- same Strong Backbone objective plus extra Rel
- LR 3e-6, carry AdamW state by default
- short alignment tail with clean-only Seen-Dev selection

The same code runs in clean and full scope. Full scope never performs Dev
selection. There is no locked-final/private/Codabench action in this runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

import realpde_loss_official_v9 as core
import realpde_sota_v2_integrated as strong
from realpde_mf01 import MF01CNO
from realpde_p0_data import H5WindowDataset
from realpde_sota_merge_spatial import SpatialPhaseExpandedDataset, SpatialPhaseExpandedSampler

STATUS = "REVIEW_REQUIRED"
PROTOCOL = "REALPDE_SOTA_MERGE_SPATIAL_LONGTRAIN_V1"
SEED = 41
BATCH_SIZE = 8
STAGE_A_LR = 1e-5
STAGE_B_LR = 3e-6
OFFICIAL_SIM_REAL_SHA256 = "82e842928a25dbf5a74c4e336bdd28e89bcf40e68bb8cdd213547f1246af4f61"
CLEAN_TRAIN_TRAJECTORIES = 51
CLEAN_DEV_TRAJECTORIES = 12
CLEAN_DENSE_WINDOWS = 41_317
CLEAN_DEV_WINDOWS = 491
FULL_TRAJECTORIES = 82
FULL_DENSE_WINDOWS = 66_755
FORBIDDEN_PATH_TOKENS = ("locked-final", "locked_final", "private_test", "private-test")


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def write_rows(path: Path, values: list[dict[str, object]]) -> None:
    if not values:
        return
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def assert_safe_path(path: Path | None) -> None:
    if path is None:
        return
    lowered = str(path).lower().replace("\\", "/")
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"forbidden locked/private path: {path}")


def point_score(raw: dict[str, float]) -> float:
    def score(error: float) -> float:
        return 100.0 / (1.0 + 0.5 * max(float(error), 0.0))
    return float(np.mean([score(raw["rel_l2"]), score(raw["tke"]), score(raw["mvpe"])]))


def strong_loss(pred: torch.Tensor, target: torch.Tensor, *, extra_rel: float):
    parts = core.loss_parts(pred, target)
    parts["vorticity"] = (
        strong.vorticity(pred, dx=1.0, dy=1.0) - strong.vorticity(target, dx=1.0, dy=1.0)
    ).square().mean()
    total = sum(strong.N2[name] * parts[name] for name in strong.N2)
    total = total + strong.LAMBDA_VORT * parts["vorticity"] + float(extra_rel) * parts["rel"]
    return total, parts


def _manifest_rows(manifest: Path, split: str) -> list[str]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get(split)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest has no non-empty {split!r} list")
    names = [str(row["file"] if isinstance(row, dict) else row) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate {split} filenames in manifest")
    return names


def resolve_scope(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    assert_safe_path(args.data_root)
    assert_safe_path(args.manifest)
    if args.scope == "clean":
        if args.manifest is None:
            raise ValueError("--manifest is required for --scope clean")
        train_names = _manifest_rows(args.manifest, "train")
        dev_names = _manifest_rows(args.manifest, "dev")
        train = [args.data_root / name for name in train_names]
        dev = [args.data_root / name for name in dev_names]
        if {p.name for p in train} & {p.name for p in dev}:
            raise ValueError("clean train/dev overlap")
        if (len(train), len(dev)) != (CLEAN_TRAIN_TRAJECTORIES, CLEAN_DEV_TRAJECTORIES):
            raise ValueError(f"clean scope must be 51/12, got {len(train)}/{len(dev)}")
    else:
        if args.manifest is not None:
            raise ValueError("--manifest is forbidden for --scope full")
        train, dev = sorted(args.data_root.glob("*.h5")), []
        if len(train) != FULL_TRAJECTORIES:
            raise ValueError(f"full scope requires exactly {FULL_TRAJECTORIES} released trajectories, got {len(train)}")
    missing = [str(path) for path in train + dev if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing trajectories: {missing[:10]}")
    return train, dev


class P00DenseSampler(Sampler[int]):
    """Historical Dense-All shuffle with exact resume offset."""

    def __init__(self, dataset: H5WindowDataset, *, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0
        self._order: list[int] = []
        self.set_epoch(0, start_index=0)

    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        if epoch < 0 or start_index < 0:
            raise ValueError("invalid P00 sampler state")
        order = list(range(len(self.dataset)))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch)]))
        rng.shuffle(order)
        if start_index > len(order):
            raise ValueError("P00 sampler offset outside epoch")
        self.epoch = int(epoch)
        self.start_index = int(start_index)
        self._order = order

    def __iter__(self):
        return iter(self._order[self.start_index :])

    def __len__(self) -> int:
        return len(self._order) - self.start_index


@torch.no_grad()
def evaluate_clean(
    model,
    builder,
    dev_paths: list[Path],
    *,
    kit_root: Path,
    workers: int,
    device: torch.device,
    out_dir: Path,
    update: int,
) -> tuple[dict[str, float], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset, loader = strong.dev_loader(dev_paths, argparse.Namespace(eval_batch_size=8, workers=workers))
    model.eval()
    preds, targets = [], []
    elapsed = 0.0
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        pred = strong.forward_mf(model, builder, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        preds.append(pred.cpu().numpy().astype(np.float32))
        targets.append(y.numpy().astype(np.float32))
    prediction, target = np.concatenate(preds), np.concatenate(targets)
    if not np.isfinite(prediction).all() or float(np.abs(prediction[..., 2]).max()) != 0.0:
        raise FloatingPointError("invalid clean evaluation prediction")
    scored = core.score_bundle(kit_root, prediction, target, elapsed / len(dataset), out_dir)
    raw = {name: float(scored["raw_errors"][name]) for name in ("rel_l2", "tke", "mvpe")}
    score = point_score(raw)
    dump(out_dir / "horizon_error_summary.json", strong.horizon_error_summary(prediction, target))
    dump(out_dir / "eval_summary.json", {
        "update": int(update),
        "point_score": score,
        "raw_errors": raw,
        "official_v9_subscores": scored.get("official_v9_subscores"),
        "mean_t_neural_s": float(scored["mean_t_neural_s"]),
        "windows": len(dataset),
    })
    return raw, score


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def checkpoint_payload(
    *,
    model,
    optimizer,
    update: int,
    stage: str,
    scope: str,
    sampler_epoch: int,
    sampler_offset: int,
    feature_config,
    metadata: dict[str, object],
    score: float | None,
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(update),
        "stage": stage,
        "scope": scope,
        "sampler_epoch": int(sampler_epoch),
        "sampler_offset": int(sampler_offset),
        "feature_set": "P0-A",
        "feature_config": vars(feature_config),
        "loss_weights": strong.N2,
        "lambda_vort": strong.LAMBDA_VORT,
        "extra_rel": 0.0 if stage == "A" else strong.EXTRA_REL,
        "selection_metric": "point_score" if scope == "clean" else None,
        "selection_score": None if score is None else float(score),
        "metadata": metadata,
    }


def save_checkpoint(path: Path, **kwargs) -> None:
    atomic_torch_save(checkpoint_payload(**kwargs), path)


def load_model_and_optimizer(
    *,
    train_paths: list[Path],
    kit_root: Path,
    init_checkpoint: Path,
    device: torch.device,
    resume_checkpoint: Path | None,
    stage: str,
    scope: str,
    lr: float,
):
    builder, feature_config = strong.build_features(train_paths, device)
    init_payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    model = MF01CNO(kit_root, len(builder.feature_names), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    resume = None
    if resume_checkpoint is None:
        strong.init_mf_from_direct(model, init_payload, len(builder.feature_names))
    else:
        resume = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if resume.get("stage") != stage or resume.get("scope") != scope:
            raise ValueError(f"resume checkpoint stage/scope mismatch: {resume.get('stage')}/{resume.get('scope')}")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = lr
    return model, optimizer, builder, feature_config, init_payload, resume


def preflight_fresh(model, builder, init_payload, *, kit_root: Path, dev_paths, train_paths, device):
    probe_paths = dev_paths if dev_paths else train_paths
    probe_ds = H5WindowDataset(
        probe_paths[:1], in_steps=20, out_steps=20, stride=20, sub_sample=2,
        include_pressure=False, window_mode="fixed",
    )
    x, y, _, _ = next(iter(DataLoader(probe_ds, batch_size=2, shuffle=False, num_workers=0)))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        mf_pred = strong.forward_mf(model, builder, x)
        direct = strong.build_direct(kit_root, builder, init_payload, device)
        direct_pred = strong.forward_direct(direct, builder, x)
    parity = float((mf_pred[..., :2] - direct_pred[..., :2]).abs().max())
    loss, _ = strong_loss(mf_pred, y, extra_rel=0.0)
    if parity > 1e-6 or not torch.isfinite(loss):
        raise RuntimeError(f"fresh Direct->MF preflight failed: parity={parity}, loss={float(loss)}")
    return {
        "passed": True,
        "direct_to_mf_uv_max_abs_diff": parity,
        "pressure_max_abs": float(mf_pred[..., 2].abs().max()),
        "initial_stage_a_loss": float(loss),
    }


def common_metadata(args, train_paths, dev_paths, feature_config, *, dense_windows: int):
    return {
        "status": STATUS,
        "protocol": PROTOCOL,
        "scope": args.scope,
        "seed": SEED,
        "train_trajectories": len(train_paths),
        "dev_trajectories": len(dev_paths),
        "dense_temporal_windows": int(dense_windows),
        "batch_size": BATCH_SIZE,
        "precision": "FP32",
        "feature_set": "P0-A",
        "feature_config": vars(feature_config),
        "backbone": "MF-CNO",
        "n2": strong.N2,
        "lambda_vort": strong.LAMBDA_VORT,
        "stage_a_lr": STAGE_A_LR,
        "stage_b_lr": STAGE_B_LR,
        "stage_b_extra_rel": strong.EXTRA_REL,
        "locked_final_accessed": False,
        "private_accessed": False,
        "codabench_accessed": False,
    }


def normalize_stage_a_sampler(sampler, epoch: int, offset: int) -> tuple[int, int]:
    sampler.set_epoch(epoch, start_index=0)
    full = sampler.full_epoch_size
    if offset == full:
        epoch += 1
        offset = 0
    elif not 0 <= offset < full:
        raise ValueError("invalid Stage-A sampler offset")
    sampler.set_epoch(epoch, start_index=offset)
    return epoch, offset


def train_stage_a(args: argparse.Namespace) -> None:
    train_paths, dev_paths = resolve_scope(args)
    for path in (args.init_checkpoint, args.kit_root):
        assert_safe_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    if strong.sha256(args.init_checkpoint) != OFFICIAL_SIM_REAL_SHA256:
        raise ValueError("official sim_real checkpoint SHA mismatch")
    if args.batch_size != BATCH_SIZE:
        raise ValueError("effective batch is frozen at 8")
    if args.out_dir.exists() and args.resume_checkpoint is None:
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    core.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")

    model, optimizer, builder, feature_config, init_payload, resume = load_model_and_optimizer(
        train_paths=train_paths,
        kit_root=args.kit_root,
        init_checkpoint=args.init_checkpoint,
        device=device,
        resume_checkpoint=args.resume_checkpoint,
        stage="A",
        scope=args.scope,
        lr=STAGE_A_LR,
    )
    dataset = SpatialPhaseExpandedDataset(
        train_paths,
        in_steps=20,
        out_steps=20,
        stride=1,
        sub_sample=2,
        include_pressure=False,
        preload_to_ram=args.preload_to_ram,
    )
    expected_dense = CLEAN_DENSE_WINDOWS if args.scope == "clean" else FULL_DENSE_WINDOWS
    if dataset.base_window_count != expected_dense:
        raise ValueError(f"dense window audit mismatch: {dataset.base_window_count} != {expected_dense}")

    sampler = SpatialPhaseExpandedSampler(dataset, seed=SEED)
    start_update = 0 if resume is None else int(resume["iteration"])
    sampler_epoch = 0 if resume is None else int(resume.get("sampler_epoch", 0))
    sampler_offset = 0 if resume is None else int(resume.get("sampler_offset", 0))
    sampler_epoch, sampler_offset = normalize_stage_a_sampler(sampler, sampler_epoch, sampler_offset)
    if args.max_updates <= start_update:
        raise ValueError(f"--max-updates {args.max_updates} must exceed checkpoint update {start_update}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.workers > 0),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        drop_last=True,
    )
    iterator = iter(loader)

    metadata = common_metadata(args, train_paths, dev_paths, feature_config, dense_windows=dataset.base_window_count)
    metadata.update({
        "stage": "A",
        "sampling": "exhaustive window x P00/P01/P10/P11",
        "spatial_phase_distribution": {name: 0.25 for name in ("P00", "P01", "P10", "P11")},
        "same_phase_for_past_and_future": True,
        "unique_spatial_views_per_epoch": len(dataset),
        "stream_views_per_epoch": sampler.full_epoch_size,
        "spatial_epoch_updates": sampler.full_epoch_size // BATCH_SIZE,
        "balance_pad_views_per_epoch": sampler.full_epoch_size - len(dataset),
        "preload_to_ram": bool(args.preload_to_ram),
        "cache": dataset.cache_summary(),
        "eval_interval": args.eval_interval if args.scope == "clean" else None,
        "checkpoint_interval": args.checkpoint_interval,
        "start_update": start_update,
        "target_update": args.max_updates,
        "resume_checkpoint": None if args.resume_checkpoint is None else str(args.resume_checkpoint),
        "full_has_dev_selection": False,
    })
    dump(args.out_dir / "run_config.json", metadata)
    dump(args.out_dir / "phase_dataset_audit.json", {
        "base_windows": dataset.base_window_count,
        "unique_views": len(dataset),
        "phase_counts": dataset.phase_counts(),
        "initial_sampler": sampler.audit(),
    })
    if resume is None:
        dump(args.out_dir / "preflight.json", preflight_fresh(
            model, builder, init_payload, kit_root=args.kit_root,
            dev_paths=dev_paths, train_paths=train_paths, device=device,
        ))

    history_path = args.out_dir / "aggregate_metrics.csv"
    history: list[dict[str, object]] = []
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = [dict(row) for row in csv.DictReader(handle)]
    best_score = -float("inf")
    best_update = None
    summary_path = args.out_dir / "summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        best_score = float(old.get("best_point_score", best_score))
        best_update = old.get("best_update")

    checkpoints = args.out_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if args.scope == "clean" and start_update == 0 and not history:
        raw, score = evaluate_clean(
            model, builder, dev_paths, kit_root=args.kit_root, workers=args.workers,
            device=device, out_dir=args.out_dir / "eval_000000", update=0,
        )
        history.append({"update": 0, "stage": "A", "point_score": score, **raw})
        write_rows(history_path, history)
        best_score, best_update = score, 0
        save_checkpoint(
            checkpoints / "model_best.pth",
            model=model, optimizer=optimizer, update=0, stage="A", scope=args.scope,
            sampler_epoch=sampler_epoch, sampler_offset=sampler_offset,
            feature_config=feature_config, metadata=metadata, score=score,
        )

    started = time.monotonic()
    update = start_update
    while update < args.max_updates:
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            sampler_epoch += 1
            sampler_offset = 0
            sampler.set_epoch(sampler_epoch, start_index=0)
            append_jsonl(args.out_dir / "phase_epoch_audit.jsonl", sampler.audit())
            iterator = iter(loader)
            x, y, _, _ = next(iterator)

        if int(x.shape[0]) != BATCH_SIZE:
            raise RuntimeError(f"Stage A batch-size drift: {x.shape[0]}")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.train()
        pred = strong.forward_mf(model, builder, x)
        loss, parts = strong_loss(pred, y, extra_rel=0.0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Stage-A loss @ {update + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update += 1
        sampler_offset += BATCH_SIZE

        if sampler_offset == sampler.full_epoch_size:
            sampler_epoch += 1
            sampler_offset = 0
            sampler.set_epoch(sampler_epoch, start_index=0)
            append_jsonl(args.out_dir / "phase_epoch_audit.jsonl", sampler.audit())
            iterator = iter(loader)

        if update == 1 or update % args.log_interval == 0:
            print(json.dumps({
                "update": update,
                "stage": "A",
                "loss": float(loss.detach()),
                "sampler_epoch": sampler_epoch,
                "sampler_offset": sampler_offset,
                **{f"loss_{name}": float(value.detach()) for name, value in parts.items()},
            }, sort_keys=True), flush=True)

        score: float | None = None
        if args.scope == "clean" and update % args.eval_interval == 0:
            raw, score = evaluate_clean(
                model, builder, dev_paths, kit_root=args.kit_root, workers=args.workers,
                device=device, out_dir=args.out_dir / f"eval_{update:06d}", update=update,
            )
            row = {
                "update": update,
                "stage": "A",
                "point_score": score,
                **raw,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            write_rows(history_path, history)
            if score > best_score:
                best_score, best_update = score, update
                save_checkpoint(
                    checkpoints / "model_best.pth",
                    model=model, optimizer=optimizer, update=update, stage="A", scope=args.scope,
                    sampler_epoch=sampler_epoch, sampler_offset=sampler_offset,
                    feature_config=feature_config, metadata=metadata, score=score,
                )

        if update % args.checkpoint_interval == 0 or update == args.max_updates:
            for target in (checkpoints / f"model_update_{update:06d}.pth", checkpoints / "model_latest.pth"):
                save_checkpoint(
                    target,
                    model=model, optimizer=optimizer, update=update, stage="A", scope=args.scope,
                    sampler_epoch=sampler_epoch, sampler_offset=sampler_offset,
                    feature_config=feature_config, metadata=metadata, score=score,
                )

        dump(summary_path, {
            "status": STATUS,
            "stage": "A",
            "scope": args.scope,
            "current_update": update,
            "target_update": args.max_updates,
            "best_update": best_update if args.scope == "clean" else None,
            "best_point_score": best_score if args.scope == "clean" and np.isfinite(best_score) else None,
            "sampler_epoch": sampler_epoch,
            "sampler_offset": sampler_offset,
            "spatial_epochs_seen": (update * BATCH_SIZE) / sampler.full_epoch_size,
            "training_wall_seconds_this_process": time.monotonic() - started,
            "locked_final_accessed": False,
            "codabench_accessed": False,
        })

    (args.out_dir / "DONE_TO_REQUESTED_UPDATE").touch()


def normalize_p00_state(dataset, sampler, epoch: int, offset: int) -> tuple[int, int]:
    usable = (len(dataset) // BATCH_SIZE) * BATCH_SIZE
    if offset == usable:
        epoch += 1
        offset = 0
    if offset < 0 or offset > usable or offset % BATCH_SIZE:
        raise ValueError("invalid Stage-B P00 sampler state")
    sampler.set_epoch(epoch, start_index=offset)
    return epoch, offset


def train_stage_b(args: argparse.Namespace) -> None:
    train_paths, dev_paths = resolve_scope(args)
    for path in (args.init_checkpoint, args.kit_root, args.backbone_checkpoint):
        assert_safe_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    if strong.sha256(args.init_checkpoint) != OFFICIAL_SIM_REAL_SHA256:
        raise ValueError("official sim_real checkpoint SHA mismatch")
    if args.batch_size != BATCH_SIZE:
        raise ValueError("effective batch is frozen at 8")
    if args.out_dir.exists() and args.resume_checkpoint is None:
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    core.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required")
    builder, feature_config = strong.build_features(train_paths, device)
    model = MF01CNO(args.kit_root, len(builder.feature_names), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE_B_LR)

    source = torch.load(args.backbone_checkpoint, map_location="cpu", weights_only=False)
    if source.get("stage") != "A" or source.get("scope") != args.scope:
        raise ValueError("Stage B must fork from a matching Stage-A checkpoint")
    source_stage_a_update = int(source["iteration"])
    model.load_state_dict(source["model_state_dict"], strict=True)
    if args.optimizer_policy == "carry":
        optimizer.load_state_dict(source["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = STAGE_B_LR

    start_update = 0
    sampler_epoch, sampler_offset = 0, 0
    if args.resume_checkpoint is not None:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if resume.get("stage") != "B" or resume.get("scope") != args.scope:
            raise ValueError("Stage-B resume checkpoint mismatch")
        if int(resume.get("metadata", {}).get("source_stage_a_update", -1)) != source_stage_a_update:
            raise ValueError("Stage-B resume source Stage-A checkpoint mismatch")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = STAGE_B_LR
        start_update = int(resume["iteration"])
        sampler_epoch = int(resume.get("sampler_epoch", 0))
        sampler_offset = int(resume.get("sampler_offset", 0))
    if args.max_updates <= start_update:
        raise ValueError("Stage-B --max-updates must exceed resume iteration")

    dataset = H5WindowDataset(
        train_paths,
        in_steps=20,
        out_steps=20,
        stride=20,
        sub_sample=2,
        include_pressure=False,
        window_mode="dense_all",
    )
    expected_dense = CLEAN_DENSE_WINDOWS if args.scope == "clean" else FULL_DENSE_WINDOWS
    if len(dataset) != expected_dense:
        raise ValueError(f"Stage-B dense window audit mismatch: {len(dataset)} != {expected_dense}")
    sampler = P00DenseSampler(dataset, seed=SEED)
    sampler_epoch, sampler_offset = normalize_p00_state(dataset, sampler, sampler_epoch, sampler_offset)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.workers > 0),
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        drop_last=True,
    )
    iterator = iter(loader)
    usable_per_epoch = (len(dataset) // BATCH_SIZE) * BATCH_SIZE

    metadata = common_metadata(args, train_paths, dev_paths, feature_config, dense_windows=len(dataset))
    metadata.update({
        "stage": "B",
        "sampling": "P00-only Dense-All",
        "source_stage_a_checkpoint": str(args.backbone_checkpoint),
        "source_stage_a_sha256": strong.sha256(args.backbone_checkpoint),
        "source_stage_a_update": source_stage_a_update,
        "optimizer_policy": args.optimizer_policy,
        "eval_interval": args.eval_interval if args.scope == "clean" else None,
        "checkpoint_interval": args.checkpoint_interval,
        "start_update": start_update,
        "target_update": args.max_updates,
        "full_has_dev_selection": False,
    })
    dump(args.out_dir / "run_config.json", metadata)

    history_path = args.out_dir / "aggregate_metrics.csv"
    history: list[dict[str, object]] = []
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = [dict(row) for row in csv.DictReader(handle)]
    best_score = -float("inf")
    best_update = None
    summary_path = args.out_dir / "summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        best_score = float(old.get("best_point_score", best_score))
        best_update = old.get("best_update")

    checkpoints = args.out_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if args.scope == "clean" and start_update == 0 and not history:
        raw, score = evaluate_clean(
            model, builder, dev_paths, kit_root=args.kit_root, workers=args.workers,
            device=device, out_dir=args.out_dir / "eval_000000", update=0,
        )
        history.append({"update": 0, "stage": "B", "point_score": score, **raw})
        write_rows(history_path, history)
        best_score, best_update = score, 0
        save_checkpoint(
            checkpoints / "model_best.pth",
            model=model, optimizer=optimizer, update=0, stage="B", scope=args.scope,
            sampler_epoch=sampler_epoch, sampler_offset=sampler_offset,
            feature_config=feature_config, metadata=metadata, score=score,
        )

    started = time.monotonic()
    update = start_update
    while update < args.max_updates:
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            sampler_epoch += 1
            sampler_offset = 0
            sampler.set_epoch(sampler_epoch, start_index=0)
            iterator = iter(loader)
            x, y, _, _ = next(iterator)

        if int(x.shape[0]) != BATCH_SIZE:
            raise RuntimeError(f"Stage B batch-size drift: {x.shape[0]}")
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.train()
        pred = strong.forward_mf(model, builder, x)
        loss, parts = strong_loss(pred, y, extra_rel=strong.EXTRA_REL)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Stage-B loss @ {update + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update += 1
        sampler_offset += BATCH_SIZE

        if sampler_offset == usable_per_epoch:
            sampler_epoch += 1
            sampler_offset = 0
            sampler.set_epoch(sampler_epoch, start_index=0)
            iterator = iter(loader)

        if update == 1 or update % args.log_interval == 0:
            print(json.dumps({
                "update": update,
                "stage": "B",
                "loss": float(loss.detach()),
                "source_stage_a_update": source_stage_a_update,
                **{f"loss_{name}": float(value.detach()) for name, value in parts.items()},
            }, sort_keys=True), flush=True)

        score: float | None = None
        if args.scope == "clean" and update % args.eval_interval == 0:
            raw, score = evaluate_clean(
                model, builder, dev_paths, kit_root=args.kit_root, workers=args.workers,
                device=device, out_dir=args.out_dir / f"eval_{update:06d}", update=update,
            )
            row = {
                "update": update,
                "stage": "B",
                "source_stage_a_update": source_stage_a_update,
                "point_score": score,
                **raw,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            write_rows(history_path, history)
            if score > best_score:
                best_score, best_update = score, update
                save_checkpoint(
                    checkpoints / "model_best.pth",
                    model=model, optimizer=optimizer, update=update, stage="B", scope=args.scope,
                    sampler_epoch=sampler_epoch, sampler_offset=sampler_offset,
                    feature_config=feature_config, metadata=metadata, score=score,
                )

        if update % args.checkpoint_interval == 0 or update == args.max_updates:
            for target in (checkpoints / f"model_update_{update:06d}.pth", checkpoints / "model_latest.pth"):
                save_checkpoint(
                    target,
                    model=model, optimizer=optimizer, update=update, stage="B", scope=args.scope,
                    sampler_epoch=sampler_epoch, sampler_offset=sampler_offset,
                    feature_config=feature_config, metadata=metadata, score=score,
                )

        dump(summary_path, {
            "status": STATUS,
            "stage": "B",
            "scope": args.scope,
            "source_stage_a_update": source_stage_a_update,
            "current_update": update,
            "target_update": args.max_updates,
            "best_update": best_update if args.scope == "clean" else None,
            "best_point_score": best_score if args.scope == "clean" and np.isfinite(best_score) else None,
            "training_wall_seconds_this_process": time.monotonic() - started,
            "locked_final_accessed": False,
            "codabench_accessed": False,
        })

    (args.out_dir / "DONE_TO_REQUESTED_UPDATE").touch()


def map_updates(args: argparse.Namespace) -> None:
    if args.clean_updates < 1:
        raise ValueError("--clean-updates must be positive")
    if args.stage == "A":
        clean_views = 4 * CLEAN_DENSE_WINDOWS + (4 if (4 * CLEAN_DENSE_WINDOWS) % 8 else 0)
        full_views = 4 * FULL_DENSE_WINDOWS + (4 if (4 * FULL_DENSE_WINDOWS) % 8 else 0)
        clean_per_epoch = clean_views // BATCH_SIZE
        full_per_epoch = full_views // BATCH_SIZE
    else:
        clean_per_epoch = CLEAN_DENSE_WINDOWS // BATCH_SIZE
        full_per_epoch = FULL_DENSE_WINDOWS // BATCH_SIZE
    mapped = int(round(args.clean_updates * full_per_epoch / clean_per_epoch))
    print(json.dumps({
        "stage": args.stage,
        "clean_reference_updates": args.clean_updates,
        "clean_updates_per_epoch": clean_per_epoch,
        "full_updates_per_epoch": full_per_epoch,
        "update_scale": full_per_epoch / clean_per_epoch,
        "mapped_full_updates": mapped,
    }, indent=2, sort_keys=True))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope", choices=("clean", "full"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, required=True, help="total update index to reach, not additional updates")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--require-cuda", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    stage_a = sub.add_parser("stage-a", help="open-ended four-phase Strong Backbone training")
    add_common(stage_a)
    stage_a.add_argument("--preload-to-ram", action="store_true")

    stage_b = sub.add_parser("stage-b", help="P00-only low-LR alignment tail forked from Stage A")
    add_common(stage_b)
    stage_b.add_argument("--backbone-checkpoint", type=Path, required=True)
    stage_b.add_argument("--optimizer-policy", choices=("carry", "reset"), default="carry")

    mapper = sub.add_parser("map-updates", help="map clean update budget to equal full-data epoch exposure")
    mapper.add_argument("--stage", choices=("A", "B"), required=True)
    mapper.add_argument("--clean-updates", type=int, required=True)

    args = parser.parse_args()
    if args.command == "stage-a":
        train_stage_a(args)
    elif args.command == "stage-b":
        train_stage_b(args)
    else:
        map_updates(args)


if __name__ == "__main__":
    main()
