#!/usr/bin/env python3
"""Matched Direct-CNO + lightweight local residual experiment.

The global branch is the registered P0-A Direct CNO.  The local branch sees
only the runtime-safe Past20 raw u/v window and is fused as
``global + residual``.  Its final projection is zero initialized so update 0
is exactly the supplied Direct checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realpde_loss_official_v9 as core  # noqa: E402
import realpde_mf01 as direct  # noqa: E402

SEED = 20260901
RELATIVE_UPDATES = 1500
N2_WEIGHTS = direct.N2_WEIGHTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class LocalResidualBranch(nn.Module):
    """Small local spatiotemporal operator over raw Past20 u/v only."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.input = nn.Conv3d(2, hidden, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.output = nn.Conv3d(hidden, 2, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[1:] != (20, x.shape[2], x.shape[3], 3):
            if x.ndim != 5 or x.shape[1] != 20 or x.shape[-1] < 2:
                raise ValueError(f"expected [B,20,H,W,C>=2], got {tuple(x.shape)}")
        uv = x[..., :2].permute(0, 4, 1, 2, 3)
        residual = self.output(self.activation(self.input(uv)))
        return residual.permute(0, 2, 3, 4, 1)


def fuse_prediction(global_prediction: Tensor, x: Tensor, local: LocalResidualBranch) -> Tensor:
    residual = local(x)
    output = global_prediction.clone()
    output[..., :2] = global_prediction[..., :2] + residual
    return output


def compare_trajectory_metrics(baseline: dict[str, dict[str, float]], candidate: dict[str, dict[str, float]]) -> tuple[dict, dict]:
    metrics = ("rel_l2", "tke", "mvpe")
    delta: dict[str, dict[str, float | bool]] = {}
    wins = {metric: 0 for metric in metrics}
    for name in sorted(set(baseline) & set(candidate)):
        delta[name] = {}
        for metric in metrics:
            value = (baseline[name][metric] - candidate[name][metric]) / max(baseline[name][metric], 1e-12) * 100.0
            delta[name][f"{metric}_pct"] = float(value)
            delta[name][f"{metric}_win"] = bool(value > 0.0)
            wins[metric] += int(value > 0.0)
    return delta, wins


def build_global(kit_root: Path, checkpoint: Path, train_paths: list[Path], device: torch.device) -> tuple[nn.Module, direct.P0FeatureBuilder]:
    builder, _ = direct.build_features(train_paths, device)
    model = direct.cno_direct(kit_root, len(builder.feature_names), device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    input_key = "lift.inter_CNOBlock.convolution.weight"
    if input_key in state and state[input_key].shape[1] == len(builder.feature_names):
        model.load_state_dict(state, strict=True)
    else:
        direct.adapt_input_weight(model, payload, len(builder.feature_names))
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("non-finite Direct checkpoint")
    return model, builder


def forward(model: nn.Module, builder: direct.P0FeatureBuilder, x: Tensor, local: LocalResidualBranch | None = None) -> Tensor:
    features = builder(x)
    global_prediction = model(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
    return global_prediction if local is None else fuse_prediction(global_prediction, x, local)


@torch.no_grad()
def evaluate(model: nn.Module, builder: direct.P0FeatureBuilder, local: LocalResidualBranch | None,
             paths: list[Path], args: argparse.Namespace, device: torch.device, out: Path,
             save_predictions: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds, loader = core.loader(paths, args, shuffle=False)
    model.eval();
    if local is not None: local.eval()
    predictions, targets, residuals, elapsed = [], [], [], 0.0
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter(); prediction = forward(model, builder, x, local)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        predictions.append(prediction.cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
        if local is not None:
            residuals.append(local(x).cpu().numpy().astype(np.float32))
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    scored = core.score_bundle(args.kit_root, prediction, target, elapsed / len(ds), out)
    rows, _ = core.trajectory_rows(ds, prediction, target, args.kit_root)
    with (out / "trajectory_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if save_predictions:
        payload = {"prediction": prediction, "target": target}
        if residuals:
            payload["local_residual_uv"] = np.concatenate(residuals)
        np.savez_compressed(out / "predictions.npz", **payload)
    return scored | {"windows": len(ds), "trajectories": len(rows), "prediction_path": str(out / "predictions.npz") if save_predictions else None}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    _, train_paths = core.read_manifest(args.manifest, "train")
    _, dev_paths = core.read_manifest(args.manifest, "dev")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, builder = build_global(args.kit_root, args.checkpoint, train_paths, device)
    local = LocalResidualBranch(hidden=args.local_hidden).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if "optimizer_state_dict" in payload and args.restore_optimizer:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    metadata = {
        "experiment_id": args.experiment_id, "reference": "T1-ID-MF-DIRECT3000-CLOSEOUT-S20260901",
        "architecture": "Direct CNO P0-A global + Conv3d local residual; prediction=global+local",
        "global_input": list(builder.feature_names),
        "local_input": "Past20 runtime-safe raw u/v only", "local_operator": "Conv3d(2,16,3,pad=1)+GELU+Conv3d(16,2,1)",
        "zero_init": True, "loss_weights": N2_WEIGHTS, "seed": args.seed, "batch_size": args.batch_size,
        "workers": args.workers, "lr": args.lr, "relative_updates": args.updates, "start_absolute_update": args.start_update,
        "final_absolute_update": args.start_update + args.updates, "manifest_sha256": sha256(args.manifest),
        "checkpoint_sha256": sha256(args.checkpoint), "scorer_sha256": sha256(args.kit_root / "scoring.py"),
        "train_trajectories": len(train_paths), "dev_trajectories": len(dev_paths), "locked_final_accessed": False,
        "codabench": False, "device": str(device), "global_parameters": sum(p.numel() for p in model.parameters()),
        "local_parameters": sum(p.numel() for p in local.parameters()), "runner_sha256": sha256(Path(__file__)),
    }
    save_json(args.out_dir / "run_metadata.json", metadata)
    if args.smoke:
        ds, loader = core.loader(train_paths[:1], args, shuffle=False)
        x, _, _, _ = next(iter(loader)); x = x.to(device)
        with torch.no_grad():
            direct_pred = forward(model, builder, x)
            fused = fuse_prediction(direct_pred, x, local)
        assert torch.equal(fused, direct_pred)
        save_json(args.out_dir / "smoke.json", {"passed": True, "zero_init_exact": True, "shape": list(fused.shape), "local_input": "raw uv", "local_parameters": metadata["local_parameters"], "windows": len(ds)})
        return
    baseline0 = evaluate(model, builder, None, dev_paths, args, device, args.out_dir / "eval_00000_direct1500", save_predictions=False)
    history = [{"absolute_update": args.start_update, "model": "A1_at_zero_local", **baseline0["raw_errors"]}]
    train_ds, loader = core.loader(train_paths, args, shuffle=True)
    iterator = iter(loader); started = time.monotonic()
    eval_steps = set(args.milestones)
    for relative in range(1, args.updates + 1):
        try: x, y, _, _ = next(iterator)
        except StopIteration: iterator = iter(loader); x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.train(); local.train()
        prediction = forward(model, builder, x, local)
        parts = core.loss_parts(prediction, y)
        loss = sum(N2_WEIGHTS[name] * parts[name] for name in N2_WEIGHTS)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(local.parameters()), 1.0); optimizer.step()
        if relative in eval_steps or relative == args.updates:
            absolute = args.start_update + relative
            result = evaluate(model, builder, local, dev_paths, args, device, args.out_dir / f"eval_{absolute:05d}", save_predictions=relative == args.updates)
            history.append({"absolute_update": absolute, "model": "A1", **result["raw_errors"], "mean_t_neural_s": result["mean_t_neural_s"], "elapsed_seconds": time.monotonic() - started})
            torch.save({"model_state_dict": model.state_dict(), "local_state_dict": local.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": absolute, "metadata": metadata}, args.out_dir / f"model_update_{absolute:05d}.pth")
    torch.save({"model_state_dict": model.state_dict(), "local_state_dict": local.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": args.start_update + args.updates, "metadata": metadata}, args.out_dir / "model_last.pth")
    with (args.out_dir / "update_curve.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(history)
    save_json(args.out_dir / "summary.json", {"metadata": metadata | {"train_windows": len(train_ds), "elapsed_seconds": time.monotonic() - started}, "history": history})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--start-update", type=int, default=1500)
    parser.add_argument("--updates", type=int, default=RELATIVE_UPDATES); parser.add_argument("--milestones", type=int, nargs="+", default=[500, 1000, 1500])
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=None); parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--local-hidden", type=int, default=16); parser.add_argument("--restore-optimizer", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__": main()
