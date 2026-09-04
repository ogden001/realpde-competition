"""MF Energy-Aware Campaign 02: matched continuation and energy mechanisms."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

import realpde_loss_official_v9 as core
from realpde_mf01 import MF01CNO, build_features, factorized_reconstruct, init_mf_from_direct

BASE_N2 = {"mse": 1.0, "tke": 0.05, "rel": 0.027514, "mvpe": 0.009757}
GAIN_NAMES = ("u_history_std", "v_history_std", "delta_u", "delta_v", "history_tke_proxy")


def scorer_tke_map(x: Tensor) -> Tensor:
    fluct = x[..., :2] - x[..., :2].mean(dim=1, keepdim=True)
    return 0.5 * fluct.square().mean(dim=1).sum(dim=-1)


def relative_tke_map_loss(pred: Tensor, target: Tensor) -> Tensor:
    p, y = scorer_tke_map(pred), scorer_tke_map(target)
    return (torch.linalg.norm(p.flatten(1) - y.flatten(1), dim=1) /
            torch.linalg.norm(y.flatten(1), dim=1).clamp_min(1e-8)).mean()


def relative_rms_loss(pred: Tensor, target: Tensor) -> Tensor:
    p = torch.sqrt(scorer_tke_map(pred).clamp_min(0.0) + 1e-8)
    y = torch.sqrt(scorer_tke_map(target).clamp_min(0.0) + 1e-8)
    return (torch.linalg.norm(p.flatten(1) - y.flatten(1), dim=1) /
            torch.linalg.norm(y.flatten(1), dim=1).clamp_min(1e-8)).mean()


def high_energy_fluctuation_loss(pred: Tensor, target: Tensor) -> Tensor:
    pf = pred[..., :2] - pred[..., :2].mean(dim=1, keepdim=True)
    yf = target[..., :2] - target[..., :2].mean(dim=1, keepdim=True)
    k = scorer_tke_map(target[..., :2])
    w = (0.5 + k / (k.mean(dim=(-2, -1), keepdim=True) + 1e-8)).clamp(0.5, 3.0)
    w = w / (w.mean(dim=(-2, -1), keepdim=True) + 1e-8)
    w = w[:, None, :, :, None]
    num = (w * (pf - yf).square()).sum(dim=(1, 2, 3, 4))
    den = (w * yf.square()).sum(dim=(1, 2, 3, 4)).clamp_min(1e-8)
    return torch.sqrt(num / den).mean()


def set_frozen_gain_trainable(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("gain_head.")


def reconstruct_with_gain(mean_raw: Tensor, fluct_raw: Tensor, pressure: Tensor, gain_raw: Tensor):
    mean_field = mean_raw.mean(dim=1, keepdim=True).expand_as(mean_raw)
    fluctuation = fluct_raw - fluct_raw.mean(dim=1, keepdim=True)
    alpha = 1.0 + 0.20 * torch.tanh(gain_raw)
    return torch.cat((mean_field + alpha.permute(0, 2, 3, 1).unsqueeze(1) * fluctuation, pressure), dim=-1), alpha


class GainModel(nn.Module):
    def __init__(self, base: MF01CNO, feature_names: tuple[str, ...], mode: str):
        super().__init__(); self.base = base; self.mode = mode
        self.register_buffer("gain_indices", torch.tensor([feature_names.index(n) for n in GAIN_NAMES]), persistent=False)
        self.gain_head = nn.Linear(5, 1) if mode == "e7" else nn.Conv2d(5, 1, 1)
        nn.init.zeros_(self.gain_head.weight); nn.init.zeros_(self.gain_head.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.base.cno(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        gain_features = features.index_select(-1, self.gain_indices)
        if self.mode == "e7":
            gain_raw = self.gain_head(gain_features.mean(dim=(1, 2, 3))).reshape(-1, 1, 1, 1)
        else:
            gain_raw = self.gain_head(gain_features.mean(dim=1).permute(0, 3, 1, 2))
        prediction, alpha = reconstruct_with_gain(raw[..., :2], raw[..., 2:4], raw[..., 4:5], gain_raw)
        return prediction, alpha


def forward(model: nn.Module, builder, x: Tensor) -> tuple[Tensor, Tensor | None]:
    features = builder(x)
    if isinstance(model, GainModel):
        return model(features)
    raw = model.cno(features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
    uv, fluct = factorized_reconstruct(raw[..., :2], raw[..., 2:4])
    return torch.cat((uv, raw[..., 4:5]), dim=-1), None


@torch.no_grad()
def evaluate(model, builder, paths, args, device, kit_root, out):
    out.mkdir(parents=True, exist_ok=True); ds, loader = core.loader(paths, args, shuffle=False)
    preds, targets, alphas, elapsed = [], [], [], 0.0; model.eval()
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter(); pred, alpha = forward(model, builder, x)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        preds.append(pred.cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
        if alpha is not None: alphas.append(alpha.cpu().numpy().astype(np.float32).ravel())
    pred, target = np.concatenate(preds), np.concatenate(targets)
    result = core.score_bundle(kit_root, pred, target, elapsed / len(ds), out)
    rows, anatomy = core.trajectory_rows(ds, pred, target, kit_root)
    result.update({"windows": len(ds), "trajectories": len(rows), "trajectory_anatomy": anatomy})
    with (out / "trajectory_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out / "predictions.npz", prediction=pred, target=target)
    if alphas:
        a = np.concatenate(alphas); result["gain_stats"] = {k: float(v) for k, v in zip(("min", "p25", "median", "p75", "max", "std"), (a.min(), np.percentile(a, 25), np.median(a), np.percentile(a, 75), a.max(), a.std()))}
        np.save(out / "gain_values.npy", a)
    return result


def loss_for(mode: str, pred: Tensor, target: Tensor) -> Tensor:
    parts = core.loss_parts(pred, target)
    base = sum(BASE_N2[k] * parts[k] for k in BASE_N2)
    if mode == "e4": return base + 0.02 * relative_tke_map_loss(pred[..., :2], target[..., :2])
    if mode == "e5": return base + 0.02 * relative_rms_loss(pred[..., :2], target[..., :2])
    if mode == "e6": return base + 0.02 * high_energy_fluctuation_loss(pred, target)
    return base


def run(args):
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    core.set_seed(args.seed); args.out_dir.mkdir(parents=True)
    _, train_paths = core.read_manifest(args.manifest, "train"); _, dev_paths = core.read_manifest(args.manifest, "dev")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder, config = build_features(train_paths, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    base = MF01CNO(args.kit_root, len(builder.feature_names), device); base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = base if args.mode == "c0" else (GainModel(base, tuple(builder.feature_names), args.mode).to(device) if args.mode in ("e7", "e8") else base)
    if args.mode in ("e7", "e8"): set_frozen_gain_trainable(model)
    metadata = {"experiment_id": args.experiment_id, "mode": args.mode, "seed": args.seed, "updates": args.updates, "milestones": args.milestones, "lr": args.lr, "batch_size": args.batch_size, "workers": args.workers, "manifest_sha256": core.sha256(args.manifest), "checkpoint_sha256": core.sha256(args.checkpoint), "scorer_sha256": core.sha256(args.kit_root / "scoring.py"), "feature_config": vars(config), "locked_final_accessed": False, "codabench_accessed": False, "device": str(device)}
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    train_ds, loader = core.loader(train_paths, args, shuffle=True); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    if args.mode == "c0" and "optimizer_state_dict" in checkpoint: optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    baseline = evaluate(model, builder, dev_paths, args, device, args.kit_root, args.out_dir / "eval_00000")
    history = [{"iteration": 0, **baseline["raw_errors"], "official_v9_subscores": baseline["official_v9_subscores"]}]; iterator = iter(loader); started = time.monotonic()
    for step in range(1, args.updates + 1):
        try: x, y, _, _ = next(iterator)
        except StopIteration: iterator = iter(loader); x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); model.train(); pred, _ = forward(model, builder, x); loss = loss_for(args.mode, pred, y)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); optimizer.step()
        if step in args.milestones:
            ev = evaluate(model, builder, dev_paths, args, device, args.kit_root, args.out_dir / f"eval_{step:05d}"); history.append({"iteration": step, **ev["raw_errors"], "official_v9_subscores": ev["official_v9_subscores"], "elapsed_seconds": time.monotonic() - started}); print(json.dumps(history[-1], sort_keys=True), flush=True)
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": step, "metadata": metadata}, args.out_dir / f"model_update_{step:05d}.pth")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": args.updates, "metadata": metadata}, args.out_dir / "model_last.pth")
    (args.out_dir / "summary.json").write_text(json.dumps({"metadata": metadata | {"train_windows": len(train_ds), "elapsed_seconds": time.monotonic() - started}, "history": history}, indent=2, default=str), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=("c0", "e4", "e5", "e6", "e7", "e8"), required=True); p.add_argument("--experiment-id", required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--kit-root", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--updates", type=int, default=1500); p.add_argument("--milestones", type=int, nargs="+", default=[500, 1000, 1500]); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--workers", type=int, default=2); p.add_argument("--max-windows", type=int, default=None); p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--seed", type=int, default=20260901); run(p.parse_args())


if __name__ == "__main__": main()
