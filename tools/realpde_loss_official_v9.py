#!/usr/bin/env python3
"""Bounded Track 1 loss experiments scored with the official v9 scorer.

The tool operates only on user-supplied real HDF5 trajectories.  It creates
trajectory-disjoint manifests, trains a CNO continuation from a supplied
checkpoint, and writes official-v9 scorer subscores plus trajectory aggregates.
It intentionally has explicit paths for every external asset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from realpde_p0_data import H5WindowDataset, list_h5  # noqa: E402


VARIANTS = {
    "E0": {"rel": 0.0, "mse": 1.0, "tke": 0.05, "mvpe": 0.0, "mean": 0.0, "fluct": 0.0},
    "E1": {"rel": 1.0, "mse": 0.0, "tke": 0.05, "mvpe": 0.0, "mean": 0.0, "fluct": 0.0},
    "E2": {"rel": 1.0, "mse": 0.0, "tke": 0.05, "mvpe": 0.10, "mean": 0.0, "fluct": 0.0},
    "E3": {"rel": 1.0, "mse": 0.0, "tke": 0.05, "mvpe": 0.10, "mean": 0.05, "fluct": 0.10},
    "N0": {"rel": 0.0, "mse": 1.0, "tke": 0.05, "mvpe": 0.0, "mean": 0.0, "fluct": 0.0},
    "N1": {"rel": 0.0, "mse": 1.0, "tke": 0.05, "mvpe": 0.0, "mean": 0.0, "fluct": 0.0},
    "N2": {"rel": 0.0, "mse": 1.0, "tke": 0.05, "mvpe": 0.0, "mean": 0.0, "fluct": 0.0},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metadata(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        u = f["u"] if "u" in f else f["measured_data/u"]
        return {"file": path.name, "aoa": float(f["aoa"][()]), "re": float(f["re"][()]), "frames": int(u.shape[0])}


def make_manifest(real_root: Path, out: Path, seed: int, kind: str, ood_aoa: float) -> None:
    rows = [metadata(p) for p in list_h5(real_root)]
    rng = np.random.default_rng(seed)
    if kind == "id":
        groups: dict[float, list[dict]] = defaultdict(list)
        for row in rows:
            groups[row["aoa"]].append(row)
        train, dev, final = [], [], []
        for aoa in sorted(groups):
            group = groups[aoa]
            rng.shuffle(group)
            n = len(group)
            n_dev = max(1, round(n * 0.2))
            n_final = max(1, round(n * 0.2))
            n_train = n - n_dev - n_final
            if n_train < 1:
                raise ValueError(f"condition aoa={aoa} has too few trajectories")
            train += group[:n_train]
            dev += group[n_train:n_train + n_dev]
            final += group[n_train + n_dev:]
        result = {"kind": "id", "seed": seed, "real_root": str(real_root), "train": train, "dev": dev, "final": final}
    else:
        ood = [r for r in rows if np.isclose(r["aoa"], ood_aoa)]
        remain = [r for r in rows if not np.isclose(r["aoa"], ood_aoa)]
        rng.shuffle(remain)
        n_dev = max(1, round(len(remain) * 0.2))
        result = {"kind": "ood_aoa", "seed": seed, "ood_aoa": ood_aoa, "real_root": str(real_root), "train": remain[n_dev:], "dev": remain[:n_dev], "ood": ood}
    result["all_metadata"] = sorted(rows, key=lambda r: r["file"])
    json_dump(out, result)


def read_manifest(path: Path, split: str) -> tuple[dict, list[Path]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = Path(manifest["real_root"])
    if split not in manifest:
        raise KeyError(f"split {split!r} absent from {path}")
    return manifest, [root / row["file"] for row in manifest[split]]


def rel_loss(p: Tensor, y: Tensor) -> Tensor:
    p, y = p.reshape(p.shape[0], -1), y.reshape(y.shape[0], -1)
    return (torch.linalg.norm(p - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-8)).mean()


def tke_map(x: Tensor) -> Tensor:
    fluct = x[..., :2] - x[..., :2].mean(dim=1, keepdim=True)
    return 0.5 * fluct.square().mean(dim=1).sum(dim=-1)


def official_mvpe_loss(p: Tensor, y: Tensor) -> Tensor:
    """Differentiable counterpart of v9 scoring.py's 32x64 probe geometry."""
    _, _, h, w, _ = p.shape
    center_y, interval_y, d, center_x = 16, min(2, max(1, h // 10)), 16, 10
    ys = [z for z in range(center_y - interval_y * 4, center_y + interval_y * 5, interval_y) if 0 <= z < h]
    xs = [int(((i + 1) * d + center_x) / 2) for i in range(4)] if 21 < w else [int(0.5 * (i + 2) * d + center_x) for i in range(4)]
    terms = [rel_loss(p[:, :, ys, x, :2].mean(dim=1), y[:, :, ys, x, :2].mean(dim=1)) for x in xs if 0 <= x < w and ys]
    return torch.stack(terms).mean() if terms else p.new_tensor(0.0)


def loss_parts(pred: Tensor, target: Tensor) -> dict[str, Tensor]:
    uv, yuv = pred[..., :2], target[..., :2]
    return {
        "rel": rel_loss(uv, yuv),
        "mse": (uv - yuv).square().mean(),
        "tke": rel_loss(tke_map(uv), tke_map(yuv)),
        "mvpe": official_mvpe_loss(uv, yuv),
        "mean": rel_loss(uv.mean(dim=1), yuv.mean(dim=1)),
        "fluct": rel_loss(uv - uv.mean(dim=1, keepdim=True), yuv - yuv.mean(dim=1, keepdim=True)),
    }


def load_cno(kit_root: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(kit_root))
    from rpde_baselines.model.cno import CNO3d
    model = CNO3d(in_dim=3, out_dim=3, out_dim_mult=1, in_size=64, N_layers=3).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=True)
    return model


def forward(model: torch.nn.Module, x: Tensor) -> Tensor:
    return model(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)


def loader(paths: list[Path], args: argparse.Namespace, shuffle: bool) -> tuple[H5WindowDataset, DataLoader]:
    ds = H5WindowDataset(paths, max_windows_per_trajectory=args.max_windows)
    generator = torch.Generator().manual_seed(args.seed) if shuffle else None
    return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                          pin_memory=True, generator=generator, persistent_workers=False)


def score_bundle(kit_root: Path, pred: np.ndarray, target: np.ndarray, mean_t_neural_s: float, out_dir: Path) -> dict:
    """Run the supplied scoring.py program, then attach its raw error measures."""
    sys.path.insert(0, str(kit_root))
    import scoring
    raw = {
        "rel_l2": float(np.mean(scoring.rel_l2_per_sample(pred, target, scoring.measured_channels(target)))),
        "tke": float(np.mean(scoring.tke_rel_l2_per_sample(pred, target, scoring.measured_channels(target)))),
        "mvpe": float(scoring.mvpe_rel_l2(pred, target)),
    }
    with tempfile.TemporaryDirectory(prefix="v9_score_", dir=out_dir) as temp:
        temp_path = Path(temp)
        (temp_path / "ref").mkdir()
        np.savez_compressed(temp_path / "predictions.npz", prediction=pred, mean_t_neural_s=np.asarray([mean_t_neural_s], dtype=np.float32))
        np.savez_compressed(temp_path / "ref" / "targets.npz", target=target)
        scored = temp_path / "scored"
        subprocess.run([sys.executable, str(kit_root / "scoring.py"), str(temp_path), str(scored)], check=True)
        subscores = json.loads((scored / "scores.json").read_text(encoding="utf-8"))
    return {"raw_errors": raw, "official_v9_subscores": subscores, "mean_t_neural_s": mean_t_neural_s}


def trajectory_rows(ds: H5WindowDataset, pred: np.ndarray, target: np.ndarray, kit_root: Path) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(kit_root))
    import scoring
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, ref in enumerate(ds.refs):
        groups[ref.path.name].append((ref.start, index))
    rows, anatomy = [], []
    for name, refs in sorted(groups.items()):
        order = [i for _, i in sorted(refs)]
        p, y = np.concatenate([pred[i] for i in order], axis=0)[None], np.concatenate([target[i] for i in order], axis=0)[None]
        c = scoring.measured_channels(y)
        rel = float(scoring.rel_l2_per_sample(p, y, c)[0])
        tke = float(scoring.tke_rel_l2_per_sample(p, y, c)[0])
        mvpe = float(scoring.mvpe_rel_l2_per_sample(p, y)[0])
        kp, kt = scoring.kinetic_energy(p), scoring.kinetic_energy(y)
        a, b = kp[0].ravel(), kt[0].ravel()
        alpha = float(a.dot(b) / max(a.dot(a), 1e-12))
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")
        pf, yf = p[..., :2] - p[..., :2].mean(axis=1, keepdims=True), y[..., :2] - y[..., :2].mean(axis=1, keepdims=True)
        ku_p, ku_y = .5 * np.mean(pf[..., 0] ** 2, axis=1), .5 * np.mean(yf[..., 0] ** 2, axis=1)
        kv_p, kv_y = .5 * np.mean(pf[..., 1] ** 2, axis=1), .5 * np.mean(yf[..., 1] ** 2, axis=1)
        rows.append({"trajectory_id": name, "windows": len(order), "rel_l2": rel, "tke": tke, "mvpe": mvpe})
        anatomy.append({"trajectory_id": name, "tke_norm_ratio": float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-12)), "tke_sum_ratio": float(a.sum() / max(b.sum(), 1e-12)), "tke_rel_l2": tke, "tke_scale_alpha": alpha, "tke_scale_corrected_rel_l2": float(np.linalg.norm(alpha * a - b) / max(np.linalg.norm(b), 1e-12)), "tke_correlation": corr, "tke_cosine": float(a.dot(b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)), "u_tke_norm_ratio": float(np.linalg.norm(ku_p) / max(np.linalg.norm(ku_y), 1e-12)), "v_tke_norm_ratio": float(np.linalg.norm(kv_p) / max(np.linalg.norm(kv_y), 1e-12))})
    def summary(key: str) -> dict:
        values = np.asarray([x[key] for x in anatomy], float)
        return {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()), "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)), "p90": float(np.percentile(values, 90))}
    return rows, {k: summary(k) for k in anatomy[0] if k != "trajectory_id"}


@torch.no_grad()
def evaluate(model: torch.nn.Module, paths: list[Path], args: argparse.Namespace, device: torch.device, kit_root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds, data = loader(paths, args, shuffle=False)
    model.eval(); predictions, targets, elapsed = [], [], 0.0
    for x, y, _, _ in data:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter(); pred = forward(model, x)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        predictions.append(pred.cpu().numpy().astype(np.float32)); targets.append(y.numpy().astype(np.float32))
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    result = score_bundle(kit_root, pred, target, elapsed / len(ds), out)
    rows, anatomy = trajectory_rows(ds, pred, target, kit_root)
    result["windows"] = len(ds); result["trajectories"] = len(rows); result["trajectory_anatomy"] = anatomy
    with (out / "trajectory_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result


def gradient_audit(model: torch.nn.Module, paths: list[Path], args: argparse.Namespace, device: torch.device, weights: dict, out: Path) -> None:
    _, data = loader(paths, args, shuffle=False); rows, cosines = [], []
    params = [p for p in model.parameters() if p.requires_grad]
    for batch_id, (x, y, _, _) in enumerate(data):
        if batch_id >= args.gradient_batches: break
        x, y = x.to(device), y.to(device); pred = forward(model, x); part = loss_parts(pred, y); vectors = {}
        for name, value in part.items():
            grad = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
            flat = torch.cat([g.detach().reshape(-1) for g in grad if g is not None])
            vectors[name] = flat; norm = float(torch.linalg.norm(flat).cpu())
            rows.append({"batch_id": batch_id, "loss_name": name, "raw_gradient_norm": norm, "weighted_gradient_norm": abs(weights.get(name, 0.0)) * norm, "loss_value": float(value.detach().cpu())})
        for left, a in vectors.items():
            for right, b in vectors.items():
                cos = float(torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b)).clamp_min(1e-12))
                cosines.append({"batch_id": batch_id, "loss_a": left, "loss_b": right, "cosine": cos})
    for name, values in (("gradient_audit.csv", rows), ("gradient_cosines.csv", cosines)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed); out = args.out_dir
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    kit_root, checkpoint, manifest_path = args.kit_root.resolve(), args.checkpoint.resolve(), args.manifest.resolve()
    if not (kit_root / "scoring.py").is_file(): raise FileNotFoundError(kit_root / "scoring.py")
    manifest, train_paths = read_manifest(manifest_path, "train")
    _, eval_paths = read_manifest(manifest_path, args.eval_split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = json.loads(args.weights_json.read_text()) if args.weights_json else VARIANTS[args.variant]
    run_meta = {"variant": args.variant, "weights": weights, "seed": args.seed, "updates": args.updates, "batch_size": args.batch_size, "workers": args.workers, "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "kit_root": str(kit_root), "kit_scorer_sha256": sha256(kit_root / "scoring.py"), "device": str(device), "start_time": time.time(), "eval_split": args.eval_split}
    json_dump(out / "run_metadata.json", run_meta)
    model = load_cno(kit_root, checkpoint, device)
    if args.mode == "calibration":
        gradient_audit(model, train_paths, args, device, weights, out)
        rows = list(csv.DictReader((out / "gradient_audit.csv").open()))
        summary = {}
        for name in {row["loss_name"] for row in rows}:
            values = [float(row["raw_gradient_norm"]) for row in rows if row["loss_name"] == name]
            summary[name] = {"median_raw_gradient_norm": float(np.median(values)), "batches": len(values)}
        json_dump(out / "calibration_summary.json", {"metadata": run_meta, "gradient_norms": summary})
        return
    if args.mode == "audit":
        result = evaluate(model, eval_paths, args, device, kit_root, out)
        if not args.skip_gradient_audit:
            gradient_audit(model, eval_paths, args, device, weights, out)
        json_dump(out / "summary.json", {"metadata": run_meta, "evaluation": result})
        return
    train_ds, train_loader = loader(train_paths, args, shuffle=True)
    baseline = evaluate(model, eval_paths, args, device, kit_root, out / "eval_baseline")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history, latest_parts = [{"iteration": 0, **baseline["raw_errors"]}], {}
    best_key, best_iter = float("inf"), 0
    torch.save({"model_state_dict": model.state_dict(), "iteration": 0, "weights": weights}, out / "model_latest.pth")
    iterator = iter(train_loader)
    train_started = time.monotonic()
    stop_reason = "max_updates"
    for step in range(1, args.updates + 1):
        if args.max_train_seconds is not None and time.monotonic() - train_started >= args.max_train_seconds:
            stop_reason = "max_train_seconds"
            break
        try: x, y, _, _ = next(iterator)
        except StopIteration: iterator = iter(train_loader); x, y, _, _ = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.train(); parts = loss_parts(forward(model, x), y); loss = sum(weights[k] * v for k, v in parts.items())
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        latest_parts = {k: float(v.detach().cpu()) for k, v in parts.items()} | {"total": float(loss.detach().cpu())}
        if step % args.eval_interval == 0 or step == args.updates:
            dev = evaluate(model, eval_paths, args, device, kit_root, out / f"eval_{step:04d}")
            raw = dev["raw_errors"]; history.append({"iteration": step, **raw, "dev_rel_score": dev["official_v9_subscores"]["rel_l2_score"], "dev_tke_score": dev["official_v9_subscores"]["tke_score"], "dev_mvpe_score": dev["official_v9_subscores"]["mvpe_score"]})
            key = raw["tke"] if raw["rel_l2"] <= baseline["raw_errors"]["rel_l2"] * 1.02 and raw["mvpe"] <= baseline["raw_errors"]["mvpe"] * 1.02 else raw["tke"] + 1e6
            torch.save({"model_state_dict": model.state_dict(), "iteration": step, "weights": weights}, out / "model_latest.pth")
            if key < best_key:
                best_key, best_iter = key, step
                torch.save({"model_state_dict": model.state_dict(), "iteration": step, "weights": weights}, out / "model_best.pth")
    actual_updates = step if "step" in locals() else 0
    if history[-1]["iteration"] != actual_updates:
        dev = evaluate(model, eval_paths, args, device, kit_root, out / f"eval_{actual_updates:04d}")
        raw = dev["raw_errors"]; history.append({"iteration": actual_updates, **raw, "dev_rel_score": dev["official_v9_subscores"]["rel_l2_score"], "dev_tke_score": dev["official_v9_subscores"]["tke_score"], "dev_mvpe_score": dev["official_v9_subscores"]["mvpe_score"]})
        key = raw["tke"] if raw["rel_l2"] <= baseline["raw_errors"]["rel_l2"] * 1.02 and raw["mvpe"] <= baseline["raw_errors"]["mvpe"] * 1.02 else raw["tke"] + 1e6
        torch.save({"model_state_dict": model.state_dict(), "iteration": actual_updates, "weights": weights}, out / "model_latest.pth")
        if key < best_key:
            best_key, best_iter = key, actual_updates
            torch.save({"model_state_dict": model.state_dict(), "iteration": actual_updates, "weights": weights}, out / "model_best.pth")
    if not (out / "model_best.pth").exists(): shutil.copy2(out / "model_latest.pth", out / "model_best.pth")
    json_dump(out / "summary.json", {"metadata": run_meta | {"end_time": time.time(), "train_seconds": time.monotonic() - train_started, "train_windows": len(train_ds), "actual_updates": actual_updates, "stop_reason": stop_reason, "best_iteration": best_iter, "last_train_loss_parts": latest_parts}, "baseline": baseline, "history": history})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--make-manifest", action="store_true")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--manifest-kind", choices=("id", "ood"), default="id")
    p.add_argument("--real-root", type=Path, required=True)
    p.add_argument("--ood-aoa", type=float, default=20.0)
    p.add_argument("--mode", choices=("audit", "train", "calibration"), default="audit")
    p.add_argument("--kit-root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--variant", choices=tuple(VARIANTS), default="E0")
    p.add_argument("--weights-json", type=Path, help="JSON loss-weight mapping; overrides the named variant defaults.")
    p.add_argument("--eval-split", default="dev")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--max-train-seconds", type=float, default=None,
                   help="Optional wall-clock cap for the training loop; evaluation is excluded.")
    p.add_argument("--eval-interval", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--gradient-batches", type=int, default=3)
    p.add_argument("--skip-gradient-audit", action="store_true",
                   help="For retrying inference/scorer-only audits when the optional gradient audit OOMs.")
    p.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args()
    if args.make_manifest:
        if args.manifest is None: p.error("--manifest is required with --make-manifest")
        make_manifest(args.real_root, args.manifest, args.seed, args.manifest_kind, args.ood_aoa); return
    for arg in (args.manifest, args.kit_root, args.checkpoint, args.out_dir):
        if arg is None: p.error("--manifest, --kit-root, --checkpoint, and --out-dir are required")
    run(args)


if __name__ == "__main__":
    main()
